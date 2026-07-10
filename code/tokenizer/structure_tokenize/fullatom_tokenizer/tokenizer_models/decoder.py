import math
import torch
from torch import nn
import einops

from esm.utils.constants import esm3 as C
from esm.utils.structure.predicted_aligned_error import (
    compute_predicted_aligned_error,
    compute_tm,
)

from structure_tokenize.fullatom_tokenizer.tokenizer_models.layers.transformer_stack import VanillaTransformerStack


class VanillaRegressionHead(nn.Module):
    """
    Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/models/vqvae.py#L104
    """
    def __init__(self, embed_dim: int, output_dim: int):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = nn.GELU()
        self.norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, output_dim)

    def forward(self, features):
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.norm(x)
        x = self.output(x)
        return x


class VanillaCategoricalMixture:
    """
    Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/models/vqvae.py#L120C1-L146C1
    """
    def __init__(self, param, bins=50, start=0, end=1):
        # All tensors are of shape ..., bins.
        self.logits = param
        bins = torch.linspace(
            start, end, bins + 1, device=self.logits.device, dtype=torch.float32
        )
        self.v_bins = (bins[:-1] + bins[1:]) / 2

    def log_prob(self, true):
        # Shapes are:
        #     self.probs: ... x bins
        #     true      : ... (floating point # for target)
        true_index = (
            (true.unsqueeze(-1) - self.v_bins[[None] * true.ndim]).abs().argmin(-1)
        )
        nll = self.logits.log_softmax(-1)
        return torch.take_along_dim(nll, true_index.unsqueeze(-1), dim=-1).squeeze(-1)

    def mean(self):
        return (
            self.logits.to(self.v_bins.dtype).softmax(-1) @ self.v_bins.unsqueeze(1)
        ).squeeze(-1)

    def median(self):
        return self.v_bins[self.logits.max(-1).indices]


class VanillaPairwisePredictionHead(nn.Module):
    """
    Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/models/vqvae.py#L55
    """
    def __init__(
        self,
        input_dim: int,
        downproject_dim: int,
        hidden_dim: int,
        n_bins: int,
        bias: bool = True,
        pairwise_state_dim: int = 0,
    ):
        super().__init__()
        self.downproject = nn.Linear(input_dim, downproject_dim, bias=bias)
        self.linear1 = nn.Linear(
            downproject_dim + pairwise_state_dim, hidden_dim, bias=bias
        )
        self.activation_fn = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, n_bins, bias=bias)

    def forward(self, x, pairwise: torch.Tensor | None = None):
        """
        Args:
            x: [B x L x D]

        Output:
            [B x L x L x K]
        """
        x = self.downproject(x)
        # Let x_i be a vector of size (B, D).
        # Input is {x_1, ..., x_L} of size (B, L, D)
        # Output is 2D where x_ij = cat([x_i * x_j, x_i - x_j])
        q, k = x.chunk(2, dim=-1)

        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]
        x_2d = [
            prod,
            diff,
        ]
        if pairwise is not None:
            x_2d.append(pairwise)
        x = torch.cat(x_2d, dim=-1)
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.norm(x)
        x = self.linear2(x)
        return x


class VanillaStructureTokenDecoder(nn.Module):
    """
    Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/models/vqvae.py#L335
    """
    def __init__(self, **kwargs):
        super().__init__()

        encoder_d_out = kwargs["encoder_d_out"]
        d_model = kwargs["d_model"]
        n_heads = kwargs["n_heads"]
        n_layers = kwargs["n_layers"]
        self.use_backbone = kwargs.get("use_backbone", False)
        self.backbone_angle_dim = kwargs.get("backbone_angle_dim", 0)
        self.num_backbone_coord_freqs = kwargs.get("num_backbone_coord_freqs", 4)
        self.backbone_coord_freq_max = max(
            1.0, float(kwargs.get("backbone_coord_freq_max", 1000.0))
        )
        self.backbone_coord_dim = 9 if self.use_backbone else 0
        if self.use_backbone:
            coord_freqs = torch.logspace(
                0.0,
                math.log10(self.backbone_coord_freq_max),
                steps=self.num_backbone_coord_freqs,
                dtype=torch.float32,
            )
            self.register_buffer("coord_freqs", coord_freqs, persistent=False)
            coord_feat_dim = (
                self.backbone_coord_dim * self.num_backbone_coord_freqs * 2
            )
            self.backbone_coord_proj = nn.Linear(coord_feat_dim, encoder_d_out)

        self.vqvae_codebook_size = C.VQVAE_CODEBOOK_SIZE
        self.special_tokens = C.VQVAE_SPECIAL_TOKENS
        self.max_pae_bin = C.VQVAE_MAX_PAE_BIN
        self.seq_vocab_size = len(C.SEQUENCE_VOCAB)

        post_vq_in_dim = encoder_d_out + (
            self.backbone_angle_dim if self.use_backbone else 0
        )
        self.post_vq_proj = nn.Linear(post_vq_in_dim, d_model)
        self.decoder_stack = VanillaTransformerStack(
            d_model, n_heads, 1, n_layers, scale_residue=False, n_layers_geom=0
        )
        self.sequence_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, self.seq_vocab_size, bias=False),
        )
        self.struct_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, 37*3, bias=False),
        )
        # Chi angle prediction: 4 chi angles × 21 bins each
        self.chi_angle_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            #torch.nn.Linear(d_model, d_model*2),
            #torch.nn.GELU(),
            torch.nn.Linear(d_model, 4 * 21),
        )

        pae_bins = C.VQVAE_PAE_BINS
        self.pairwise_bins = [
            64,  # distogram
            pae_bins,  # predicted aligned error
        ]
        self.pairwise_classification_head = VanillaPairwisePredictionHead(
            d_model,
            downproject_dim=128,
            hidden_dim=128,
            n_bins=sum(self.pairwise_bins),
            bias=False,
        )

        plddt_bins = C.VQVAE_PLDDT_BINS
        self.plddt_head = VanillaRegressionHead(
            embed_dim=d_model, output_dim=plddt_bins
        )

    def decode(
        self,
        quantized_z: torch.Tensor,
        structure_tokens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        backbone_coords: torch.Tensor | None = None,
        backbone_mask: torch.Tensor | None = None,
        backbone_angles: torch.Tensor | None = None,
        skip_pairwise: bool = False,
    ):
        if sequence_id is None:
            sequence_id = torch.zeros_like(structure_tokens, dtype=torch.int64)
        # not supported for now
        chain_id = torch.zeros_like(structure_tokens, dtype=torch.int64)

        assert (
            (structure_tokens < 0).sum() == 0
        ), "All structure tokens set to -1 should be replaced with BOS, EOS, PAD, or MASK tokens by now, but that isn't the case!"

        decoder_input = quantized_z
        if self.use_backbone:
            if (
                backbone_coords is None
                or backbone_mask is None
                or backbone_angles is None
            ):
                raise ValueError(
                    "Backbone coordinates, mask, and angles must be provided when decoder expects backbone input."
                )
            coord_embedding = self._sinusoidal_backbone_embedding(backbone_coords)
            coord_embedding = coord_embedding.masked_fill(
                ~backbone_mask.unsqueeze(-1), 0.0
            )
            base_input = quantized_z + coord_embedding
            if backbone_angles.shape[-1] != self.backbone_angle_dim:
                raise ValueError(
                    f"Backbone angle dim mismatch: expected {self.backbone_angle_dim}, "
                    f"got {backbone_angles.shape[-1]}"
                )
            angle_features = backbone_angles.masked_fill(
                ~backbone_mask.unsqueeze(-1), 0.0
            )
            decoder_input = torch.cat([base_input, angle_features], dim=-1)

        x = self.post_vq_proj(decoder_input) # [B, L, hidden_dim=128] -> [B, L, d_model=1024]
        # Attention mask is applied inside VanillaMultiHeadAttention via sequence_id path
        x, _ = self.decoder_stack.forward(
            x,
            attention_mask=attention_mask,
            affine=None,
            affine_mask=None,
            sequence_id=sequence_id,
            chain_id=chain_id
        ) # [B, L, d_model], [B, L, d_model]

        seq_logits = self.sequence_proj(x)

        coord_flat = self.struct_proj(x)
        coord_pred = einops.rearrange(
            coord_flat, "b n (a t) -> b n a t", a=37, t=3
        )  # [b, n, 37, 3]

        chi_logits = self.chi_angle_proj(x)  # [b, n, 84]
        chi_logits = chi_logits.view(chi_logits.shape[0], chi_logits.shape[1], 4, 21)  # [b, n, 4, 21]

        # --- Pairwise heads: PAE, pTM, distogram, pLDDT ---
        # These are O(L²) and NOT used in the training loss.
        # Skip during training to save significant compute and memory.
        if skip_pairwise:
            return dict(
                coord_pred=coord_pred,
                plddt=None,
                ptm=None,
                predicted_aligned_error=None,
                pairwise_dist_logits=None,
                seq_logits=seq_logits,
                chi_logits=chi_logits,
            )

        aatype_max = torch.argmax(seq_logits, dim=-1)  # [b, n]
        aatype_max = aatype_max * attention_mask  # [b, n]

        pae, ptm = None, None
        pairwise_logits = self.pairwise_classification_head(x) # [B, L, L, 64 + 96 + 64]
        pairwise_dist_logits, pae_logits = [
            (o if o.numel() > 0 else None)
            for o in pairwise_logits.split(self.pairwise_bins, dim=-1)
        ] # [B, L, L, 64], [B, L, L, 96], [B, L, L, 64]

        special_tokens_mask = structure_tokens >= min(self.special_tokens.values())
        pae = compute_predicted_aligned_error(
            pae_logits,  # type: ignore
            aa_mask=~special_tokens_mask,
            sequence_id=sequence_id,
            max_bin=self.max_pae_bin,
        ) # [B, L, L]
        # This might be broken for chainbreak tokens? We might align to the chainbreak
        ptm = compute_tm(
            pae_logits,  # type: ignore
            aa_mask=~special_tokens_mask,
            max_bin=self.max_pae_bin,
        ) # [B,]

        plddt_logits = self.plddt_head(x) # [B, L, 50]
        plddt_value = VanillaCategoricalMixture(
            plddt_logits, bins=plddt_logits.shape[-1]
        ).mean() # [B, L]

        return dict(
            coord_pred=coord_pred,
            plddt=plddt_value,
            ptm=ptm,
            predicted_aligned_error=pae,
            pairwise_dist_logits=pairwise_dist_logits,
            seq_logits=seq_logits,
            chi_logits=chi_logits,
        )

    def _sinusoidal_backbone_embedding(self, backbone_coords: torch.Tensor) -> torch.Tensor:
        """
        backbone_coords: [B, L, 3, 3]
        """
        if not hasattr(self, "coord_freqs"):
            raise ValueError("Coordinate frequencies are not initialized for backbone-aware decoding.")
        b, l = backbone_coords.shape[0], backbone_coords.shape[1]
        coords_flat = backbone_coords.reshape(b, l, -1)
        freqs = self.coord_freqs.to(backbone_coords.dtype).view(1, 1, 1, -1)
        coords_scaled = coords_flat.unsqueeze(-1) * freqs
        sin_feat = torch.sin(coords_scaled)
        cos_feat = torch.cos(coords_scaled)
        feat = torch.cat([sin_feat, cos_feat], dim=-1)
        feat = feat.reshape(b, l, -1)
        return self.backbone_coord_proj(feat)