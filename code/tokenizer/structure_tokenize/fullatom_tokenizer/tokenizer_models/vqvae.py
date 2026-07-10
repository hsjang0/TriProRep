"""
Code adopted from StructTokenBench (https://github.com/KatarinaYuan/StructTokenBench.git).
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from openfold.np.residue_constants import restypes as openfold_restypes

from esm.utils.structure.protein_structure import infer_cbeta_from_atom37
from esm.utils.constants import esm3 as C
from structure_tokenize.fullatom_tokenizer.utils.protein_chain import WrappedProteinChain
from structure_tokenize.fullatom_tokenizer.tokenizer_models.fullatom_image_encoder import AtomisticImageEncoder
from structure_tokenize.fullatom_tokenizer.tokenizer_models.decoder import VanillaStructureTokenDecoder
from structure_tokenize.fullatom_tokenizer.tokenizer_models.quantizers import *
from structure_tokenize.fullatom_tokenizer.utils.angle_utils import bond_angles
from structure_tokenize.fullatom_tokenizer.utils.coord_utils import compute_residue_frames, canonicalize_residue_coords, uncanonicalize_residue_coords


class VQVAEModel(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.model_cfg = model_cfg
        quantizer_cfg = model_cfg.quantizer
        self.loss_weight = quantizer_cfg["loss_weight"]
        self.inverse_folding_loss_weight = self.loss_weight.get("inverse_folding_loss_weight", 0.0)
        self.chi_angle_loss_weight = self.loss_weight.get("chi_angle_loss_weight", 0.0)
        self.quantizer = eval(quantizer_cfg["quantizer_type"])(**quantizer_cfg)

        if model_cfg.encoder.name == "AtomisticImageEncoder":
            self.encoder = AtomisticImageEncoder(**model_cfg.encoder)
        else:
            raise ValueError(f"Unsupported encoder for inference: {model_cfg.encoder.name}")

        self.use_canonical_frames = model_cfg.encoder.get("use_canonical_frames", False)

        # When canonical frames are enabled, the decoder does not need backbone
        # conditioning — the codebook encodes local geometry and the decoder
        # predicts directly in the canonical (local-frame) coordinate space.
        self.encoder_requires_backbone = (
            model_cfg.encoder.name in {"AtomisticLocalEncoder", "AtomisticImageEncoder"}
            and not self.use_canonical_frames
        )
        self.backbone_angle_dim = 63 if self.encoder_requires_backbone else 0

        model_cfg.decoder["encoder_d_out"] = model_cfg.encoder.d_out
        model_cfg.decoder["use_backbone"] = self.encoder_requires_backbone
        model_cfg.decoder["backbone_angle_dim"] = self.backbone_angle_dim
        self.decoder = VanillaStructureTokenDecoder(**model_cfg.decoder)

        self._step_count = 0
        self._eval_every_n = 50  # RMSD/lDDT only every N steps (expensive CPU ops)

        # --- Residue subsampling config ---
        # During training, randomly select at most this many residues per protein
        # to reduce O(L) encoder and O(L²) decoder cost.
        # Set to 0 or None to disable.
        self.max_train_residues = model_cfg.get("max_train_residues", 0)

        # --- Optional: freeze everything except chi angle head ---
        self.freeze_except_chi_head = model_cfg.get("freeze_except_chi_head", False)
        if self.freeze_except_chi_head:
            for param in self.parameters():
                param.requires_grad = False
            for param in self.decoder.chi_angle_proj.parameters():
                param.requires_grad = True

        # Pre-build aatype lookup table on CPU (avoid rebuilding every step)
        self._restype_lut = torch.zeros(128, dtype=torch.long)  # ASCII range
        for idx, aa in enumerate(openfold_restypes):
            self._restype_lut[ord(aa)] = idx

    def _fast_aatype_encode(self, sequences, shape, device):
        """Vectorized aatype encoding — no Python loop per residue."""
        B, L = shape
        aatypes = torch.zeros(B, L, dtype=torch.long, device=device)
        lut = self._restype_lut.to(device)
        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), L)
            if seq_len:
                ascii_codes = torch.tensor(
                    [ord(c) for c in seq[:seq_len]], dtype=torch.long, device=device
                )
                aatypes[i, :seq_len] = lut[ascii_codes]
        return aatypes

    # ------------------------------------------------------------------
    # Residue subsampling: select N random valid residues per sample.
    # Uses topk (fixed output shape) so the graph is compile-friendly.
    # ------------------------------------------------------------------
    def _subsample_residues(self, coords, attention_mask, residue_index,
                            seq_residue_tokens, aatypes,
                            bb_angles=None, sc_angles=None,
                            decoder_bb_angles=None):
        """Randomly subsample residues for training efficiency.

        Returns subsampled tensors with L' = min(L, max_train_residues).
        Indices are sorted to preserve sequential order.

        When bb_angles / sc_angles are provided (pre-computed on the full
        sequence), they are subsampled with the same indices so that
        cross-residue backbone angles remain correct.
        """
        B, L = attention_mask.shape
        N = self.max_train_residues
        if N <= 0 or L <= N:
            return coords, attention_mask, residue_index, seq_residue_tokens, aatypes, bb_angles, sc_angles, decoder_bb_angles

        # Random scores; invalid positions get -1 so they're never selected
        scores = torch.rand(B, L, device=coords.device)
        scores = scores.masked_fill(~attention_mask.bool(), -1.0)

        # Select top-N valid residues by random score
        _, indices = scores.topk(N, dim=1, sorted=False)
        indices = indices.sort(dim=1).values  # preserve sequential order

        # Gather all tensors along the residue dimension
        idx_1d = indices                                          # [B, N]
        idx_3d = indices[:, :, None, None].expand(-1, -1, 37, 3) # [B, N, 37, 3]

        coords = coords.gather(1, idx_3d)
        attention_mask = attention_mask.gather(1, idx_1d)
        residue_index = residue_index.gather(1, idx_1d)
        seq_residue_tokens = seq_residue_tokens.gather(1, idx_1d)
        aatypes = aatypes.gather(1, idx_1d)

        if bb_angles is not None:
            idx_bb = indices[:, :, None].expand(-1, -1, bb_angles.shape[-1])
            bb_angles = bb_angles.gather(1, idx_bb)
        if sc_angles is not None:
            idx_sc = indices[:, :, None].expand(-1, -1, sc_angles.shape[-1])
            sc_angles = sc_angles.gather(1, idx_sc)
        if decoder_bb_angles is not None:
            idx_dec = indices[:, :, None].expand(-1, -1, decoder_bb_angles.shape[-1])
            decoder_bb_angles = decoder_bb_angles.gather(1, idx_dec)

        return coords, attention_mask, residue_index, seq_residue_tokens, aatypes, bb_angles, sc_angles, decoder_bb_angles

    def forward(self, input_list, use_as_tokenizer=False):
        """
        coords: [B, L, 37, 3]
        attention_mask: [B, L]
        residue_index: [B, L]
        seq_residue_tokens: [B, L]
        """
        self._step_count += 1

        coords, attention_mask, residue_index, seq_residue_tokens, pdb_chain = input_list

        # --- aatype preparation (vectorized) ---
        sequences = [pdb_chain[i].sequence for i in range(len(pdb_chain))]
        aatypes = self._fast_aatype_encode(
            sequences, residue_index.shape, seq_residue_tokens.device
        )
        sequence_id = None

        # --- residue subsampling (training only) ---
        # Pre-compute angles on FULL sequence so cross-residue backbone
        # angles (theta_2, theta_3) use true consecutive neighbours,
        # then subsample both coords and angles with the same indices.
        subsampled = False
        precomputed_angles = None
        precomputed_decoder_bb_angles = None
        if self.training and self.max_train_residues > 0 and coords.shape[1] > self.max_train_residues:
            bb_angles, sc_angles = self.encoder.precompute_angles(
                coords, attention_mask, aatypes, residue_index
            )
            # Also pre-compute decoder backbone angles on full sequence (same
            # cross-residue theta_2/theta_3 issue applies to the decoder).
            dec_bb_angles = None
            if self.encoder_requires_backbone:
                full_atom_mask = torch.all(
                    torch.isfinite(coords) & (coords < 1e6), dim=-1,
                ) & attention_mask[..., None]
                dec_bb_angles = self._compute_backbone_angles(
                    coords, full_atom_mask, residue_index
                )
            coords, attention_mask, residue_index, seq_residue_tokens, aatypes, \
                bb_angles, sc_angles, dec_bb_angles = self._subsample_residues(
                    coords, attention_mask, residue_index,
                    seq_residue_tokens, aatypes,
                    bb_angles, sc_angles, dec_bb_angles,
                )
            precomputed_angles = (bb_angles, sc_angles)
            precomputed_decoder_bb_angles = dec_bb_angles
            subsampled = True

        # --- encoder ---
        z = self.encoder.encode(
            coords, attention_mask, sequence_id, aatypes, residue_index,
            precomputed_angles=precomputed_angles,
        )

        # --- quantizer ---
        quantized_z, quantized_indices, partial_loss, partial_metrics = self.quantizer(z)

        if use_as_tokenizer:
            return quantized_z, quantized_indices, z

        # --- codebook utilization (cheap, every step) ---
        with torch.no_grad():
            valid_indices = quantized_indices[attention_mask.bool()]
            codebook_metrics = get_codebook_utility(
                valid_indices.flatten(), self.quantizer.codebook.weight
            )

        # --- mask & canonical frame prep ---
        atom_mask = torch.all(
            torch.isfinite(coords) & (coords < 1e6),
            dim=-1,
        )  # [B, L, 37] boolean
        atom_mask = atom_mask & attention_mask[..., None]

        coords_masked = coords.masked_fill(~atom_mask.unsqueeze(-1), 0.0)
        residue_frames = None
        if self.use_canonical_frames:
            residue_frames = compute_residue_frames(coords_masked, atom_mask)
            coords_target = canonicalize_residue_coords(
                coords_masked, atom_mask, frames=residue_frames
            )
        else:
            # Per-residue CA centering: decoder predicts local geometry relative
            # to CA, not global coordinates. This removes global translation from
            # the loss and matches what the encoder sees (coords - CA).
            ca_coords = coords_masked[:, :, 1:2, :]  # [B, L, 1, 3]
            coords_target = coords_masked - ca_coords
            coords_target = coords_target.masked_fill(~atom_mask.unsqueeze(-1), 0.0)

        backbone_coords = None
        backbone_mask = None
        backbone_angles = None
        if self.encoder_requires_backbone:
            backbone_coords, backbone_mask, backbone_angles = self._prepare_backbone_features(
                coords, atom_mask, attention_mask, residue_index,
                precomputed_backbone_angles=precomputed_decoder_bb_angles,
            )

        # --- decoder ---
        # During training, skip the expensive O(L²) pairwise head (PAE, pTM,
        # distogram, pLDDT) — none of these contribute to the training loss.
        decoded_states = self.decoder.decode(
            quantized_z,
            quantized_indices,
            attention_mask,
            sequence_id,
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
            backbone_angles=backbone_angles,
            skip_pairwise=self.training,
        )

        # reconstructed proteins
        coords_pred = decoded_states["coord_pred"]

        # For evaluation metrics, transform predictions back to global space
        if self.use_canonical_frames and residue_frames is not None:
            R, origin, backbone_valid = residue_frames
            coords_pred_global = uncanonicalize_residue_coords(
                coords_pred, R, origin, backbone_valid, atom_mask
            )
        else:
            # Undo per-residue CA centering: add back CA positions
            ca_coords = coords_masked[:, :, 1:2, :]  # [B, L, 1, 3]
            coords_pred_global = coords_pred + ca_coords
            coords_pred_global = coords_pred_global.masked_fill(~atom_mask.unsqueeze(-1), 0.0)

        # --- RMSD / lDDT evaluation (CPU-bound, only every N steps) ---
        # Skip entirely when residues are subsampled (pdb_chain lengths won't match)
        if self.training and (subsampled or self._step_count % self._eval_every_n != 0):
            rmsd_val = torch.tensor(float("nan"), device=coords.device)
            lddt_val = torch.tensor(float("nan"), device=coords.device)
        else:
            rmsd_list, lddt_list = [], []
            for i in range(len(coords_pred_global)):
                pdb_chain_recon = WrappedProteinChain.from_atom37(
                    coords_pred_global[i].detach()
                )
                pdb_chain_recon = pdb_chain_recon[:len(pdb_chain[i])]
                coord_rmsd = pdb_chain_recon.rmsd(
                    pdb_chain[i], only_compute_backbone_rmsd=False
                )
                lddt = np.array(pdb_chain_recon.lddt_ca(pdb_chain[i]))
                rmsd_list.append(coord_rmsd)
                lddt_list.append(lddt.mean())
            rmsd_val = torch.tensor(rmsd_list, device=coords.device).mean()
            lddt_val = torch.tensor(lddt_list, device=coords.device).mean()

        # --- loss computation ---
        struct_recon_loss, struct_recon_metric = self.compute_structure_loss(
            coords_pred, coords_target, attention_mask, atom_mask
        )
        geom_dist_loss, geom_dist_metrics = self.compute_geometric_distance(
            coords_pred, coords_target, atom_mask
        )
        reconstruction_loss = (struct_recon_loss + geom_dist_loss).mean()

        if self.inverse_folding_loss_weight > 0:
            inverse_folding_loss, inverse_folding_metrics = self.compute_inverse_folding(
                decoded_states["seq_logits"], seq_residue_tokens, attention_mask
            )
            reconstruction_loss = reconstruction_loss + self.inverse_folding_loss_weight * inverse_folding_loss.mean()
        else:
            inverse_folding_metrics = {}

        if self.chi_angle_loss_weight > 0:
            # Compute ground truth chi angles from input coords
            if precomputed_angles is not None:
                sc_angles_gt = precomputed_angles[1]
            else:
                coords_prep, mask_prep, _ = self.encoder._prepare_inputs(coords)
                mask_prep = mask_prep & attention_mask.bool().unsqueeze(-1)
                sc_angles_gt = self.encoder.get_sidechain_angles(coords_prep, aatypes, mask_prep)

            chi_angle_loss, chi_angle_metrics = self.compute_chi_angle_loss(
                decoded_states["chi_logits"], sc_angles_gt, attention_mask
            )
            reconstruction_loss = reconstruction_loss + self.chi_angle_loss_weight * chi_angle_loss
        else:
            chi_angle_metrics = {}

        loss = reconstruction_loss * self.loss_weight["reconstruction_loss_weight"] + partial_loss

        metrics = {
            **struct_recon_metric,
            **geom_dist_metrics,
            **inverse_folding_metrics,
            **chi_angle_metrics,
            **partial_metrics,
            **{f"codebook_{k}": v for k, v in codebook_metrics.items()},
            "reconstruction_loss": reconstruction_loss,
            "rmsd": rmsd_val,
            "lddt": lddt_val,
        }
        loss_and_metrics = (loss, metrics)

        return (loss_and_metrics, )

    def compute_structure_loss(self, x_recon, x, mask, atom_mask):
        """
        x_recon: [B, L, 37, 3]
        x: [B, L, 37, 3]
        mask: [B, L]
        atom_mask: [B, L, 37]
        """
        # keep only valid atoms/residues
        atom_mask = atom_mask & mask[..., None]
        x_recon = x_recon.masked_fill(~atom_mask.unsqueeze(-1), 0.)
        x = x.masked_fill(~atom_mask.unsqueeze(-1), 0.)

        # squared distance per atom
        sq_err = (x_recon - x) ** 2  # [B, L, 37, 3]
        sq_err = sq_err.sum(dim=-1)  # [B, L, 37]
        sq_err = sq_err * atom_mask

        num_atoms = atom_mask.sum().clamp(min=1)
        loss = sq_err.sum() / num_atoms

        metric = {
            "struct_recon_loss": loss,
            "struct_recon_rmsd": torch.sqrt(loss + 1e-8),
        }
        return loss, metric

    def compute_ca_geometric_distance(self, x_recon, x, residue_mask, clamp_value=50):
        """
        x_recon: [B, L, 3]
        x: [B, L, 3]
        residue_mask: [B, L]
        """

        # ignore padding regions
        valid_residue_mask = residue_mask[..., None]
        x_recon = x_recon.masked_fill(~valid_residue_mask, 0.0)
        x = x.masked_fill(~valid_residue_mask, 0.0)

        dist_pred = torch.cdist(x_recon, x_recon, p=2.0) # [B, L, L]
        dist_true = torch.cdist(x, x, p=2.0) # [B, L, L]

        dist_mask = residue_mask.unsqueeze(1) * residue_mask.unsqueeze(2) # [B, L, L]
        dist_pred = dist_pred[dist_mask]
        dist_true = dist_true[dist_mask]
        loss = F.mse_loss(dist_pred, dist_true, reduction="none") # flattened
        loss = torch.clamp(loss, max=clamp_value)
        metric = {
            f"ca_geom_dist_loss": loss.mean(),
            f"ca_geom_dist_loss_below_clamp": loss[loss != clamp_value].mean(),
            f"ca_geom_dist_loss_clamp_ratio_{clamp_value}": (loss != clamp_value).float().mean(),
        }
        # metrics like spearman R is too time consuming to calculate
        return loss.mean(), metric

    def compute_geometric_distance(self, x_recon, x, atom_mask, clamp_value=50):
        """
        x_recon: [B, L, 37, 3]
        x: [B, L, 37, 3]
        atom_mask: [B, L, 37]
        """

        # ignore padding regions
        valid_atom_mask = atom_mask[..., None]
        x_recon = x_recon.masked_fill(~valid_atom_mask, 0.0)
        x = x.masked_fill(~valid_atom_mask, 0.0)

        dist_pred = torch.cdist(x_recon, x_recon, p=2.0) # [B, L, 37, 37]
        dist_true = torch.cdist(x, x, p=2.0) # [B, L, 37, 37]

        dist_mask = atom_mask.unsqueeze(2) * atom_mask.unsqueeze(3) # [B, L, 37, 37]
        dist_pred = dist_pred[dist_mask]
        dist_true = dist_true[dist_mask]
        loss = F.mse_loss(dist_pred, dist_true, reduction="none") # flattened
        loss = torch.clamp(loss, max=clamp_value)
        metric = {
            f"geom_dist_loss": loss.mean(),
            f"geom_dist_loss_below_clamp": loss[loss != clamp_value].mean(),
            f"geom_dist_loss_clamp_ratio_{clamp_value}": (loss != clamp_value).float().mean(),
        }
        # metrics like spearman R is too time consuming to calculate
        return loss.mean(), metric

    def compute_binned_distance(self, pairwise_logits, coords, attention_mask):
        """
        pairwise_logits: [B, L, L, 64]
        coords: [B, L, 37, 3]
        attention_mask: [B, L]
        """

        # calculate Cbeta
        cbeta = infer_cbeta_from_atom37(coords) # [B, L, 3]

        # pairwise Cbeta distance
        NUM_BIN = 64
        dist_true = torch.cdist(cbeta, cbeta, p=2.0)
        bin_edges = [0] + [(2.3125 + 0.3075 * i) ** 2 for i in range(NUM_BIN)]
        bin_edges = torch.tensor(bin_edges, device=pairwise_logits.device)
        binned_labels = torch.bucketize(dist_true, bin_edges, right=True) - 1 # [B, L, L]
        binned_labels = torch.clamp(binned_labels, max=NUM_BIN - 1, min=0)
        assert binned_labels.min() >= 0 and binned_labels.max() < NUM_BIN

        mask = torch.logical_and(attention_mask.unsqueeze(-1), attention_mask.unsqueeze(1)) # [B, L, L]
        pairwise_logits, binned_labels = pairwise_logits[mask], binned_labels[mask]

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(pairwise_logits, binned_labels)

        metric = {
            f"binned_dist_loss": loss.mean(),
            f"binned_dist_accuracy": (pairwise_logits.argmax(dim=-1) == binned_labels).float().mean(),
        }
        return loss.mean(), metric

    def compute_chi_angle_loss(self, chi_logits, sc_angles_gt, attention_mask):
        """
        chi_logits: [B, L, 4, 21] predicted logits from decoder
        sc_angles_gt: [B, L, 88] ground truth (4×21 one-hot + 4 mask)
        attention_mask: [B, L]
        """
        B, L = attention_mask.shape
        # Extract targets: one-hot [B, L, 4, 21] → bin indices [B, L, 4]
        chi_onehot = sc_angles_gt[:, :, :84].view(B, L, 4, 21)
        chi_targets = chi_onehot.argmax(dim=-1)  # [B, L, 4]
        # Extract mask: which chi angles exist for each residue
        chi_mask = sc_angles_gt[:, :, 84:88]  # [B, L, 4]
        combined_mask = chi_mask * attention_mask.unsqueeze(-1).float()  # [B, L, 4]

        # Cross-entropy loss
        logits_flat = chi_logits.reshape(-1, 21)  # [B*L*4, 21]
        targets_flat = chi_targets.reshape(-1)  # [B*L*4]
        ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")  # [B*L*4]
        ce_loss = ce_loss.view(B, L, 4)

        num_valid = combined_mask.sum().clamp(min=1)
        loss = (ce_loss * combined_mask).sum() / num_valid

        # Accuracy
        preds = chi_logits.argmax(dim=-1)  # [B, L, 4]
        correct = (preds == chi_targets).float() * combined_mask
        accuracy = correct.sum() / num_valid

        metric = {
            "chi_angle_loss": loss,
            "chi_angle_accuracy": accuracy,
        }
        return loss, metric

    def compute_inverse_folding(self, logits, residue_labels, attention_mask):
        """
        h: [B, L, d_model=1024]
        residue_labels: [B, L]
        attention_mask: [B, L]
        """
        if not (logits.shape[0] == attention_mask.shape[0] and logits.shape[1] == attention_mask.shape[1]):
            raise ValueError

        logits, residue_labels = logits[attention_mask], residue_labels[attention_mask]

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits, residue_labels)

        metric = {
            f"inverse_folding_loss": loss.mean(),
            f"inverse_folding_accuracy": (logits.argmax(dim=-1) == residue_labels).float().mean(),
        }
        return loss.mean(), metric

    def _prepare_backbone_features(self, coords, atom_mask, attention_mask, residue_index,
                                    precomputed_backbone_angles=None):
        coords = coords.masked_fill(~atom_mask.unsqueeze(-1), 0.0)
        backbone_mask = atom_mask[:, :, :3].all(dim=-1)
        backbone_mask = backbone_mask & attention_mask
        backbone_coords = coords[:, :, :3, :].masked_fill(
            ~backbone_mask[..., None, None],
            0.0,
        )
        if precomputed_backbone_angles is not None:
            backbone_angles = precomputed_backbone_angles
        else:
            backbone_angles = self._compute_backbone_angles(coords, atom_mask, residue_index)
        backbone_angles = backbone_angles.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
        backbone_angles = backbone_angles.masked_fill(~backbone_mask.unsqueeze(-1), 0.0)
        return backbone_coords, backbone_mask, backbone_angles

    def _compute_backbone_angles(self, coords, atom_mask, residue_index):
        """
        coords: [B, L, 37, 3]
        atom_mask: [B, L, 37]
        residue_index: [B, L]
        """
        b, l = coords.shape[0], coords.shape[1]
        if residue_index is None:
            residue_idx = torch.arange(l, device=coords.device).unsqueeze(0).expand(b, -1)
        else:
            residue_idx = residue_index.to(coords.device)
        residue_idx = residue_idx.long()

        N = coords[:, :, 0, :]
        CA = coords[:, :, 1, :]
        C = coords[:, :, 2, :]

        theta_1 = bond_angles(N, CA, C)
        theta_2 = bond_angles(CA[:, :-1, :], C[:, :-1, :], N[:, 1:, :])
        theta_3 = bond_angles(C[:, :-1, :], N[:, 1:, :], CA[:, 1:, :])

        valid_residue_mask = atom_mask[:, :, :3].all(dim=-1)
        residue_delta = residue_idx[:, 1:] - residue_idx[:, :-1]
        good_pair = residue_delta == 1
        pair_mask = valid_residue_mask[:, :-1] & valid_residue_mask[:, 1:] & good_pair
        theta_1 = theta_1.masked_fill(~valid_residue_mask, 0.0)
        theta_2 = theta_2.masked_fill(~pair_mask, 0.0)
        theta_3 = theta_3.masked_fill(~pair_mask, 0.0)

        zero_pad = torch.zeros((b, 1), device=coords.device, dtype=coords.dtype)
        theta_2 = torch.cat([theta_2, zero_pad], dim=-1)
        theta_3 = torch.cat([theta_3, zero_pad], dim=-1)

        bb_angles = torch.stack([theta_1, theta_2, theta_3], dim=-1)
        bin_limits = torch.linspace(-torch.pi, torch.pi, 20, device=coords.device, dtype=coords.dtype)
        bin_indices = torch.bucketize(bb_angles, bin_limits)
        angles_feat = F.one_hot(bin_indices, len(bin_limits) + 1).float()
        angles_feat = angles_feat.reshape(b, l, -1)

        valid_mask = atom_mask[:, :, :3].all(dim=-1)
        angles_feat = angles_feat * valid_mask.unsqueeze(-1)
        return angles_feat
