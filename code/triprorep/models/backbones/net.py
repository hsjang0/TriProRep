import torch
import torch.nn as nn
from typing import Optional

from models.backbones.layers import (
    ESM1LayerNorm,
    MultiheadAttention,
    gelu,
)


class TransformerLayer(nn.Module):
    """Transformer encoder block with rotary attention support."""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        add_bias_kv=True,
        use_rotary_embeddings: bool = False,
        use_qk_norm: bool = False,
        attn_logit_softcap: float = 0.0,
        relax_temperature_scaling: Optional[float] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.use_rotary_embeddings = use_rotary_embeddings
        self.use_qk_norm = use_qk_norm
        self.attn_logit_softcap = attn_logit_softcap
        self.relax_temperature_scaling = relax_temperature_scaling
        self._init_submodules(add_bias_kv)

    def _init_submodules(self, add_bias_kv):
        BertLayerNorm = ESM1LayerNorm
        self.self_attn = MultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            use_rotary_embeddings=self.use_rotary_embeddings,
            dropout=0.1,
            use_checkpoint=False,
            use_qk_norm=self.use_qk_norm,
            attn_logit_softcap=self.attn_logit_softcap,
            relax_temperature_scaling=self.relax_temperature_scaling,
        )
        self.self_attn_layer_norm = BertLayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = BertLayerNorm(self.embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        self_attn_padding_mask=None,
        need_weights: bool = False,
        attn_bias: Optional[object] = None,
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            position_ids=position_ids,
            key_padding_mask=self_attn_padding_mask,
            need_weights=need_weights,
            attn_bias=attn_bias,
        )
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, attn


class Net(nn.Module):
    """Backbone network with chain embedding for Stage 1/2 compatibility.

    Chain embedding distinguishes chain A (0) vs chain B (1).
    At Stage 1 (single chain), all tokens use chain_id=0.
    At Stage 2 (complex), chain A tokens use 0, chain B tokens use 1.
    Per-chain positional encoding resets positions for each chain.
    """
    def __init__(
        self,
        max_residues: int = 512,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_heads: int = 12,
        decoder_dim: int = 384,
        decoder_depth: int = 8,
        decoder_heads: int = 6,
        seq_vocab_size: int = 21,
        bb_vocab_size: int = 512,
        fa_vocab_size: int = 0,
        use_qk_norm: bool = False,
        attn_logit_softcap: float = 0.0,
        relax_temperature_scaling: Optional[float] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.max_residues = max_residues
        self.seq_vocab_size = seq_vocab_size
        self.bb_vocab_size = bb_vocab_size
        self.fa_vocab_size = fa_vocab_size
        self.use_qk_norm = use_qk_norm
        self.attn_logit_softcap = attn_logit_softcap
        self.relax_temperature_scaling = relax_temperature_scaling

        from models.backbones.layers import HAS_XFORMERS
        print(f"[Model] seq_vocab={seq_vocab_size}, bb_vocab={bb_vocab_size}, fa_vocab={fa_vocab_size}, embed_dim={embed_dim}, xformers={HAS_XFORMERS}")
        self.seq_embedding = nn.Embedding(seq_vocab_size, embed_dim)
        self.bb_embedding = nn.Embedding(bb_vocab_size, embed_dim)
        if fa_vocab_size > 4:
            self.fa_embedding = nn.Embedding(fa_vocab_size, embed_dim)
            self.fuse_layer = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),
                ESM1LayerNorm(embed_dim),
            )
        else:
            self.fuse_layer = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                ESM1LayerNorm(embed_dim),
            )

        # Chain embedding: 0=chain A (or single chain), 1=chain B
        # `chain_embedding` is added at the ENCODER input (pretraining used
        # chain_idx=0 only; row 0 is the learned single-chain bias).
        self.chain_embedding = nn.Embedding(100, embed_dim)
        # `decoder_chain_embedding` is added at the DECODER input. Fresh at
        # stage-2 fine-tuning; used to differentiate chain A (idx=0) vs chain
        # B (idx=1) AFTER the shared encoder pass, so the encoder can run on
        # a single chain (e.g. corrupted apo) and the decoder produces
        # chain-specific outputs.
        self.decoder_chain_embedding = nn.Embedding(100, decoder_dim)

        # --- Encoder ---
        self.encoder = nn.ModuleList([
            TransformerLayer(
                self.embed_dim, 4 * self.embed_dim, encoder_heads,
                add_bias_kv=False, use_rotary_embeddings=True,
                use_qk_norm=use_qk_norm,
                attn_logit_softcap=attn_logit_softcap,
                relax_temperature_scaling=relax_temperature_scaling,
            ) for _ in range(encoder_depth)
        ])
        self.encoder_norm_after = ESM1LayerNorm(embed_dim)

        # --- Decoder ---
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.decoder = nn.ModuleList([
            TransformerLayer(
                self.decoder_dim, 4 * self.decoder_dim, decoder_heads,
                add_bias_kv=False, use_rotary_embeddings=True,
                use_qk_norm=use_qk_norm,
                attn_logit_softcap=attn_logit_softcap,
                relax_temperature_scaling=relax_temperature_scaling,
            ) for _ in range(decoder_depth)
        ])
        self.decoder_norm_after = ESM1LayerNorm(decoder_dim)

        # --- Prediction Heads ---
        self.lm_head_seq = nn.Linear(decoder_dim, seq_vocab_size)
        self.lm_head_bb = nn.Linear(decoder_dim, bb_vocab_size)
        if fa_vocab_size > 4:
            self.lm_head_fa = nn.Linear(decoder_dim, fa_vocab_size)

        # --- Interface Head (skeleton for Stage 2) ---
        # self.interface_head = nn.Linear(embed_dim, 1)

    def forward(
        self,
        input_seq: Optional[torch.Tensor] = None,
        input_bb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_fa: Optional[torch.Tensor] = None,
        chain_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        ret_embeddings: bool = False,
        seqlens: Optional[list] = None,
        encoder_out: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        decode_only: bool = False,
    ):
        """
        Args:
            input_seq: [L, B] sequence token IDs (packed: [N, 1])
            input_bb: [L, B] backbone structure token IDs
            attention_mask: [B, L] True=valid, False=padding (ignored when seqlens given)
            input_fa: [L, B] full-atom token IDs (optional)
            chain_ids: [B, L] 0=chain A, 1=chain B (optional)
            position_ids: [B, L] per-chain position indices (optional)
            ret_embeddings: if True, return encoder embeddings early
            ret_hidden_at: if set, also return encoder hidden state after layer `ret_hidden_at`
            ret_hidden_layers: list of HF-style negative offsets to capture in a
                     single forward. Returns dict {offset: [B, L, D]}.
                     Convention (matches HF `hidden_states[offset]`):
                       -1 → last-block output AFTER encoder_norm_after
                       -k → output of encoder block (depth - k)   for k >= 2
                     When set, all other return modes are ignored and the dict
                     is the sole return value.
            seqlens: list[int] per-sequence lengths for packed mode. When provided,
                     inputs must be in [N_total, 1] layout (bsz=1, no padding) and
                     attention uses xformers BlockDiagonalMask (flash-compatible).
            encoder_out: [L, B, embed_dim] post-`encoder_norm_after` features.
                     Only consulted when `decode_only=True`.
            padding_mask: [B, L] True=padding boolean mask for the decoder pass.
                     Only consulted when `decode_only=True` (otherwise derived
                     from `attention_mask`).
            decode_only: if True, run ONLY the decoder stack on `encoder_out`.
                     Skips token embeddings and the encoder. Used so that the
                     decoder forward is dispatched through `Net.__call__` and
                     therefore picks up `torch.compile`.
        """
        # --- decode-only fast path ----------------------------------------
        # Lets callers route the decoder forward through `Net.__call__` (and
        # thus through the torch.compile wrapper) without doing the encoder.
        # If `chain_ids` is provided, applies `decoder_chain_embedding` at the
        # decoder input — letting a single encoder pass feed both chains in
        # the complex decoder with chain identity injected here only.
        if decode_only:
            x = self.decoder_embed(encoder_out)  # [L, B, decoder_dim]
            if chain_ids is not None:
                # chain_ids: [B, L] → emb: [B, L, decoder_dim] → [L, B, decoder_dim]
                dec_chain_emb = self.decoder_chain_embedding(chain_ids).transpose(0, 1)
                x = x + dec_chain_emb
            for layer in self.decoder:
                x, _ = layer(
                    x,
                    position_ids=position_ids,
                    self_attn_padding_mask=padding_mask,
                    need_weights=False,
                )
            return self.decoder_norm_after(x)

        packed = seqlens is not None
        if packed:
            from xformers.ops.fmha.attn_bias import BlockDiagonalMask
            # seqlens may arrive as a Tensor (from Lightning batch transfer) or list.
            if isinstance(seqlens, torch.Tensor):
                seqlens_list = seqlens.tolist()
            else:
                seqlens_list = list(seqlens)
            attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=seqlens_list)
            seqlens = seqlens_list
            padding_mask = None
        else:
            if attention_mask.dtype == torch.bool:
                padding_mask = ~attention_mask
            else:
                padding_mask = (attention_mask == 0)
            padding_mask = padding_mask.to(input_seq.device)
            attn_bias = None

        # Embed tokens
        x_seq = self.seq_embedding(input_seq)
        x_bb = self.bb_embedding(input_bb)
        if input_fa is not None:
            x_fa = self.fa_embedding(input_fa)
            x = self.fuse_layer(torch.cat([x_seq, x_bb, x_fa], dim=-1))
        else:
            x = self.fuse_layer(torch.cat([x_seq, x_bb], dim=-1))

        seq_len, batch_size = x.shape[:2]

        # Add chain embedding
        if chain_ids is None:
            chain_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=x.device)
        # chain_ids: [B, L] -> chain_emb: [L, B, D] (transpose to match x)
        chain_emb = self.chain_embedding(chain_ids).transpose(0, 1)
        x = x + chain_emb

        # Default position_ids: sequential 0..L-1 (padded) or per-seq concat (packed)
        if position_ids is None:
            if packed:
                position_ids = torch.cat(
                    [torch.arange(L, device=x.device) for L in seqlens]
                ).unsqueeze(0)  # [1, N]
            else:
                position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)


        for i, layer in enumerate(self.encoder):
            x, _ = layer(
                x, position_ids=position_ids,
                self_attn_padding_mask=padding_mask,
                attn_bias=attn_bias,
                need_weights=False,
            )

        encoder_out = self.encoder_norm_after(x)  # [L, B, embed_dim]
        if ret_embeddings:
            return encoder_out.transpose(0, 1)  # [B, L, embed_dim]

        # Decoder
        x = self.decoder_embed(encoder_out)
        for layer in self.decoder:
            x, _ = layer(
                x, position_ids=position_ids,
                self_attn_padding_mask=padding_mask,
                attn_bias=attn_bias,
                need_weights=False,
            )

        x = self.decoder_norm_after(x)
        return x

    def apply_lm_head(self, x: torch.Tensor):
        logits_seq = self.lm_head_seq(x)
        logits_bb = self.lm_head_bb(x)
        if self.fa_vocab_size > 4:
            logits_fa = self.lm_head_fa(x)
        else:
            logits_fa = None
        return logits_seq, logits_bb, logits_fa

    # def apply_interface_head(self, encoder_embeddings: torch.Tensor, len_A: int, len_B: int):
    #     """Skeleton interface prediction head.
    #
    #     Args:
    #         encoder_embeddings: [B, L, D] encoder output for concatenated complex
    #         len_A: number of residues in chain A
    #         len_B: number of residues in chain B
    #
    #     Returns:
    #         interface_logits: [B, len_A, len_B] pairwise interface logits (skeleton)
    #     """
    #     emb_A = encoder_embeddings[:, :len_A, :]   # [B, N, D]
    #     emb_B = encoder_embeddings[:, len_A:len_A + len_B, :]  # [B, M, D]
    #
    #     # Skeleton: outer product of per-residue scores
    #     score_A = self.interface_head(emb_A).squeeze(-1)  # [B, N]
    #     score_B = self.interface_head(emb_B).squeeze(-1)  # [B, M]
    #     interface_logits = score_A.unsqueeze(-1) + score_B.unsqueeze(-2)  # [B, N, M]
    #
    #     return interface_logits
