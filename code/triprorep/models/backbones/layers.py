"""
Code adapted from ESM2 (https://github.com/facebookresearch/esm).
"""
import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from typing import Optional, Tuple, Dict, Any
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

try:
    import xformers.ops as xops
    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False


def utils_softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim)


class RMSNorm(nn.Module):
    """Root-Mean-Square LayerNorm (no mean subtraction, no bias).

    Used for QK-norm in Qwen3, Gemma, Llama. Preserves Q/K direction better
    than LayerNorm while still bounding the magnitude.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Compute in fp32 for precision (matches Llama/Qwen impl)
        orig_dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x = x * rms
        return (x * self.weight).to(orig_dtype)


## Rotary Embedding
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, : x.shape[-2], :]
    sin = sin[:, : x.shape[-2], :]
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(torch.nn.Module):
    """Rotary position embeddings from RoFormer (Su et. al)."""

    def __init__(self, dim: int, *_, **__):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=1):
        seq_len = x.shape[seq_dimension]
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self._cos_cached = emb.cos()[None, :, :]
            self._sin_cached = emb.sin()[None, :, :]
        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)
        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )

    def forward_with_position_ids(
        self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor,
        bsz: int, num_heads: int, head_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings using explicit position IDs (for per-chain positions).

        Args:
            q: [bsz*num_heads, seq_len, head_dim]
            k: [bsz*num_heads, seq_len, head_dim]
            position_ids: [bsz, seq_len] integer position indices
        """
        device = q.device
        dtype = q.dtype
        inv_freq = self.inv_freq.to(device=device, dtype=dtype)

        # position_ids: [B, L] -> freqs: [B, L, head_dim//2]
        pos = position_ids.to(dtype=dtype)  # [B, L]
        freqs = torch.einsum("bl,d->bld", pos, inv_freq)  # [B, L, D//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B, L, D]

        cos = emb.cos()  # [B, L, D]
        sin = emb.sin()  # [B, L, D]

        # Expand for num_heads: [B, L, D] -> [B*H, L, D]
        cos = cos.unsqueeze(1).expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, -1, head_dim)
        sin = sin.unsqueeze(1).expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, -1, head_dim)

        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return q, k


# MHA attention
class MultiheadAttention(nn.Module):
    """Multi-headed attention with optional rotary embeddings."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        self_attention: bool = False,
        encoder_decoder_attention: bool = False,
        use_rotary_embeddings: bool = False,
        use_checkpoint: bool = True,
        use_qk_norm: bool = False,
        attn_logit_softcap: float = 0.0,
        relax_temperature_scaling: Optional[float] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.use_checkpoint = use_checkpoint

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and value to be of the same size"
        )

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        #if add_bias_kv:
        #    self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
        #    self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        #else:
        self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.reset_parameters()

        self.onnx_trace = False
        self.rot_emb = None
        if use_rotary_embeddings:
            self.rot_emb = RotaryEmbedding(dim=self.head_dim)

        # --- QK-Norm (Dehghani et al. 2023, ViT-22B; also Gemma, Llama-3) ---
        # Bounds attention logits regardless of Q/K norm growth during training.
        # Applied per-head on the head_dim axis, before RoPE.
        # Disables the F.multi_head_attention_forward fast path (since it doesn't
        # support injecting norm between proj and attention).
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            # RMSNorm on head_dim (no mean subtraction) — matches Qwen3, Gemma, Llama.
            # Preserves Q/K direction better than LayerNorm while bounding magnitude.
            self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

        # Attention logit soft-capping (Gemma-2): bounds logits regardless of
        # Q/K norm growth. 0.0 disables. Typical value: 50.0 for stability.
        # Note: incompatible with xformers fast path (can't inject op between
        # QK and softmax), so enabling this falls back to manual attention.
        self.attn_logit_softcap = float(attn_logit_softcap)

        # Attention temperature. The kernel (xformers / SDPA) already applies
        # the standard 1/sqrt(head_dim) scaling internally; on top of that we
        # apply an explicit `relax_temperature_scaling` factor to Q. The
        # default (1/sqrt(head_dim)) yields an effective 1/head_dim attention
        # temperature — a deliberately softer (higher-entropy) attention
        # distribution than vanilla 1/sqrt(head_dim). Override via config to
        # tune the temperature (e.g. 1.0 recovers the textbook scaling).
        self.relax_temperature_scaling = (
            float(relax_temperature_scaling)
            if relax_temperature_scaling is not None
            else self.head_dim ** -0.5
        )

        self.enable_torch_version = hasattr(F, "multi_head_attention_forward")

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        #if self.bias_k is not None:
        #   nn.init.xavier_normal_(self.bias_k)
        #if self.bias_v is not None:
        #    nn.init.xavier_normal_(self.bias_v)

    def _use_xformers(self, need_weights, need_head_weights, attn_mask):
        """Decide whether the xformers fast-path can be used."""
        return (
            HAS_XFORMERS
            and not need_weights
            and not need_head_weights
            and not self.onnx_trace
            and attn_mask is None
            and not self.add_zero_attn
            and self.bias_k is None
            and self.attn_logit_softcap <= 0.0  # soft-cap needs manual softmax path
        )

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        position_ids: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        need_head_weights: bool = False,
        attn_bias: Optional[object] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            position_ids: [Batch, Time] per-token position indices for rotary embedding.
                          If None, uses sequential 0..T-1.
            key_padding_mask: [Batch, src_len] mask where True = padding.
            attn_bias: xformers AttentionBias (e.g. BlockDiagonalMask) for packed
                       sequence training. When set, query/key/value must already
                       be in [1, N_total, ...] layout and key_padding_mask is
                       ignored. Uses xformers flash path regardless of other flags.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        # --- Packed sequence fast path (BlockDiagonalMask via xformers) ---
        if attn_bias is not None:
            return self._forward_xformers_bias(query, position_ids, attn_bias, bsz, tgt_len)

        # --- xformers fast path ---
        if self._use_xformers(need_weights, need_head_weights, attn_mask):
            return self._forward_xformers(query, key, value, position_ids, key_padding_mask, bsz, tgt_len)

        # --- Fallback: manual attention path (need_weights=True, softcap>0,
        # xformers unavailable, etc.). F.multi_head_attention_forward can't
        # inject RoPE/QK-norm between proj and attention, and Net always uses
        # rotary, so that fast path is removed. ---
        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        # Scaling is applied AFTER QK-norm (below) when use_qk_norm=True,
        # otherwise keep legacy behavior: scale here.
        if not self.use_qk_norm:
            q *= self.scaling

        #if self.bias_k is not None:
        #    assert self.bias_v is not None
        #    k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
        #    v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
        #    if attn_mask is not None:
        #        attn_mask = torch.cat(
        #            [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
        #        )
        #    if key_padding_mask is not None:
        #        key_padding_mask = torch.cat(
        #            [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)],
        #            dim=1,
        #        )

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # QK-Norm: normalize per-head Q/K on head_dim (before RoPE).
        # Apply scaling AFTER the norm so it isn't erased.
        if self.use_qk_norm:
            q = self.q_norm(q)
            q = q * self.scaling
            if k is not None:
                k = self.k_norm(k)

        assert k is not None
        src_len = k.size(1)

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)],
                    dim=1,
                )

        # Apply rotary embeddings
        if self.rot_emb is not None:
            if position_ids is not None:
                q, k = self.rot_emb.forward_with_position_ids(
                    q, k, position_ids, bsz, self.num_heads, self.head_dim
                )
            else:
                q, k = self.rot_emb(q, k)

        def _attn_forward(q_t, k_t, v_t, key_pad_t, attn_mask_t):
            # This manual bmm path has no built-in scaling, so apply the same
            # `relax_temperature_scaling` temperature factor used by the
            # xformers/SDPA paths to keep all three paths numerically
            # consistent (matters for need_weights=True diagnostics / softcap).
            q_t = q_t * self.relax_temperature_scaling
            attn_weights = torch.bmm(q_t, k_t.transpose(1, 2))

            # Gemma-style logit soft-capping: bound logits before softmax.
            # Prevents runaway attention scores regardless of Q/K norm growth.
            if self.attn_logit_softcap > 0.0:
                cap = self.attn_logit_softcap
                attn_weights = torch.tanh(attn_weights / cap) * cap

            if attn_mask_t is not None:
                if attn_mask_t.dim() == 2:
                    attn_mask_t_local = attn_mask_t.unsqueeze(0)
                elif attn_mask_t.dim() == 3 and attn_mask_t.size(0) == bsz:
                    attn_mask_t_local = attn_mask_t.unsqueeze(1).expand(
                        bsz, self.num_heads, tgt_len, src_len
                    ).reshape(bsz * self.num_heads, tgt_len, src_len)
                else:
                    attn_mask_t_local = attn_mask_t
                attn_weights = attn_weights + attn_mask_t_local

            if key_pad_t is not None:
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                attn_weights = attn_weights.masked_fill(
                    key_pad_t.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
                )
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

            attn_weights_float = utils_softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
            attn_weights = attn_weights_float.type_as(attn_weights)
            attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
            attn = torch.bmm(attn_probs, v_t)
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
            attn = self.out_proj(attn)

            if not need_weights:
                return attn, attn_weights_float.new_zeros((0,))

            attn_weights_out = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).type_as(attn).transpose(1, 0)
            if not need_head_weights:
                attn_weights_out = attn_weights_out.mean(dim=0)
            return attn, attn_weights_out

        use_ckpt = self.use_checkpoint and self.training and not self.onnx_trace
        if use_ckpt:
            key_padding_mask_cp = (
                key_padding_mask if key_padding_mask is not None
                else torch.zeros((bsz, src_len), device=q.device, dtype=torch.bool)
            )
            attn_mask_cp = (
                attn_mask if attn_mask is not None
                else torch.zeros((1, tgt_len, src_len), device=q.device, dtype=q.dtype)
            )
            attn, attn_weights = checkpoint(
                _attn_forward, q, k, v, key_padding_mask_cp, attn_mask_cp, use_reentrant=False
            )
        else:
            attn, attn_weights = _attn_forward(q, k, v, key_padding_mask, attn_mask)

        if attn_weights.numel() == 0:
            attn_weights = None
        return attn, attn_weights

    def _forward_xformers(
        self,
        query: Tensor,
        key: Optional[Tensor],
        value: Optional[Tensor],
        position_ids: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        bsz: int,
        tgt_len: int,
    ) -> Tuple[Tensor, None]:
        """xformers memory-efficient attention fast path.

        Avoids materializing the full [B, H, L, L] attention matrix,
        giving O(L) memory instead of O(L^2).
        """
        # Project Q, K, V
        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            q = self.q_proj(query)
            assert key is not None and value is not None
            k = self.k_proj(key)
            v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        # Reshape: [L, B, D] -> [B*H, L, head_dim] for rotary, then -> [B, L, H, head_dim] for xformers
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        # QK-Norm: normalize per-head Q/K on head_dim (before RoPE)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply rotary embeddings (needs [B*H, L, D] format)
        if self.rot_emb is not None:
            if position_ids is not None:
                q, k = self.rot_emb.forward_with_position_ids(
                    q, k, position_ids, bsz, self.num_heads, self.head_dim
                )
            else:
                q, k = self.rot_emb(q, k)

        # Explicit temperature factor on Q. The xformers/SDPA kernels below add
        # their own 1/sqrt(head_dim); together with the default factor this
        # gives an effective 1/head_dim attention temperature.
        q = q * self.relax_temperature_scaling

        # Reshape to [B, H, L, D] -> [B, L, H, D] for xformers
        q = q.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2)
        k = k.view(bsz, self.num_heads, src_len, self.head_dim).transpose(1, 2)
        v = v.view(bsz, self.num_heads, src_len, self.head_dim).transpose(1, 2)

        # Padding mask: additive -inf on padded keys (correct masking).
        # xformers dense attn_bias is unsupported on many backends (fa3/fa2/cutlass
        # reject it on SM100+), so we route to torch SDPA when a padding mask is
        # present — it dispatches to flash/efficient kernels and handles the mask
        # natively. No-mask path keeps the xformers fast path.
        p = self.dropout if self.training else 0.0
        orig_dtype = q.dtype
        if q.dtype == torch.float32:
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)

        if key_padding_mask is not None:
            # SDPA expects [B, H, L, D]; q/k/v are currently [B, L, H, D].
            q_s = q.transpose(1, 2)
            k_s = k.transpose(1, 2)
            v_s = v.transpose(1, 2)
            attn_mask = (~key_padding_mask[:, None, None, :].to(torch.bool))
            out = F.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask=attn_mask, dropout_p=p)
            out = out.transpose(1, 2)  # back to [B, L, H, D]
        else:
            out = xops.memory_efficient_attention(q, k, v, attn_bias=None, p=p)
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        # out: [B, L, H, D] -> [L, B, D]
        out = out.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        out = self.out_proj(out)
        return out, None

    def _forward_xformers_bias(
        self,
        query: Tensor,
        position_ids: Optional[Tensor],
        attn_bias: object,
        bsz: int,
        tgt_len: int,
    ) -> Tuple[Tensor, None]:
        """xformers flash path with AttentionBias (e.g. BlockDiagonalMask).

        Used for packed-sequence training. Expects bsz==1; total tokens live
        in tgt_len. attn_bias must be an xformers AttentionBias instance.
        """
        # Self-attention only in packed mode.
        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)

        # [L, B, D] -> [B*H, L, head_dim] for rotary + QK-norm
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.rot_emb is not None:
            if position_ids is not None:
                q, k = self.rot_emb.forward_with_position_ids(
                    q, k, position_ids, bsz, self.num_heads, self.head_dim
                )
            else:
                q, k = self.rot_emb(q, k)

        # Explicit temperature factor on Q (see _forward_xformers).
        q = q * self.relax_temperature_scaling

        # [B*H, L, D] -> [B, L, H, D] for xformers
        q = q.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2)
        k = k.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2)
        v = v.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2)

        p = self.dropout if self.training else 0.0
        orig_dtype = q.dtype
        if q.dtype == torch.float32:
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)

        out = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias, p=p)
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        out = out.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        out = self.out_proj(out)
        return out, None

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.size(1)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.size(1)),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat([filler.float(), key_padding_mask.float()], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def apply_sparse_mask(attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights


class ESM1LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12, affine=True):
        super().__init__()
        self.hidden_size = (hidden_size,) if isinstance(hidden_size, int) else tuple(hidden_size)
        self.eps = eps
        self.affine = bool(affine)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.weight, self.bias = None, None

    def forward(self, x):
        dims = tuple(-(i + 1) for i in range(len(self.hidden_size)))
        means = x.mean(dims, keepdim=True)
        x_zeromean = x - means
        variances = x_zeromean.pow(2).mean(dims, keepdim=True)
        x = x_zeromean / torch.sqrt(variances + self.eps)
        if self.affine:
            x = (self.weight * x) + self.bias
        return x

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
