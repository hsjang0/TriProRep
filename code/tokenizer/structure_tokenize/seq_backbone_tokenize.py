
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from pathlib import Path
import sys
from typing import List, Dict, Optional, Union
from einops import rearrange
import functools
import math

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if PROJECT_ROOT.name == "FullAtomPLM" and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.structure.normalize_coordinates import normalize_coordinates
from esm.utils.structure.affine3d import build_affine3d_from_coordinates, Affine3D
from esm.utils.misc import knn_graph
from esm.utils import residue_constants as RC
from esm.utils.constants import esm3 as C
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.layers.rotary import RotaryEmbedding
from esm.layers.structure_proj import Dim6RotStructureHead
from esm.utils.structure.predicted_aligned_error import (
    compute_predicted_aligned_error,
    compute_tm,
)

from biotite.structure.io.pdbx import CIFFile, convert
import biotite.structure.io.pdb as pdb
import biotite.structure as bs
from Bio.Data import PDBData
from Bio import PDB

# ============================================================================
# INLINED CODE FROM CODEBASE - NO LOCAL IMPORTS
# ============================================================================

def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)
    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]

def node_gather(s: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    return batched_gather(s.unsqueeze(-3), edges, -2, no_batch_dims=len(s.shape) - 1)

class VanillaRelativePositionEmbedding(nn.Module):
    def __init__(self, bins, embedding_dim, init_std=0.02):
        super().__init__()
        self.bins = bins
        self.embedding = nn.Embedding(2 * bins + 2, embedding_dim)
        self.embedding.weight.data.normal_(0, init_std)

    def forward(self, query_residue_index, key_residue_index):
        diff = key_residue_index - query_residue_index.unsqueeze(1)
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1
        return self.embedding(diff)

class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2

def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)

def swiglu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, swiglu_correction_fn(expansion_ratio, d_model) * 2, bias=bias),
        SwiGLU(),
        nn.Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model, bias=bias),
    )

def gelu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    hidden_dim = int(expansion_ratio * d_model)
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, hidden_dim, bias=bias),
        nn.GELU(),
        nn.Linear(hidden_dim, d_model, bias=bias),
    )

class VanillaMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, bias: bool = False, qk_layernorm: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = self.d_model // self.n_heads
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=bias)
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()
        self.rotary = RotaryEmbedding(d_model // n_heads)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, x, attention_mask, seq_id):
        qkv_BLD3 = self.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
        query_BLD, key_BLD = self.q_ln(query_BLD), self.k_ln(key_BLD)
        query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)
        n_heads = self.n_heads
        reshaper = functools.partial(rearrange, pattern="b s (h d) -> b h s d", h=n_heads)
        query_BHLD, key_BHLD, value_BHLD = map(reshaper, (query_BLD, key_BLD, value_BLD))
        if seq_id is not None:
            mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
            attn_mask_BLL = torch.logical_and(attention_mask.unsqueeze(-1), attention_mask.unsqueeze(1))
            mask_BHLL = torch.logical_and(mask_BLL, attn_mask_BLL).unsqueeze(1)
            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD, mask_BHLL.to(torch.float)
            )
        else:
            context_BHLD = F.scaled_dot_product_attention(query_BHLD, key_BHLD, value_BHLD)
        context_BLD = rearrange(context_BHLD, "b h s d -> b s (h d)")
        return self.out_proj(context_BLD)

class VanillaGeometricReasoningOriginalImpl(nn.Module):
    def __init__(self, c_s: int, v_heads: int, num_vector_messages: int = 1,
                 mask_and_zero_frameless: bool = True, divide_residual_by_depth: bool = False, bias: bool = False):
        super().__init__()
        self.c_s = c_s
        self.v_heads = v_heads
        self.num_vector_messages = num_vector_messages
        self.mask_and_zero_frameless = mask_and_zero_frameless
        self.s_norm = nn.LayerNorm(c_s, bias=bias)
        dim_proj = 4 * self.v_heads * 3 + self.v_heads * 3 * self.num_vector_messages
        self.proj = nn.Linear(c_s, dim_proj, bias=bias)
        channels_out = self.v_heads * 3 * self.num_vector_messages
        self.out_proj = nn.Linear(channels_out, c_s, bias=bias)
        self.distance_scale_per_head = nn.Parameter(torch.zeros((self.v_heads)))
        self.rotation_scale_per_head = nn.Parameter(torch.zeros((self.v_heads)))

    def forward(self, s, attention_mask, affine, affine_mask, sequence_id, chain_id):
        attn_bias = torch.zeros(attention_mask.shape[0], 1, attention_mask.shape[1], attention_mask.shape[1], device=s.device)
        attn_bias = attn_bias.masked_fill(
            ~torch.logical_and(affine_mask, attention_mask)[:, None, None, :], torch.finfo(attn_bias.dtype).min
        )
        chain_id_mask = chain_id.unsqueeze(1) != chain_id.unsqueeze(2)
        attn_bias = attn_bias.masked_fill(chain_id_mask.unsqueeze(1), torch.finfo(s.dtype).min)
        ns = self.s_norm(s)
        vec_rot, vec_dist = self.proj(ns).split([
            self.v_heads * 2 * 3 + self.v_heads * 3 * self.num_vector_messages,
            self.v_heads * 2 * 3,
        ], dim=-1)
        query_rot, key_rot, value = (
            affine.rot[..., None]
            .apply(rearrange(vec_rot, "... (h c) -> ... h c", c=3))
            .split([self.v_heads, self.v_heads, self.v_heads * self.num_vector_messages], dim=-2)
        )
        query_dist, key_dist = (
            affine[..., None]
            .apply(rearrange(vec_dist, "... (h c) -> ... h c", c=3))
            .chunk(2, dim=-2)
        )
        query_dist = rearrange(query_dist, "b s h d -> b h s 1 d")
        key_dist = rearrange(key_dist, "b s h d -> b h 1 s d")
        query_rot = rearrange(query_rot, "b s h d -> b h s d")
        key_rot = rearrange(key_rot, "b s h d -> b h d s")
        value = rearrange(value, "b s (h m) d -> b h s (m d)", m=self.num_vector_messages)
        distance_term = (query_dist - key_dist).norm(dim=-1) / math.sqrt(3)
        rotation_term = query_rot.matmul(key_rot) / math.sqrt(3)
        distance_term_weight = rearrange(F.softplus(self.distance_scale_per_head), "h -> h 1 1")
        rotation_term_weight = rearrange(F.softplus(self.rotation_scale_per_head), "h -> h 1 1")
        attn_weight = rotation_term * rotation_term_weight - distance_term * distance_term_weight
        if attn_bias is not None:
            s_q = attn_weight.size(2)
            s_k = attn_weight.size(3)
            _s_q = max(0, attn_bias.size(2) - s_q)
            _s_k = max(0, attn_bias.size(3) - s_k)
            attn_bias = attn_bias[:, :, _s_q:, _s_k:]
            attn_weight = attn_weight + attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_out = attn_weight.matmul(value)
        attn_out = (
            affine.rot[..., None]
            .invert()
            .apply(rearrange(attn_out, "b h s (m d) -> b s (h m) d", m=self.num_vector_messages))
        )
        attn_out = rearrange(attn_out, "b s (h m) d -> b s (h m d)", m=self.num_vector_messages)
        if self.mask_and_zero_frameless:
            attn_out = attn_out.masked_fill(~affine_mask[..., None], 0.0)
        s = self.out_proj(attn_out)
        return s

class VanillaUnifiedTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, use_geom_attn: bool = False,
                 use_plain_attn: bool = True, v_heads: int | None = None, bias: bool = False,
                 expansion_ratio: float = 4.0, residue_scaling_factor: float = 1,
                 mask_and_zero_frameless: bool = False, qk_layernorm: bool = True, ffn_type: str = "swiglu"):
        super().__init__()
        self.use_plain_attn = use_plain_attn
        if self.use_plain_attn:
            self.attn = VanillaMultiHeadAttention(d_model, n_heads, bias, qk_layernorm=qk_layernorm)
        self.use_geom_attn = use_geom_attn
        if self.use_geom_attn:
            if v_heads is None:
                raise ValueError("v_heads must be specified when use_geom_attn is True")
            self.geom_attn = VanillaGeometricReasoningOriginalImpl(
                c_s=d_model, v_heads=v_heads, bias=bias, mask_and_zero_frameless=mask_and_zero_frameless,
            )
        if ffn_type == "swiglu":
            self.ffn = swiglu_ln_ffn(d_model, expansion_ratio, bias)
        elif ffn_type == "gelu":
            self.ffn = gelu_ln_ffn(d_model, expansion_ratio, bias)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")
        self.scaling_factor = residue_scaling_factor

    def forward(self, x, attention_mask, sequence_id, frames, frames_mask, chain_id):
        if self.use_plain_attn:
            r1 = self.attn(x, attention_mask, sequence_id)
            x = x + r1 / self.scaling_factor
        if self.use_geom_attn:
            r2 = self.geom_attn(x, attention_mask, frames, frames_mask, sequence_id, chain_id)
            x = x + r2 / self.scaling_factor
        r3 = self.ffn(x) / self.scaling_factor
        x = x + r3
        return x

class VanillaTransformerStack(nn.Module):
    def __init__(self, d_model: int, n_heads: int, v_heads: int | None, n_layers: int,
                 n_layers_geom: int = 1, scale_residue: bool = True, mask_and_zero_frameless: bool = False,
                 bias: bool = False, qk_layernorm: bool = True, ffn_type: str = "swiglu", expansion_ratio: float = 8 / 3):
        super().__init__()
        self.blocks = nn.ModuleList([
            VanillaUnifiedTransformerBlock(
                d_model, n_heads, v_heads=v_heads,
                use_geom_attn=i < n_layers_geom,
                residue_scaling_factor=(math.sqrt(n_layers / 36) if scale_residue else 1.0),
                expansion_ratio=expansion_ratio,
                mask_and_zero_frameless=mask_and_zero_frameless,
                bias=bias, qk_layernorm=qk_layernorm, ffn_type=ffn_type,
            )
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model, bias=False)

    def forward(self, x, attention_mask=None, sequence_id=None, affine=None, affine_mask=None, chain_id=None):
        *batch_dims, _ = x.shape
        if chain_id is None:
            chain_id = torch.ones(size=batch_dims, dtype=torch.int64, device=x.device)
        for block in self.blocks:
            x = block(x, attention_mask, sequence_id, affine, affine_mask, chain_id)
        return self.norm(x), x

class VanillaGeometricEncoderStack(VanillaTransformerStack):
    def __init__(self, d_model, n_heads, v_heads, n_layers):
        super().__init__(d_model, n_heads, v_heads, 0)
        self.blocks = nn.ModuleList([
            VanillaUnifiedTransformerBlock(
                d_model, n_heads, v_heads=v_heads,
                use_geom_attn=True, use_plain_attn=False,
                expansion_ratio=4, bias=True,
            )
            for i in range(n_layers)
        ])
        self.norm = nn.Identity()

class VanillaStructureTokenEncoder(nn.Module):
    def __init__(self, d_model, n_heads, v_heads, n_layers, d_out, n_codes):
        super().__init__()
        self.transformer = VanillaGeometricEncoderStack(d_model, n_heads, v_heads, n_layers)
        self.pre_vq_proj = nn.Linear(d_model, d_out)
        self.relative_positional_embedding = VanillaRelativePositionEmbedding(32, d_model, init_std=0.02)
        self.knn = 16
        self.d_out = d_out

    def encode_local_structure(self, coords, affine, attention_mask, sequence_id, affine_mask, residue_index=None):
        assert coords.size(-1) == 3 and coords.size(-2) == 3, "need N, CA, C"
        with torch.no_grad():
            knn_edges, knn_edge_mask = self.find_knn_edges(
                coords, ~attention_mask, coord_mask=affine_mask, sequence_id=sequence_id, knn=self.knn,
            )
            B, L, E = knn_edges.shape
            knn_edge_mask = knn_edge_mask.view(-1, E)
            affine_tensor = affine.tensor
            T_D = affine_tensor.size(-1)
            knn_affine_tensor = node_gather(affine_tensor, knn_edges)
            knn_affine_tensor = knn_affine_tensor.view(-1, E, T_D).contiguous()
            affine = Affine3D.from_tensor(knn_affine_tensor)
            knn_sequence_id = (
                node_gather(sequence_id.unsqueeze(-1), knn_edges).view(-1, E)
                if sequence_id is not None
                else torch.zeros(B * L, E, dtype=torch.int64, device=coords.device)
            )
            knn_attention_mask = (
                node_gather(attention_mask.unsqueeze(-1), knn_edges).view(-1, E)
                if attention_mask is not None
                else torch.zeros(B * L, E, dtype=torch.int64, device=coords.device)
            )
            knn_attention_mask = torch.logical_and(knn_attention_mask, knn_edge_mask)
            knn_affine_mask = node_gather(affine_mask.unsqueeze(-1), knn_edges).view(-1, E)
            knn_affine_mask = torch.logical_and(knn_affine_mask, knn_edge_mask)
            knn_chain_id = torch.zeros(B * L, E, dtype=torch.int64, device=coords.device)
            if residue_index is None:
                res_idxs = knn_edges.view(-1, E)
            else:
                res_idxs = node_gather(residue_index.unsqueeze(-1), knn_edges).view(-1, E)
        z = self.relative_positional_embedding(res_idxs[:, 0], res_idxs)
        z, _ = self.transformer.forward(
            x=z, attention_mask=knn_attention_mask, sequence_id=knn_sequence_id,
            affine=affine, affine_mask=knn_affine_mask, chain_id=knn_chain_id,
        )
        z = z.view(B, L, E, -1)
        z = z[:, :, 0, :]
        return z

    @staticmethod
    def find_knn_edges(coords, padding_mask, coord_mask, sequence_id=None, knn=None):
        assert knn is not None
        coords = coords.clone()
        coords[~coord_mask] = 0
        if sequence_id is None:
            sequence_id = torch.zeros((coords.shape[0], coords.shape[1]), device=coords.device).long()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            ca = coords[..., 1, :]
            edges, edge_mask = knn_graph(ca, coord_mask, padding_mask, sequence_id, no_knn=knn)
        return edges, edge_mask

    def encode(self, coords, attention_mask=None, sequence_id=None, residue_index=None):
        coords = coords[..., :3, :]
        affine, affine_mask = build_affine3d_from_coordinates(coords=coords)
        if sequence_id is None:
            sequence_id = torch.zeros_like(affine_mask, dtype=torch.int64)
        z = self.encode_local_structure(coords, affine, attention_mask, sequence_id, affine_mask, residue_index)
        z = z.masked_fill(~affine_mask.unsqueeze(2), 0)
        z = self.pre_vq_proj(z)
        return z

class BaseQuantizer(nn.Module):
    def __init__(self, codebook_size=None, codebook_embed_size=None, loss_weight=None,
                 _need_init=True, freeze_codebook=False, use_linear_project=False, **kwargs):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_embed_size = codebook_embed_size
        self.codebook = nn.Embedding(codebook_size, codebook_embed_size)
        self.loss_weight = loss_weight
        self._need_init = _need_init
        self.freeze_codebook = freeze_codebook
        self.use_linear_project = use_linear_project
        if self.use_linear_project:
            self.linear_proj = nn.Linear(codebook_embed_size, codebook_embed_size)

    def embedding2indices(self, z: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, dim_size = z.shape
        flat_z = rearrange(z, "b l h -> (b l) h")
        if self.use_linear_project:
            weight = self.linear_proj(self.codebook.weight)
        else:
            weight = self.codebook.weight
        dist = (torch.sum(flat_z ** 2, dim=1, keepdim=True) 
                + torch.sum(weight ** 2, dim=1)
                - 2 * torch.matmul(flat_z, weight.t()))
        quantized_indices = torch.argmin(dist, dim=1)
        quantized_indices = rearrange(quantized_indices, "(b l) -> b l", b=batch_size, l=seq_length).detach()
        return quantized_indices

class StraightThroughQuantizer(BaseQuantizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _tile(self, x):
        d, ew = x.shape
        if d < self.codebook_size:
            n_repeats = (self.codebook_size + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        self._need_init = False
        flat_inputs = z.view(-1, self.codebook_embed_size)
        y = self._tile(flat_inputs)
        _k_rand = y[torch.randperm(y.shape[0])][: self.codebook_size]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.codebook.weight.detach().copy_(_k_rand)
        if self.freeze_codebook:
            for name, p in self.codebook.named_parameters():
                p.requires_grad = False

    def forward(self, z: torch.Tensor):
        if self._need_init and self.training:
            self._init_embeddings(z)
        quantized_indices = self.embedding2indices(z)
        batch_size, seq_length, dim_size = z.shape
        flat_z = rearrange(z, "b l h -> (b l) h")
        flat_indices = rearrange(quantized_indices, "b l -> (b l)")
        quantized_z_pos = torch.zeros((flat_indices.shape[0], self.codebook_size), device=z.device)
        quantized_z_pos = quantized_z_pos.scatter_(1, flat_indices.unsqueeze(1), 1)
        if self.use_linear_project:
            quantized_z = torch.matmul(quantized_z_pos, self.linear_proj(self.codebook.weight))
        else:
            quantized_z = torch.matmul(quantized_z_pos, self.codebook.weight)
        commitment_loss = F.mse_loss(quantized_z.detach(), flat_z)
        loss = self.loss_weight["commitment_loss_weight"] * commitment_loss
        quantization_loss = F.mse_loss(quantized_z, flat_z.detach())
        loss += self.loss_weight["quantization_loss_weight"] * quantization_loss
        quantized_z = flat_z + (quantized_z - flat_z).detach()
        quantized_z = rearrange(quantized_z, "(b l) h -> b l h", b=batch_size, l=seq_length, h=dim_size)
        metrics = {"commitment_loss": commitment_loss, "quantization_loss": quantization_loss}
        return quantized_z, quantized_indices, loss, metrics

class VanillaCategoricalMixture:
    def __init__(self, param, bins=50, start=0, end=1):
        self.logits = param
        bins = torch.linspace(start, end, bins + 1, device=self.logits.device, dtype=torch.float32)
        self.v_bins = (bins[:-1] + bins[1:]) / 2

    def log_prob(self, true):
        true_index = ((true.unsqueeze(-1) - self.v_bins[[None] * true.ndim]).abs().argmin(-1))
        nll = self.logits.log_softmax(-1)
        return torch.take_along_dim(nll, true_index.unsqueeze(-1), dim=-1).squeeze(-1)

    def mean(self):
        return (self.logits.to(self.v_bins.dtype).softmax(-1) @ self.v_bins.unsqueeze(1)).squeeze(-1)

    def median(self):
        return self.v_bins[self.logits.max(-1).indices]

class VanillaPairwisePredictionHead(nn.Module):
    def __init__(self, input_dim: int, downproject_dim: int, hidden_dim: int, n_bins: int,
                 bias: bool = True, pairwise_state_dim: int = 0):
        super().__init__()
        self.downproject = nn.Linear(input_dim, downproject_dim, bias=bias)
        self.linear1 = nn.Linear(downproject_dim + pairwise_state_dim, hidden_dim, bias=bias)
        self.activation_fn = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, n_bins, bias=bias)

    def forward(self, x, pairwise: torch.Tensor | None = None):
        x = self.downproject(x)
        q, k = x.chunk(2, dim=-1)
        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]
        x_2d = [prod, diff]
        if pairwise is not None:
            x_2d.append(pairwise)
        x = torch.cat(x_2d, dim=-1)
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.norm(x)
        x = self.linear2(x)
        return x

class VanillaRegressionHead(nn.Module):
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

class VanillaStructureTokenDecoder(nn.Module):
    def __init__(self, encoder_d_out, d_model, n_heads, n_layers):
        super().__init__()
        self.decoder_channels = d_model
        self.vqvae_codebook_size = C.VQVAE_CODEBOOK_SIZE
        self.special_tokens = C.VQVAE_SPECIAL_TOKENS
        self.max_pae_bin = C.VQVAE_MAX_PAE_BIN
        self.post_vq_proj = nn.Linear(encoder_d_out, d_model)
        self.decoder_stack = VanillaTransformerStack(
            d_model, n_heads, 1, n_layers, scale_residue=False, n_layers_geom=0
        )
        self.affine_output_projection = Dim6RotStructureHead(
            self.decoder_channels, 10, predict_torsion_angles=False
        )
        direction_loss_bins = C.VQVAE_DIRECTION_LOSS_BINS
        pae_bins = C.VQVAE_PAE_BINS
        self.pairwise_bins = [64, direction_loss_bins * 6, pae_bins]
        self.pairwise_classification_head = VanillaPairwisePredictionHead(
            self.decoder_channels, downproject_dim=128, hidden_dim=128,
            n_bins=sum(self.pairwise_bins), bias=False,
        )
        plddt_bins = C.VQVAE_PLDDT_BINS
        self.plddt_head = VanillaRegressionHead(embed_dim=self.decoder_channels, output_dim=plddt_bins)

    def decode(self, quantized_z, structure_tokens=None, attention_mask=None, sequence_id=None):
        if sequence_id is None:
            sequence_id = torch.zeros_like(structure_tokens, dtype=torch.int64)
        chain_id = torch.zeros_like(structure_tokens, dtype=torch.int64)
        assert (structure_tokens < 0).sum() == 0
        x = self.post_vq_proj(quantized_z)
        x, _ = self.decoder_stack.forward(
            x, attention_mask=attention_mask, affine=None, affine_mask=None,
            sequence_id=sequence_id, chain_id=chain_id
        )
        tensor7_affine, bb_pred = self.affine_output_projection(
            x, affine=None, affine_mask=torch.zeros_like(attention_mask)
        )
        pairwise_logits = self.pairwise_classification_head(x)
        pairwise_dist_logits, pairwise_dir_logits, pae_logits = [
            (o if o.numel() > 0 else None)
            for o in pairwise_logits.split(self.pairwise_bins, dim=-1)
        ]
        special_tokens_mask = structure_tokens >= min(self.special_tokens.values())
        pae = compute_predicted_aligned_error(
            pae_logits, aa_mask=~special_tokens_mask,
            sequence_id=sequence_id, max_bin=self.max_pae_bin,
        )
        ptm = compute_tm(
            pae_logits, aa_mask=~special_tokens_mask, max_bin=self.max_pae_bin,
        )
        plddt_logits = self.plddt_head(x)
        plddt_value = VanillaCategoricalMixture(plddt_logits, bins=plddt_logits.shape[-1]).mean()
        return dict(
            tensor7_affine=tensor7_affine, bb_pred=bb_pred, plddt=plddt_value,
            ptm=ptm, predicted_aligned_error=pae,
            pairwise_dist_logits=pairwise_dist_logits, pairwise_dir_logits=pairwise_dir_logits,
            last_hidden_state=x,
        )

class VQVAEModel(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        quantizer_cfg = model_cfg.quantizer
        self.loss_weight = quantizer_cfg["loss_weight"]
        self.quantizer = StraightThroughQuantizer(**quantizer_cfg)
        self.encoder = VanillaStructureTokenEncoder(**model_cfg.encoder, n_codes=quantizer_cfg.codebook_size)
        self.encoder_cfg = model_cfg.encoder
        # Decoder needed to load checkpoint (has decoder weights), but not used for tokenization
        model_cfg.decoder["encoder_d_out"] = model_cfg.encoder.d_out
        self.decoder = VanillaStructureTokenDecoder(**model_cfg.decoder)
        self.inverse_folding_head = VanillaRegressionHead(
            embed_dim=model_cfg.decoder.d_model,
            output_dim=len(C.SEQUENCE_VOCAB)
        )

    def forward(self, input_list, use_as_tokenizer=False):
        coords, attention_mask, residue_index, seq_residue_tokens, pdb_chain = input_list
        sequence_id = None
        if attention_mask is None:
            attention_mask = torch.ones_like(seq_residue_tokens, dtype=torch.bool)
        else:
            attention_mask = ~attention_mask
        attention_mask = attention_mask.bool()
        z = self.encoder.encode(coords, attention_mask, sequence_id, residue_index)
        assert self.quantizer.codebook_embed_size == self.encoder.d_out
        quantized_z, quantized_indices, partial_loss, partial_metrics = self.quantizer(z)
        assert not z.isnan().any() and not quantized_indices.isnan().any()
        if use_as_tokenizer:
            return quantized_z, quantized_indices, z
        return quantized_z, quantized_indices, partial_loss, partial_metrics

class WrappedProteinChain(ProteinChain):
    @classmethod
    def from_cif(cls, path, chain_id="detect", id=None, is_predicted=False, atom_array=None):
        if id is None:
            file_id = Path(path).with_suffix("").name
        else:
            file_id = id
        if atom_array is None:
            atom_array = convert.get_structure(CIFFile.read(path), model=1, extra_fields=["b_factor"])
        if chain_id == "detect":
            chain_id = atom_array.chain_id[0]
        if not (atom_array.chain_id == chain_id).any():
            atom_array = convert.get_structure(CIFFile.read(path), model=1, extra_fields=["b_factor"], use_author_fields=False)
        atom_array = atom_array[bs.filter_amino_acids(atom_array) & ~atom_array.hetero & (atom_array.chain_id == chain_id)]
        sequence = "".join(
            (r if len(r := PDBData.protein_letters_3to1.get(monomer[0].res_name, "X")) == 1 else "X")
            for monomer in bs.residue_iter(atom_array)
        )
        num_res = len(sequence)
        atom_positions = np.full([num_res, RC.atom_type_num, 3], np.nan, dtype=np.float32)
        atom_mask = np.full([num_res, RC.atom_type_num], False, dtype=bool)
        residue_index = np.full([num_res], -1, dtype=np.int64)
        insertion_code = np.full([num_res], "", dtype="<U4")
        confidence = np.ones([num_res], dtype=np.float32)
        for i, res in enumerate(bs.residue_iter(atom_array)):
            res_index = res[0].res_id
            residue_index[i] = res_index
            insertion_code[i] = res[0].ins_code
            for atom in res:
                atom_name = atom.atom_name
                if atom_name == "SE" and atom.res_name == "MSE":
                    atom_name = "SD"
                if atom_name in RC.atom_order:
                    atom_positions[i, RC.atom_order[atom_name]] = atom.coord
                    atom_mask[i, RC.atom_order[atom_name]] = True
                    if is_predicted and atom_name == "CA":
                        confidence[i] = atom.b_factor
        assert all(sequence), "Some residue name was not specified correctly"
        return cls(
            id=file_id, sequence=sequence, chain_id=chain_id, entity_id=1,
            atom37_positions=atom_positions, atom37_mask=atom_mask,
            residue_index=residue_index, insertion_code=insertion_code, confidence=confidence,
        )

    def to_structure_encoder_inputs(self, device="cpu", should_normalize_coordinates=True):
        coords = torch.tensor(self.atom37_positions, dtype=torch.float32, device=device)
        plddt = torch.tensor(self.confidence, dtype=torch.float32, device=device)
        residue_index = torch.tensor(self.residue_index, dtype=torch.long, device=device)
        if should_normalize_coordinates:
            coords = normalize_coordinates(coords)
        return coords.unsqueeze(0), plddt.unsqueeze(0), residue_index.unsqueeze(0)

class WrappedOurPretrainedTokenizer:
    def __init__(self, device="cpu", model_cfg=None, pretrained_ckpt_path=None, ckpt_name=None):
        self.device = device
        self.model = VQVAEModel(model_cfg=model_cfg)
        model_states = torch.load(pretrained_ckpt_path, map_location=self.device)["module"]
        new_model_states = {}
        for k, v in model_states.items():
            assert k.startswith("model.")
            new_model_states[k[6:]] = v
        self.model.load_state_dict(new_model_states)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.to(self.device)
        self.seq_tokenizer = EsmSequenceTokenizer()
        self.ckpt_name = ckpt_name
        self.pad_token_id = self.model.quantizer.codebook.weight.shape[0] + 3

    def encode_structure(self, pdb_chain, use_continuous=False, use_sequence=False):
        assert use_sequence
        coords, plddt, residue_index = pdb_chain.to_structure_encoder_inputs(self.device)
        attention_mask = coords[:, :, 0, 0] == torch.inf
        sequence = pdb_chain.sequence
        sequence = sequence.replace(C.MASK_STR_SHORT, "<mask>")
        seq_ids = self.seq_tokenizer.encode(sequence, add_special_tokens=False)
        seq_ids = torch.tensor(seq_ids, dtype=torch.int64, device=self.device)
        assert len(seq_ids) == len(coords[0])
        input_list = (coords, attention_mask, residue_index, seq_ids, pdb_chain)
        quantized_reprs, quantized_indices, reprs = self.model(input_list, use_as_tokenizer=True)
        seqs = [PDB.Polypeptide.one_to_index(x) if x != "X" else 20 for x in pdb_chain.sequence]
        if use_continuous:
            return reprs.squeeze(0), np.array(residue_index.squeeze(0).cpu()), seqs
        else:
            return quantized_indices.squeeze(0), np.array(residue_index.squeeze(0).cpu()), seqs

# ============================================================================
# CONFIG CLASS - supports both dict and attribute access
# ============================================================================

class ConfigDict(dict):
    """Dict-like object that also supports attribute access (like OmegaConf)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert nested dicts to ConfigDict
        for k, v in self.items():
            if isinstance(v, dict) and not isinstance(v, ConfigDict):
                self[k] = ConfigDict(v)
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
        self[key] = value
    
    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
        super().__setitem__(key, value)

# ============================================================================
# MAIN API FUNCTIONS
# ============================================================================

def load_tokenizer(checkpoint_path, device="cuda", quantizer_codebook_size=512,
                   quantizer_codebook_embed_size=1024, model_encoder_dmodel=1024,
                   model_encoder_nlayers=2, model_encoder_vheads=128, model_encoder_dout=1024,
                   quantizer_use_linear_project=True, ckpt_name="AminoAseed"):
    """Load AminoAseed tokenizer from checkpoint."""
    from types import SimpleNamespace
    # Use ConfigDict to support both dict["key"] and dict.key access
    quantizer_cfg = ConfigDict({
        "quantizer_type": "StraightThroughQuantizer",
        "loss_weight": {"commitment_loss_weight": 0.25, "quantization_loss_weight": 1.0, "reconstruction_loss_weight": 1.0},
        "codebook_size": quantizer_codebook_size,
        "codebook_embed_size": quantizer_codebook_embed_size,
        "_need_init": False,
        "freeze_codebook": True,
        "use_linear_project": quantizer_use_linear_project,
    })
    encoder_cfg = ConfigDict({
        "d_model": model_encoder_dmodel,
        "n_heads": 1,
        "v_heads": model_encoder_vheads,
        "n_layers": model_encoder_nlayers,
        "d_out": model_encoder_dout,
        # n_codes is passed separately, not in encoder_cfg
    })
    decoder_cfg = ConfigDict({
        "encoder_d_out": model_encoder_dout,
        "d_model": 1024,
        "n_heads": 16,
        "n_layers": 8,
    })
    model_cfg = SimpleNamespace(quantizer=quantizer_cfg, encoder=encoder_cfg, decoder=decoder_cfg)
    return WrappedOurPretrainedTokenizer(device=device, model_cfg=model_cfg,
                                         pretrained_ckpt_path=checkpoint_path, ckpt_name=ckpt_name)


def extract_plddt(pdb_path: str, chain: str) -> np.ndarray:
    """Per-residue plDDT (mean atom B-factor) from a PDB / CIF file.

    Used to optionally drop low-confidence residues (``filter_low_plddt``).
    Inlined here so the release carries no extra tokenizer dependency.
    """
    if pdb_path.endswith(".cif"):
        from Bio.PDB import MMCIFParser
        parser = MMCIFParser()
    elif pdb_path.endswith(".pdb"):
        from Bio.PDB import PDBParser
        parser = PDBParser()
    else:
        raise ValueError("plddt extraction needs a '.cif' or '.pdb' file.")

    structure = parser.get_structure("protein", pdb_path)
    if structure is None or len(structure) == 0:
        raise ValueError(f"Failed to parse structure from {pdb_path}")

    chain_obj = structure[0][chain]
    plddts = [np.mean([atom.get_bfactor() for atom in residue]) for residue in chain_obj]
    return np.asarray(plddts)


def get_struct_seq(tokenizer, path, chains: Optional[List[str]] = None, filter_low_plddt=False) -> Dict[str, Dict]:
    """
    Tokenize PDB/CIF file into structure and sequence tokens.
    
    Returns dict mapping chain_id to {
        "struct_id": torch.Tensor [L] - structure token IDs (codebook indices),
        "seq_id": torch.Tensor [L] - sequence token IDs (ESM tokens),
        "residue_index": np.ndarray [L],
        "sequence": str
    }
    """
    path = Path(path)
    pdb_id = path.stem
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    results = {}
    is_cif = path.suffix.lower() in [".cif", ".mmcif"]
    if chains is None:
        if is_cif:
            atom_array = convert.get_structure(CIFFile.read(str(path)), model=1)
            chains = list(set(atom_array.chain_id)) if len(atom_array.chain_id) > 0 else ["A"]
        else:
            atom_array = pdb.PDBFile.read(str(path)).get_structure(model=1)
            chains = list(dict.fromkeys(atom_array.chain_id)) if len(atom_array.chain_id) > 0 else ["A"]
    for chain_id in chains:
        try:
            if is_cif:
                pdb_chain = WrappedProteinChain.from_cif(str(path), chain_id=chain_id, id=path.stem)
            else:
                pdb_chain = WrappedProteinChain.from_pdb(str(path), chain_id=chain_id)
            struct_ids, residue_index, seqs = tokenizer.encode_structure(pdb_chain, use_continuous=False, use_sequence=True)
            sequence = pdb_chain.sequence
            sequence = sequence.replace(C.MASK_STR_SHORT, "<mask>")
            seq_ids = tokenizer.seq_tokenizer.encode(sequence, add_special_tokens=False)
            seq_ids = torch.tensor(seq_ids, dtype=torch.int64, device=tokenizer.device)

            if filter_low_plddt:
                plddts = extract_plddt(str(path), chain_id)
                indices = np.where(plddts < 70)[0]
                results[chain_id] = {
                    "pdb_id": pdb_id,
                    "struct_id": struct_ids.cpu().numpy(),
                    "seq_id": seq_ids.cpu().numpy(),
                    "low_plddt_residues": indices
                }
            else:
                results[chain_id] = {
                    "pdb_id": pdb_id,
                    "struct_id": struct_ids.cpu().numpy(),
                    "seq_id": seq_ids.cpu().numpy(),
                }
        except Exception as e:
            print(f"Warning: Failed to process chain {chain_id}: {e}")
            continue
    # assume only one chain
    if len(results) == 0:
        return None
    return results[chain_id]

def get_codebook_embedding(model):
    return model.model.quantizer.codebook.weight.detach().cpu().numpy()


def main():
    pdb_path = "/scratch/ssahn_taewon/GeneOntology/train/5MYJ-B7_30219.pdb"
    ckpt_path = "/home/ssahn_taewon/simplefold/saprot/data/structure_tokenize/bin/codebook_512x1024-1e+19-linear-fixed-last.ckpt/checkpoint/mp_rank_00_model_states.pt"
    tokenizer = load_tokenizer(ckpt_path, device="cuda")
    print("Tokenizer loaded!")
    results = get_struct_seq(tokenizer, pdb_path, chains=["B7"])
    import pdb; pdb.set_trace()
    print(results)

if __name__ == "__main__":
    main()
