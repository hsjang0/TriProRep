"""
Code adopted from La-Proteina (https://github.com/NVIDIA-Digital-Bio/la-proteina).
"""

import torch

from structure_tokenize.fullatom_tokenizer.tokenizer_models.modules.pair_bias_attn import MultiHeadPairBiasedAttention
from structure_tokenize.fullatom_tokenizer.tokenizer_models.modules.esm_modules import ESMMultiheadAttention, ESM1LayerNorm, gelu_esm

class MultiheadAttnAndTransition(torch.nn.Module):
    """Layer that applies mha and transition to a sequence representation. Both layers are their adaptive versions
    which rely on conditining variables (see above).

    Args:
        dim_token: Token dimension in sequence representation.
        dim_pair: Dimension of pair representation.
        nheads: Number of attention heads.
        dim_cond: Dimension of conditioning variables.
        residual_mha: Whether to use a residual connection in the mha layer.
        residual_transition: Whether to use a residual connection in the transition layer.
        parallel_mha_transition: Whether to run mha and transition in parallel or sequentially.
        use_attn_pair_bias: Whether to use a pair represnetation to bias attention.
        use_qkln: Whether to use layer norm on keyus and queries for attention.
        dropout: droput use in the self-attention layer.
    """

    def __init__(
        self,
        dim_token,
        dim_pair,
        nheads,
        residual_mha,
        residual_transition,
        parallel_mha_transition,
        use_qkln,
        dropout=0.0,
        mode="atom"
    ):
        super().__init__()
        self.parallel = parallel_mha_transition

        # If parallel do not allow both layers to have a residual connection since it leads to adding x twice
        if self.parallel and residual_mha and residual_transition:
            residual_transition = False

        self.residual_mha = residual_mha
        self.residual_transition = residual_transition

        self.mhba = MultiHeadPairBiasedAttention(
            dim_token=dim_token,
            dim_pair=dim_pair,
            nheads=nheads,
            use_qkln=use_qkln,
            mode=mode
        )
    
        self.norm = torch.nn.LayerNorm(dim_token)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_token, dim_token * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim_token * 2, dim_token),
        )

    def _apply_mha(self, x, pair_rep, mask):
        x_attn = self.mhba(x, pair_rep, mask)
        if self.residual_mha:
            x_attn = x_attn + x
        return x_attn * mask[..., None]

    def _apply_transition(self, x, mask):
        x_tr = x + self.mlp(self.norm(x))
        return x_tr * mask[..., None]

    def forward(self, x, pair_rep, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            mask: binary mask, shape [b, n]
            pair_rep: Pair representation (if provided, if no bias will be ignored), shape [b, n, n, dim_pair] or None

        Returns:
            Updated sequence representation, shape [b, n, dim].
        """
        x = x * mask[..., None]
        if self.parallel:
            x = self._apply_mha(x, pair_rep, mask) + self._apply_transition(
                x, mask
            )
        else:
            x = self._apply_mha(x, pair_rep, mask)
            x = self._apply_transition(x, mask)
        return x * mask[..., None]
    

class ESMTransformerLayer(torch.nn.Module):
    """ESM Transformer layer block."""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        add_bias_kv=True,
        use_rotary_embeddings: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.use_rotary_embeddings = use_rotary_embeddings
        self._init_submodules(add_bias_kv)

    def _init_submodules(self, add_bias_kv):
        self.self_attn = ESMMultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )
        self.self_attn_layer_norm = ESM1LayerNorm(self.embed_dim)

        self.fc1 = torch.nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = torch.nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = ESM1LayerNorm(self.embed_dim)

    def forward(
        self, 
        x, 
        self_attn_mask=None, 
        self_attn_padding_mask=None, 
        need_head_weights=False
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = gelu_esm(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, attn
