"""
Code adopted from La-Proteina (https://github.com/NVIDIA-Digital-Bio/la-proteina).
"""

import torch
from torch.utils.checkpoint import checkpoint

from openfold.model.pair_transition import PairTransition

from structure_tokenize.fullatom_tokenizer.tokenizer_models.modules.atom_triangular_update import TriangleMultiplicativeUpdate


class PairReprUpdate(torch.nn.Module):
    """Layer to update the pair representation."""

    def __init__(
        self,
        token_dim,
        pair_dim,
        expansion_factor_transition=2,
        use_tri_mult=False,
        tri_mult_c=196,
        checkpointing=True
    ):
        super().__init__()

        self.use_tri_mult = use_tri_mult
        self.layer_norm_in = torch.nn.LayerNorm(token_dim)
        self.linear_x = torch.nn.Linear(token_dim, int(2 * pair_dim), bias=False)
        self.checkpointing = checkpointing

        if use_tri_mult:
            tri_mult_c = min(pair_dim, tri_mult_c)
            self.tri_mult_out = TriangleMultiplicativeUpdate(
                c_z=pair_dim, c_hidden=tri_mult_c, _outgoing=True
            )
            self.tri_mult_in = TriangleMultiplicativeUpdate(
                c_z=pair_dim, c_hidden=tri_mult_c, _outgoing=False
            )
        self.transition_out = PairTransition(
            c_z=pair_dim, n=expansion_factor_transition
        )

    def _apply_mask(self, pair_rep, pair_mask):
        """
        pair_rep has shape [B, L, A, A, pair_dim]
        pair_mask has shape [b, L, A, A]
        """
        return pair_rep * pair_mask[..., None]

    def forward(self, x, pair_rep, mask):
        """
        Args:
            x: Input sequence, shape [B, L, A, token_dim]
            pair_rep: Input pair representation, shape [B, L, L, A, pair_dim]
            mask: binary mask, shape [B, L, A]

        Returns:
            Updated pair representation, shape [B, L, A, A, pair_dim].
        """
        pair_mask = mask.unsqueeze(2) * mask.unsqueeze(3)  # [B, L, A, A]
        x = x * mask[..., None]  # [B, L, A, token_dim]
        x_proj_1, x_proj_2 = self.linear_x(self.layer_norm_in(x)).chunk(
            2, dim=-1
        )  # [B, L, A, pair_dim] each
        pair_rep = (
            pair_rep + x_proj_1[:, :, None, :, :] + x_proj_2[:, :, :, None, :]
        )  # [B, L, A, A, pair_dim]
        pair_rep = self._apply_mask(pair_rep, pair_mask)  # [B, L, A, A, pair_dim]
        if self.use_tri_mult:
            if self.checkpointing:
                pair_rep = pair_rep + checkpoint(
                    self.tri_mult_out, *(pair_rep, pair_mask * 1.0), use_reentrant=False
                )
            else:
                pair_rep = pair_rep + self.tri_mult_out(pair_rep, pair_mask * 1.0)
            pair_rep = self._apply_mask(pair_rep, pair_mask)  # [B, L, A, A, pair_dim]
            if self.checkpointing:
                pair_rep = pair_rep + checkpoint(
                    self.tri_mult_in, *(pair_rep, pair_mask * 1.0), use_reentrant=False
                )
            else:
                pair_rep = pair_rep + self.tri_mult_in(pair_rep, pair_mask * 1.0)
            pair_rep = self._apply_mask(pair_rep, pair_mask)  # [B, L, A, A, pair_dim]
        if self.checkpointing:
            pair_rep = pair_rep + checkpoint(
                self.transition_out, *(pair_rep, pair_mask * 1.0), use_reentrant=False
            )
        else:
            pair_rep = pair_rep + self.transition_out(pair_rep, pair_mask * 1.0)
        pair_rep = self._apply_mask(pair_rep, pair_mask)  # [B, L, A, A, pair_dim]
        return pair_rep
