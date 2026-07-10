"""
Separate masking strategy for seq/bb/fa tokens.

Per-modality dropout (5% chance): each modality independently may be
fully masked (prob=1.0), but at least one modality is guaranteed to
remain un-dropped.
"""
import torch
from typing import Optional
from dataloader.mask_strategy.base import MaskStrategy, MaskOutput


class SeparateMaskStrategy(MaskStrategy):
    def __init__(
        self,
        mask_prob: float = 0.15,
        modality_drop_prob: float = 0.05,
        mask_token_id: int = 3,
        num_special_tokens: int = 4,
        seq_vocab_size: int = 21,
        bb_vocab_size: int = 21,
        fa_vocab_size: int = 21,
        token_mask_ratio: str = "1_1_1",
        random_seed: Optional[int] = None,
        **kwargs,
    ):
        self.mask_prob = mask_prob
        self.modality_drop_prob = modality_drop_prob
        self.mask_token_id = mask_token_id
        self.num_special_tokens = num_special_tokens
        self.seq_vocab_size = seq_vocab_size
        self.bb_vocab_size = bb_vocab_size
        self.fa_vocab_size = fa_vocab_size
        self.random_seed = random_seed

    def apply_mask(
        self,
        seq_ids: torch.Tensor,
        struct_ids: torch.Tensor,
        struct_ids_fa: Optional[torch.Tensor] = None,
    ) -> MaskOutput:
        seq_len = len(seq_ids)
        device = seq_ids.device

        generator = None
        if self.random_seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.random_seed)
        rand = torch.rand
        rand_kwargs = {"device": device}
        if generator is not None:
            rand_kwargs["generator"] = generator

        has_fa = struct_ids_fa is not None
        num_modalities = 3 if has_fa else 2

        # Per-modality dropout: 5% chance to fully mask each modality
        drop_seq = rand((), **rand_kwargs).item() < self.modality_drop_prob
        drop_bb = rand((), **rand_kwargs).item() < self.modality_drop_prob
        drop_fa = rand((), **rand_kwargs).item() < self.modality_drop_prob if has_fa else False

        drops = [drop_seq, drop_bb] + ([drop_fa] if has_fa else [])

        # If ALL modalities would be dropped, randomly keep one
        if all(drops):
            keep_idx = torch.randint(num_modalities, (), **rand_kwargs).item()
            drops[keep_idx] = False
            drop_seq, drop_bb = drops[0], drops[1]
            if has_fa:
                drop_fa = drops[2]

        seq_p = 1.0 if drop_seq else self.mask_prob
        bb_p = 1.0 if drop_bb else self.mask_prob
        fa_p = 1.0 if drop_fa else self.mask_prob

        # Generate masks
        seq_mask = rand(seq_len, **rand_kwargs) < seq_p
        bb_mask = rand(seq_len, **rand_kwargs) < bb_p
        fa_mask = rand(seq_len, **rand_kwargs) < fa_p if has_fa else None

        # Apply masks
        masked_seq_ids = seq_ids.clone()
        masked_struct_ids = struct_ids.clone()
        masked_struct_ids_fa = struct_ids_fa.clone() if has_fa else None

        if seq_mask.any():
            masked_seq_ids[seq_mask] = self.mask_token_id
        if bb_mask.any():
            masked_struct_ids[bb_mask] = self.mask_token_id
        if has_fa and fa_mask.any():
            masked_struct_ids_fa[fa_mask] = self.mask_token_id

        return MaskOutput(
            masked_seq_ids=masked_seq_ids,
            masked_bb_ids=masked_struct_ids,
            masked_fa_ids=masked_struct_ids_fa,
            seq_mask=seq_mask,
            bb_mask=bb_mask,
            fa_mask=fa_mask,
        )
