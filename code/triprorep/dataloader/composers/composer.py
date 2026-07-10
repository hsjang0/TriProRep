import torch
from typing import Dict, Optional, Callable

from dataloader.mask_strategy import (
    MaskStrategy,
    SeparateMaskStrategy,
)
from dataloader.format_strategy import DataFormatStrategy, SeparateFormatStrategy
from dataloader.collate import collate_separate


class TransformComposer:
    """Apply masking and formatting to seq/bb/fa tokens for pretraining."""
    def __init__(
        self,
        mask_strategy: MaskStrategy,
        format_strategy: DataFormatStrategy,
        eos_token_id: int = 0,
        bos_token_id: int = 1,
        pad_token_id: int = 2,
        mask_token_id: int = 3,
    ):
        self.mask_strategy = mask_strategy
        self.format_strategy = format_strategy
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id

    def __call__(self, sample: Dict[str, torch.Tensor], is_test: bool = False) -> Dict[str, torch.Tensor]:
        input_tokens_seq = sample["seq_ids"].clone()
        target_tokens_seq = sample["seq_ids"].clone()
        input_tokens_bb = sample["bb_ids"].clone()
        target_tokens_bb = sample["bb_ids"].clone()
        if "fa_ids" in sample:
            input_tokens_fa = sample["fa_ids"].clone()
            target_tokens_fa = sample["fa_ids"].clone()
        else:
            input_tokens_fa, target_tokens_fa = None, None

        # Apply masking
        mask_result = self.mask_strategy.apply_mask(
            input_tokens_seq, input_tokens_bb, input_tokens_fa
        )
        input_tokens_seq = mask_result.masked_seq_ids
        input_tokens_bb = mask_result.masked_bb_ids
        input_tokens_fa = mask_result.masked_fa_ids

        is_seq_masked = mask_result.seq_mask
        is_bb_masked = mask_result.bb_mask
        is_fa_masked = mask_result.fa_mask

        if not is_test:
            target_tokens_seq = torch.where(is_seq_masked, target_tokens_seq, torch.full_like(target_tokens_seq, self.pad_token_id))
            if is_bb_masked is not None:
                target_tokens_bb = torch.where(is_bb_masked, target_tokens_bb, torch.full_like(target_tokens_bb, self.pad_token_id))
            if is_fa_masked is not None:
                target_tokens_fa = torch.where(is_fa_masked, target_tokens_fa, torch.full_like(target_tokens_fa, self.pad_token_id))

        final_seq_len = len(input_tokens_seq)
        attention_mask = torch.ones(final_seq_len, dtype=torch.bool)

        output = {
            "input_seq": input_tokens_seq,
            "target_seq": target_tokens_seq,
            "input_bb": input_tokens_bb,
            "target_bb": target_tokens_bb,
            "input_fa": input_tokens_fa if is_fa_masked is not None else None,
            "target_fa": target_tokens_fa if is_fa_masked is not None else None,
            "attention_mask": attention_mask,
        }

        if "pdb_id" in sample:
            output["pdb_id"] = sample["pdb_id"]
        if "boltz_s" in sample:
            output["boltz_s"] = sample["boltz_s"]

        # Pass through Stage 2 complex pretraining metadata
        if "chain_ids" in sample:
            output["chain_ids"] = sample["chain_ids"]
        if "position_ids" in sample:
            output["position_ids"] = sample["position_ids"]
        if "len_A" in sample:
            output["len_A"] = sample["len_A"]
        if "len_B" in sample:
            output["len_B"] = sample["len_B"]
        if "interface_info" in sample:
            output["interface_info"] = sample["interface_info"]

        return output


def create_transform(
    mask_strategy: str,
    format_strategy: str,
    mask_prob: float = 0.15,
    mask_token_id: int = 3,
    eos_token_id: int = 0,
    bos_token_id: int = 1,
    pad_token_id: int = 2,
    num_special_tokens: int = 4,
    seq_vocab_size: int = 21,
    bb_vocab_size: int = 21,
    fa_vocab_size: int = 21,
    random_seed: int = None,
    token_mask_ratio: str = "1_1_1",
    **kwargs,
) -> TransformComposer:

    if mask_strategy == "separate_mask":
        mask_strat = SeparateMaskStrategy(
            mask_prob=mask_prob,
            mask_token_id=mask_token_id,
            num_special_tokens=num_special_tokens,
            seq_vocab_size=seq_vocab_size,
            bb_vocab_size=bb_vocab_size,
            fa_vocab_size=fa_vocab_size,
            random_seed=random_seed,
            token_mask_ratio=token_mask_ratio,
        )
    else:
        raise ValueError(f"Unsupported mask_strategy: {mask_strategy}")

    if format_strategy == "separate":
        format_strat = SeparateFormatStrategy(
            num_special_tokens=num_special_tokens,
            seq_vocab_size=seq_vocab_size,
            bb_vocab_size=bb_vocab_size,
            fa_vocab_size=fa_vocab_size,
            mask_token_id=mask_token_id,
        )
    else:
        raise ValueError(f"Unsupported format_strategy: {format_strategy}")

    transform = TransformComposer(
        mask_strategy=mask_strat,
        format_strategy=format_strat,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
        mask_token_id=mask_token_id,
    )

    return transform


def create_collate_fn(
    format_strategy: str,
    pad_token_id: int = 2,
    struct_format: str = None,
    max_len: int = 512,
    **kwargs,
) -> Callable:
    if format_strategy == "separate":
        def collate_fn(batch):
            return collate_separate(
                batch,
                pad_token_id=pad_token_id,
                struct_format=struct_format,
                max_len=max_len,
            )
    else:
        raise ValueError(f"Unsupported format_strategy: {format_strategy}")

    return collate_fn
