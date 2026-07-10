import torch
from typing import List, Dict


def collate_separate(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 2,
    start_token_id: int = 0,
    struct_format: str = None,
    max_len: int = 1024,
) -> Dict[str, torch.Tensor]:
    """Collate function for pretraining (Stage 1 and Stage 2)."""
    batch_size = len(batch)

    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    input_tokens_seq = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    target_tokens_seq = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    input_tokens_bb = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    target_tokens_bb = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)

    if "input_fa" in batch[0] and batch[0]["input_fa"] is not None and struct_format and 'fullatom' in struct_format:
        use_fa = True
        input_tokens_fa = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        target_tokens_fa = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    else:
        use_fa = False

    # Chain IDs: 0 for chain A, 1 for chain B (Stage 2)
    has_chain_ids = "chain_ids" in batch[0]
    if has_chain_ids:
        chain_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        position_ids = torch.zeros((batch_size, max_len), dtype=torch.long)

    pdb_ids = []
    boltz_s_list = []

    for i, item in enumerate(batch):
        seq_len = len(item["input_seq"])
        input_tokens_seq[i, :seq_len] = item["input_seq"]
        target_tokens_seq[i, :seq_len] = item["target_seq"]
        input_tokens_bb[i, :seq_len] = item["input_bb"]
        target_tokens_bb[i, :seq_len] = item["target_bb"]
        if use_fa:
            input_tokens_fa[i, :seq_len] = item["input_fa"]
            target_tokens_fa[i, :seq_len] = item["target_fa"]
        attention_mask[i, :seq_len] = item["attention_mask"]

        if has_chain_ids:
            chain_ids[i, :seq_len] = item["chain_ids"]
            position_ids[i, :seq_len] = item["position_ids"]

        if "pdb_id" in item:
            pdb_ids.append(item["pdb_id"])
        if "boltz_s" in item:
            boltz_s_list.append(item["boltz_s"])

    result = {
        "input_seq": input_tokens_seq,
        "target_seq": target_tokens_seq,
        "input_bb": input_tokens_bb,
        "target_bb": target_tokens_bb,
        "attention_mask": attention_mask,
    }

    if use_fa:
        result["input_fa"] = input_tokens_fa
        result["target_fa"] = target_tokens_fa
    if has_chain_ids:
        result["chain_ids"] = chain_ids
        result["position_ids"] = position_ids
    if pdb_ids:
        result["pdb_ids"] = pdb_ids
    if boltz_s_list:
        result["boltz_s"] = torch.stack(boltz_s_list)
    return result


def collate_complex_pretrain(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 2,
    struct_format: str = None,
    max_len: int = 2048,
) -> Dict[str, torch.Tensor]:
    """Collate function for Stage 2 complex pretraining.

    Each sample has concatenated chain A + chain B tokens with chain_ids
    and position_ids already set by the dataset/transform.
    """
    return collate_separate(
        batch,
        pad_token_id=pad_token_id,
        struct_format=struct_format,
        max_len=max_len,
    )


def collate_packed(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 2,
    start_token_id: int = 0,
    struct_format: str = None,
    max_len: int = 1024,
    max_tokens: int = None,
) -> Dict[str, torch.Tensor]:
    """Pack variable-length samples into one super-sequence (bsz=1).

    Every per-token 1-D tensor field in the sample (those whose length equals
    the sample's input_seq length) gets concatenated along the token axis and
    unsqueezed to [1, N_total]. Everything else (scalars, pdb_ids, boltz_s) is
    collected as-is / stacked.

    If `max_tokens` is given, the packed super-sequence is padded to exactly
    that length by appending one extra "pad block" (not a dense mask — just
    another entry in `seqlens` that `BlockDiagonalMask.from_seqlens` treats
    as its own attention block). This keeps shape stable across steps so
    `torch.compile` and CUDA graphs remain happy. Since `max_tokens` is sized
    to `batch_size * max_len`, `sum(real_seqlens) <= max_tokens` always holds
    and no sample is ever dropped.

    A `seqlens` key holds per-block lengths so `Net.forward` can build an
    xformers `BlockDiagonalMask`. Loss skips pad positions via ignore_index.
    """
    # Per-sample length (computed once from input_seq, then clamped to max_len).
    raw_lens = [int(len(item["input_seq"])) for item in batch]
    seqlens = [min(L, max_len) for L in raw_lens]
    N_real = sum(seqlens)

    # Target total length (padded so shape is stable across steps).
    if max_tokens is not None:
        assert N_real <= max_tokens, (
            f"packed batch exceeds max_tokens ({N_real} > {max_tokens}); "
            f"lower per-sample max_len or raise max_tokens"
        )
        N_total = max_tokens
    else:
        N_total = N_real
    pad_len = N_total - N_real

    # Discover per-token fields: 1-D tensors whose length matches input_seq.
    first = batch[0]
    per_token_keys = []
    per_sample_keys = []
    other_keys = []
    for k, v in first.items():
        if isinstance(v, torch.Tensor):
            if v.dim() == 1 and v.shape[0] == raw_lens[0]:
                per_token_keys.append(k)
            else:
                per_sample_keys.append(k)
        else:
            other_keys.append(k)

    def _pad_value_for(key: str, ref: torch.Tensor) -> int:
        """Pick a pad value that makes the loss and metrics ignore pad positions."""
        # For bool fields (attention_mask, mask_seq, mask_bb, mask_fa): False.
        if ref.dtype == torch.bool:
            return 0
        # Targets and originals: pad_token_id so CrossEntropyLoss(ignore_index)
        # skips them. Inputs: pad_token_id (neutral placeholder).
        # chain_ids, position_ids: 0 (within-range index).
        if key in ("chain_ids", "position_ids"):
            return 0
        return int(pad_token_id)

    # Concat per-token fields + append pad segment.
    result: Dict[str, torch.Tensor] = {}
    for k in per_token_keys:
        pieces = [item[k][:L] for item, L in zip(batch, seqlens)]
        if pad_len > 0:
            ref = pieces[0]
            pad_val = _pad_value_for(k, ref)
            pieces.append(torch.full((pad_len,), pad_val, dtype=ref.dtype))
        result[k] = torch.cat(pieces, dim=0).unsqueeze(0)  # [1, N_total]

    # Ensure chain_ids / position_ids exist (some datasets don't emit them).
    if "chain_ids" not in result:
        chain_pieces = [torch.zeros(L, dtype=torch.long) for L in seqlens]
        if pad_len > 0:
            chain_pieces.append(torch.zeros(pad_len, dtype=torch.long))
        result["chain_ids"] = torch.cat(chain_pieces, dim=0).unsqueeze(0)
    if "position_ids" not in result:
        pos_pieces = [torch.arange(L, dtype=torch.long) for L in seqlens]
        if pad_len > 0:
            pos_pieces.append(torch.arange(pad_len, dtype=torch.long))
        result["position_ids"] = torch.cat(pos_pieces, dim=0).unsqueeze(0)

    # attention_mask: True for real tokens, False for pad block.
    attn = torch.zeros((1, N_total), dtype=torch.bool)
    attn[0, :N_real] = True
    result["attention_mask"] = attn

    # Seqlens list includes the trailing pad block (if any) so BlockDiagonalMask
    # isolates it from real sequences. Loss ignores pad positions by target=pad.
    seqlens_full = list(seqlens)
    if pad_len > 0:
        seqlens_full.append(pad_len)
    result["seqlens"] = seqlens_full
    # Number of real (non-pad) sequences — some downstream logic needs to
    # distinguish real blocks from the trailing pad block.
    result["num_real_seqs"] = len(seqlens)

    # Per-sample tensors: stack (same shape across batch).
    for k in per_sample_keys:
        try:
            result[k] = torch.stack([item[k] for item in batch])
        except RuntimeError:
            result[k] = [item[k] for item in batch]

    # Non-tensor fields: collect into lists.
    for k in other_keys:
        result[k] = [item[k] for item in batch]
    return result
