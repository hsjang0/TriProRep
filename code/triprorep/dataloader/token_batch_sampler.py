"""Token-budget based batch samplers for packed-sequence training.

Standard fixed-batch sampling wastes tokens when sequence lengths vary: a
batch of 128 × max_length=512 reserves 65536 slots, but only a fraction are
real tokens. `TokenBatchSampler` instead groups samples until the sum of
(clipped) lengths approaches a token budget, maximizing effective tokens
per step.

Trade-off: the number of samples per batch (and hence the number of
BlockDiagonalMask blocks) varies per step. This is *dynamic* and therefore
incompatible with `torch.compile` / CUDA graphs. Use only when compile is
disabled.
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import lmdb
import torch
from torch.utils.data import Sampler
from tqdm import tqdm


def scan_lmdb_lengths(
    lmdb_path: str,
    keys: Sequence[bytes],
    seq_field_candidates: Sequence[str] = ("seq_id", "seq_ids"),
    cache_path: Optional[str] = None,
    show_progress: bool = True,
) -> torch.Tensor:
    """Return a 1-D LongTensor of sequence lengths, one per dataset sample.

    Reads each LMDB entry once and extracts the length of the sequence field.
    Caches the result to disk (defaults to `<lmdb_path>.lengths.pt`) so
    subsequent training runs load instantly.
    """
    if cache_path is None:
        cache_path = str(lmdb_path).rstrip("/") + ".lengths.pt"
    cp = Path(cache_path)
    if cp.exists():
        cached = torch.load(cp)
        if len(cached) == len(keys):
            return cached
        # Cache stale — regenerate.

    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    lengths = torch.empty(len(keys), dtype=torch.long)
    iterator = tqdm(keys, desc=f"scan lengths {Path(lmdb_path).name}") if show_progress else keys
    with env.begin() as txn:
        for i, key in enumerate(iterator):
            raw = txn.get(key)
            if raw is None:
                lengths[i] = 0
                continue
            data = pickle.loads(raw)
            L = 0
            for field in seq_field_candidates:
                if field in data:
                    L = len(data[field])
                    break
            lengths[i] = L
    env.close()
    try:
        torch.save(lengths, cp)
    except OSError:
        pass  # Read-only fs — skip caching silently.
    return lengths


class TokenBatchSampler(Sampler[List[int]]):
    """Greedy pack indices into batches whose clipped-length sum ≤ max_tokens.

    Args:
        lengths: 1-D tensor or list of raw sample lengths.
        max_tokens: upper bound on sum(min(length_i, max_length)) per batch.
        max_length: per-sample length cap (mirrors dataset truncation).
        shuffle: shuffle indices before packing.
        drop_last: drop the tail batch if below `min_batch_size`.
        min_batch_size: minimum samples per batch (default 1).
        seed: base seed for shuffling; combined with epoch for reproducibility.

    Distributed:
        Pass `rank`/`world_size` to shard indices across ranks. Each rank
        sees its own disjoint slice and packs independently. Number of
        batches may differ slightly across ranks — Lightning handles this
        via gradient averaging, but if strict sync is required, set
        `even_batches=True` to trim all ranks to the minimum.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        max_tokens: int,
        max_length: int,
        shuffle: bool = True,
        drop_last: bool = True,
        min_batch_size: int = 1,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        even_batches: bool = True,
    ):
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.tolist()
        self.lengths = [min(int(L), int(max_length)) for L in lengths]
        self.max_tokens = int(max_tokens)
        self.max_length = int(max_length)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.min_batch_size = min_batch_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.even_batches = even_batches
        self.epoch = 0
        if self.max_tokens < self.max_length:
            raise ValueError(
                f"max_tokens ({self.max_tokens}) < max_length ({self.max_length}); "
                "cannot fit even one sample."
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rank_indices(self) -> List[int]:
        """Return this rank's slice of indices, optionally shuffled."""
        n = len(self.lengths)
        indices = list(range(n))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(n, generator=g).tolist()
            indices = perm
        if self.world_size > 1:
            indices = indices[self.rank :: self.world_size]
        return indices

    def _pack(self, indices: List[int]) -> List[List[int]]:
        batches: List[List[int]] = []
        cur: List[int] = []
        total = 0
        for idx in indices:
            L = self.lengths[idx]
            if L == 0:
                continue
            if total + L > self.max_tokens and cur:
                batches.append(cur)
                cur = []
                total = 0
            cur.append(idx)
            total += L
        if cur and not (self.drop_last and len(cur) < self.min_batch_size):
            batches.append(cur)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        indices = self._rank_indices()
        batches = self._pack(indices)

        if self.world_size > 1 and self.even_batches:
            # All-reduce minimum #batches across ranks to keep DDP in sync.
            n_local = torch.tensor(len(batches))
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(n_local, op=torch.distributed.ReduceOp.MIN)
            batches = batches[: int(n_local)]

        yield from batches

    def __len__(self) -> int:
        # Estimate: total clipped tokens divided by budget. Sampler-iter may
        # emit slightly fewer (due to greedy boundary fragmentation).
        total = sum(self.lengths)
        if self.world_size > 1:
            total = total // self.world_size
        return max(1, total // self.max_tokens)
