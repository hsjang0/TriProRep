import os
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Tuple
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataloader.dataset import ProteinDataset, ComplexPretrainDataset
from dataloader.composers import create_transform, create_collate_fn
from dataloader.collate import collate_complex_pretrain, collate_packed


def worker_init_fn(worker_id):
    """Reset LMDB handles per worker to avoid sharing state after fork."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if hasattr(dataset, 'env'):
            dataset.env = None


class ProteinDataModule(pl.LightningDataModule):
    """DataModule for Stage 1 single-chain pretraining."""

    def __init__(
        self,
        lmdb_dir: str = "/path/to/lmdb",
        struct_format: str = "foldseek",
        batch_size: int = 32,
        num_workers: int = 8,
        prefetch_factor: int = 2,
        max_length: Optional[int] = None,
        pin_memory: bool = True,
        mask_strategy: str = "bert_mask",
        format_strategy: str = "sequential",
        mask_prob: float = 0.15,
        bb_vocab_size: int = 512,
        fa_vocab_size: int = 1024,
        token_mask_ratio: str = "1_1_1",
        load_codebook: bool = False,
        fa_remap_index: str = None,
        random_cropping: bool = False,
        pack_sequences: bool = False,
        dynamic_batch_tokens: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.lmdb_dir = Path(lmdb_dir)
        self.struct_format = struct_format
        self.mask_strategy = mask_strategy
        self.format_strategy = format_strategy
        self.mask_prob = mask_prob

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.max_length = max_length
        self.pin_memory = pin_memory
        self.bb_vocab_size = bb_vocab_size
        self.fa_vocab_size = fa_vocab_size
        self.token_mask_ratio = token_mask_ratio

        self.load_codebook = load_codebook
        self.fa_remap_index = fa_remap_index
        self.random_cropping = random_cropping
        self.pack_sequences = pack_sequences
        # When set (and pack_sequences=True), each batch greedily packs samples
        # up to this many tokens via TokenBatchSampler. Otherwise fixed-#sample
        # packing with no pad block. Mirrors ELECTRADataModule semantics.
        self.dynamic_batch_tokens = dynamic_batch_tokens

        # Token configuration
        self.eos_token_id = 0
        self.bos_token_id = 1
        self.pad_token_id = 2
        self.mask_token_id = 3
        self.num_special_tokens = 0 if self.struct_format == "foldseek" else 4
        self.seq_vocab_size = 33

        self.train_dataset = None

        if self.load_codebook:
            codebook_path = os.path.join(f"{self.lmdb_dir}", "codebook.npy")
            if os.path.exists(codebook_path):
                print(f"Loading codebook from {codebook_path}")
                self.codebook = torch.from_numpy(np.load(codebook_path))
            else:
                self.codebook = None
                print(f"Codebook not found at {codebook_path}")
        else:
            self.codebook = None

        self.transform, self.collate_fn = self._create_transform_collate_fn()

    def _create_transform_collate_fn(self) -> Tuple[Callable, Callable]:
        transform = create_transform(
            mask_strategy=self.mask_strategy,
            format_strategy=self.format_strategy,
            mask_prob=self.mask_prob,
            mask_token_id=self.mask_token_id,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            num_special_tokens=self.num_special_tokens,
            seq_vocab_size=self.seq_vocab_size,
            bb_vocab_size=self.bb_vocab_size,
            fa_vocab_size=self.fa_vocab_size,
            token_mask_ratio=self.token_mask_ratio,
        )

        if self.pack_sequences:
            # Two modes:
            #   (a) dynamic_batch_tokens set → TokenBatchSampler feeds variable
            #       #samples; pack super-sequence to that budget.
            #   (b) dynamic_batch_tokens None → fixed #samples per batch, no
            #       pad block. Variable super-sequence length per step.
            pack_max_tokens = (
                int(self.dynamic_batch_tokens)
                if self.dynamic_batch_tokens is not None
                else None
            )
            collate_fn = lambda batch: collate_packed(
                batch,
                pad_token_id=self.pad_token_id,
                struct_format=self.struct_format,
                max_len=self.max_length,
                max_tokens=pack_max_tokens,
            )
        else:
            collate_fn = create_collate_fn(
                format_strategy=self.format_strategy,
                pad_token_id=self.pad_token_id,
                struct_format=self.struct_format,
                max_len=self.max_length,
            )

        return transform, collate_fn

    def get_seq_vocab_size(self) -> int:
        return self.transform.format_strategy.get_seq_vocab_size()

    def get_bb_vocab_size(self) -> int:
        return self.transform.format_strategy.get_bb_vocab_size()

    def get_fa_vocab_size(self) -> int:
        return self.transform.format_strategy.get_fa_vocab_size()

    def setup(self, stage: Optional[str] = None):
        # Use a single data.lmdb for training (no train/valid/test split)
        data_lmdb = self.lmdb_dir #/ "data.lmdb"
        if not data_lmdb.exists():
            raise FileNotFoundError(f"data.lmdb not found: {data_lmdb}")

        if stage == "fit" or stage is None:
            self.train_dataset = ProteinDataset(
                lmdb_path=str(data_lmdb),
                struct_format=self.struct_format,
                max_length=self.max_length,
                transform=self.transform,
                num_special_tokens=self.num_special_tokens,
                seq_vocab_size=self.seq_vocab_size,
                fa_remap_index=self.fa_remap_index,
                random_cropping=self.random_cropping,
            )

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        prefetch_kwargs = (
            {"prefetch_factor": self.prefetch_factor}
            if self.num_workers > 0 and self.prefetch_factor is not None
            else {}
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=worker_init_fn if self.num_workers > 0 else None,
            drop_last=True,
            **prefetch_kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        if self.pack_sequences and self.dynamic_batch_tokens is not None:
            from dataloader.token_batch_sampler import TokenBatchSampler, scan_lmdb_lengths
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            ws = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
            lengths = scan_lmdb_lengths(str(self.lmdb_dir), self.train_dataset.keys)
            batch_sampler = TokenBatchSampler(
                lengths=lengths,
                max_tokens=int(self.dynamic_batch_tokens),
                max_length=int(self.max_length),
                shuffle=True,
                drop_last=True,
                rank=rank, world_size=ws,
            )
            prefetch_kwargs = (
                {"prefetch_factor": self.prefetch_factor}
                if self.num_workers > 0 and self.prefetch_factor is not None
                else {}
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=self.pin_memory,
                persistent_workers=False,  # must re-seed each epoch
                worker_init_fn=worker_init_fn if self.num_workers > 0 else None,
                **prefetch_kwargs,
            )
        return self._make_loader(self.train_dataset, shuffle=True)


class ComplexPretrainDataModule(pl.LightningDataModule):
    """DataModule for Stage 2 complex pretraining."""

    def __init__(
        self,
        lmdb_dir: str,
        struct_format: str = "fullatom",
        batch_size: int = 16,
        num_workers: int = 8,
        prefetch_factor: int = 2,
        max_length_per_chain: Optional[int] = 512,
        max_length: Optional[int] = 1024,
        train_split: str = "train",
        val_split: str = "valid",
        test_split: str = "test",
        pin_memory: bool = True,
        mask_strategy: str = "separate_mask",
        format_strategy: str = "separate",
        mask_prob: float = 0.5,
        bb_vocab_size: int = 512,
        fa_vocab_size: int = 256,
        token_mask_ratio: str = "1_1_1",
        fa_remap_index: str = None,
        **kwargs,
    ):
        super().__init__()
        self.lmdb_dir = Path(lmdb_dir)
        self.struct_format = struct_format
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.max_length_per_chain = max_length_per_chain
        self.max_length = max_length
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.pin_memory = pin_memory
        self.bb_vocab_size = bb_vocab_size
        self.fa_vocab_size = fa_vocab_size
        self.fa_remap_index = fa_remap_index

        self.eos_token_id = 0
        self.bos_token_id = 1
        self.pad_token_id = 2
        self.mask_token_id = 3
        self.num_special_tokens = 0 if struct_format == "foldseek" else 4
        self.seq_vocab_size = 33

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Create transform for masking concatenated complex sequences
        self.transform = create_transform(
            mask_strategy=mask_strategy,
            format_strategy=format_strategy,
            mask_prob=mask_prob,
            mask_token_id=self.mask_token_id,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            num_special_tokens=self.num_special_tokens,
            seq_vocab_size=self.seq_vocab_size,
            bb_vocab_size=bb_vocab_size,
            fa_vocab_size=fa_vocab_size,
            token_mask_ratio=token_mask_ratio,
        )

        self.collate_fn = lambda batch: collate_complex_pretrain(
            batch,
            pad_token_id=self.pad_token_id,
            struct_format=self.struct_format,
            max_len=self.max_length,
        )

    def get_seq_vocab_size(self) -> int:
        return self.transform.format_strategy.get_seq_vocab_size()

    def get_bb_vocab_size(self) -> int:
        return self.transform.format_strategy.get_bb_vocab_size()

    def get_fa_vocab_size(self) -> int:
        return self.transform.format_strategy.get_fa_vocab_size()

    def _make_dataset(self, split: str, is_test: bool = False) -> ComplexPretrainDataset:
        lmdb_path = self.lmdb_dir / f"{split}.lmdb"
        if not lmdb_path.exists():
            raise FileNotFoundError(f"LMDB not found: {lmdb_path}")
        return ComplexPretrainDataset(
            lmdb_path=str(lmdb_path),
            struct_format=self.struct_format,
            max_length_per_chain=self.max_length_per_chain,
            transform=self.transform,
            num_special_tokens=self.num_special_tokens,
            seq_vocab_size=self.seq_vocab_size,
            fa_remap_index=self.fa_remap_index,
        )

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            self.train_dataset = self._make_dataset(self.train_split)
            self.val_dataset = self._make_dataset(self.val_split)
            test_lmdb = self.lmdb_dir / f"{self.test_split}.lmdb"
            if test_lmdb.exists():
                self.test_dataset = self._make_dataset(self.test_split)
        if stage in ("test", None):
            self.test_dataset = self._make_dataset(self.test_split)

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        prefetch_kwargs = (
            {"prefetch_factor": self.prefetch_factor}
            if self.num_workers > 0 and self.prefetch_factor is not None
            else {}
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=worker_init_fn if self.num_workers > 0 else None,
            drop_last=True,
            **prefetch_kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            return None
        return self._make_loader(self.test_dataset, shuffle=False)


# ---------------------------------------------------------------------------
# ELECTRA DataModule — corrective MLM with generator-corrupted input
# ---------------------------------------------------------------------------


class ELECTRATransform:
    """Independent per-modality masking for ELECTRA corrective MLM."""

    def __init__(self, mask_prob=0.40, mask_token_id=3, pad_token_id=2,
                 num_special_tokens=4, seq_vocab_size=33,
                 bb_vocab_size=512, fa_vocab_size=512, **kwargs):
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.num_special_tokens = num_special_tokens
        self._seq_vocab_size = seq_vocab_size
        self._bb_vocab_size = bb_vocab_size
        self._fa_vocab_size = fa_vocab_size

    def get_seq_vocab_size(self):
        return self._seq_vocab_size + self.num_special_tokens

    def get_bb_vocab_size(self):
        return self._bb_vocab_size + self.num_special_tokens

    def get_fa_vocab_size(self):
        return self._fa_vocab_size + self.num_special_tokens

    def __call__(self, sample, is_test=False):
        seq_ids, bb_ids = sample["seq_ids"], sample["bb_ids"]
        fa_ids = sample.get("fa_ids", None)
        L = len(seq_ids)

        mask_seq = torch.rand(L) < self.mask_prob
        mask_bb = torch.rand(L) < self.mask_prob
        mask_fa = torch.rand(L) < self.mask_prob if fa_ids is not None else None

        original_seq, original_bb = seq_ids.clone(), bb_ids.clone()
        original_fa = fa_ids.clone() if fa_ids is not None else None

        input_seq, input_bb = seq_ids.clone(), bb_ids.clone()
        input_seq[mask_seq] = self.mask_token_id
        input_bb[mask_bb] = self.mask_token_id

        target_seq = torch.full_like(seq_ids, self.pad_token_id)
        target_bb = torch.full_like(bb_ids, self.pad_token_id)
        target_seq[mask_seq] = seq_ids[mask_seq]
        target_bb[mask_bb] = bb_ids[mask_bb]

        out = {
            "input_seq": input_seq, "input_bb": input_bb,
            "target_seq": target_seq, "target_bb": target_bb,
            "original_seq": original_seq, "original_bb": original_bb,
            "mask_seq": mask_seq, "mask_bb": mask_bb,
            "attention_mask": torch.ones(L, dtype=torch.bool),
        }
        if fa_ids is not None:
            input_fa = fa_ids.clone()
            input_fa[mask_fa] = self.mask_token_id
            target_fa = torch.full_like(fa_ids, self.pad_token_id)
            target_fa[mask_fa] = fa_ids[mask_fa]
            out.update({"input_fa": input_fa, "target_fa": target_fa,
                        "original_fa": original_fa, "mask_fa": mask_fa})
        if "pdb_id" in sample:
            out["pdb_id"] = sample["pdb_id"]
        return out


def collate_electra(batch, pad_token_id=2, struct_format="fullatom", max_len=512):
    """Collate for ELECTRA: pads all fields including originals and per-modality masks."""
    B = len(batch)
    has_fa = "input_fa" in batch[0] and batch[0]["input_fa"] is not None and "fullatom" in struct_format

    fields = ["input_seq", "input_bb", "target_seq", "target_bb", "original_seq", "original_bb"]
    if has_fa:
        fields += ["input_fa", "target_fa", "original_fa"]

    t = {f: torch.full((B, max_len), pad_token_id, dtype=torch.long) for f in fields}
    t["attention_mask"] = torch.zeros((B, max_len), dtype=torch.bool)
    t["mask_seq"] = torch.zeros((B, max_len), dtype=torch.bool)
    t["mask_bb"] = torch.zeros((B, max_len), dtype=torch.bool)
    if has_fa:
        t["mask_fa"] = torch.zeros((B, max_len), dtype=torch.bool)

    pdb_ids = []
    for i, item in enumerate(batch):
        sl = len(item["input_seq"])
        for f in fields:
            if f in item and item[f] is not None:
                t[f][i, :sl] = item[f]
        t["attention_mask"][i, :sl] = item["attention_mask"]
        t["mask_seq"][i, :sl] = item["mask_seq"]
        t["mask_bb"][i, :sl] = item["mask_bb"]
        if has_fa and "mask_fa" in item:
            t["mask_fa"][i, :sl] = item["mask_fa"]
        if "pdb_id" in item:
            pdb_ids.append(item["pdb_id"])
    if pdb_ids:
        t["pdb_ids"] = pdb_ids
    return t


class ELECTRADataModule(pl.LightningDataModule):
    """DataModule for ELECTRA corrective MLM pretraining."""

    def __init__(self, lmdb_dir, struct_format="fullatom",
                 batch_size=32, num_workers=8, prefetch_factor=2,
                 max_length=512, pin_memory=True, mask_prob=0.40,
                 bb_vocab_size=512, fa_vocab_size=512,
                 fa_remap_index=None, random_cropping=False,
                 pack_sequences: bool = False,
                 dynamic_batch_tokens: Optional[int] = None, **kwargs):
        super().__init__()
        self.lmdb_dir = Path(lmdb_dir)
        self.struct_format = struct_format
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.max_length = max_length
        self.pin_memory = pin_memory
        self.fa_remap_index = fa_remap_index
        self.random_cropping = random_cropping
        self.pack_sequences = pack_sequences
        # When set (and pack_sequences=True), each batch greedily packs samples
        # up to this many tokens — variable #samples per batch. Maximizes real
        # tokens per step; incompatible with torch.compile (dynamic shapes).
        self.dynamic_batch_tokens = dynamic_batch_tokens
        self.eos_token_id = 0
        self.bos_token_id = 1
        self.pad_token_id = 2
        self.mask_token_id = 3
        self.num_special_tokens = 0 if struct_format == "foldseek" else 4
        self.seq_vocab_size = 33
        self.bb_vocab_size = bb_vocab_size
        self.fa_vocab_size = fa_vocab_size
        self.train_dataset = None

        self.transform = ELECTRATransform(
            mask_prob=mask_prob, mask_token_id=self.mask_token_id,
            pad_token_id=self.pad_token_id, num_special_tokens=self.num_special_tokens,
            seq_vocab_size=self.seq_vocab_size,
            bb_vocab_size=bb_vocab_size, fa_vocab_size=fa_vocab_size,
        )
        if self.pack_sequences:
            # Three modes:
            #   (a) dynamic_batch_tokens set → TokenBatchSampler feeds variable
            #       #samples; pack super-sequence to that budget (fixed-within-
            #       sampler but variable-across-steps).
            #   (b) dynamic_batch_tokens None → fixed #samples per batch, no
            #       padding at all. Super-sequence length = sum(real_seqlens),
            #       variable per step. Incompatible with torch.compile.
            if self.dynamic_batch_tokens is not None:
                pack_max_tokens = int(self.dynamic_batch_tokens)
            else:
                pack_max_tokens = None  # no pad block
            self.collate_fn = lambda batch: collate_packed(
                batch, pad_token_id=self.pad_token_id,
                struct_format=self.struct_format, max_len=self.max_length,
                max_tokens=pack_max_tokens,
            )
        else:
            self.collate_fn = lambda batch: collate_electra(
                batch, pad_token_id=self.pad_token_id,
                struct_format=self.struct_format, max_len=self.max_length,
            )

    def get_seq_vocab_size(self):
        return self.transform.get_seq_vocab_size()

    def get_bb_vocab_size(self):
        return self.transform.get_bb_vocab_size()

    def get_fa_vocab_size(self):
        return self.transform.get_fa_vocab_size()

    def setup(self, stage=None):
        if not self.lmdb_dir.exists():
            raise FileNotFoundError(f"LMDB not found: {self.lmdb_dir}")
        if stage == "fit" or stage is None:
            self.train_dataset = ProteinDataset(
                lmdb_path=str(self.lmdb_dir),
                struct_format=self.struct_format, max_length=self.max_length,
                transform=self.transform, num_special_tokens=self.num_special_tokens,
                seq_vocab_size=self.seq_vocab_size, fa_remap_index=self.fa_remap_index,
                random_cropping=self.random_cropping,
            )

    def train_dataloader(self):
        pf = ({"prefetch_factor": self.prefetch_factor}
              if self.num_workers > 0 and self.prefetch_factor is not None else {})

        if self.pack_sequences and self.dynamic_batch_tokens is not None:
            from dataloader.token_batch_sampler import TokenBatchSampler, scan_lmdb_lengths
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            ws = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
            lengths = scan_lmdb_lengths(str(self.lmdb_dir), self.train_dataset.keys)
            batch_sampler = TokenBatchSampler(
                lengths=lengths,
                max_tokens=int(self.dynamic_batch_tokens),
                max_length=int(self.max_length),
                shuffle=True,
                drop_last=True,
                rank=rank, world_size=ws,
            )
            return DataLoader(
                self.train_dataset, batch_sampler=batch_sampler,
                num_workers=self.num_workers, collate_fn=self.collate_fn,
                pin_memory=self.pin_memory,
                persistent_workers=False,  # must re-seed each epoch
                worker_init_fn=worker_init_fn if self.num_workers > 0 else None,
                **pf,
            )

        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=self.collate_fn,
            pin_memory=self.pin_memory, persistent_workers=self.num_workers > 0,
            worker_init_fn=worker_init_fn if self.num_workers > 0 else None,
            drop_last=True, **pf,
        )
