import os
import json
import lmdb
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional, Callable


class ProteinDataset(Dataset):
    """LMDB-backed dataset for pretraining inputs (seq, bb, fa tokens)."""

    def __init__(
        self,
        lmdb_path: str,
        struct_format: str = "foldseek",
        max_length: Optional[int] = None,
        transform: Optional[Callable] = None,
        num_special_tokens: int = 4,
        seq_vocab_size: int = 21,
        load_boltz_embeddings: bool = False,
        boltz_dir: str = "/path/to/boltz",
        is_test: bool = False,
        fa_remap_index: str = None,
        random_cropping: bool = False,
        **kwargs,
    ):
        self.lmdb_path = lmdb_path
        self.struct_format = struct_format
        self.max_length = max_length
        self.transform = transform
        self.num_special_tokens = num_special_tokens
        self.seq_vocab_size = seq_vocab_size
        self.load_boltz_embeddings = load_boltz_embeddings
        self.boltz_dir = Path(boltz_dir) if load_boltz_embeddings else None
        self.is_test = is_test
        self.random_cropping = random_cropping

        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin() as txn:
            metadata_bytes = txn.get(b"__metadata__")
            self.metadata = pickle.loads(metadata_bytes) if metadata_bytes else {}

            # Fast path: if __keys__ blob exists, load it directly (no cursor scan)
            keys_blob = txn.get(b"__keys__")
            if keys_blob is not None:
                self.keys = pickle.loads(keys_blob)
                print(f"Loaded {len(self.keys)} keys from __keys__ blob ({lmdb_path})")
            else:
                # Fallback: full cursor scan (slow for large LMDBs)
                n_entries = txn.stat()["entries"]
                print(f"No __keys__ blob, scanning {n_entries} entries... (this may take a while)")
                self.keys = [key for key, _ in txn.cursor()
                             if key not in (b"__metadata__", b"__keys__")]
                print(f"Loaded {len(self.keys)} samples from {lmdb_path}")
        env.close()

        self.env = None
        self.fa_remap = None

        if fa_remap_index:
            mapping_path = Path(fa_remap_index)
            if not mapping_path.exists():
                raise FileNotFoundError(f"FA remap codebook not found: {mapping_path}")
            with mapping_path.open("r") as f:
                idx2cluster = json.load(f)
            remap_size = max(int(k) for k in idx2cluster.keys()) + 1
            remap = torch.empty(remap_size, dtype=torch.long)
            for idx, cluster in idx2cluster.items():
                remap[int(idx)] = int(cluster)
            self.fa_remap = remap
            print(f"Code book: {remap}")

    def _init_db(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._init_db()
        key = self.keys[idx]

        with self.env.begin() as txn:
            value = txn.get(key)
            if value is None:
                raise KeyError(f"Key {key} not found in LMDB")
            data = pickle.loads(value)

        pdb_id = data.get("pdb_id", key.decode())

        if "seq_id" in data:
            seq_ids = torch.from_numpy(data["seq_id"]).long()
        elif "seq_ids" in data:
            seq_ids = torch.from_numpy(data["seq_ids"]).long()

        struct_ids = None
        struct_ids_fa = None
        if 'fullatom' in self.struct_format:
            if "struct_id_aminoaseed" in data and "struct_id_fullatom" in data:
                struct_ids = torch.from_numpy(data["struct_id_aminoaseed"]).long()
                struct_ids_fa = torch.from_numpy(data["struct_id_fullatom"]).long()
        else:
            if "struct_ids" in data:
                struct_ids = torch.from_numpy(data["struct_ids"]).long()
            elif "struct_id" in data:
                struct_ids = torch.from_numpy(data["struct_id"]).long()
            elif "struct_id_aminoaseed" in data:
                struct_ids = torch.from_numpy(data["struct_id_aminoaseed"]).long()

        if struct_ids_fa is not None and self.fa_remap is not None:
            struct_ids_fa = self.fa_remap[struct_ids_fa]

        # Shift token IDs
        if self.struct_format.count("foldseek") == 0:
            seq_ids = seq_ids + self.num_special_tokens

        if struct_ids is not None:
            struct_ids = struct_ids + self.num_special_tokens
        if struct_ids_fa is not None:
            struct_ids_fa = struct_ids_fa + self.num_special_tokens

        # Truncate or random crop
        if self.max_length is not None and len(seq_ids) > self.max_length:
            if self.random_cropping:
                start = torch.randint(0, len(seq_ids) - self.max_length + 1, (1,)).item()
            else:
                start = 0
            end = start + self.max_length
            seq_ids = seq_ids[start:end]
            if struct_ids is not None:
                struct_ids = struct_ids[start:end]
            if struct_ids_fa is not None:
                struct_ids_fa = struct_ids_fa[start:end]

        sample = {
            "seq_ids": seq_ids,
            "bb_ids": struct_ids if struct_ids is not None else torch.zeros_like(seq_ids),
            "pdb_id": pdb_id,
        }

        if struct_ids_fa is not None:
            sample["fa_ids"] = struct_ids_fa

        if self.load_boltz_embeddings:
            boltz_s = self._load_boltz_embeddings(pdb_id)
            if boltz_s is not None:
                sample["boltz_s"] = boltz_s

        if self.transform is not None:
            sample = self.transform(sample, is_test=self.is_test)

        return sample

    def _load_boltz_embeddings(self, pdb_id: str) -> Optional[torch.Tensor]:
        if self.boltz_dir is None:
            return None
        pdb_chain_id = f"{pdb_id}_A"
        emb_path = (
            self.boltz_dir / pdb_chain_id / f"boltz_results_{pdb_chain_id}" /
            "predictions" / pdb_chain_id / f"embeddings_{pdb_chain_id}.npz"
        )
        try:
            data = np.load(str(emb_path))
            s = data['s'][0]
            s_avg = torch.from_numpy(s.mean(axis=0)).float()
            return s_avg
        except Exception:
            return torch.zeros((384,)).float()

    def __del__(self):
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()


class ComplexPretrainDataset(Dataset):
    """LMDB-backed dataset for Stage 2 complex pretraining.

    Each LMDB entry contains two chains (seq_ids_A/B, struct_ids_A/B, fa_ids_A/B)
    and optional interface information (e.g., distance matrix).

    Returns concatenated tokens with chain_ids and position_ids for
    per-chain positional encoding.
    """

    def __init__(
        self,
        lmdb_path: str,
        struct_format: str = "fullatom",
        max_length_per_chain: Optional[int] = 512,
        transform: Optional[Callable] = None,
        num_special_tokens: int = 4,
        seq_vocab_size: int = 21,
        fa_remap_index: str = None,
        **kwargs,
    ):
        self.lmdb_path = lmdb_path
        self.struct_format = struct_format
        self.max_length_per_chain = max_length_per_chain
        self.transform = transform
        self.num_special_tokens = num_special_tokens
        self.seq_vocab_size = seq_vocab_size

        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin() as txn:
            self.keys = [key for key, _ in txn.cursor() if key != b"__metadata__"]
        with env.begin() as txn:
            metadata_bytes = txn.get(b"__metadata__")
            self.metadata = pickle.loads(metadata_bytes) if metadata_bytes else {}
        env.close()

        self.env = None
        self.fa_remap = None
        print(f"[ComplexPretrainDataset] {len(self.keys)} samples from {lmdb_path}")

        if fa_remap_index:
            mapping_path = Path(fa_remap_index)
            if not mapping_path.exists():
                raise FileNotFoundError(f"FA remap codebook not found: {mapping_path}")
            with mapping_path.open("r") as f:
                idx2cluster = json.load(f)
            remap_size = max(int(k) for k in idx2cluster.keys()) + 1
            remap = torch.empty(remap_size, dtype=torch.long)
            for idx_str, cluster in idx2cluster.items():
                remap[int(idx_str)] = int(cluster)
            self.fa_remap = remap

    def _init_db(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self) -> int:
        return len(self.keys)

    def _load_chain_tokens(self, data, chain_suffix):
        """Load and process tokens for one chain (A or B)."""
        # Sequence
        seq_key = f"seq_ids_{chain_suffix}"
        if seq_key not in data:
            seq_key = f"seq_id_{chain_suffix}"
        if seq_key in data:
            seq_ids = torch.from_numpy(data[seq_key]).long()
        else:
            return None, None, None

        # Backbone structure
        bb_ids = None
        bb_key = f"struct_ids_{chain_suffix}"
        if bb_key not in data:
            bb_key = f"struct_id_aminoaseed_{chain_suffix}"
        if bb_key in data:
            bb_ids = torch.from_numpy(data[bb_key]).long()

        # Full-atom structure
        fa_ids = None
        if 'fullatom' in self.struct_format:
            fa_key = f"struct_id_fullatom_{chain_suffix}"
            if fa_key not in data:
                fa_key = f"fa_ids_{chain_suffix}"
            if fa_key in data:
                fa_ids = torch.from_numpy(data[fa_key]).long()
                if self.fa_remap is not None:
                    fa_ids = self.fa_remap[fa_ids]

        # Shift token IDs
        if self.struct_format.count("foldseek") == 0:
            seq_ids = seq_ids + self.num_special_tokens
        if bb_ids is not None:
            bb_ids = bb_ids + self.num_special_tokens
        if fa_ids is not None:
            fa_ids = fa_ids + self.num_special_tokens

        # Truncate
        if self.max_length_per_chain is not None and len(seq_ids) > self.max_length_per_chain:
            seq_ids = seq_ids[:self.max_length_per_chain]
            if bb_ids is not None:
                bb_ids = bb_ids[:self.max_length_per_chain]
            if fa_ids is not None:
                fa_ids = fa_ids[:self.max_length_per_chain]

        return seq_ids, bb_ids, fa_ids

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._init_db()
        key = self.keys[idx]

        with self.env.begin() as txn:
            value = txn.get(key)
            if value is None:
                raise KeyError(f"Key {key} not found in LMDB")
            data = pickle.loads(value)

        pdb_id = data.get("pdb_id", key.decode())

        seq_A, bb_A, fa_A = self._load_chain_tokens(data, "A")
        seq_B, bb_B, fa_B = self._load_chain_tokens(data, "B")

        if seq_A is None or seq_B is None:
            raise ValueError(f"Missing chain data for {pdb_id}")

        len_A = len(seq_A)
        len_B = len(seq_B)

        # Concatenate chains
        seq_ids = torch.cat([seq_A, seq_B], dim=0)
        bb_ids = torch.cat([
            bb_A if bb_A is not None else torch.zeros_like(seq_A),
            bb_B if bb_B is not None else torch.zeros_like(seq_B),
        ], dim=0)

        fa_ids = None
        if fa_A is not None and fa_B is not None:
            fa_ids = torch.cat([fa_A, fa_B], dim=0)

        # Chain IDs: 0 for chain A, 1 for chain B
        chain_ids = torch.cat([
            torch.zeros(len_A, dtype=torch.long),
            torch.ones(len_B, dtype=torch.long),
        ], dim=0)

        # Per-chain position IDs (restart at 0 for chain B)
        position_ids = torch.cat([
            torch.arange(len_A, dtype=torch.long),
            torch.arange(len_B, dtype=torch.long),
        ], dim=0)

        # Interface information (skeleton - placeholder)
        # TODO: determine interface_info format (e.g., NxM distance matrix)
        interface_info = None
        if "interface_dist" in data and data["interface_dist"] is not None:
            interface_info = torch.from_numpy(data["interface_dist"]).float()
            # Truncate to match chain lengths
            interface_info = interface_info[:len_A, :len_B]

        sample = {
            "seq_ids": seq_ids,
            "bb_ids": bb_ids,
            "chain_ids": chain_ids,
            "position_ids": position_ids,
            "pdb_id": pdb_id,
            "len_A": len_A,
            "len_B": len_B,
        }
        if fa_ids is not None:
            sample["fa_ids"] = fa_ids
        if interface_info is not None:
            sample["interface_info"] = interface_info

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __del__(self):
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()
