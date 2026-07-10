#!/usr/bin/env python
"""Pack REPSP_PDB/{monomer,homodimer}/*.pdb into sharded .tar.gz files.

Reads the split lists (`train.txt`, `valid.txt`, `test.txt`) and streams
matching PDBs into gzipped tar shards, closing the current shard once its
compressed size passes SHARD_TARGET_BYTES (default 40 GB, well under
HF's 50 GB LFS limit).

Output layout:
    <out_dir>/monomer/valid.tar.gz
    <out_dir>/monomer/test.tar.gz
    <out_dir>/monomer/train_000.tar.gz [train_001.tar.gz ...]
    <out_dir>/homodimer/valid.tar.gz
    <out_dir>/homodimer/test.tar.gz
    <out_dir>/homodimer/train_000.tar.gz [...]

Usage:
    python scripts/pack_repsp_pdbs.py \\
        --pdbs_root /mnt/.../REPSP_PDB \\
        --splits_dir /mnt/.../repsp-benchmark/splits/folding \\
        --out_dir /mnt/.../repsp-benchmark-stage/REPSP_PDB \\
        --shard_target_gb 40
"""
from __future__ import annotations

import argparse
import gzip
import os
import tarfile
import time
from pathlib import Path


def _open_shard(out_dir: Path, kind: str, split: str, idx: int, is_sharded: bool):
    """Return (path, gzip_file, tarfile)."""
    if is_sharded:
        name = f"{split}_{idx:03d}.tar.gz"
    else:
        name = f"{split}.tar.gz"
    path = out_dir / kind / name
    path.parent.mkdir(parents=True, exist_ok=True)
    gz = gzip.open(path, "wb", compresslevel=6)
    tar = tarfile.open(fileobj=gz, mode="w|")
    return path, gz, tar


def _close_shard(path, gz, tar):
    tar.close()
    gz.close()
    sz = path.stat().st_size
    print(f"  wrote {path.name}  {sz/1e9:.2f} GB", flush=True)


def pack_one(pdbs_root: Path, splits_dir: Path, out_dir: Path,
             kind: str, split: str, shard_target: int, is_sharded_split: bool):
    """kind = 'monomer' | 'homodimer', split = 'train' | 'valid' | 'test'."""
    src_dir = pdbs_root / kind
    split_file = splits_dir / f"{split}.txt"
    if not split_file.exists():
        print(f"  ! missing {split_file}, skip", flush=True)
        return

    afids = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
    print(f"=== pack {kind}/{split}: {len(afids)} AFids ===", flush=True)

    shard_idx = 0
    n_packed = 0
    n_missing = 0
    path, gz, tar = _open_shard(out_dir, kind, split, shard_idx, is_sharded_split)
    t0 = time.time()

    for i, afid in enumerate(afids, 1):
        if kind == "monomer":
            pdb = src_dir / f"{afid}_monomer.pdb"
        else:
            pdb = src_dir / f"{afid}.pdb"
        if not pdb.exists():
            n_missing += 1
            continue
        tar.add(str(pdb), arcname=pdb.name)
        n_packed += 1

        if is_sharded_split and n_packed > 0 and n_packed % 500 == 0:
            gz.flush()
            cur_sz = path.stat().st_size
            if cur_sz >= shard_target:
                _close_shard(path, gz, tar)
                shard_idx += 1
                path, gz, tar = _open_shard(out_dir, kind, split, shard_idx, True)

        if i % 20000 == 0:
            rate = i / (time.time() - t0 + 1e-9)
            print(f"  [{i:>6}/{len(afids)}] packed={n_packed} miss={n_missing} "
                  f"{rate:.0f} pdb/s", flush=True)

    _close_shard(path, gz, tar)
    print(f"  DONE {kind}/{split}: packed={n_packed} miss={n_missing} "
          f"shards={shard_idx+1} wall={time.time()-t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdbs_root", required=True, type=Path,
                    help="REPSP_PDB root with monomer/ and homodimer/ subdirs.")
    ap.add_argument("--splits_dir", required=True, type=Path,
                    help="Dir with train.txt / valid.txt / test.txt.")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--shard_target_gb", type=int, default=40)
    ap.add_argument("--kinds", default="monomer,homodimer",
                    help="Comma-separated: monomer|homodimer.")
    ap.add_argument("--splits", default="valid,test,train",
                    help="Comma-separated: train|valid|test.")
    args = ap.parse_args()

    shard_target = args.shard_target_gb * (1 << 30)

    for kind in args.kinds.split(","):
        for split in args.splits.split(","):
            kind, split = kind.strip(), split.strip()
            # Only 'train' gets sharded; valid + test always fit in one file.
            pack_one(args.pdbs_root, args.splits_dir, args.out_dir,
                     kind, split, shard_target, is_sharded_split=(split == "train"))


if __name__ == "__main__":
    main()
