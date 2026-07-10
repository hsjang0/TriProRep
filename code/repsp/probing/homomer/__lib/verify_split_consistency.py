#!/usr/bin/env python
"""Verify split consistency across staged encoders.

For each split, the staged `.pt` files should have **identical**
(pid, L) sequences — i.e. the same set of proteins in the same order
with the same per-protein residue counts. The X tensor will differ
(different encoder dim D, different embedding values) but pid_slices
must match for fair head-to-head comparison.

Usage:
  python verify_split_consistency.py \
      --root /path/to/probing_features \
      --encoders apo_ours apo_saprot apo_esm2 ...
"""
import argparse
import hashlib
import sys
from pathlib import Path

import torch


def pid_slices_signature(slices):
    """Hash of (pid, L) sequence — order-sensitive."""
    h = hashlib.sha256()
    for pid, s, e in slices:
        h.update(f"{pid}\t{e - s}\n".encode())
    return h.hexdigest()[:16]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--encoders", nargs="+", required=True)
    args = p.parse_args()

    root = Path(args.root)
    print(f"[verify] root={root}", flush=True)
    print(f"[verify] encoders={args.encoders}", flush=True)
    print()

    rows = []  # (encoder, split, n_proteins, n_res, D, sig)
    by_split = {"train": [], "valid": [], "test": []}
    missing = []
    corrupt = []
    for enc in args.encoders:
        for split in ("train", "valid", "test"):
            pt = root / enc / f"{split}.pt"
            if not pt.exists():
                missing.append(str(pt))
                continue
            # mmap=True: don't read X tensor bytes; just load metadata + pid_slices.
            # Drops verify cost from ~5 min (15×30 GB load) to seconds.
            try:
                d = torch.load(pt, map_location="cpu", weights_only=False, mmap=True)
            except Exception as e:
                corrupt.append((str(pt), type(e).__name__, str(e)[:80]))
                continue
            sig = pid_slices_signature(d["pid_slices"])
            n_proteins = len(d["pid_slices"])
            n_res = d["X"].shape[0]
            D = d["X"].shape[1]
            rows.append((enc, split, n_proteins, n_res, D, sig))
            by_split[split].append((enc, sig, n_proteins, n_res))
            del d  # release mmap explicitly

    if missing:
        print("[verify] MISSING staged files:", flush=True)
        for m in missing:
            print(f"  {m}", flush=True)
        print()

    if corrupt:
        print("[verify] CORRUPT staged files (skipped — re-stage required):", flush=True)
        for path, etype, msg in corrupt:
            print(f"  {path}  [{etype}] {msg}", flush=True)
        print()

    print(f"{'encoder':22s}  {'split':5s}  {'n_proteins':>10s}  {'n_res':>10s}  {'D':>5s}  {'sig':>16s}",
          flush=True)
    print("-" * 80)
    for r in rows:
        print(f"{r[0]:22s}  {r[1]:5s}  {r[2]:>10d}  {r[3]:>10d}  {r[4]:>5d}  {r[5]:>16s}",
              flush=True)
    print()

    # Consistency check: per-split, all encoders must share the same pid_slices sig
    # Corrupt files don't fail consistency directly (skipped above), but exit
    # non-zero so callers know something needs fixing.
    ok = (len(corrupt) == 0)
    for split, entries in by_split.items():
        if not entries:
            continue
        sigs = {e[1] for e in entries}
        nps = {e[2] for e in entries}
        nrs = {e[3] for e in entries}
        if len(sigs) == 1 and len(nps) == 1 and len(nrs) == 1:
            print(f"[OK]  {split}: all {len(entries)} encoders consistent  "
                  f"(sig={next(iter(sigs))} n_proteins={next(iter(nps))} n_res={next(iter(nrs))})",
                  flush=True)
        else:
            ok = False
            print(f"[FAIL] {split}: inconsistent across encoders!", flush=True)
            for e, sig, np_, nr in entries:
                print(f"    {e:22s}  sig={sig}  n_proteins={np_}  n_res={nr}",
                      flush=True)

    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
