#!/usr/bin/env python
"""Extract per-encoder probing features.

Reads an encoder's full features LMDB (`[L_A, D]` fp16 per `AF-{id}`),
filters to the train/valid/test pids from the splits dir, and writes one
flat `.pt` per split that the probe loads directly — no further staging.

Output (per split):
  {out_dir}/{split}.pt = {
      'X':          fp16 [N_res, D],         # concatenated chain-A features
      'pid_slices': list[(pid, start, end)], # X[start:end] = residues of `pid`
  }

D / n_proteins / n_res are derivable: X.shape[1] / len(pid_slices) / X.shape[0].
Labels live in the probing labels pkl (separate), NOT here.
"""
import argparse
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lmdb
import numpy as np
import torch


def _read_split_pids(splits_dir: Path, split: str) -> list[str]:
    for candidate in (
        splits_dir / f"{split}.txt",
        splits_dir / f"splits_{split}.txt",
    ):
        if candidate.exists():
            return [l.strip() for l in candidate.read_text().splitlines() if l.strip()]
    return []


def _read_target_lengths(target_pkl: str) -> dict[str, int]:
    """Per-pid expected L from the homomer target pkl (truncate features to it)."""
    with open(target_pkl, "rb") as f:
        d = pickle.load(f)
    return {pid: rec["L"] for pid, rec in d.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features_lmdb", required=True,
                   help="Per-encoder features LMDB (chain-A only, [L_A, D] fp16).")
    p.add_argument("--splits_dir", required=True,
                   help="Dir with train/valid/test pid lists "
                        "(`{split}.txt` or upload-style `splits_{split}.txt`).")
    p.add_argument("--target_pkl", required=True,
                   help="Probing labels pkl — used here for canonical L per pid.")
    p.add_argument("--out_dir", required=True,
                   help="Where to write {train,valid,test}.pt for the probe.")
    p.add_argument("--max_per_split", type=int, default=None)
    p.add_argument("--n_workers", type=int, default=32,
                   help="parallel LMDB read threads (cold-cache page latency mask)")
    args = p.parse_args()

    print(f"[extract] features_lmdb={args.features_lmdb}", flush=True)
    print(f"[extract] splits_dir={args.splits_dir}", flush=True)
    print(f"[extract] target_pkl={args.target_pkl}", flush=True)
    print(f"[extract] out_dir={args.out_dir}", flush=True)

    L_lookup = _read_target_lengths(args.target_pkl)
    print(f"[extract] {len(L_lookup)} target entries (expected L)", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = Path(args.splits_dir)

    env = lmdb.open(args.features_lmdb, readonly=True, lock=False, readahead=False)

    # One thread-local txn per worker; LMDB read txns are cheap.
    def _probe_one(pid_idx):
        idx, pid = pid_idx
        L_target = L_lookup.get(pid)
        if L_target is None:
            return ("miss_target", idx, pid, None, None)
        with env.begin() as txn:
            raw = txn.get(pid.lower().encode())
        if raw is None:
            return ("miss_emb", idx, pid, None, None)
        arr = pickle.loads(raw)  # fp16 [L_A, D] or [2*L_A, D] tiled A|A
        L_arr = arr.shape[0]
        if L_arr == L_target:
            pass                       # chain-A features exactly.
        elif L_arr == 2 * L_target:
            arr = arr[:L_target]       # legacy A|A tiling. Take chain A.
        else:
            return ("miss_len", idx, pid, None, None)
        return ("ok", idx, pid, arr, L_target)

    for split in ("train", "valid", "test"):
        ids = _read_split_pids(splits_dir, split)
        if args.max_per_split:
            ids = ids[: args.max_per_split]
        if not ids:
            print(f"[extract] {split}: no ids, skip.", flush=True)
            continue

        # First pass: parallel LMDB read + decode → unordered results
        miss_emb = miss_target = miss_len = 0
        D = None
        results = [None] * len(ids)  # idx → (pid, arr, L) or None
        n_done = 0
        n_kept = 0
        t0 = time.time()
        print(f"[extract] {split} probe start (n_workers={args.n_workers})", flush=True)
        with ThreadPoolExecutor(max_workers=args.n_workers) as ex:
            for status, idx, pid, arr, L_target in ex.map(
                _probe_one, enumerate(ids), chunksize=8
            ):
                n_done += 1
                if status == "ok":
                    if D is None:
                        D = arr.shape[1]
                    results[idx] = (pid, arr, L_target)
                    n_kept += 1
                elif status == "miss_target":
                    miss_target += 1
                elif status == "miss_emb":
                    miss_emb += 1
                elif status == "miss_len":
                    miss_len += 1
                if n_done % 5000 == 0:
                    rate = n_done / (time.time() - t0 + 1e-9)
                    print(f"  {split} probe {n_done}/{len(ids)} kept={n_kept} "
                          f"miss_emb={miss_emb} miss_target={miss_target} "
                          f"miss_len={miss_len}  rate={rate:.0f} pid/s",
                          flush=True)

        # Reassemble plan in original (deterministic) keep-list order so
        # pid_slices are reproducible across encoders.
        plan = [r for r in results if r is not None]
        N_res = sum(p[2] for p in plan)
        print(f"[extract] {split} plan: kept={len(plan)} miss_emb={miss_emb} "
              f"miss_target={miss_target} miss_len={miss_len} "
              f"N_res={N_res} D={D}", flush=True)
        if not plan:
            continue

        # Second pass: copy into flat tensor
        X = torch.empty((N_res, D), dtype=torch.float16)
        pid_slices = []
        offset = 0
        t0 = time.time()
        for j, (pid, arr, L) in enumerate(plan):
            X[offset : offset + L].copy_(
                torch.from_numpy(np.ascontiguousarray(arr[:L])))
            pid_slices.append((pid, offset, offset + L))
            offset += L
            if (j + 1) % 5000 == 0:
                print(f"  {split} copy {j+1}/{len(plan)} "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)
        # Atomic write: tmp file then rename. SIGKILL during torch.save would
        # otherwise leave a partial zip file at the final path, corrupting any
        # prior valid .pt that was being overwritten. Rename on POSIX is atomic
        # so the final path either has the old or new file, never partial.
        out_pt = out_dir / f"{split}.pt"
        out_pt_tmp = out_dir / f"{split}.pt.tmp"
        torch.save({"X": X, "pid_slices": pid_slices}, out_pt_tmp)
        out_pt_tmp.replace(out_pt)  # atomic
        sz_gb = out_pt.stat().st_size / 1e9
        print(f"[extract] {split} → {out_pt}  N_res={N_res} D={D} "
              f"({sz_gb:.2f} GB)", flush=True)

    env.close()


if __name__ == "__main__":
    main()
