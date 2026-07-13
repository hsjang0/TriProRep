"""Encode our pre-tokenized probing subset into a features LMDB.

Reads `benchmark/probing/tokens.lmdb` (our 40K probing subset, chain-A
apo tokens per AFid) and runs `encode()` for each record, writing
`work/features_<size>.lmdb` in the same on-disk schema `run_probing.sh`
expects.

This avoids the slow PDB-to-token step (~0.5 PDB/s) and runs the
encoder directly on tokens (~50+ records/s).
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import lmdb
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "code" / "triprorep"))

from inference import encode, load_encoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens_lmdb", required=True, type=Path)
    ap.add_argument("--output",      required=True, type=Path)
    ap.add_argument("--model",       default="650M",
                    choices=("35M", "150M", "650M", "3B"))
    ap.add_argument("--hf_repo",     default=None)
    ap.add_argument("--device",      default="cuda")
    ap.add_argument("--map_size_gb", type=int, default=64)
    args = ap.parse_args()

    hf_repo = args.hf_repo or f"k-fold-structure/triprorep-{args.model}"
    encoder = load_encoder(args.model, hf_repo=hf_repo, device=args.device)
    encoder_tag = f"ours_{args.model}"

    src = lmdb.open(str(args.tokens_lmdb), readonly=True, lock=False, readahead=False)
    dst = lmdb.open(str(args.output), map_size=args.map_size_gb * (1 << 30), subdir=True)

    n_ok = 0
    D = None
    t0 = time.time()
    with src.begin() as st, dst.begin(write=True) as dt:
        cur = st.cursor()
        # Deterministic order helps reproducibility.
        keys = sorted(k for k, _ in cur if k != b"__metadata__")
        print(f"[extract-tokens] {len(keys)} records, model={args.model}", flush=True)
        for i, k in enumerate(keys, 1):
            rec = pickle.loads(st.get(k))
            feats = encode(encoder, rec["seq_A"], rec["bb_A"], rec["fa_A"])
            feats = np.asarray(feats, dtype=np.float16)
            if D is None:
                D = int(feats.shape[1])
            dt.put(k, pickle.dumps(feats))
            n_ok += 1
            if i % 500 == 0:
                rate = i / (time.time() - t0 + 1e-9)
                eta  = (len(keys) - i) / max(rate, 1e-9)
                print(f"  [{i:>6}/{len(keys)}] {rate:.1f} rec/s  eta {eta/60:.1f} min", flush=True)

        dt.put(b"__metadata__", pickle.dumps({
            "n_samples": n_ok, "output_dim": D,
            "encoder": encoder_tag, "axis": "chain_A_only",
        }))

    src.close(); dst.close()
    print(f"[extract-tokens] DONE  ok={n_ok}  D={D}  wall={time.time()-t0:.1f}s  "
          f"({args.output})", flush=True)


if __name__ == "__main__":
    main()
