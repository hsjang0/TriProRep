"""
Verify fullatom_tokenizer reproduces the structure tokens snapshotted by
build_reference.py (which uses the original favqvae package).

This script imports ONLY fullatom_tokenizer — the old package is not required
at runtime, only at snapshot-build time.

Usage:
    python -m structure_tokenize.fullatom_tokenizer.verify
    # (--ckpt defaults to the bundled fullatom_tokenizer.pt)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent

from structure_tokenize.fullatom_tokenizer.integration import load_fullatom_tokenizer, tokenize_fullatom_structure

DEFAULT_CKPT = HERE.parents[1] / "fullatom_tokenizer.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ref", default=str(HERE / "reference" / "tokens.npz"))
    args = parser.parse_args()

    ref_path = Path(args.ref)
    if not ref_path.exists():
        sys.exit(f"Reference file not found: {ref_path}\n"
                 f"Run `python -m structure_tokenize.fullatom_tokenizer.build_reference ...` first.")

    ref = np.load(ref_path, allow_pickle=True)
    pdb_paths = [str(p) for p in ref["_paths"]]
    keys = [Path(p).name for p in pdb_paths]
    print(f"Verifying {len(keys)} PDBs against {ref_path}.\n")

    torch.manual_seed(0)
    model = load_fullatom_tokenizer(args.ckpt, "pretrain_fullatom_image", device=args.device)

    n_pass = 0
    failures = []
    for pdb, key in zip(pdb_paths, keys):
        expected = ref[key]
        try:
            out = tokenize_fullatom_structure(model, Path(pdb))
        except Exception as e:
            failures.append((key, f"exception: {e}"))
            print(f"[FAIL] {key}: {e}")
            continue
        if out is None:
            failures.append((key, "tokenize returned None"))
            print(f"[FAIL] {key}: tokenize returned None")
            continue
        new_tok = np.asarray(out["fullatom_id"], dtype=np.int64)

        if expected.shape != new_tok.shape:
            failures.append((key, f"shape: ref={expected.shape} new={new_tok.shape}"))
            print(f"[FAIL] {key}: shape mismatch ref={expected.shape} new={new_tok.shape}")
            continue

        diff = expected != new_tok
        n_diff = int(diff.sum())
        if n_diff == 0:
            n_pass += 1
            print(f"[PASS] {key}: L={len(new_tok)}")
        else:
            failures.append((key, f"{n_diff}/{len(new_tok)} differ"))
            idx = np.where(diff)[0][:5]
            print(f"[FAIL] {key}: {n_diff}/{len(new_tok)} differ at {idx.tolist()}")

    print(f"\nSummary: {n_pass}/{len(keys)} match reference.")
    if failures:
        print("Failures:")
        for key, msg in failures:
            print(f"  {key}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
