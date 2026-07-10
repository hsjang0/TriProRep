#!/usr/bin/env python
"""Extract per-residue features from a PDB directory using our encoder.

Reads `<split>_monomer.txt` lists, looks up each PDB in `--pdbs_dir`, runs
`embed_pdb` (lazy-loads bb/fa tokenizers from the HF model repo on first
call), and writes a single features LMDB:

    key   = `<af-id>` (lowercase, e.g. `af-0000000065760022`)
    value = `pickle.dumps(np.ndarray[L_A, D], dtype=fp16)`
    `__metadata__` = {n_samples, output_dim, encoder, axis: "chain_A_only"}

The output LMDB matches the schema consumed by both
`code/repsp/probing/homomer/__lib/extract_probing_features.py` (stages
per-split `.pt`) and the folding/co-folding launchers
(`++data.feature_paths.{apo_s,repa_target_s}`).

Default flow (when `examples/run_benchmark.sh` calls this):

    python examples/extract_features_from_pdbs.py \\
        --pdbs_dir ./pdbs/monomer \\
        --splits_dir ./benchmark/splits/probing \\
        --model 650M \\
        --output ./work/features.lmdb
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
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "code" / "triprorep"))

from inference import embed_pdb, load_encoder  # noqa: E402


def _read_input_pids(splits_dir: Path, split: str) -> list[str]:
    for candidate in (
        splits_dir / f"{split}.txt",
        splits_dir / f"splits_{split}.txt",
        splits_dir / f"{split}_monomer.txt",
    ):
        if candidate.exists():
            return [l.strip() for l in candidate.read_text().splitlines() if l.strip()]
    return []


def _resolve_pdb(pdbs_dir: Path, pid: str) -> Path | None:
    """Look up the monomer PDB for a homodimer AFid.

    Accepts both `<AF-id>_monomer.pdb` (the convention shipped in the
    REPSP_PDB/monomer/ folder) and the bare `<AF-id>.pdb` fallback.
    """
    for name in (
        f"{pid}_monomer.pdb", f"{pid.upper()}_monomer.pdb", f"{pid.lower()}_monomer.pdb",
        f"{pid}.pdb", f"{pid.upper()}.pdb", f"{pid.lower()}.pdb",
    ):
        p = pdbs_dir / name
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdbs_dir", required=True,
                    help="Directory containing PDB files.")
    ap.add_argument("--splits_dir", default=None,
                    help="Directory with `<split>_monomer.txt` (one AF-id per "
                         "line). Use this for the benchmark splits. For an "
                         "arbitrary user PDB collection, use --pdb_glob instead.")
    ap.add_argument("--pdb_glob", default=None,
                    help="Glob (relative to --pdbs_dir) to pull all matching "
                         "PDBs without a splits file. Example: '*.pdb' or "
                         "'**/*.pdb'. Mutually exclusive with --splits_dir.")
    ap.add_argument("--key_from_filename",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="When using --pdb_glob, use the file stem as the LMDB "
                         "key (default). Pass --no-key_from_filename to use the "
                         "absolute path instead.")
    ap.add_argument("--model", default="650M",
                    choices=("35M", "150M", "650M", "3B"),
                    help="Encoder size to load from HuggingFace Hub.")
    ap.add_argument("--hf_repo", default=None,
                    help="Override HF repo (default: k-fold-structure/triprorep-<model>).")
    ap.add_argument("--ckpt", default=None,
                    help="Use a local .ckpt instead of HF download.")
    ap.add_argument("--output", required=True,
                    help="Output features LMDB path.")
    ap.add_argument("--device", default="cuda",
                    help="`cuda` (default), `cuda:N`, or `cpu`.")
    ap.add_argument("--map_size_gb", type=int, default=1024,
                    help="LMDB map_size in GB (overprovision, file grows lazily).")
    ap.add_argument("--splits", default="train,valid,test",
                    help="Comma-separated split names to extract.")
    args = ap.parse_args()

    hf_repo = args.hf_repo or f"k-fold-structure/triprorep-{args.model}"
    print(f"[extract] model={args.model}  hf_repo={hf_repo}", flush=True)
    encoder = load_encoder(args.model, hf_repo=hf_repo, ckpt=args.ckpt,
                           device=args.device)
    encoder_tag = f"ours_{args.model}"

    pdbs_dir = Path(args.pdbs_dir)
    if (args.splits_dir is None) == (args.pdb_glob is None):
        raise SystemExit("[extract] specify exactly one of --splits_dir / --pdb_glob.")

    out_env = lmdb.open(args.output,
                        map_size=args.map_size_gb * (1 << 30),
                        subdir=True)

    pid_order: list[tuple[str, Path]] = []  # (lmdb_key, pdb_path)

    if args.splits_dir is not None:
        # Benchmark-style: pull AFid lists from `<split>_monomer.txt`.
        seen: set[str] = set()
        splits_dir = Path(args.splits_dir)
        for split in args.splits.split(","):
            split = split.strip()
            ids = _read_input_pids(splits_dir, split)
            if not ids:
                print(f"[extract] {split}: no ids found in {splits_dir}", flush=True)
                continue
            n_new = 0
            for pid in ids:
                if pid in seen:
                    continue
                seen.add(pid)
                p = _resolve_pdb(pdbs_dir, pid)
                if p is None:
                    continue  # missing PDB will be counted as miss below
                pid_order.append((pid, p))
                n_new += 1
            print(f"[extract] {split}: {len(ids)} ids ({n_new} found on disk), "
                  f"total queue {len(pid_order)}", flush=True)
    else:
        # Free-form: walk a PDB collection without a splits file.
        for pdb_path in sorted(pdbs_dir.glob(args.pdb_glob)):
            if not pdb_path.is_file():
                continue
            key = pdb_path.stem if args.key_from_filename else str(pdb_path)
            pid_order.append((key, pdb_path))
        print(f"[extract] --pdb_glob='{args.pdb_glob}': {len(pid_order)} PDBs", flush=True)

    if not pid_order:
        raise SystemExit("[extract] empty queue: nothing to do.")

    n_ok = n_err = 0
    D = None
    t0 = time.time()
    with out_env.begin(write=True) as txn:
        for i, (key, pdb_path) in enumerate(pid_order):
            try:
                feats = embed_pdb(encoder, str(pdb_path), hf_repo=hf_repo)
            except Exception as e:
                n_err += 1
                print(f"[extract]   ! {key}: {type(e).__name__}: {e}", flush=True)
                continue
            feats = np.asarray(feats, dtype=np.float16)
            if D is None:
                D = int(feats.shape[1])
            txn.put(key.lower().encode(), pickle.dumps(feats))
            n_ok += 1
            if (i + 1) % 100 == 0:
                rate = (i + 1) / (time.time() - t0 + 1e-9)
                eta = (len(pid_order) - i - 1) / max(rate, 1e-9)
                print(f"  [{i+1:>6}/{len(pid_order)}] ok={n_ok} err={n_err}  "
                      f"{rate:.1f} pdb/s  eta {eta/60:.1f} min", flush=True)

        txn.put(b"__metadata__", pickle.dumps({
            "n_samples": n_ok,
            "output_dim": D,
            "encoder": encoder_tag,
            "axis": "chain_A_only",
        }))

    out_env.close()
    dt = time.time() - t0
    print(f"[extract] DONE  ok={n_ok}  err={n_err}  D={D}  "
          f"wall={dt/60:.1f} min  ({args.output})", flush=True)


if __name__ == "__main__":
    main()
