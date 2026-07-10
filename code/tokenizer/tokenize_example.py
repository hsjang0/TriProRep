"""
Build monomer tokenization LMDB from AFDB monomer PDBs (filter_alphafold).

Per-sample LMDB entry (key = pdb_id = AF-{16digit}):

    {
      "pdb_id": str,
      "seq":    np.int64[L],    # AA tokens
      "bb":     np.int64[L],    # aminoaseed backbone tokens
      "fa":     np.int64[L],    # fullatom tokens
    }

Key convention: ``pdb_id = <name>`` where ``<name>.pdb`` matches
``/path/to/afdb_apo_pdb/chunk_*/<name>.pdb``. Downstream code can join on
pdb_id across LMDBs without any mapping.

Single-chain reference example: each PDB yields one (seq, bb, fa) token triple
written to LMDB.  Use it as a template for batch-tokenizing your own structures
(the released LMDBs were produced by the same encode_structure / worker-pool
path exercised here).

Usage:
  python tokenize_example.py \\
      --pdb_root /path/to/afdb_apo_pdb \\
      --output_dir /path/to/lmdb/afdb_monomer_alphafold_fullatom_image \\
      --struct_format fullatom_image \\
      --gpu_ids 0,1,2,3,4,5,6,7
"""
import argparse
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import List, Tuple

import lmdb
import numpy as np
from tqdm import tqdm

# The tokenizer component root (this directory) goes on sys.path so the helper
# packages — preprocess/, structure_tokenize/ (incl. fullatom_tokenizer/),
# openfold/ — resolve as top-level imports.
TOKENIZER_ROOT = os.path.dirname(os.path.abspath(__file__))
if TOKENIZER_ROOT not in sys.path:
    sys.path.insert(0, TOKENIZER_ROOT)

warnings.filterwarnings("ignore")

from preprocess import (
    StructInitConfig,
    configure_spawn_start_method,
    encode_structure,
    init_structure_worker,
    run_worker_pool,
    seed_all,
    select_num_workers,
)


# Same mapping as _preprocess_apo_holo.py: encode_structure returns
# seq_id / struct_id_aminoaseed / struct_id_fullatom — we relabel to the
# short seq/bb/fa form per the user's schema.
KEY_MAP = {
    "seq_id": "seq",
    "struct_id_aminoaseed": "bb",
    "struct_id_fullatom": "fa",
    "struct_id": "bb",  # fallback for aminoaseed-only format
}


def _to_numpy(obj):
    import torch
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        return {k: _to_numpy(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def enumerate_pdbs(pdb_root: str) -> List[Tuple[str, str]]:
    """Return [(pdb_name, full_path)] for every .pdb under pdb_root/*/."""
    items = []
    root = Path(pdb_root)
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        for pdb in sorted(subdir.glob("*.pdb")):
            items.append((pdb.name, str(pdb)))
    return items


# ---------------------------------------------------------------------------
# Worker: tokenize one monomer PDB
# ---------------------------------------------------------------------------
def process_monomer_worker(args_tuple):
    """Tokenize one AFDB monomer PDB.

    Returns (pdb_id, entry_dict) or None.  AFDB monomer files have a single
    chain — ``encode_structure(..., chain=None)`` picks it up automatically.
    """
    pdb_name, pdb_path, struct_format = args_tuple
    try:
        result = encode_structure(pdb_path, struct_format, chain=None)
        if result is None:
            return None

        pdb_id = pdb_name[:-4] if pdb_name.endswith(".pdb") else pdb_name
        entry = {"pdb_id": pdb_id}
        for raw_key, short_key in KEY_MAP.items():
            if raw_key in result:
                entry[short_key] = _to_numpy(result[raw_key])
        # Require at least seq + (bb or fa), else the sample is useless.
        if "seq" not in entry or ("bb" not in entry and "fa" not in entry):
            return None
        return pdb_id, entry

    except Exception as e:
        sys.stderr.write(f"[WARN] {pdb_name}: {e}\n")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def create_monomer_lmdb(
    pdb_root: str,
    output_dir: str,
    struct_format: str = "fullatom_image",
    seed: int = 42,
    num_workers: int | None = None,
    gpu_ids: list[int] | None = None,
    workers_per_gpu: int = 1,
    bb_ckpt_path: str | None = None,
    fa_ckpt_path: str | None = None,
    fa_cfg: str | None = None,
    map_size: int = 1099511627776 * 2,
    n_shards: int = 1,
    shard_i: int = 0,
):
    configure_spawn_start_method()
    seed_all(seed)

    items = enumerate_pdbs(pdb_root)
    if not items:
        print(f"No PDBs under {pdb_root}")
        return
    print(f"Found {len(items):,} monomer PDBs under {pdb_root}", flush=True)

    # Sort by pdb name so sharding is deterministic across nodes regardless
    # of filesystem walk order.
    items.sort(key=lambda nm: nm[0])

    if n_shards > 1:
        items = [x for i, x in enumerate(items) if i % n_shards == shard_i]
        print(f"[shard] shard {shard_i}/{n_shards}: {len(items):,} samples",
              flush=True)

    process_args = [
        (name, path, struct_format) for name, path in items
    ]

    num_workers_resolved = select_num_workers(num_workers, gpu_ids, workers_per_gpu)
    print(f"Processing with {num_workers_resolved} workers ...", flush=True)

    init_config = StructInitConfig(
        struct_format=struct_format,
        bb_ckpt_path=bb_ckpt_path,
        fa_ckpt_path=fa_ckpt_path,
        fa_cfg=fa_cfg,
        label_map={},
        gpu_ids=gpu_ids,
    )

    t0 = time.time()
    results = run_worker_pool(
        process_args,
        process_monomer_worker,
        init_config,
        num_workers_resolved,
        desc="monomer tokenize",
    )
    elapsed = time.time() - t0
    print(f"Tokenized {len(results):,} / {len(items):,} samples in {elapsed:.0f}s",
          flush=True)

    # Write LMDB — per-shard filename so concurrent nodes don't collide.
    os.makedirs(output_dir, exist_ok=True)
    lmdb_name = "train.lmdb" if n_shards == 1 else f"train.shard{shard_i:02d}.lmdb"
    lmdb_path = os.path.join(output_dir, lmdb_name)
    print(f"Writing LMDB to {lmdb_path} ...", flush=True)
    env = lmdb.open(lmdb_path, map_size=map_size)
    with env.begin(write=True) as txn:
        for pdb_id, entry in results:
            txn.put(pdb_id.encode(), pickle.dumps(entry))
        metadata = {
            "num_samples": len(results),
            "struct_format": struct_format,
            "pdb_root": pdb_root,
            "task": "monomer_alphafold",
            "key_convention": "pdb_id matches the source PDB filename stem",
            "schema": {
                "seq": "AA tokens (np.int64[L])",
                "bb":  "aminoaseed backbone tokens (np.int64[L])",
                "fa":  "fullatom tokens (np.int64[L])",
            },
        }
        txn.put(b"__metadata__", pickle.dumps(metadata))
    env.sync()
    env.close()
    print(f"Done. {len(results):,} entries → {lmdb_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Build monomer tokenization LMDB "
                                            "from filter_alphafold AFDB PDBs")
    p.add_argument("--pdb_root", type=str,
                   default="/path/to/afdb_apo_pdb",
                   help="Root dir: {pdb_root}/{subdir}/<name>.pdb")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output LMDB dir.  Default: "
                        "/path/to/lmdb/"
                        "afdb_monomer_alphafold_{fa_cfg}")
    p.add_argument("--struct_format", type=str, default="fullatom_image",
                   choices=["fullatom_image", "aminoaseed"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument("--workers_per_gpu", type=int, default=2)
    p.add_argument("--bb_ckpt_path", type=str,
                   default=os.path.join(TOKENIZER_ROOT, "backbone_tokenizer.pt"),
                   help="aminoaseed backbone tokenizer ckpt (bundled).")
    p.add_argument("--fa_ckpt_path", type=str,
                   default=os.path.join(TOKENIZER_ROOT, "fullatom_tokenizer.pt"),
                   help="full-atom VQ-VAE tokenizer ckpt (bundled).")
    p.add_argument("--fa_cfg", type=str, default="pretrain_fullatom_image")
    p.add_argument("--map_size_tb", type=float, default=2.0,
                   help="LMDB map size in TB")
    p.add_argument("--n_shards", type=int, default=1,
                   help="Total parallel shards across all nodes")
    p.add_argument("--shard_i", type=int, default=0,
                   help="This node's shard index in [0, n_shards)")
    args = p.parse_args()
    if not (0 <= args.shard_i < args.n_shards):
        sys.exit(f"--shard_i must be in [0, {args.n_shards})")

    output_dir = (args.output_dir
                  or f"/path/to/lmdb/"
                     f"afdb_monomer_alphafold_{args.fa_cfg}")
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()] or None

    create_monomer_lmdb(
        pdb_root=args.pdb_root,
        output_dir=output_dir,
        struct_format=args.struct_format,
        seed=args.seed,
        num_workers=args.num_workers,
        gpu_ids=gpu_ids,
        workers_per_gpu=args.workers_per_gpu,
        bb_ckpt_path=args.bb_ckpt_path,
        fa_ckpt_path=args.fa_ckpt_path,
        fa_cfg=args.fa_cfg,
        map_size=int(args.map_size_tb * 1024**4),
        n_shards=args.n_shards,
        shard_i=args.shard_i,
    )


if __name__ == "__main__":
    main()
