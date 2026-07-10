"""
Standalone FA-VQVAE tokenizer inference entry point.

Usage:
    python -m structure_tokenize.fullatom_tokenizer.inference \
        --pdb /path/to/protein.pdb \
        --ckpt /path/to/fullatom_tokenizer.pt \
        [--config pretrain_fullatom_image] \
        [--device cuda] \
        [--output tokens.npy]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# allow running as a script as well as a module
_here = Path(__file__).resolve()
if __package__ in (None, ""):
    # add the tokenizer component root (parent of structure_tokenize/)
    sys.path.insert(0, str(_here.parents[2]))

from structure_tokenize.fullatom_tokenizer.integration import (
    load_fullatom_tokenizer,
    tokenize_fullatom_structure,
)

DEFAULT_CKPT = _here.parents[2] / "fullatom_tokenizer.pt"


def tokenize(pdb_path: str, ckpt_path: str = str(DEFAULT_CKPT),
             cfg_name: str = "pretrain_fullatom_image",
             device: str = "cuda") -> np.ndarray:
    """Load tokenizer + tokenize a single PDB. Returns int array of struct token ids [L]."""
    tokenizer = load_fullatom_tokenizer(ckpt_path, cfg_name, device=device)
    out = tokenize_fullatom_structure(tokenizer, Path(pdb_path))
    if out is None:
        raise RuntimeError(f"Tokenization failed for {pdb_path}")
    return np.asarray(out["fullatom_id"], dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="FA-VQVAE PDB → structure-token tokenizer.")
    parser.add_argument("--pdb", required=True, help="Path to input PDB file.")
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT),
                        help=f"Path to FA-VQVAE checkpoint (default: {DEFAULT_CKPT.name} bundled with the package).")
    parser.add_argument("--config", default="pretrain_fullatom_image",
                        help="Config name under structure_tokenize/fullatom_tokenizer/configs/ (without .yaml).")
    parser.add_argument("--device", default="cuda", help="cuda | cpu")
    parser.add_argument("--output", default=None,
                        help="Optional path to write tokens (.npy). If omitted, prints to stdout.")
    args = parser.parse_args()

    tokens = tokenize(args.pdb, args.ckpt, args.config, args.device)

    if args.output:
        np.save(args.output, tokens)
        print(f"Saved {len(tokens)} tokens to {args.output}")
    else:
        # print one token per line for easy diffing
        for t in tokens.tolist():
            print(t)


if __name__ == "__main__":
    main()
