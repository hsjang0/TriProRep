#!/usr/bin/env bash
# One-shot setup for the probing benchmark.
#
# Pulls only what the four probing tasks need:
#   * splits/probing/{train,valid,test}.txt
#   * probing/labels.pkl                          (per-residue labels)
#   * REPSP_PDB/monomer/probing.tar.gz            (40K monomer PDBs, ~2 GB)
#
# After this script, run `bash examples/run_probing.sh`.

set -euo pipefail

BENCHMARK=./benchmark
PDBS_DIR=./REPSP_PDB/monomer

command -v hf >/dev/null 2>&1 || {
    echo "FATAL: \`hf\` CLI not found. Install with \`pip install huggingface_hub[cli]\`." >&2
    exit 2
}

mkdir -p "$BENCHMARK" "$PDBS_DIR"

echo "=== [1/2] splits + labels ==="
hf download k-fold-structure/repsp-benchmark --repo-type dataset --local-dir "$BENCHMARK" \
    --include "splits/probing/*.txt" "probing/labels.pkl"

echo
echo "=== [2/2] probing monomer PDBs (about 2 GB, 40K records) ==="
hf download k-fold-structure/repsp-benchmark --repo-type dataset --local-dir "$BENCHMARK" \
    --include "REPSP_PDB/monomer/probing.tar.gz"
tar xzf "$BENCHMARK/REPSP_PDB/monomer/probing.tar.gz" -C "$PDBS_DIR"
rm "$BENCHMARK/REPSP_PDB/monomer/probing.tar.gz"

n=$(find "$PDBS_DIR" -name "*.pdb" -printf . 2>/dev/null | wc -c)
echo
echo "REPSP_PDB/monomer/  $n pdb files"
echo
echo "Next: bash examples/run_probing.sh"
