#!/usr/bin/env bash
# One-shot setup for the probing benchmark.
#
# Pulls only what the four probing tasks need:
#   * splits/probing/{train,valid,test}.txt         (about 500 KB)
#   * probing/labels.pkl                            (per-residue labels)
#   * probing/tokens.lmdb                           (about 430 MB, our
#                                                    pre-tokenized subset,
#                                                    lets run_probing.sh
#                                                    skip PDB tokenization)
#
# After this script, run `bash examples/run_probing.sh`.

set -euo pipefail

BENCHMARK=./benchmark

command -v hf >/dev/null 2>&1 || {
    echo "FATAL: \`hf\` CLI not found. Install with \`pip install huggingface_hub[cli]\`." >&2
    exit 2
}

mkdir -p "$BENCHMARK"

echo "=== fetching probing assets (about 430 MB) ==="
hf download k-fold-structure/repsp-benchmark --repo-type dataset --local-dir "$BENCHMARK" \
    --include "splits/probing/*.txt" \
              "probing/labels.pkl" \
              "probing/tokens.lmdb/*"

echo
echo "Next: bash examples/run_probing.sh"
