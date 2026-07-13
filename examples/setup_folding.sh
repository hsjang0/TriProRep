#!/usr/bin/env bash
# One-shot setup for REPA-supervised folding training.
#
# Pulls everything the folding pipeline reads:
#   * splits/folding/{train,valid,test}.txt         (390,627 / 400 / 1,000)
#   * REPSP_PDB/monomer/{valid,test,train_000}.tar.gz   (~18 GB compressed)
#   * boltz_apo_{tokens,targets}/shard*.tar         (Boltz-format apo structures)
#
# After this script, build features.lmdb with your encoder (see README
# section 'Folding (REPA)') and launch training.

set -euo pipefail

BENCHMARK=./benchmark
PDBS_DIR=./REPSP_PDB/monomer

command -v hf >/dev/null 2>&1 || {
    echo "FATAL: \`hf\` CLI not found. Install with \`pip install huggingface_hub[cli]\`." >&2
    exit 2
}

mkdir -p "$BENCHMARK" "$PDBS_DIR"

echo "=== [1/3] folding splits + Boltz apo tokens + apo structure targets ==="
hf download k-fold-structure/repsp-benchmark --repo-type dataset --local-dir "$BENCHMARK" \
    --include "splits/folding/*.txt" \
              "boltz_apo_tokens/*" "boltz_apo_targets/*"

for d in boltz_apo_tokens boltz_apo_targets; do
    if compgen -G "$BENCHMARK/$d/shard*.tar" > /dev/null; then
        ( cd "$BENCHMARK/$d" && for t in shard*.tar; do tar xf "$t"; done && rm shard*.tar )
    fi
done

echo
echo "=== [2/3] monomer PDBs (18 GB compressed, all 392K records) ==="
hf download k-fold-structure/repsp-benchmark --repo-type dataset --local-dir "$BENCHMARK" \
    --include "REPSP_PDB/monomer/valid.tar.gz" \
              "REPSP_PDB/monomer/test.tar.gz" \
              "REPSP_PDB/monomer/train_*.tar.gz"
for tar in "$BENCHMARK/REPSP_PDB/monomer"/*.tar.gz; do
    [[ -f "$tar" ]] || continue
    echo "  unpack $(basename "$tar")"
    tar xzf "$tar" -C "$PDBS_DIR"
    rm "$tar"
done

n=$(find "$PDBS_DIR" -name "*.pdb" -printf . 2>/dev/null | wc -c)
echo
echo "REPSP_PDB/monomer/  $n pdb files"
echo
echo "Next: build features.lmdb with your encoder (see README section"
echo "'Folding (REPA)'), then launch training."
