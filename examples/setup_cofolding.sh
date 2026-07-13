#!/usr/bin/env bash
# One-shot setup for co-folding training.
#
# Pulls everything the co-folding pipeline reads:
#   * splits/folding/{train,valid,test}.txt
#   * REPSP_PDB/monomer/{valid,test,train_000}.tar.gz   (~18 GB, apo conditioning)
#   * boltz_holo_{tokens,targets}/shard*.tar            (~120 GB, dimer training data)
#
# After this script, build features.lmdb with your encoder (see README
# section 'Co-folding (homodimer)') and launch training.

set -euo pipefail

BENCHMARK=./benchmark
PDBS_DIR=./REPSP_PDB/monomer

command -v hf >/dev/null 2>&1 || {
    echo "FATAL: \`hf\` CLI not found. Install with \`pip install huggingface_hub[cli]\`." >&2
    exit 2
}

mkdir -p "$BENCHMARK" "$PDBS_DIR"

echo "=== [1/3] folding splits + Boltz holo tokens + holo structure targets ==="
hf download k-fold-structure/repsp-benchmark --repo-type dataset --local-dir "$BENCHMARK" \
    --include "splits/folding/*.txt" \
              "boltz_holo_tokens/*" "boltz_holo_targets/*"

for d in boltz_holo_tokens boltz_holo_targets; do
    if compgen -G "$BENCHMARK/$d/shard*.tar" > /dev/null; then
        ( cd "$BENCHMARK/$d" && for t in shard*.tar; do tar xf "$t"; done && rm shard*.tar )
    fi
done

echo
echo "=== [2/3] monomer PDBs for apo conditioning (18 GB compressed) ==="
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
echo "'Co-folding (homodimer)'), then launch training."
