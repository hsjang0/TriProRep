#!/usr/bin/env bash
# Download + prepare the benchmark.
#
# Pulls the benchmark assets (splits, labels, Boltz tokens, and the
# monomer REPSP_PDB tarballs) from `k-fold-structure/repsp-benchmark` on
# HuggingFace and untars them into place.
#
# Layout after this script:
#
#   ./benchmark/
#     splits/{folding,probing}/{train,valid,test}.txt
#     probing/labels.pkl
#     boltz_apo_{tokens,targets}/    boltz_holo_{tokens,targets}/
#   ./REPSP_PDB/monomer/<AF-id>_monomer.pdb    # chain A, apo prediction
#
# The AFid in every PDB filename is the same homodimer identifier taken
# from `benchmark/splits/{folding,probing}/{train,valid,test}.txt`. Raw
# homodimer PDBs are not fetched by this script (none of the shipped
# benchmarks need them — co-folding trains on the Boltz-tokenized dimer
# under `boltz_holo_{tokens,targets}` above, not on raw dimer PDB). If
# you need them for label recomputation or custom analysis, fetch them
# directly:
#
#   hf download k-fold-structure/repsp-benchmark --repo-type dataset \
#       --include "REPSP_PDB/homodimer/*"
#
# Env vars:
#   SPLIT=test|valid|train|all   which monomer PDB shard(s) to fetch
#                                (default: test, ~130 MB compressed).
#                                Use `all` for the full training set.
#
# After this script, run `bash examples/run_benchmark.sh`.

set -euo pipefail

BENCHMARK=${BENCHMARK:-./benchmark}
PDBS_DIR=${PDBS_DIR:-./REPSP_PDB}
SPLIT=${SPLIT:-test}
REPO=k-fold-structure/repsp-benchmark

command -v hf >/dev/null 2>&1 || {
    echo "FATAL: \`hf\` CLI not found. Install with \`pip install huggingface_hub[cli]\`." >&2
    exit 2
}

mkdir -p "$BENCHMARK" "$PDBS_DIR/monomer"

echo "=== [1/2] benchmark assets (splits + labels + Boltz tokens) ==="
hf download "$REPO" --repo-type dataset --local-dir "$BENCHMARK" \
    --exclude "REPSP_PDB/*"
if compgen -G "$BENCHMARK/boltz_holo_tokens/shard*.tar" > /dev/null; then
    ( cd "$BENCHMARK/boltz_holo_tokens" && for t in shard*.tar; do tar xf "$t"; done && rm shard*.tar )
fi

# Which monomer PDB shards do we want?
case "$SPLIT" in
    test)  monomer_incl=("REPSP_PDB/monomer/test.tar.gz") ;;
    valid) monomer_incl=("REPSP_PDB/monomer/valid.tar.gz") ;;
    train) monomer_incl=("REPSP_PDB/monomer/train_*.tar.gz") ;;
    all)   monomer_incl=("REPSP_PDB/monomer/*.tar.gz") ;;
    *) echo "FATAL: SPLIT must be test|valid|train|all (got '$SPLIT')" >&2; exit 1 ;;
esac

echo
echo "=== [2/2] fetch monomer PDB shards (SPLIT=$SPLIT) ==="
hf download "$REPO" --repo-type dataset --local-dir "$BENCHMARK" \
    --include "${monomer_incl[@]}"
for tar in "$BENCHMARK/REPSP_PDB/monomer"/*.tar.gz; do
    [[ -f "$tar" ]] || continue
    echo "  unpack $(basename "$tar") -> $PDBS_DIR/monomer/"
    tar xzf "$tar" -C "$PDBS_DIR/monomer/"
    rm "$tar"
done

n_mono=$(find "$PDBS_DIR/monomer" -name "*.pdb" -printf . 2>/dev/null | wc -c)
echo
echo "monomer/  $n_mono pdb files"

echo
echo "Setup complete. Next:"
echo "  bash examples/run_benchmark.sh"
