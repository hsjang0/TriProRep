#!/usr/bin/env bash
# Download + prepare the benchmark.
#
# Pulls the benchmark assets (splits, labels, Boltz tokens, and REPSP_PDB
# tarballs) from `k-fold-structure/repsp-benchmark` on HuggingFace and
# untars everything into place.
#
# Layout after this script:
#
#   ./benchmark/
#     splits/{folding,probing}/{train,valid,test}.txt
#     probing/labels.pkl
#     boltz_apo_{tokens,targets}/    boltz_holo_{tokens,targets}/
#   ./REPSP_PDB/
#     monomer/<AF-id>_monomer.pdb          # chain A, apo prediction
#     homodimer/<AF-id>.pdb                # chain A + chain B, holo dimer
#
# The AFid in every PDB filename is the same homodimer identifier taken
# from `benchmark/splits/{folding,probing}/{train,valid,test}.txt`.
#
# Env vars:
#   SPLIT=test|valid|train|all   which PDB shards to fetch (default test)
#   NEED_HOMODIMER=1             also fetch the homodimer shards
#                                (required only for folding / co-folding)
#
# After this script, run `bash examples/run_benchmark.sh`.

set -euo pipefail

BENCHMARK=${BENCHMARK:-./benchmark}
PDBS_DIR=${PDBS_DIR:-./REPSP_PDB}
SPLIT=${SPLIT:-test}
NEED_HOMODIMER=${NEED_HOMODIMER:-0}
REPO=k-fold-structure/repsp-benchmark

command -v hf >/dev/null 2>&1 || {
    echo "FATAL: \`hf\` CLI not found. Install with \`pip install huggingface_hub[cli]\`." >&2
    exit 2
}

mkdir -p "$BENCHMARK" "$PDBS_DIR/monomer"
[[ "$NEED_HOMODIMER" == "1" ]] && mkdir -p "$PDBS_DIR/homodimer"

echo "=== [1/3] benchmark assets (splits + labels + Boltz tokens) ==="
hf download "$REPO" --repo-type dataset --local-dir "$BENCHMARK" \
    --exclude "REPSP_PDB/*"
if compgen -G "$BENCHMARK/boltz_holo_tokens/shard*.tar" > /dev/null; then
    ( cd "$BENCHMARK/boltz_holo_tokens" && for t in shard*.tar; do tar xf "$t"; done && rm shard*.tar )
fi

# Which PDB shards do we want?
case "$SPLIT" in
    test)  monomer_incl=("REPSP_PDB/monomer/test.tar.gz")
           homodimer_incl=("REPSP_PDB/homodimer/test.tar.gz") ;;
    valid) monomer_incl=("REPSP_PDB/monomer/valid.tar.gz")
           homodimer_incl=("REPSP_PDB/homodimer/valid.tar.gz") ;;
    train) monomer_incl=("REPSP_PDB/monomer/train_*.tar.gz")
           homodimer_incl=("REPSP_PDB/homodimer/train_*.tar.gz") ;;
    all)   monomer_incl=("REPSP_PDB/monomer/*.tar.gz")
           homodimer_incl=("REPSP_PDB/homodimer/*.tar.gz") ;;
    *) echo "FATAL: SPLIT must be test|valid|train|all (got '$SPLIT')" >&2; exit 1 ;;
esac

echo
echo "=== [2/3] fetch monomer PDB shards (SPLIT=$SPLIT) ==="
mkdir -p "$BENCHMARK"
hf download "$REPO" --repo-type dataset --local-dir "$BENCHMARK" \
    --include "${monomer_incl[@]}"
for tar in "$BENCHMARK/REPSP_PDB/monomer"/*.tar.gz; do
    [[ -f "$tar" ]] || continue
    echo "  unpack $(basename "$tar") -> $PDBS_DIR/monomer/"
    tar xzf "$tar" -C "$PDBS_DIR/monomer/"
    rm "$tar"
done

if [[ "$NEED_HOMODIMER" == "1" ]]; then
    echo
    echo "=== [3/3] fetch homodimer PDB shards (SPLIT=$SPLIT) ==="
    hf download "$REPO" --repo-type dataset --local-dir "$BENCHMARK" \
        --include "${homodimer_incl[@]}"
    for tar in "$BENCHMARK/REPSP_PDB/homodimer"/*.tar.gz; do
        [[ -f "$tar" ]] || continue
        echo "  unpack $(basename "$tar") -> $PDBS_DIR/homodimer/"
        tar xzf "$tar" -C "$PDBS_DIR/homodimer/"
        rm "$tar"
    done
fi

n_mono=$(find "$PDBS_DIR/monomer"   -name "*.pdb" -printf . 2>/dev/null | wc -c)
echo
echo "monomer/   $n_mono pdb files"
if [[ "$NEED_HOMODIMER" == "1" ]]; then
    n_homo=$(find "$PDBS_DIR/homodimer" -name "*.pdb" -printf . 2>/dev/null | wc -c)
    echo "homodimer/ $n_homo pdb files"
fi

echo
echo "Setup complete. Next:"
echo "  bash examples/run_benchmark.sh"
