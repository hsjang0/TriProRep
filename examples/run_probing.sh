#!/usr/bin/env bash
# Run the probing benchmark (four homodimer tasks) with our encoder.
#
# Prereq: `bash examples/setup_probing.sh` (populates ./benchmark and
# ./REPSP_PDB/monomer).
#
# Outputs `./work/results_<size>/<task>.json` for the four tasks:
#   binding_site, delta_sasa_mean, levy_tier, bond_type_plip.
#
# Default encoder: 650M. Override with `MODEL_SIZE=3B bash examples/run_probing.sh`
# or one of {35M, 150M, 650M, 3B}.

set -euo pipefail

MODEL_SIZE=${MODEL_SIZE:-650M}
BENCHMARK=./benchmark
PDBS_DIR=./REPSP_PDB/monomer
WORK=./work
FEATS_LMDB="$WORK/features_${MODEL_SIZE}.lmdb"
FEATS_PT="$WORK/probing_features_${MODEL_SIZE}"
RESULTS="$WORK/results_${MODEL_SIZE}"

for req in "$BENCHMARK/probing/labels.pkl" "$BENCHMARK/splits/probing/train.txt" "$PDBS_DIR"; do
    [[ -e "$req" ]] || { echo "FATAL: missing $req. Run bash examples/setup_probing.sh first." >&2; exit 2; }
done

mkdir -p "$WORK" "$FEATS_PT" "$RESULTS"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

echo "=== [1/3] extract per-residue features ($MODEL_SIZE) ==="
if [[ ! -f "$FEATS_LMDB/data.mdb" ]]; then
    python "$HERE/extract_features_from_pdbs.py" \
        --pdbs_dir "$PDBS_DIR" --pdb_glob "*.pdb" \
        --model "$MODEL_SIZE" --output "$FEATS_LMDB"
else
    echo "  reusing $FEATS_LMDB"
fi

echo
echo "=== [2/3] stage per-split .pt ==="
if [[ ! -f "$FEATS_PT/test.pt" ]]; then
    python "$REPO/code/repsp/probing/homomer/__lib/extract_probing_features.py" \
        --features_lmdb "$FEATS_LMDB" \
        --splits_dir    "$BENCHMARK/splits/probing" \
        --target_pkl    "$BENCHMARK/probing/labels.pkl" \
        --out_dir       "$FEATS_PT"
else
    echo "  reusing $FEATS_PT"
fi

echo
echo "=== [3/3] four probing tasks ==="
for TASK in binding_site delta_sasa_mean levy_tier bond_type_plip; do
    echo "  -> $TASK"
    python "$REPO/code/repsp/probing/homomer/__lib/probe_residue_homomer.py" \
        --features_dir "$FEATS_PT" \
        --target_pkl   "$BENCHMARK/probing/labels.pkl" \
        --task         "$TASK" \
        --run_name     "ours_${MODEL_SIZE}_${TASK}" \
        --results_dir  "$RESULTS"
done

echo
echo "Done. Results: $RESULTS"
ls "$RESULTS"
