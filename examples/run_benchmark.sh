#!/usr/bin/env bash
# End-to-end benchmark runner.
#
# Reproduces the probing benchmark on the four homodimer tasks
# (`binding_site`, `delta_sasa_mean`, `levy_tier`, `bond_type_plip`)
# starting from compressed PDB archives.
#
# Expected layout AFTER `setup_benchmark.sh` (or manual download + untar):
#
#   ./benchmark/                                  # repsp-benchmark (HF dataset)
#     ├── splits/probing/{train,valid,test}_monomer.txt
#     ├── splits/probing/{train,valid,test}_homodimer.txt
#     └── probing/labels.pkl
#
#   ./pdbs/monomer/<AF-id>.pdb                    # fetched from EBI AFDB (see setup_benchmark.sh)
#   ./pdbs/homodimer/<AF-id>.pdb                  # optional, only if folding/co-folding
#
# Usage:
#
#   bash examples/run_benchmark.sh                    # defaults: 650M, ./work output
#   MODEL_SIZE=3B bash examples/run_benchmark.sh
#   PDBS_DIR=./pdbs/monomer BENCHMARK=./benchmark bash examples/run_benchmark.sh
#
# Outputs land under $WORKDIR/results/.

set -euo pipefail

MODEL_SIZE=${MODEL_SIZE:-650M}
PDBS_DIR=${PDBS_DIR:-./REPSP_PDB/monomer}
BENCHMARK=${BENCHMARK:-./benchmark}
WORKDIR=${WORKDIR:-./work}
DEVICE=${DEVICE:-cuda}

SPLITS_DIR="$BENCHMARK/splits/probing"
LABELS="$BENCHMARK/probing/labels.pkl"
FEATS_LMDB="$WORKDIR/features_${MODEL_SIZE}.lmdb"
FEATS_PT_DIR="$WORKDIR/probing_features_${MODEL_SIZE}"
RESULTS_DIR="$WORKDIR/results_${MODEL_SIZE}"

for required in "$SPLITS_DIR" "$LABELS" "$PDBS_DIR"; do
    if [[ ! -e "$required" ]]; then
        echo "FATAL: missing $required" >&2
        echo "       run examples/setup_benchmark.sh first, or set the env vars." >&2
        exit 2
    fi
done

mkdir -p "$WORKDIR" "$FEATS_PT_DIR" "$RESULTS_DIR"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

# 1. PDB -> features.lmdb (cached; skip if already built).
if [[ ! -f "$FEATS_LMDB/data.mdb" ]]; then
    echo "=== [1/3] Extract per-residue features ($MODEL_SIZE) ==="
    python "$HERE/extract_features_from_pdbs.py" \
        --pdbs_dir   "$PDBS_DIR" \
        --splits_dir "$SPLITS_DIR" \
        --model      "$MODEL_SIZE" \
        --output     "$FEATS_LMDB" \
        --device     "$DEVICE"
else
    echo "=== [1/3] Reusing features at $FEATS_LMDB ==="
fi

# 2. features.lmdb -> per-split .pt for the probe.
if [[ ! -f "$FEATS_PT_DIR/test.pt" ]]; then
    echo "=== [2/3] Stage per-split .pt ==="
    python "$REPO/code/repsp/probing/homomer/__lib/extract_probing_features.py" \
        --features_lmdb "$FEATS_LMDB" \
        --splits_dir    "$SPLITS_DIR" \
        --target_pkl    "$LABELS" \
        --out_dir       "$FEATS_PT_DIR"
else
    echo "=== [2/3] Reusing .pt at $FEATS_PT_DIR ==="
fi

# 3. Probing - one job per task, defaults reproduce the benchmark
#    (10 epochs, residue batch 16,824, lr 5e-4).
echo "=== [3/3] Probing (4 tasks) ==="
for TASK in binding_site delta_sasa_mean levy_tier bond_type_plip; do
    echo "  -> $TASK"
    python "$REPO/code/repsp/probing/homomer/__lib/probe_residue_homomer.py" \
        --features_dir "$FEATS_PT_DIR" \
        --target_pkl   "$LABELS" \
        --task         "$TASK" \
        --run_name     "ours_${MODEL_SIZE}_${TASK}" \
        --results_dir  "$RESULTS_DIR" \
        --device       "$DEVICE"
done

echo
echo "=== Done. Results: $RESULTS_DIR ==="
ls -1 "$RESULTS_DIR"
