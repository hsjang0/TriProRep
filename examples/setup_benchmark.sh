#!/usr/bin/env bash
# Download + prepare the benchmark.
#
# Pulls the small benchmark assets (splits, labels, Boltz tokens) from
# HuggingFace and points at your local `REPSP_PDB/` directory of PDBs.
# We do NOT redistribute the PDBs (AFDB license). Users obtain them by
# either:
#
#   (a) Running scripts/npz_to_pdb.py against a local checkout of the
#       AFDB-Multimer Boltz-format structures, or
#   (b) Downloading the NVIDIA/AFDB-Multimer chunk tars from
#       https://ftp.ebi.ac.uk/pub/databases/alphafold/collaborations/nvda/
#       and extracting.
#
# The layout this script expects afterwards:
#
#   ./REPSP_PDB/
#     monomer/<AF-id>_monomer.pdb          # chain A, apo prediction
#     homodimer/<AF-id>.pdb                # chain A + chain B, holo dimer
#
# The AFid in both filenames is the same homodimer identifier taken from
# `benchmark/splits/{folding,probing}/{train,valid,test}.txt`.
#
# After this script, run `bash examples/run_benchmark.sh`.

set -euo pipefail

BENCHMARK=${BENCHMARK:-./benchmark}
PDBS_DIR=${PDBS_DIR:-./REPSP_PDB}

command -v hf >/dev/null 2>&1 || {
    echo "FATAL: \`hf\` CLI not found. Install with \`pip install huggingface_hub[cli]\`." >&2
    exit 2
}

mkdir -p "$BENCHMARK"

echo "=== [1/2] benchmark assets ==="
hf download k-fold-structure/repsp-benchmark --repo-type dataset \
    --local-dir "$BENCHMARK"
if compgen -G "$BENCHMARK/boltz_holo_tokens/shard*.tar" > /dev/null; then
    ( cd "$BENCHMARK/boltz_holo_tokens" && for t in shard*.tar; do tar xf "$t"; done && rm shard*.tar )
fi

echo
echo "=== [2/2] REPSP_PDB check ==="
if [[ ! -d "$PDBS_DIR/monomer" ]] || [[ ! -d "$PDBS_DIR/homodimer" ]]; then
    cat >&2 <<'EOM'

REPSP_PDB is not populated at $PDBS_DIR. Options:

  * If you have the AFDB-Multimer Boltz-tokenized structures locally,
    convert them to PDB with:

        python scripts/npz_to_pdb.py \
            --split ./benchmark/splits/folding/train.txt \
            --apo_dir  /path/to/co-folding/full/apo_targets/structures \
            --holo_dir /path/to/co-folding/full/holo_targets/structures \
            --out_dir  ./REPSP_PDB \
            --workers 32

  * Otherwise fetch the NVIDIA chunk tars from
    https://ftp.ebi.ac.uk/pub/databases/alphafold/collaborations/nvda/
    (uses homodimer_metadata.csv → chunk_NNNN.tar mapping) and untar
    into the REPSP_PDB/ layout above.

EOM
    exit 3
fi

n_mono=$(find "$PDBS_DIR/monomer"   -name "*.pdb" -printf . 2>/dev/null | wc -c)
n_homo=$(find "$PDBS_DIR/homodimer" -name "*.pdb" -printf . 2>/dev/null | wc -c)
echo "monomer/   $n_mono pdb files"
echo "homodimer/ $n_homo pdb files"

echo
echo "Setup complete. Next:"
echo "  bash examples/run_benchmark.sh"
