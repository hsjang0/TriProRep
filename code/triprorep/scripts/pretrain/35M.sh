#!/usr/bin/env bash
#SBATCH --job-name=35M
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=gpu-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=192
#SBATCH --mem-per-gpu=128G

set -euo pipefail

CONFIG_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/configs/pretrain_35M/pretrain_electra.yaml"
EXTRA_ARGS=("$@")

LOG_DIR="slurm_logs"
mkdir -p "$LOG_DIR"

NNODES="${SLURM_JOB_NUM_NODES:-5}"
NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"

MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
MASTER_PORT="${MASTER_PORT:-29500}"
RDZV_ID="${RDZV_ID:-${SLURM_JOB_ID:-1}}"
NODE_RANK="${SLURM_NODEID:-0}"

export MASTER_ADDR MASTER_PORT RDZV_ID NODE_RANK

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,ENV,NET}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONFAULTHANDLER=1


# ----------------------------
# Network / NCCL settings
# ----------------------------
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond.2561}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond.2561}"
export NCCL_OOB_NET_IFNAME="${NCCL_OOB_NET_IFNAME:-bond.2561}"

export NCCL_IB_DISABLE=0
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9}"
export NCCL_IB_PKEY=1
export NCCL_IB_GID_INDEX=0
export NCCL_NET=IB
#export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"

# ----------------------------
# Conda env
# ----------------------------


ulimit -n 65000

# Activate your conda env (adjust to your install):
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate <your-env>

echo "Starting multi-node training"
echo "  NNODES=${NNODES}"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "  NODE_RANK=${NODE_RANK}"
echo "  MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "  RDZV_ID=${RDZV_ID}"
echo "  Config=${CONFIG_FILE}"
echo "  NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "  NCCL_IB_HCA=${NCCL_IB_HCA}"

# keep srun so each node launches exactly one torchrun
srun --nodes="${NNODES}" --ntasks-per-node=1 --kill-on-bad-exit=1 --cpu-bind=none --accel-bind=gn \
  torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --rdzv_backend=c10d \
    --rdzv_id="${RDZV_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/train_multinode.py \
      --config "${CONFIG_FILE}" \
      "${EXTRA_ARGS[@]}"
