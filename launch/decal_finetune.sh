#!/usr/bin/env bash
set -euo pipefail


export MASTER_ADDR=${MASTER_ADDR:-"172.26.10.76"}
export MASTER_PORT=${MASTER_PORT:-6379}
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

PROC_PER_NODE="${PROC_PER_NODE:-4}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
    export CUDA_HOME
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export WANDB_MODE=online
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false

# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

###############################################################################
############################## TRAINING config ################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"

cd ${PROJ_ROOT}

# 1. policy config
POLICY="decal"
PRETRAINED_PATH="${PROJ_ROOT}/ckpt/pretrained_ckpts/models--InternRobotics--InternVLA-A1-3B/snapshots/19e953fb2d4b49d2b181c0b55981f2bd3444d546"

# 2. dataset config
DATASET_REPO_ID="$1"
ACTION_TYPE=${2:-abs}          # abs | delta
USE_EXTERNAL_STATS=${3:-false} # true | false

# 3. output config
BASE_OUTPUT_DIR="outputs/${POLICY}"
PRETRAINED_DETAIL="a1_agibotworld_700k"
JOB_NAME="$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-${DATASET_REPO_ID//[\/ ]/_}-${ACTION_TYPE}-${PRETRAINED_DETAIL}-finetune-split_arm_hand_action-tactile_gate-image_transforms"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"

echo "PROC_PER_NODE = ${PROC_PER_NODE}"
echo "NUM_PROCESSES = ${NUM_PROCESSES}"
ARGS=(
    # --multi_gpu
    --num_processes="${NUM_PROCESSES}"
    --num_machines="${NODE_COUNT}"
    --machine_rank="${NODE_RANK}"
    --main_process_ip="${MASTER_ADDR}"
    --main_process_port="${MASTER_PORT}" 
    src/lerobot/scripts/lerobot_train.py

    --output_dir="${OUTPUT_DIR}"
    --num_workers=12
    --job_name="${JOB_NAME}"

    # ---- Policy ----
    --policy.type=${POLICY}
    --policy.repo_id=lerobot_lab/${POLICY}
    --policy.pretrained_path=${PRETRAINED_PATH}
    --policy.push_to_hub=false
    --policy.gradient_checkpointing=false
    --policy.dtype=bfloat16
    --policy.optimizer_lr=5.0e-5
    --policy.scheduler_warmup_steps=2000
    --policy.scheduler_decay_steps=100000
    --policy.scheduler_decay_lr=5.0e-6
    --policy.freeze_vision_encoder=false
    --policy.train_expert_only=false
    --policy.train_vlm_only=false
    --policy.qwen3_vl_variant=qwen3_vl_28l
    --policy.action_expert_variant=qwen3_28l
    --policy.chunk_size=50
    --policy.n_action_steps=50
    # --policy.use_tactile_gen_branch=false
    # --policy.use_visual_gen=false
    --policy.split_arm_hand_action=true
    --policy.use_tactile_gate=true
    # --policy.use_consistency_action_loss=true
    --policy.use_tactile_force=true
    # --policy.use_tactile_deform=true
    # --policy.use_tactile_batchnorm=false

    # ---- Dataset ----
    --dataset.type=${POLICY}
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.action_mode="${ACTION_TYPE}"
    --dataset.use_external_stats=${USE_EXTERNAL_STATS}
    --dataset.external_stats_path=${HF_LEROBOT_HOME}/stats/${ACTION_TYPE}/${DATASET_REPO_ID}/stats.json
    --dataset.image_transforms.enable=true

    # ---- Training ----
    --seed=42
    --batch_size=4
    --steps=100000
    # --eval_freq=60000
    --save_freq=20000
    --log_freq=200
    # --resume=true

    # ---- Logging ----
    --wandb.enable=true
    --wandb.project=lerobot_lab_${POLICY}
    --wandb.mode=online
)

accelerate launch "${ARGS[@]}"
