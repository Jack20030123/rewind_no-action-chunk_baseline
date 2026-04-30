#!/usr/bin/env bash
#SBATCH --job-name=rw_on_smoke
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Minimal online smoke test for the pure ReWiND baseline.
# Submit from the baseline repo root and pass the existing cluster env/data paths:
#   CONDA_ENV, OFFLINE_H5, OFFLINE_CKPT, REWIND_CKPT

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"

: "${CONDA_ENV:?Set CONDA_ENV to the existing rewind_nochunk conda env path}"
: "${OFFLINE_H5:?Set OFFLINE_H5 to an existing labeled H5 file}"
: "${OFFLINE_CKPT:?Set OFFLINE_CKPT to an existing offline checkpoint path without .zip}"
: "${REWIND_CKPT:?Set REWIND_CKPT to an existing ReWiND reward-model checkpoint}"

ENV_ID="${ENV_ID:-coffee-button-v2}"
SEED="${SEED:-0}"
ONLINE_STEPS="${ONLINE_STEPS:-5000}"
LEARNING_STARTS="${LEARNING_STARTS:-100}"
EVAL_FREQ="${EVAL_FREQ:-1000}"
VIDEO_FREQ="${VIDEO_FREQ:-1000}"
N_CRITICS="${N_CRITICS:-10}"

WANDB_ENTITY="${WANDB_ENTITY:-rewind-valuemodel}"
WANDB_PROJECT="${WANDB_PROJECT:-rewind-policy-training}"
GROUP_NAME="${GROUP_NAME:-smoke_rewind_baseline_online}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/logs/smoke_online}"
LOG_DIR="${LOG_ROOT}/${ENV_ID}_seed${SEED}_${RUN_TAG}"

module purge
module load gcc/13.3.0
module load mesa/24.2.3
module load mesa-glu/9.0.2
module load glew/2.2.0
module load glx/1.4

# Some conda activate.d scripts read optional variables before defining them.
# Keep strict mode for our script, but relax nounset only while activating conda.
set +u
source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
set -u

export LD_PRELOAD="$(g++ -print-file-name=libstdc++.so.6)"
export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
GCC_LIBDIR="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="${GCC_LIBDIR}:${MUJOCO_PY_MUJOCO_PATH}/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$HOME/.config/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$HOME/.cache/wandb}"
export WANDB_DIR="${WANDB_DIR:-$HOME/wandb}"
export WANDB_MODE="${WANDB_MODE:-online}"

mkdir -p "${PROJECT_DIR}/logs" "${LOG_ROOT}"
cd "${PROJECT_DIR}"

echo "===== ReWiND online smoke test ====="
echo "project_dir=${PROJECT_DIR}"
echo "conda_env=${CONDA_ENV}"
echo "env_id=${ENV_ID}"
echo "seed=${SEED}"
echo "online_steps=${ONLINE_STEPS}"
echo "offline_h5=${OFFLINE_H5}"
echo "offline_ckpt=${OFFLINE_CKPT}"
echo "rewind_ckpt=${REWIND_CKPT}"
echo "log_dir=${LOG_DIR}"
echo "wandb=${WANDB_ENTITY}/${WANDB_PROJECT} group=${GROUP_NAME}"
echo "node=${SLURM_NODELIST:-}"
nvidia-smi || true

check_file() {
  local label="$1"
  local path="$2"
  if [ ! -f "${path}" ]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    return 1
  fi
  echo "found ${label}: ${path}"
}

check_file "offline H5" "${OFFLINE_H5}"
check_file "offline checkpoint zip" "${OFFLINE_CKPT}.zip"
check_file "ReWiND reward checkpoint" "${REWIND_CKPT}"

srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=rewind \
  reward_model.model_path="${REWIND_CKPT}" \
  reward_model.success_bonus=0 \
  reward_model.reward_divisor=1 \
  general_training.seed="${SEED}" \
  environment.env_id="${ENV_ID}" \
  offline_training.offline_h5_path="${OFFLINE_H5}" \
  offline_training.ckpt_path="${OFFLINE_CKPT}" \
  offline_training.offline_training_steps=0 \
  online_training.total_time_steps="${ONLINE_STEPS}" \
  online_training.mix_buffers_ratio=0.0 \
  online_training.learning_starts="${LEARNING_STARTS}" \
  logging.wandb=true \
  logging.wandb_entity_name="${WANDB_ENTITY}" \
  logging.wandb_project_name="${WANDB_PROJECT}" \
  logging.wandb_group_name="${GROUP_NAME}" \
  logging.log_dir="${LOG_DIR}" \
  logging.eval_freq="${EVAL_FREQ}" \
  logging.video_freq="${VIDEO_FREQ}" \
  hydra.run.dir=. \
  general_training.n_critics="${N_CRITICS}"
