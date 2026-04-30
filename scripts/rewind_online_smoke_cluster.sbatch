#!/usr/bin/env bash
#SBATCH --job-name=rw_pipe_smoke
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# One-job smoke pipeline for the pure ReWiND baseline:
#   1) make a tiny raw/embedding H5 with one task, one trajectory, one language embedding
#   2) label that tiny H5 with the ReWiND reward model
#   3) train one short offline checkpoint
#   4) run one short online job from that checkpoint
#
# Submit from the baseline repo root. Required env vars:
#   CONDA_ENV, RAW_H5, EMBEDDING_H5, REWIND_CKPT

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"

: "${CONDA_ENV:?Set CONDA_ENV to the rewind_nochunk conda env path}"
: "${RAW_H5:?Set RAW_H5 to metaworld_generation.h5}"
: "${EMBEDDING_H5:?Set EMBEDDING_H5 to metaworld_embeddings_train.h5}"
: "${REWIND_CKPT:?Set REWIND_CKPT to the ReWiND reward-model checkpoint}"

ENV_ID="${ENV_ID:-coffee-button-v2}"
SEED="${SEED:-0}"
OFFLINE_STEPS="${OFFLINE_STEPS:-100}"
ONLINE_STEPS="${ONLINE_STEPS:-1000}"
LEARNING_STARTS="${LEARNING_STARTS:-100}"
EVAL_FREQ="${EVAL_FREQ:-1000}"
VIDEO_FREQ="${VIDEO_FREQ:-1000}"
N_CRITICS="${N_CRITICS:-10}"

LABEL_KEY="${LABEL_KEY:-}"
TRAJ_INDEX="${TRAJ_INDEX:-0}"
ANNOTATION_INDEX="${ANNOTATION_INDEX:-0}"

WANDB_ENTITY="${WANDB_ENTITY:-rewind-valuemodel}"
WANDB_PROJECT="${WANDB_PROJECT:-rewind-policy-training}"
GROUP_NAME="${GROUP_NAME:-smoke_rewind_baseline_pipeline}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SMOKE_ROOT="${SMOKE_ROOT:-${PROJECT_DIR}/logs/smoke_pipeline}"
RUN_DIR="${SMOKE_ROOT}/${ENV_ID}_seed${SEED}_${RUN_TAG}"

TINY_RAW_H5="${RUN_DIR}/metaworld_generation_one.h5"
TINY_EMBEDDING_H5="${RUN_DIR}/metaworld_embeddings_one.h5"
LABELED_H5="${RUN_DIR}/metaworld_labeled_one.h5"
OFFLINE_LOG_DIR="${RUN_DIR}/offline"
ONLINE_LOG_DIR="${RUN_DIR}/online"
OFFLINE_CKPT="${OFFLINE_LOG_DIR}/last_offline"

export PROJECT_DIR RAW_H5 EMBEDDING_H5 REWIND_CKPT
export LABEL_KEY TRAJ_INDEX ANNOTATION_INDEX
export TINY_RAW_H5 TINY_EMBEDDING_H5 LABELED_H5
export OFFLINE_LOG_DIR ONLINE_LOG_DIR OFFLINE_CKPT RUN_DIR

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

mkdir -p "${PROJECT_DIR}/logs" "${RUN_DIR}" "${OFFLINE_LOG_DIR}" "${ONLINE_LOG_DIR}"
cd "${PROJECT_DIR}"

echo "===== ReWiND label + offline + online smoke pipeline ====="
echo "project_dir=${PROJECT_DIR}"
echo "conda_env=${CONDA_ENV}"
echo "raw_h5=${RAW_H5}"
echo "embedding_h5=${EMBEDDING_H5}"
echo "rewind_ckpt=${REWIND_CKPT}"
echo "env_id=${ENV_ID}"
echo "seed=${SEED}"
echo "offline_steps=${OFFLINE_STEPS}"
echo "online_steps=${ONLINE_STEPS}"
echo "label_key=${LABEL_KEY:-<first common key>}"
echo "traj_index=${TRAJ_INDEX}"
echo "annotation_index=${ANNOTATION_INDEX}"
echo "run_dir=${RUN_DIR}"
echo "labeled_h5=${LABELED_H5}"
echo "offline_ckpt=${OFFLINE_CKPT}"
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

check_file "raw trajectory H5" "${RAW_H5}"
check_file "embedding H5" "${EMBEDDING_H5}"
check_file "ReWiND reward checkpoint" "${REWIND_CKPT}"

echo
echo "===== Step 1/4: create one-task one-trajectory H5 inputs ====="
python - <<'PY'
import os
import h5py

raw_path = os.environ["RAW_H5"]
emb_path = os.environ["EMBEDDING_H5"]
tiny_raw_path = os.environ["TINY_RAW_H5"]
tiny_emb_path = os.environ["TINY_EMBEDDING_H5"]
label_key = os.environ.get("LABEL_KEY", "")
traj_index = int(os.environ.get("TRAJ_INDEX", "0"))
annotation_index = int(os.environ.get("ANNOTATION_INDEX", "0"))

def sort_key(value):
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)

os.makedirs(os.path.dirname(tiny_raw_path), exist_ok=True)

with h5py.File(raw_path, "r") as raw, h5py.File(emb_path, "r") as emb:
    raw_keys = set(raw.keys())
    emb_keys = sorted(emb.keys())

    if label_key:
        if label_key not in raw:
            raise KeyError(f"LABEL_KEY={label_key!r} is not in RAW_H5")
        if label_key not in emb:
            raise KeyError(f"LABEL_KEY={label_key!r} is not in EMBEDDING_H5")
        key = label_key
    else:
        common = [k for k in emb_keys if k in raw_keys]
        if not common:
            raise RuntimeError("RAW_H5 and EMBEDDING_H5 do not share any task keys")
        key = common[0]

    traj_names = sorted(raw[key].keys(), key=sort_key)
    if not traj_names:
        raise RuntimeError(f"Task {key!r} has no trajectories")
    if traj_index < 0 or traj_index >= len(traj_names):
        raise IndexError(f"TRAJ_INDEX={traj_index} out of range for {len(traj_names)} trajectories")
    traj_name = traj_names[traj_index]

    lang = emb[key]["minilm_lang_embedding"]
    if annotation_index < 0 or annotation_index >= len(lang):
        raise IndexError(
            f"ANNOTATION_INDEX={annotation_index} out of range for {len(lang)} language embeddings"
        )

    for path in (tiny_raw_path, tiny_emb_path):
        if os.path.exists(path):
            os.remove(path)

    with h5py.File(tiny_raw_path, "w") as tiny_raw:
        task_group = tiny_raw.create_group(key)
        raw.copy(raw[key][traj_name], task_group, name=traj_name)

    with h5py.File(tiny_emb_path, "w") as tiny_emb:
        task_group = tiny_emb.create_group(key)
        for attr_name, attr_value in emb[key].attrs.items():
            task_group.attrs[attr_name] = attr_value
        for name, item in emb[key].items():
            if isinstance(item, h5py.Dataset) and name == "minilm_lang_embedding":
                ds = task_group.create_dataset(name, data=item[annotation_index:annotation_index + 1])
                for attr_name, attr_value in item.attrs.items():
                    ds.attrs[attr_name] = attr_value
            else:
                emb.copy(item, task_group, name=name)

    print(f"selected_task={key}")
    print(f"selected_traj={traj_name}")
    print(f"selected_annotation={annotation_index}")
    print(f"raw_steps={len(raw[key][traj_name]['done'])}")
    print(f"raw_images={len(raw[key][traj_name]['img'])}")
    print(f"tiny_raw_h5={tiny_raw_path}")
    print(f"tiny_embedding_h5={tiny_emb_path}")
PY

echo
echo "===== Step 2/4: label tiny H5 with ReWiND ====="
python -u scripts/generate_labeled_dataset.py \
  --h5_video_path "${TINY_RAW_H5}" \
  --h5_embedding_path "${TINY_EMBEDDING_H5}" \
  --reward_model_path "${REWIND_CKPT}" \
  --output_path "${LABELED_H5}"

echo
echo "===== Labeled H5 summary ====="
python - <<'PY'
import os
import h5py

path = os.environ["LABELED_H5"]
with h5py.File(path, "r") as h5:
    for key in h5.keys():
        ds = h5[key]
        print(f"{key}: shape={ds.shape} dtype={ds.dtype}")
    n = len(h5["rewards"])
    env = h5["env_id"][0].decode() if n else "<empty>"
    print(f"num_transitions={n}")
    print(f"first_env_id={env}")
    print(f"reward_min={h5['rewards'][:].min() if n else 'NA'}")
    print(f"reward_max={h5['rewards'][:].max() if n else 'NA'}")
PY

check_file "labeled H5" "${LABELED_H5}"

echo
echo "===== Step 3/4: short offline training ====="
srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=rewind \
  reward_model.model_path="${REWIND_CKPT}" \
  reward_model.success_bonus=0 \
  reward_model.reward_divisor=1 \
  general_training.seed="${SEED}" \
  general_training.n_critics="${N_CRITICS}" \
  environment.env_id="${ENV_ID}" \
  offline_training.offline_h5_path="${LABELED_H5}" \
  offline_training.offline_tasks=all \
  offline_training.offline_training_steps="${OFFLINE_STEPS}" \
  online_training.total_time_steps=0 \
  online_training.mix_buffers_ratio=0.0 \
  logging.wandb=true \
  logging.wandb_entity_name="${WANDB_ENTITY}" \
  logging.wandb_project_name="${WANDB_PROJECT}" \
  logging.wandb_group_name="${GROUP_NAME}" \
  logging.log_dir="${OFFLINE_LOG_DIR}" \
  logging.eval_freq="${EVAL_FREQ}" \
  logging.video_freq="${VIDEO_FREQ}" \
  hydra.run.dir=.

check_file "offline checkpoint zip" "${OFFLINE_CKPT}.zip"

echo
echo "===== Step 4/4: short online training from the new offline checkpoint ====="
srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=rewind \
  reward_model.model_path="${REWIND_CKPT}" \
  reward_model.success_bonus=0 \
  reward_model.reward_divisor=1 \
  general_training.seed="${SEED}" \
  general_training.n_critics="${N_CRITICS}" \
  environment.env_id="${ENV_ID}" \
  offline_training.offline_h5_path="${LABELED_H5}" \
  offline_training.offline_tasks=all \
  offline_training.ckpt_path="${OFFLINE_CKPT}" \
  offline_training.offline_training_steps=0 \
  online_training.total_time_steps="${ONLINE_STEPS}" \
  online_training.mix_buffers_ratio=0.0 \
  online_training.learning_starts="${LEARNING_STARTS}" \
  logging.wandb=true \
  logging.wandb_entity_name="${WANDB_ENTITY}" \
  logging.wandb_project_name="${WANDB_PROJECT}" \
  logging.wandb_group_name="${GROUP_NAME}" \
  logging.log_dir="${ONLINE_LOG_DIR}" \
  logging.eval_freq="${EVAL_FREQ}" \
  logging.video_freq="${VIDEO_FREQ}" \
  hydra.run.dir=.

echo
echo "===== ReWiND smoke pipeline finished ====="
echo "run_dir=${RUN_DIR}"
echo "labeled_h5=${LABELED_H5}"
echo "offline_ckpt=${OFFLINE_CKPT}.zip"
echo "online_log_dir=${ONLINE_LOG_DIR}"
