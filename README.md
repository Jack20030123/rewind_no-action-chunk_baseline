# ReWiND reward model -> pure ReWiND policy baseline

This checkout is meant to mirror the **pure ReWiND baseline path** in the current
`rewind_no-action-chunk` project, while leaving the current baseline repo's non-ReWiND
features alone. In particular, this README does not try to port progress-diff,
base-reward, ROBOMETER, or TOPReward support into this baseline repo.

The official ReWiND reward-model repo has been cloned here:

```bash
ReWiND/
```

Use `ReWiND/` to train the official ReWiND reward model. Then use this baseline repo's
current-project-aligned ReWiND label script to produce the H5 used by offline and online
policy training.

Expected runtime assumptions:

```text
Linux + conda
CUDA GPU for labeling and policy training
Internet access, or pre-cached Hugging Face / torch.hub models, for first-time DINOv2
and MiniLM loading
```

The current project's pure ReWiND baseline SLURM jobs used a local `rewind_nochunk`
conda environment. The portable environment files in `ReWiND/rewind.yml` and
`environment_roboclip.yml` are based on the full package export from that working
environment. They are intentionally kept equivalent; prefer `ReWiND/rewind.yml` in the
commands below. The most important versions are:

```text
python: 3.10.16
torch: 2.5.1+cu124
torchvision: 0.20.1+cu124
transformers: 4.25.0
gym: 0.21.0
stable-baselines3: 1.8.0
hydra-core: 1.3.2
omegaconf: 2.3.0
h5py: 3.12.1
numpy: 1.23.4
```

The original export also contained user-specific editable installs:

```text
metaworld from the local Metaworld/ checkout
roboclipv2 from this local repo checkout
mjrl from the aravindr93/mjrl git checkout
```

Those entries are intentionally not hard-coded into `ReWiND/rewind.yml`. They are
installed portably with `pip install -e` in the setup steps below.

This is the only README kept in this deliverable. Nested README files from bundled
dependencies and the cloned official ReWiND repo were intentionally removed so that all
required setup and experiment instructions live in one place.

## What Was Aligned With The Current Project

The following files are aligned to the current project's pure ReWiND baseline path:

```text
models/reward_model/rewind_reward_model.py
models/encoders/dino_miniLM_encoder.py
scripts/generate_labeled_dataset.py
offline_rl_algorithms/offline_replay_buffers.py
```

The label script keeps the same reward/row construction logic as the current project, but
uses portable default paths through `REWIND_ROOT` and `REWIND_CKPT`. The ReWiND checkpoint
and labeled-H5 config paths were also made portable. Optional diff/base-reward/ROBOMETER/
TOPReward fields were intentionally not added.

The documented flow below is the supported pure ReWiND path. Older RoboCLIP scripts that
are not part of this flow may still contain legacy example paths from the upstream repo.

## 0. Paths

Local or workstation example:

```bash
cd /path/to/rewind_no-action-chunk_baseline
export BASELINE_ROOT="$PWD"
export REWIND_ROOT="$BASELINE_ROOT/ReWiND"
```

Cluster example:

```bash
export BASELINE_ROOT=/path/to/rewind_no-action-chunk_baseline
export REWIND_ROOT="$BASELINE_ROOT/ReWiND"
cd "$BASELINE_ROOT"
```

## 1. Train The Official ReWiND Reward Model

Set up the ReWiND environment. This creates a local conda env named `rewind_nochunk`
from the full exported dependency list in `ReWiND/rewind.yml`; it does not assume any
user-specific absolute path:

```bash
cd "$REWIND_ROOT"
bash -i setup_ReWiND_env.sh
conda activate rewind_nochunk
```

Prepare the official ReWiND data:

```bash
python download_data.py --download_path datasets

python data_generation/metaworld_generation.py --save_path datasets
python data_preprocessing/metaworld_center_crop.py \
  --video_path datasets \
  --target_path datasets
python data_preprocessing/generate_dino_embeddings.py \
  --video_path_folder datasets \
  --target_path datasets
```

Train the reward model:

```bash
python train_reward.py \
  --wandb_entity YOUR_WANDB_ENTITY \
  --wandb_project rewind-reward-training \
  --rewind \
  --subsample_video \
  --clip_grad \
  --cosine_scheduler \
  --batch_size 1024 \
  --worker 1
```

The expected checkpoint is:

```bash
export REWIND_CKPT="$REWIND_ROOT/checkpoints/rewind_metaworld_epoch_19.pth"
```

Sanity check:

```bash
python - <<'PY'
import os, torch
ckpt = torch.load(os.environ["REWIND_CKPT"], map_location="cpu")
print(ckpt.keys())
print("has model_state_dict:", "model_state_dict" in ckpt)
PY
```

The official checkpoint should contain `model_state_dict`.

## 2. Label The Offline Dataset In The Baseline Repo

On a new machine, set up the policy environment in Section 3 before running this step.
The label script uses the baseline repo's PyTorch/H5/DINO/MiniLM dependencies; it is not
meant to be run from a bare shell.

Use the baseline repo script copied from the current project:

```bash
cd "$BASELINE_ROOT"
mkdir -p datasets

python scripts/generate_labeled_dataset.py \
  --h5_video_path "$REWIND_ROOT/datasets/metaworld_generation.h5" \
  --h5_embedding_path "$REWIND_ROOT/datasets/metaworld_embeddings_train.h5" \
  --reward_model_path "$REWIND_CKPT" \
  --output_path "$BASELINE_ROOT/datasets/metaworld_labeled_rewind_official.h5"
```

This matches the current project's pure ReWiND label format:

```text
state
action
rewards
done
policy_lang_embedding
img_embedding
env_id
```

It does not write `next_state` or `next_img_embedding`; that is intentional, because the
goal here is to match the current project's pure ReWiND baseline behavior instead of
introducing a new replay-buffer format.

Sanity check:

```bash
python - <<'PY'
import h5py
path = "datasets/metaworld_labeled_rewind_official.h5"
with h5py.File(path, "r") as f:
    print(list(f.keys()))
    for k in ["state", "action", "rewards", "done", "policy_lang_embedding", "img_embedding", "env_id"]:
        print(k, f[k].shape, f[k].dtype)
PY
```

## 3. Policy Environment

Use the same `rewind_nochunk` environment for labeling, offline training, and online
fine-tuning. This matches the packages from the current project's pure ReWiND baseline
environment, but the env is created by name so other users do not need your cluster path.

```bash
cd "$BASELINE_ROOT"
conda activate rewind_nochunk
```

Install the local code and the MetaWorld fork used by this repo:

```bash
cd "$BASELINE_ROOT"
pip install -e Metaworld
pip install -e .
```

If you only want to run the policy baseline and already have a ReWiND checkpoint, you can
create the environment directly from the exported file without running the reward-model
setup script:

```bash
cd "$BASELINE_ROOT"
conda env create -f ReWiND/rewind.yml
conda activate rewind_nochunk
pip install -e Metaworld
pip install -e .
```

If this repo was cloned with Git rather than received as a complete archive, initialize
submodules before the editable installs:

```bash
git submodule init
git submodule update --recursive
```

If this repo was received as a zip/tar archive, the pure ReWiND MetaWorld path only
requires `ReWiND/`, `Metaworld/`, this repo's `models/`, `offline_rl_algorithms/`,
`envs/`, `configs/`, `scripts/`, and `test_scripts/` directories to be present. The
legacy `S3D_HowTo100M`, `mjrl`, and `models/LIV` components are not used by the
documented `reward=rewind` commands.

On the cluster, if you already have a working policy env, activate that instead.

For headless MetaWorld rendering:

```bash
export MUJOCO_GL=egl
```

## 4. Offline WSRL-IQL

```bash
cd "$BASELINE_ROOT"

export LABELED_H5="$BASELINE_ROOT/datasets/metaworld_labeled_rewind_official.h5"
export OFFLINE_LOGDIR="$BASELINE_ROOT/logs/offline_rewind_official_wsrl_iql_seed0"

python test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=rewind \
  reward_model.model_path="$REWIND_CKPT" \
  reward_model.success_bonus=0 \
  reward_model.reward_divisor=1 \
  offline_training.offline_h5_path="$LABELED_H5" \
  offline_training.offline_training_steps=100000 \
  online_training.total_time_steps=0 \
  online_training.mix_buffers_ratio=0.0 \
  general_training.seed=0 \
  environment.env_id=coffee-button-v2 \
  logging.wandb=false \
  logging.log_dir="$OFFLINE_LOGDIR" \
  hydra.run.dir=.
```

Expected checkpoint:

```bash
$OFFLINE_LOGDIR/last_offline.zip
```

## 5. Online Fine-Tuning

```bash
cd "$BASELINE_ROOT"

export LABELED_H5="$BASELINE_ROOT/datasets/metaworld_labeled_rewind_official.h5"
export OFFLINE_CKPT="$OFFLINE_LOGDIR/last_offline"

for ENV_ID in \
  window-close-v2 \
  reach-wall-v2 \
  faucet-close-v2 \
  coffee-button-v2 \
  button-press-wall-v2 \
  door-lock-v2 \
  handle-press-side-v2 \
  sweep-into-v2
do
  python test_scripts/test_iql.py \
    metaworld=off_on_15 \
    algorithm=wsrl_iql \
    reward=rewind \
    reward_model.model_path="$REWIND_CKPT" \
    reward_model.success_bonus=0 \
    reward_model.reward_divisor=1 \
    offline_training.offline_h5_path="$LABELED_H5" \
    offline_training.offline_training_steps=0 \
    offline_training.ckpt_path="$OFFLINE_CKPT" \
    online_training.total_time_steps=100000 \
    online_training.mix_buffers_ratio=0.0 \
    general_training.seed=0 \
    environment.env_id="$ENV_ID" \
    logging.wandb=false \
    logging.wandb_group_name="rewind_official_wsrl_iql_${ENV_ID}_seed0" \
    logging.log_dir="$BASELINE_ROOT/logs/online_rewind_official_wsrl_iql_${ENV_ID}_seed0" \
    hydra.run.dir=.
done
```

Repeat with `general_training.seed=32` or other seeds as needed.

## Notes

The raw ReWiND trajectories may have one more image than action because they contain the
terminal image after the last action. The current-project label script handles that by
using `progress_values[1:]` as rewards and `img_embedding[:-1]` as policy observations,
so the flat labeled H5 has the same length for `state`, `action`, `rewards`, `done`,
`policy_lang_embedding`, `img_embedding`, and `env_id`.
