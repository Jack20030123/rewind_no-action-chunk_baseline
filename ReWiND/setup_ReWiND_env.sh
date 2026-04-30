#!/bin/bash
set -e

echo "Setting up ReWiND environment..."

# Create conda environment
echo "Creating conda environment 'rewind_nochunk'..."
conda env create -f rewind.yml
conda activate rewind_nochunk

# Install official ReWiND package.
echo "Installing official ReWiND package..."
pip install -e .

# Setup metaworld_policy_training
echo "Setting up metaworld_policy_training..."
cd metaworld_policy_training


# Clone and install dependencies
echo "Installing mjrl..."
if [ ! -d "mjrl" ]; then
    git clone https://github.com/aravindr93/mjrl.git
fi
git -C mjrl checkout 3871d93763d3b49c4741e6daeaebbc605fe140dc
pip install -e mjrl

echo "Installing Metaworld..."
if [ ! -d "Metaworld" ]; then
    git clone https://github.com/sumedh7/Metaworld.git
fi
pip install -e Metaworld


# RL, Hydra, MuJoCo, PyTorch, and CUDA-wheel dependencies are pinned in rewind.yml.

# Final installation
pip install -e .

echo "Setup complete! Activate the environment with: conda activate rewind_nochunk"
