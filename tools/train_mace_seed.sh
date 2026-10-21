#!/bin/bash
# MACE control run as tools/train_mace.sh, with a seed argument and evaluation of the FINAL epoch
# (eval_interval=1, so the stage-two model is the epoch-60 model rather than the last 10-epoch checkpoint).
# usage: tools/train_mace_seed.sh <mol> <epochs> <hidden_irreps> <tag> <seed>
mol=${1:-aspirin}; ep=${2:-60}; irreps=${3:-"64x0e+64x1o"}; tag=${4:-mace_small}; seed=${5:-0}
swa=$(( ep * 3 / 4 ))
export PYTHONIOENCODING=utf-8 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
${PYTHON:-../.venv/Scripts/python} -c "
import sys, runpy, torch
from face import safety; safety.install()
sys.argv = ['run_train'] + sys.argv[1:]
runpy.run_module('mace.cli.run_train', run_name='__main__')
" --name="${tag}_${mol}" --model=MACE --train_file="data/xyz/${mol}_train.xyz" \
  --valid_file="data/xyz/${mol}_val.xyz" --test_file="data/xyz/${mol}_test.xyz" \
  --energy_key=REF_energy --forces_key=REF_forces --E0s=average \
  --hidden_irreps="$irreps" --r_max=5.0 --num_radial_basis=8 --correlation=3 --max_L=1 \
  --batch_size=5 --valid_batch_size=50 --max_num_epochs=$ep --lr=0.01 --amsgrad \
  --ema --ema_decay=0.99 --energy_weight=1 --forces_weight=1000 \
  --swa --start_swa=$swa --swa_energy_weight=1000 --swa_forces_weight=100 \
  --default_dtype=float32 --device=cuda --seed=$seed --eval_interval=1 \
  --model_dir=runs/mace --log_dir=runs/mace --checkpoints_dir=runs/mace/ckpt --results_dir=runs/mace
