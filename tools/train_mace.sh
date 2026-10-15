#!/bin/bash
# MACE control run on the identical rMD17 split (tools/make_xyz.py), same epoch budget as FACE.
# usage: tools/train_mace.sh <mol> <epochs> <hidden_irreps> <tag>
mol=${1:-aspirin}; ep=${2:-60}; irreps=${3:-"64x0e+64x1o"}; tag=${4:-mace_small}
swa=$(( ep * 3 / 4 ))
export PYTHONIOENCODING=utf-8 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# 75% GPU cap: same safety guard as FACE runs
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
  --default_dtype=float32 --device=cuda --seed=0 --eval_interval=10 \
  --model_dir=runs/mace --log_dir=runs/mace --checkpoints_dir=runs/mace/ckpt --results_dir=runs/mace
