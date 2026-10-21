#!/bin/bash
# rMD17 multi-molecule suite: FACE flagship (C96 B16 K32 rc4.5) vs MACE-small, identical splits, 60 epochs each, sequential.
export PYTHONIOENCODING=utf-8 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
for mol in ${MOLS:-ethanol malonaldehyde uracil}; do
  ${PYTHON:-../.venv/Scripts/python} tools/make_xyz.py $mol > /dev/null
  ${PYTHON:-../.venv/Scripts/python} -u -m face.train --mol $mol --epochs 60 --lr 0.04 --transport 1 --eqmsg 1 --factor 1 \
      --C 96 --B 16 --K 32 --rc 4.5 --unit 1 --tag suite_flag > runs/suite_flag_$mol.log 2>&1
  bash tools/train_mace.sh $mol 60 "64x0e+64x1o" suite_mace > runs/suite_mace_$mol.log 2>&1
  ${PYTHON:-../.venv/Scripts/python} tools/eval_compare.py $mol mace=runs/mace/suite_mace_${mol}_stagetwo.model 2>/dev/null \
      | grep stagetwo > runs/suite_mace_${mol}_eval.txt
done
echo SUITE_DONE
