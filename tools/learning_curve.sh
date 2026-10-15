#!/usr/bin/env bash
# Learning curve of the flagship on rMD17 aspirin: nested training subsets, same validation/test sets.
# epochs = max(400, 20000 * 5 / N), i.e. 400 epochs or at least 20k optimiser steps.
PY=${PYTHON:-../.venv/Scripts/python}
for N in 50 100 250 500; do
  E=$(( 100000 / N )); [ $E -lt 400 ] && E=400
  $PY -m face.train --mol aspirin --n_train $N --epochs $E --log_every $(( E / 40 )) --lr 0.04 --transport 1 --eqmsg 1 \
      --factor 1 --C 96 --B 16 --K 32 --rc 4.5 --unit 1 --tag lc_n${N} > runs/lc_n${N}_aspirin.log 2>&1
done
