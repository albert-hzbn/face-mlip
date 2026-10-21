#!/usr/bin/env bash
# Second review queue (runs after tools/review_queue.sh): cost breakdown and a longer cutoff-matched timing.
PY=${PYTHON:-../.venv/Scripts/python}
export PYTHONIOENCODING=utf-8 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
source <(sed -n '/^log()/,/^run()/p' tools/review_queue.sh)
run "bench profile" $PY tools/bench_review.py profile
run "bench cutoff (15 rounds)" env BENCH_ROUNDS=15 $PY tools/bench_review.py cutoff
run "bench 168 atoms (15 rounds)" env BENCH_ROUNDS=15 $PY tools/bench_review.py small
log "QUEUE2 DONE"
