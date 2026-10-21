#!/usr/bin/env bash
# Experiments requested in review, run one at a time. Before each job: no NVIDIA driver fault (nvlddmkm) in the last
# 10 minutes and no other training process on the machine. Create runs/STOP_review to stop after the current job.
PY=${PYTHON:-../.venv/Scripts/python}
export PYTHONIOENCODING=utf-8 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
log() { echo "$(date +%H:%M:%S) $*" | tee -a runs/review_queue.log; }

gate() {
  while true; do
    [ -f runs/STOP_review ] && { log "STOP file found, exiting"; exit 0; }
    faults=$(powershell -NoProfile -c "@(Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=(Get-Date).AddMinutes(-10)} -ErrorAction SilentlyContinue | ? ProviderName -match 'nvlddmkm').Count" | tr -d '\r')
    others=$(powershell -NoProfile -c "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { \$_.CommandLine -match 'train|bench|md_nve' -and \$_.CommandLine -like '*.venv*' }).Count" | tr -d '\r')
    # each job appears as two python.exe processes (venv launcher + interpreter): allow at most one other job
    if [ "${faults:-0}" -eq 0 ] && [ "${others:-0}" -le 2 ]; then return; fi
    log "waiting: faults(10 min)=$faults, other GPU jobs=$others"; sleep 120
  done
}
run() { gate; log "START $1"; shift; "$@"; log "END rc=$?"; }

# 1. timing (cutoff-matched, Figure 5b at one size, cost breakdown, qeq cost)
for t in cutoff small profile qeq; do run "bench $t" $PY tools/bench_review.py $t; done

# 2. unit-parameterisation control: default parameterisation at the higher learning rates (v4 B32 K32 config)
for lr in 0.02 0.04; do
  run "default-param lr $lr" $PY -m face.train --mol aspirin --epochs 60 --lr $lr --transport 1 --eqmsg 1 --factor 1 \
      --C 96 --B 32 --K 32 --rc 5.0 --unit 0 --tag ctrl_default_lr$lr
done

# 3. distillation controls (aspirin): fine-tune only, and a C=160 FACE-CP trained on DFT data only
run "fine-tune-only control" $PY -m face.train --mol aspirin --epochs 60 --lr 0.004 --swa_frac 0.5 --log_every 10 \
    --transport 1 --eqmsg 1 --factor 1 --C 96 --B 16 --cp 1 --unit 1 --init runs/long_cp_aspirin.pt --tag ctrl_ftonly_cp
run "CP C160 DFT only, 400 epochs" $PY -m face.train --mol aspirin --epochs 400 --lr 0.04 --transport 1 --eqmsg 1 \
    --factor 1 --C 160 --B 16 --cp 1 --unit 1 --tag ctrl_cp160_dft

# 4. long-range benchmarks with the current trainer (unit parameterisation), same settings otherwise
KO="--n_train 2000 --n_val 200 --n_test 1000 --C 64 --rc 5.0 --epochs 60 --bs 16 --lr 0.005 --wF 100 --unit 1"
run "Ko Ag local" $PY -m face.train_general --data data/ko2021/datasets/Ag_cluster/input.data $KO --tag ko6_none
run "Ko Ag coulomb" $PY -m face.train_general --data data/ko2021/datasets/Ag_cluster/input.data $KO --terms coulomb --tag ko6_coulomb
KO3="${KO/--rc 5.0/--rc 3.0}"
run "Ko C10 rc3 local" $PY -m face.train_general --data data/ko2021/datasets/Carbon_chain/input.data $KO3 --tag ko6_rc3_none
run "Ko C10 rc3 qeq" $PY -m face.train_general --data data/ko2021/datasets/Carbon_chain/input.data $KO3 --terms qeq --tag ko6_rc3_qeq

# 5. seeds for the equal-budget comparison; MACE evaluated at its final epoch
for mol in aspirin malonaldehyde uracil toluene ethanol; do
  $PY tools/make_xyz.py $mol > /dev/null
  if [ "$mol" != ethanol ]; then
    for s in 1 2; do
      run "FACE $mol seed $s" $PY -u -m face.train --mol $mol --epochs 60 --lr 0.04 --transport 1 --eqmsg 1 --factor 1 \
          --C 96 --B 16 --K 32 --rc 4.5 --unit 1 --seed $s --tag seedsuite_flag_s$s
    done
  fi
  for s in 0 1 2; do
    run "MACE $mol seed $s" bash tools/train_mace_seed.sh $mol 60 "64x0e+64x1o" seedsuite_mace_s$s $s
    $PY tools/eval_compare.py $mol mace=runs/mace/seedsuite_mace_s${s}_${mol}_stagetwo.model 2>/dev/null \
        | grep stagetwo > runs/seedsuite_mace_s${s}_${mol}_eval.txt
  done
done

# 6. molecular dynamics with the main models (NVE, 0.5 fs, 20 ps from 500 K)
for spec in "long_flag_aspirin aspirin" "gen4_eft_aspirin aspirin" "eth_gen2_cp160_ft_ethanol ethanol" \
            "mal_gen2_cp160_ft_malonaldehyde malonaldehyde"; do
  set -- $spec
  run "MD $1" bash -c "$PY tools/md_nve.py runs/$1.pt $2 500 40000 0.5 2>/dev/null | sed -n '/^{/,/^}/p' > runs/md_main_$1.json"
done
log "QUEUE DONE"
