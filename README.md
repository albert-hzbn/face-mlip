# FACE — Factorised Cartesian Atomic Cluster Expansion

An efficient Cartesian, ACE-family machine-learning interatomic potential with pluggable physics operators
(latent-charge electrostatics with PME, charge equilibration, environment-dependent dispersion, ZBL repulsion),
fused GPU kernels, and an uncertainty-gated ensemble-distillation training protocol.

## Layout

| path | content |
|---|---|
| `face/` | the package: `model.py` (architecture, physics-operator interface), `cartesian.py` (symmetric Cartesian tensors), `electrostatics.py` (latent charges, QEq, Ewald, smooth PME), `dispersion.py`, `kernels.py` + `node_kernel.py` (fused CUDA kernels via NVRTC/CuPy), `train.py` / `train_general.py` (trainers incl. distillation, recovery, fine-tuning), `calculator.py` (ASE calculators incl. ensemble with uncertainty), `safety.py` (GPU-memory / RAM / temperature guards) |
| `tests/` | symmetry, finite-difference forces, cutoff continuity, periodic invariance, Ewald/Madelung/PME, transport map, fused-kernel equivalence |
| `tools/` | benchmarks (`bench_interleaved.py` = load-robust speed ratio vs MACE), evaluation (`eval_compare.py`, `eval_ensemble.py`), teacher-MD distillation (`make_distill_md.py`, `relabel.py`), MD (`md_nve.py`), MACE baselines (`train_mace.sh`), stability probe, profilers |
| `data/` | generated distillation sets (`distill*.npz`); the third-party data sets are downloaded separately (see "Data" below) |
| `runs/` | training logs (`*.log`), result JSON, FACE checkpoints (`*.pt`), per-structure predictions and MD trajectories (`paper_*.npz`) |

## Setup

```bash
py -3.12 -m venv ../.venv
../.venv/Scripts/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
../.venv/Scripts/python -m pip install -r requirements.txt
```
All commands below are run from this folder. Shell scripts use `$PYTHON` (default `../.venv/Scripts/python`).
On torch ≥ 2.6, the MACE baseline needs `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (e3nn 0.4.4 loads its own constants).

## Data
The third-party reference data are not redistributed here; download them from the original archives:

| data set | source | place in |
|---|---|---|
| rMD17 (Christensen & von Lilienfeld 2020) | figshare, article 12672038: `rmd17_<molecule>.npz` and the split files `index_train_0k.csv`, `index_test_0k.csv` | `data/rmd17/` |
| Ko et al. 2021 4G-HDNNP sets | Materials Cloud Archive, record bk6kw-4b895 (`datasets.tar.gz`, unpacked to `datasets/`) | `data/ko2021/` |

The extxyz files used for the MACE baselines are generated with `tools/make_xyz.py`, and the MACE baselines are
retrained with `tools/train_mace.sh`. The distillation data sets and all FACE checkpoints used in the paper are
included.

## Tests
```bash
../.venv/Scripts/python tests/test_symmetry.py
../.venv/Scripts/python tests/test_periodic.py
../.venv/Scripts/python tests/test_electrostatics.py
../.venv/Scripts/python tests/test_transport_map.py
../.venv/Scripts/python tests/test_kernels.py        # needs CUDA + CuPy
../.venv/Scripts/python tests/test_node_kernel.py    # needs CUDA + CuPy
```

## Reproducing the headline results (aspirin, rMD17 split 1)
```bash
# flagship, 400 epochs (C96 B16 K32, r_c 4.5 A)
../.venv/Scripts/python -m face.train --mol aspirin --epochs 400 --lr 0.04 --transport 1 --eqmsg 1 --factor 1 --C 96 --B 16 --K 32 --rc 4.5 --unit 1 --tag long_flag
# CP accuracy model, 400 epochs
../.venv/Scripts/python -m face.train --mol aspirin --epochs 400 --lr 0.04 --transport 1 --eqmsg 1 --factor 1 --C 96 --B 16 --cp 1 --unit 1 --tag long_cp
# teacher MD (uncertainty-gated) -> distillation set
../.venv/Scripts/python tools/make_distill_md.py aspirin 500 400 20 16.5 data/distill_md_aspirin.npz <teacher checkpoints...>
# student + DFT fine-tune (+ energy-weighted fine-tune)
../.venv/Scripts/python -m face.train --mol aspirin --epochs 25 --log_every 2 --lr 0.04 --transport 1 --eqmsg 1 --factor 1 --C 160 --B 16 --cp 1 --unit 1 --aug data/distill_md_gen4_aspirin.npz --dft_rep 4 --tag distill_gen4_cp160
../.venv/Scripts/python -m face.train --mol aspirin --epochs 60 --lr 0.004 --swa_frac 0.5 --transport 1 --eqmsg 1 --factor 1 --C 160 --B 16 --cp 1 --unit 1 --init runs/distill_gen4_cp160_aspirin.pt --tag distill_gen4_cp160_ft
../.venv/Scripts/python -m face.train --mol aspirin --epochs 60 --lr 0.002 --swa_frac 1.0 --wE2 1000 --wF2 10 --transport 1 --eqmsg 1 --factor 1 --C 160 --B 16 --cp 1 --unit 1 --init runs/distill_gen4_cp160_ft_aspirin.pt --tag gen4_eft
# evaluate / ensemble / speed
../.venv/Scripts/python tools/eval_ensemble.py aspirin runs/gen4_eft_aspirin.pt
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 ../.venv/Scripts/python tools/bench_interleaved.py --side 4 --rounds 7
```
Teacher ensembles (aspirin): generation 1 = 7 flagship-class models (5 flagship seeds + 2 rank-3-transport
variants); each later generation adds the previous generation's fine-tuned student (9, 10 and 11 models for
generations 2–4). Pass the teacher checkpoints from `runs/` to `tools/make_distill_md.py` (new MD) or
`tools/relabel.py` (relabel existing snapshots).

## Hardware used
Single NVIDIA RTX 2000 Ada (16 GB), Windows 11, float32 throughout; all speed numbers are energy + forces.
`face/safety.py` caps GPU memory at 75 % and aborts on low RAM / high GPU temperature.

## Licence
MIT (see `LICENSE`).
