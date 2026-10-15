# Handoff: pivot to a genuinely novel MLIP foundation

*Written 2026-10-15 ~08:50 from a side (forked) session, for pickup by the main session.*

## 1. What the user asked (in this side session)

1. "How is FACE different from / similar to MACE?" — answered (summary in §2).
2. The user then objected: **FACE is essentially a variant of ACE, not the "highly novel approach" the original
   goal asked for.**
3. When offered non-ACE directions (learned density functional, equilibrium message passing, exact-frame nets), the
   user answered:
   > "the theoretical foundation needs to be not just non-ACE but it should **not be present in the literature in
   > the MLIP community**."

**Requirement going forward:** the core theoretical foundation of the proposed potential must be absent from the
published MLIP literature (verified by a literature search), while still meeting the original goals: near-DFT
accuracy, lower cost than other approaches including all interaction terms, customisable/extensible.

## 2. Honest assessment of FACE's novelty (agreed with the user)

| FACE component | Closest prior work |
|---|---|
| neighbour-density moments → body-ordered invariants | ACE, MACE |
| Cartesian moment tensors instead of spherical harmonics | MTP (2016), CACE, TensorNet, HotPP |
| factorised edge weights (radial × message) | Allegro; MACE radial-times-channel weights |
| latent charges + Ewald / QEq | 4G-HDNNP, LES, AIMNet2 |
| ensemble UQ; distillation into one model | deep ensembles; MLIP distillation papers |

Genuinely new but incremental: constant-map tensor transport, edge cost decoupled from node width + fused kernels,
the parameterisation diagnosis of training divergence, uncertainty-gated teacher-MD distillation.
**Conclusion:** FACE is a well-engineered ACE-family model; it does NOT satisfy the novelty requirement.

The three non-ACE ideas first offered were also rejected as already published in MLIP literature:
learned density functionals (ML-HK maps, DeePKS), deep-equilibrium force fields, local-frame / frame-averaging models.

## 3. Candidate novel foundations under literature check

A background research agent (launched from the side session) is checking these; its report goes to
**`docs/novelty_check.md`** (not yet written at the time of this note — check whether it exists).

1. **Wave-interference potential** — atoms are sources of a learned multi-channel complex field obeying a screened
   Helmholtz equation, solved in closed form via Green's functions (e.g. exp(ikr − κr)/r, learned complex
   wavenumbers, element-dependent complex amplitudes). Energy = learned function of the coherent field at each
   nucleus. Angular many-body dependence comes from phase interference, not spherical harmonics / Cartesian tensors /
   body-order products; long range is native.
2. **Variational latent-state potential** — E(R) = min over continuous per-atom latent vectors s of
   [Σ_i f_θ(s_i; env_i) + Σ_ij J_θ(r_ij) g(s_i, s_j)] with learned *nonlinear* on-site energies and couplings
   (QEq = quadratic scalar special case); forces by the envelope (Hellmann–Feynman) theorem; multi-stability
   (bond breaking, charge transfer) arises naturally.
3. **Optimal-transport potential** — environments as measures; energy from differentiable (Sinkhorn) Wasserstein
   distances to learned prototype environments.
4. **Differentiable topological potential** — energy from smooth persistent-homology summaries of the configuration,
   trained on energies and forces.

The agent may also propose up to two additional foundations it found absent from the literature.

### 3.1 Result of the literature check (completed; full report in `docs/novelty_check.md`)

**No candidate is fully novel; all four are PARTIALLY NOVEL.** Some citations in the report are marked unverified —
check before citing.

| candidate | verdict | closest prior work | notes |
|---|---|---|---|
| 1 wave-interference | partially novel; **angular claim is wrong** | Euclidean Fast Attention (arXiv:2412.08541); LODE + 1/r^p (arXiv:2308.13208); Ewald MP (arXiv:2303.04791); NEMORA (arXiv:2610.10776, Yukawa); PSWF-LR (arXiv:2606.06617) | an isotropic point-source Green's function makes the field at a nucleus depend only on distances → no angular information, weaker than radial ACE; angles need off-site sampling / field gradients / anisotropic sources, which lead back to prior work |
| 2 variational latent-state | partially novel; **recommended** | Xie, Persson & Small, JCTC 2020 (doi:10.1021/acs.jctc.0c00217), NN energy minimised over atomic populations; M-OFDFT (arXiv:2309.16578); magnetic MTP (minimises over moments); Shaidu et al. npj 2024 (label-free QEq); DEQuify (arXiv:2509.08734, fixed point not minimum) | **open niche:** unsupervised *vector* latents with learned nonlinear *convex* on-site energies, trained only on E/F, as a fast local potential |
| 3 optimal transport | partially novel, weak | OT-GNN (arXiv:2006.04804); REMatch (PCCP 2016) | rotation-invariance problematic; much slower than ACE/MACE |
| 4 persistent homology | partially novel, weakest | Minamitani et al., JCP 2023 (arXiv:2206.13727), energies only | forces discontinuous at persistence-pairing changes → breaks energy conservation |

Agent-proposed additional foundations with **no MLIP hits found**:
- **P1 latent free-energy potential** — *marginalise* (log-integral / free energy) over latent states instead of
  minimising; a refinement of candidate 2 that keeps forces smooth without requiring convexity.
- **P2 soft-registration template potential** — align each neighbourhood to learned templates; lower priority (near
  OT-GNN).

Agent recommendation: candidate 2 in its open form (vector latents, learned convex on-site energies, E/F-only
training, fast local), with P1 as the fallback if smoothness is a problem. **Not yet presented to / chosen by the user.**

## 4. Next steps for the main session

1. Read `docs/novelty_check.md` when present (verdicts: NOVEL / PARTIALLY NOVEL / ALREADY PUBLISHED, with citations).
2. Present the surviving candidates to the user and **let the user choose** before building (the user explicitly set
   the novelty bar; do not pick unilaterally).
3. Build the chosen foundation in a new module (keep `face/` intact as the ACE-family baseline/comparator).
   Reuse existing infrastructure: `face/data.py`, `face/datasets.py`, `face/safety.py`, the trainer patterns in
   `face/train.py` (unit parameterisation, EMA with buffer sync, NaN recovery), `tests/test_symmetry.py` style
   tests, `tools/bench_interleaved.py` (load-robust speed vs MACE), `tools/eval_compare.py`, the PME/Ewald and
   dispersion operators.
4. Benchmark identically to FACE: rMD17 aspirin/ethanol (same splits, 60- and 400-epoch protocols), Ko et al.
   long-range sets, speed vs MACE-small/medium on the same GPU.

## 5. Background jobs still running at handoff (all launched from the main line of work)

| job | log | state at ~08:48 |
|---|---|---|
| ethanol gen-2 distillation, CP C160 student (25 epochs) → DFT fine-tune (queued in `/tmp/eth_gen2.sh`) | `runs/eth_gen2_cp160_ethanol.log`, then `runs/eth_gen2_cp160_ft_ethanol.log` | epoch 8/25, val 3.23 / 6.58 |
| malonaldehyde CP teachers, seeds 0–2, 400 epochs (breadth test of distillation) | `runs/mal_cp_s{0,1,2}_malonaldehyde.log` | epoch 20/400 |

Pending manual step: after `runs/eth_gen2_cp160_ft_ethanol.log` shows TEST, run the energy-weighted fine-tune
(check first that `runs/eth_gen2_eft_ethanol.log` does not already exist — a duplicate may have been queued):

```
.venv/Scripts/python -u -m face.train --mol ethanol --epochs 60 --lr 0.002 --swa_frac 1.0 --wE2 1000 --wF2 10 --transport 1 --eqmsg 1 --factor 1 --C 160 --B 16 --cp 1 --unit 1 --seed 0 --init runs/eth_gen2_cp160_ft_ethanol.pt --tag eth_gen2_eft
```

## 6. Best FACE results to date (for comparison with whatever is built next)

| model | aspirin E / F (meV, meV/Å) | ethanol E / F | speed vs MACE-small |
|---|---|---|---|
| FACE-CP C160 gen-4 + energy-weighted ft (single) | 2.26 / 6.74 | – | ≈2.0× |
| FACE-CP C96 distilled + ft (single) | 2.50 / 6.97 | 0.68 / 4.22 | ≈2.9× |
| FACE flagship distilled + ft (single) | 3.13 / 8.04 | – | 5.7× (10.9× vs MACE-medium) |
| MACE-small, same 400-epoch protocol | – | 1.11 / 6.76 | 1.0× |
| published MACE (256 ch, L=2, float64) | 2.2 / 6.6 | 0.4 / 2.1 | ≈0.12× |

Full experiment log: `docs/FACE.md`. Literature review: `docs/literature_review.md`.
