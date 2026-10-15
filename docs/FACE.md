# FACE — Factorised Cartesian Atomic Cluster Expansion

*(Version labels v0–v8 below refer to development stages.)*

*Design document and living experiment log. Literature basis: [literature_review.md](literature_review.md).*

## 1. What the literature says is missing (summary of §7 of the review)

| # | Open problem | Who suffers | FACE design response |
|---|---|---|---|
| P1 | Clebsch–Gordan tensor-product cost O(L⁶) (O(L³) with eSCN), memory limits at 10³–10⁴ atoms | NequIP, MACE (high L), Equiformer | **Cartesian moment algebra** (ranks 0–3, symmetric storage: 20 numbers per edge); all couplings are fixed broadcast contractions — no CG tables |
| P2 | Edge work scales with node width C × number of irreps | MACE, NequIP, Allegro, eSEN | **Factorised edges**: edge carries φ_b(r) ⊗ m_k (B·K numbers), pooled by one GEMM; channel mixing to width C happens *per atom* → edge cost independent of C |
| P3 | Body-order incompleteness of 2/3-body descriptors | BPNN, SOAP-GAP, SchNet | explicit 2–5-body invariants per pass + **gated equivariant products** carried to the next pass (body order grows multiplicatively) |
| P4 | Non-conservative forces → energy drift | Orb, eqV2, EScAIP, PET-NC | forces are always −∇E (autograd); C¹-smooth cutoff envelope; tested by finite differences |
| P5 | Missing long-range electrostatics, total charge, non-local charge transfer | nearly all local models & foundation models | **latent-charge operator** with Gaussian charges, Ewald for PBC; two modes: `coulomb` (μ-equilibration, O(N·K)) and `qeq` (full electrostatically coupled charge equilibration, 4G-HDNNP-like) |
| P6 | Dispersion as an untrained bolt-on (D3) | most MLIPs | **environment-dependent C6 operator** (TS free-atom C6 × learned modulation, BJ damping, long pair list) trained jointly |
| P7 | Short-range collapse / OOD at high P, T | most uMLIPs | **ZBL operator** (universal screened nuclear repulsion, switched off beyond 1.2 Å) |
| P8 | Poor extensibility: physics hard-wired | most | **operator split**: E = E_local + Σ_k E_k, every operator has the same interface `term(graph, r, h) -> per-atom energy`, receives the shared learned atom features h, and is unit-tested for symmetry/forces/periodicity |
| P9 | Strong scaling / halo cost of deep MPNNs | MACE, NequIP, eSEN, UMA | 2 passes only (receptive field 2 r_c); padded-dense neighbour layout → regular memory access, GEMM pooling |
| P10 | Workstation stability during training (practical) | – | resource guard: GPU memory cap (prevents WDDM spill-over), RAM/temperature watchdog |

Not (yet) addressed and explicitly out of scope of this iteration: magnetism/spin, excited states, NQE,
uncertainty quantification (LLPR-style last-layer UQ is a natural add-on operator), multi-fidelity heads.

## 2. Theory

### 2.1 Cartesian moments (P1)
For atom i and neighbours j within r_c, u_ij = r_ij / |r_ij|. Symmetric outer powers u^{⊗l}, l = 0..3, are
stored in symmetric form (1 + 3 + 6 + 10 = 20 components). Under a rotation Q, u^{⊗l} → Q^{⊗l} u^{⊗l}
exactly, so any linear map acting only on channel indices preserves equivariance, and any full index
contraction of products is invariant. No spherical harmonics or CG coefficients are needed. Rank ≤ 3
Cartesian tensors carry the same information as spherical L ≤ 3 (the traces add redundant lower-rank
copies, harmless for learning).

### 2.2 Factorised edge densities (P2) — v4
Per edge only two small objects are formed:
* φ_b(r_ij; Z_i, Z_j), b = 1..B: learned radial basis (Bessel × polynomial envelope → MLP with an
  element-pair embedding in the first layer; a pure function of (r, Z_i, Z_j) → can be tabulated as splines)
* m_{j,k}, k = 1..K: low-rank neighbour message (linear map of the neighbour's scalar features; from pass 2 on,
  additively modulated by *bond-projected invariants* u·V_j, uᵀM_j u, u·V_i, uᵀM_i u)

Density: D_i[(b,k), x] = Σ_j φ_b m_{j,k} Y_x(u_ij), computed as one batched GEMM (N, BK, M) @ (N, M, 20).
Moments: A^l_i = W_l D^l_i (+ element-dependent per-channel scale), W_l : BK → C, acting on the channel index
only. Hence **per-edge cost is O(B·K·20) and independent of the atom width C**; widening C costs only
node-level work.

### 2.3 Tensor transport (pass ≥ 2)
Neighbour equivariant features (V_j ∈ ℝ^{K_d×3}, M_j ∈ Sym(3)^{K_d}) are transported and coupled with the bond:
rank 1: Σ_j φ_b V_j, Σ_j φ_b M_j u; rank 2: Σ_j φ_b M_j, Σ_j φ_b sym(u⊗V_j). These are the Cartesian equivalents
of the 1⊗0→1, 2⊗1→1, 2⊗0→2, 1⊗1→2 CG paths, again pooled by GEMMs over B.

### 2.4 Invariants and gated equivariant products (P3)
Per rank, three learned projections P, Q, S (channel maps) of all available moment sets. Channel-diagonal
contractions give 9 invariant families per pass:
2-body P0; 3-body P1·Q1, P2:Q2, P3⋮Q3; 4-body P1ᵀQ2S1, tr(P2Q2S2), (P3:Q2)·S1, (P3:Q3):S2; 5-body P3(Q1,S1,Q1).
They are written as broadcast multiply-sums (an einsum lowering to tiny bmm's was 10–20× slower).
Equivariant outputs for the next pass are built from the same tensors —
V = a₀P1 + a₁Q2S1 + a₂P3:Q2 + a₃P2Q1, M = sym(b₀P2 + b₁P1⊗Q1 + b₂P2Q2 + b₃P3·S1) — and gated per channel by a
linear function of the invariants (identity at init). Body order of the transported features thus already
reaches 4–5 after pass 1.

### 2.5 Physics operators (P5–P8)
* Latent charges: χ_i, J_i = head(h_i). E = Σχq + ½ΣJq² + ½k_e qᵀAq with Σq = Q (total charge is an input).
  A = Gaussian-smeared Coulomb (σ = 0.5 Å); PBC via Ewald with real-space kernel [erf(r/γ) − erf(r/γ')]/r and
  reciprocal kernel 4π/k² exp(−γ'²k²/4); charged cells get the neutralising-background term.
  `coulomb`: q_i = −(χ_i − μ)/J_i (no matrix; Ewald energy O(N·K)); `qeq`: full constrained linear solve.
* Dispersion: C6_i = C6_free(Z_i)·exp(w·h_i), C6_ij = √(C6_i C6_j), BJ damping R0 = a₁(R_i+R_j)/2 + a₂.
* ZBL: universal screened repulsion with a C²-smooth switch between 0.6 and 1.2 Å.

## 3. Verification (all pass; `tests/`)
* rotation, reflection, translation, permutation invariance of E and equivariance of F (1e-9)
* forces = finite-difference −dE/dx (1e-6 relative), incl. through QEq linear solve and Ewald
* C¹ continuity across r_c
* periodic: lattice-translation / wrapping invariance, exact supercell extensivity, rotation of cell
* Ewald: NaCl Madelung energy reproduced to 2×10⁻⁶ relative for two different Ewald splits; dense-matrix and
  O(N·K) paths agree to 1e-14

## 4. Experiment log
(see §5 tables; every number below is on the identical rMD17 aspirin split, 950 train / 50 val / 1000 test,
60 epochs, batch 5, EMA 0.99, energy-weighted last 25 %; MAE of total energy (meV) and force components (meV/Å))

### 4.1 Architecture iterations on rMD17 aspirin (equal 60-epoch budget)

| Version | Change (hypothesis) | E MAE (meV) | F MAE (meV/Å) | Outcome |
|---|---|---|---|---|
| MACE-small (control, 64×0e+64×1o, 241k params) | – | 10.6 | 19.5 | reference |
| v0 | Cartesian moments, scalar messages | 16.4 | 30.5 | baseline |
| v1 | + bond-projected directional invariants, element-pair radial | 14.1 | 27.3 | ✓ −10 % |
| v1-affine | LayerNorm → affine (hyp.: LN destroys magnitude info) | 17.9 | 33.7 | ✗ rejected |
| v1 lr=0.01 | learning rate | 13.6 | 27.0 | ≈ no effect |
| v2 C=64 | + rank-preserving tensor transport (1⊗1, 2⊗1 paths) | 12.2 | 25.0 | ✓ |
| v2 C=96 | width | 9.2 | 21.2 | ✓ |
| v3 C=64 | + gated equivariant products (hyp.: body order carried between passes is the gap) | 9.8 | 21.6 | ✓ |
| **v3 C=96** | | **6.8** | **16.7** | ✓ best accuracy; but edge cost ∝ C → slower than MACE |
| v4 C=96 B16 K32 | factorised edges (edge cost independent of C) | 10.2 | 20.5 | ≈ MACE accuracy, 1.7× faster |
| v4 C=128 | width now cheap — does it help? | 11.0 | 22.0 | ✗ bottleneck is edge rank, not C |
| v4 K=64 | wider messages | 10.5 | 22.6 | ✗ |
| v4 B=32 | more radial functions | 8.8 | 18.1 | ✓ |
| v4 K_dir=48 | wider transported tensors | 8.6 | 18.0 | ✓ |
| v4 1 pass (B32) | strictly local, 5× cheaper | 20.3 | 42.5 | ✗ two passes required |
| v4 1 pass, r_c = 6 Å | | 23.4 | 43.7 | ✗ |

### 4.2 Cost (energy + forces, µs per atom per step, RTX 2000 Ada 16 GB, float32, idle GPU, aspirin boxes ~1.4 g/cm³)

| Model | 168 atoms | 1,344 atoms | 4,536 atoms | peak mem @4.5k |
|---|---|---|---|---|
| MACE paper setting (256 ch, L=2) | 1634 | OOM (>12 GB cap) | OOM | – |
| MACE-small (64×0e+64×1o) | 195 | 192 | 196 | 9.2 GB |
| FACE v4 B16 K32 (PyTorch) | 130 | 102 | 106 | 4.9 GB |
| **FACE v4 B16 K32 (fused kernel)** | 114 | **73** | **77** | 3.7 GB |
| FACE v4 B32 K32 (fused) | 129 | 103 | 106 | 4.6 GB |
| FACE v4 B16 K_dir=48 (fused) | 139 | 132 | 135 | 5.4 GB |

Kernel engineering steps (each verified bit-compatible with the reference path):
einsum→broadcast contractions (10–19× on those ops); index_select on the innermost dim → constant-matrix GEMM;
per-rank GEMMs on the small Y instead of slicing the large density; fancy indexing removed; fused NVRTC
density kernel (forward 2.1× faster than PyTorch; backward rewritten from per-edge to tiled three-phase with
bank-conflict padding: 150 ms → 40 ms).

### 4.3 Long-range / charge benchmarks (Ko et al., Nat. Commun. 2021 datasets; 2000 train / 1000 test)

| System | Model | E MAE (meV/atom) | F MAE (meV/Å) | Note |
|---|---|---|---|---|
| Ag₃⁺ / Ag₃⁻ | local | 1189 | 44.3 | 6529/7500 steps non-finite (float32 overflow of high-order products) → stuck |
| | local + equivariant saturation | 214 | 440 | stable (0 skipped); geometry partly reveals charge (see note) |
| | **+ latent charges (`coulomb`)** | **1.94** | **9.7** | total charge enters as a constraint; ≈100× lower E error |
| C₁₀H₂ / C₁₀H₃⁺, **r_c 3 Å** (reach 6 Å < 13 Å chain) — controlled test | local | 0.71 | 32.8 | |
| | **+ `qeq`** | **0.37** | **17.4** | **≈2× lower E and F error from non-local charge transfer** |
| C₁₀H₂ / C₁₀H₃⁺ (r_c 5 Å, 2 passes = 10 Å reach) | local | 0.33 | 12.6 | **correction**: an earlier epoch-30 comparison (33.5 vs 1.4) was a schedule artefact |
| | + `qeq` | 0.36 | 14.5 | reach ≈ molecule length → not a discriminating test |

Note (correction): Ag₃⁺ and Ag₃⁻ were sampled by separate MD runs and occupy mostly different geometries
(largest bond angle 67° ± 5° vs 134° ± 34°; overlap only 71–88°), so this set is *not* a same-geometry test,
and the "≈1.2 eV/atom floor" argued earlier does not hold; the clean statement is the 100× gap above.

### 4.4 Equivariant saturation (stability operator) — partially rejected
A_{l,c} ← A_{l,c} / sqrt(1 + ‖A_{l,c}‖²_F / s²) (Frobenius norm with symmetric-storage multiplicities — an
unweighted norm broke rotation equivariance and was caught by the test suite). Removes all non-finite steps on
Ag₃ but with s = 4 doubles aspirin errors (19.5 / 40.0 vs 8.8 / 18.1) → kept as opt-in, not default.

### 4.5 Scale instability of polynomial invariants — diagnosis and fix (v5)
*Symptom.* Training on ethanol produced NaN gradients from step 16 onward on every batch, with all weights finite
(|w| ≤ 3.4). Anomaly mode located the NaN in the double-backward of the invariant LayerNorm.
*Diagnosis.* After 16 updates the LayerNorm inputs had grown to 1e7 (pass 0) and 3e20 (pass 1). Invariants of body
order p are degree-p polynomials of the moments; the gated products transported to pass 2 are already degree 2–3,
so pass-2 invariants are polynomials of degree up to ≈15 in the edge inputs. LayerNorm is invariant to an overall
scale, so the optimiser can (and does) use the moment *magnitude* as a free knob that re-weights body orders,
inflating it without bound until float32 overflows in the second derivative. At initialisation the opposite holds:
4-/5-body invariants are 1e-6–1e-9 of the 2-body ones and invisible after LayerNorm (row variance 5e-6 < LN eps).
Both facts explain why fixed saturation (§4.4) hurt accuracy: it clipped the growth the model relied on.
*Fix (v5).* Equivariant RMS normalisation of every tensor before it enters a product: per atom and rank,
A_l ← A_l / sqrt(mean_c ‖A_{l,c}‖²_F + ε) (rotation-invariant scalar, Frobenius weights in symmetric storage), and
the same for the gated vectors/matrices passed to the next pass; log(1 + n²/ε) of each set is appended to the
MLP input so no magnitude information is lost. Products are then O(1) independent of the weights.

### 4.6 MD stability and energy conservation (NVE, velocity Verlet, dt = 0.5 fs, aspirin, v4 B32 checkpoint)
| T₀ | length | mean T | bonds broken | drift (meV/atom/ps) | E_tot std (meV/atom) |
|---|---|---|---|---|---|
| 500 K | 1 ps | 463 K | no | 0.009 | 0.057 |
| 1000 K (2× training T) | 10 ps | 658 K | no | **0.0009** | 0.110 |

*Result (60 epochs).* v5 trains with 0 non-finite steps on ethanol (was: every step from step 16) — stability fixed —
but accuracy dropped: aspirin 20.7 meV / 32.7 meV/Å (v4 B32: 8.8 / 18.1), ethanol 9.8 / 33.6 (MACE-small not yet run).
Per-atom RMS normalisation over channels removes too much information (per-atom magnitude enters only via 4 log-norm
scalars). Open next step: normalise with a *global* (data-calibrated, fixed) scale instead of per-atom, or keep per-atom
normalisation but feed per-channel log-norms; re-test aspirin/ethanol, then relaunch the 600-epoch run and the
multi-molecule suite (tools/suite.sh) on the winning variant.

### 4.7 Root cause of training divergence: parameterisation, not normalisation (v6 → v7)
Earlier fixes treated the symptom (scale of invariants). A controlled probe (tools/probe_stability.py: constant
lr, 400–1500 Adam steps, first non-finite step reported) located the cause:

| config (lr 0.01 constant) | aspirin: first NaN step | ethanol (1500 steps) |
|---|---|---|
| v6: calibrated scales, no LayerNorm | 2 | – |
| v6 + max body order 3 / no transport / 1 pass | 3–17 | – |
| v4 (LayerNorm) | none in 400 | **85** |
| v4 + calibrated scales (LN kept) | – | 24 |
| **v4 + unit parameterisation (v7)** | **none** | **none** (F-RMSE 283 → 107 meV/Å) |

PyTorch's nn.Linear stores weights at scale 1/sqrt(fan_in) (≈0.03 for the 1024-wide edge-mixing maps), while Adam's
step is ≈lr in *absolute* size per weight → ≈30 % relative change per step at lr 0.01, summed over 1024 inputs.
Dividing by small calibrated scales amplifies this further (why v6 was worse). The e3nn/MACE convention fixes it:
store N(0,1) weights and apply 1/sqrt(fan_in) in the forward pass (`UnitLinear`, identical function at init),
making Adam updates ≈lr *relative*. v7 = v4 architecture + unit parameterisation: 0 non-finite steps on aspirin and
ethanol at lr 0.01–0.04. Because relative updates are now ~30× smaller, the optimal lr moves up:

| aspirin, 60 epochs, v7 (B32 K32) | E MAE (meV) | F MAE (meV/Å) |
|---|---|---|
| lr 0.01 | 12.5 | 24.1 |
| lr 0.02 | 9.4 | 19.4 |
| **lr 0.04** | **8.9** | **17.6** |
| MACE-small control | 10.6 | 19.5 |

Ethanol, 60 epochs: v7 (lr 0.01) 3.18 / 15.2 vs MACE-small 3.38 / 14.5.

Also found while debugging: the EMA copy of the model did not receive *buffers* (calibration scales) — fixed by
copying buffers each step; earlier results were unaffected (no buffers were in use).

### 4.8 Negative result: spline-compiled radial functions for wide radial heads
Hermite tables are exact (6e-10 rel. force error in float64 at 4096 points; h⁴ convergence), but for a CP model
with B + 4C = 400 radial outputs per edge the 4 table reads per edge cost more bandwidth than the small radial GEMM
(346 vs 262 µs/atom, interleaved benchmark). Compilation only pays off for narrow bases (B ≤ 32).

### 4.9 Stable-training accuracy (v7, unit parameterisation, lr 0.04, 60 epochs, aspirin)
| model | E MAE (meV) | F MAE (meV/Å) | speed vs MACE-small (interleaved, same load) |
|---|---|---|---|
| MACE-small (control) | 10.6 | 19.5 | 1.00× |
| **v7 B16 K32 (flagship)** | **9.0** | **18.0** | **3.22×** (fused density + fused node + compiled radial) |
| v7 B32 K32 | 8.9 | 17.6 | 1.68× (before node fusion) |
| v8 CP (per-(rank,channel) radial) | 7.2 | 16.4 | 1.68× |
Ethanol: v7 3.35 / 15.4 vs MACE-small 3.38 / 14.5.

### 4.10 Fused node-level algebra (face/node_kernel.py)
One thread per (atom, channel) evaluates all 8 polynomial invariant families and the 8 gated-product paths in
registers; hand-derived backward (product rule) matches autograd to 2e-7 for outputs and all 10 gradients.
6.3× faster than the PyTorch broadcast formulation (17 vs 107 ms per pass fwd+bwd at 4.5k atoms, C=96); lifted
the flagship from 2.69× to 3.22× MACE-small.

### 4.11 CUDA-graph inference
All per-call host→device constant copies removed (cached per device/dtype) and the structure count can be passed
in → the full energy+force evaluation (incl. the CuPy kernels) is capturable. Replay vs eager (shared GPU):
168 atoms 2.4×, 1,344 atoms 1.06×, 4,536 atoms 1.0× → launch overhead matters only for small systems; forces
agree to 1e-5 eV/Å.

### 4.12 Accuracy–cost comparison against two MACE sizes (aspirin, identical split, 60 epochs, same GPU)
| model | E MAE (meV) | F MAE (meV/Å) | µs/atom @1,344 atoms (interleaved) | vs MACE-medium |
|---|---|---|---|---|
| MACE-medium (128×0e+128×1o) | 9.1 | 17.8 | 913 (OOM at 2.6k atoms under a 12 GB cap) | 1.0× |
| MACE-small (64×0e+64×1o) | 10.6 | 19.5 | 484 | 1.9× |
| **FACE flagship: C96 B16 K32, r_c 4.5 Å, fused + compiled** | **8.5** | **18.3** | **115** | **8.0×** |
| FACE C64 r_c 4.5 (fastest; 4.6× MACE-small) | 11.6 | 21.5 | – | – |
| FACE v8 CP (accuracy option) | 7.2 | 16.4 | 265 | 3.4× |
Ethanol (60 epochs): FACE C64 r_c 4.5 2.63 / 12.8 vs MACE-small 3.38 / 14.5.

Transport post-processing (≈25 slicing/stack ops) replaced by one constant (36 × 18) linear map — outputs identical
(test energies to 4 decimals for all 9 test configurations); flagship 3.22× → 3.65× MACE-small. r_c 5 → 4.5 Å:
−27 % edges, accuracy unchanged on aspirin (9.0/18.0 → 8.5/18.3).

### 4.13 Long training (400 epochs, v7 B32 K32 K_dir32 C96 r_c 5 Å, lr 0.04, float32) — aspirin test (1000 configs)
| model (Ntrain ≈ 1000) | E MAE (meV) | F MAE (meV/Å) | source |
|---|---|---|---|
| linear ACE | 6.1 | 17.9 | published |
| **FACE v7 (this work, 400 epochs)** | **3.0** | **9.1** | measured |
| TensorNet-2L | 2.4 | 8.9 | published |
| BOTNet | 2.3 | 8.5 | published |
| NequIP | 2.3 | 8.2 | published |
| Allegro | 2.3 | 7.3 | published |
| MACE | 2.2 | 6.6 | published (longer schedules, float64) |
Validation force MAE plateaued at 8.45 meV/Å from epoch 360 while the training force RMSE fell to ≈1.6 meV/Å:
the remaining gap is generalisation (overfitting at 950 configurations), not optimisation.

### 4.14 Seed variance of 60-epoch runs (ethanol, flagship C96 B16 r_c 4.5)
seed 0: 3.93 / 16.3; seed 1: 2.87 / 13.3 → single 60-epoch runs vary by ≈20–30 %; comparisons in §4.1–4.12
that differ by less than this are within noise (MACE seeds pending).

### 4.15 Five-molecule rMD17 suite: FACE flagship vs MACE-small (identical splits, 60 epochs, same GPU)
Flagship = C96 B16 K32, r_c 4.5 Å, unit parameterisation, lr 0.04; 4.8× faster than MACE-small at inference.
| molecule | FACE E / F | MACE-small E / F | better |
|---|---|---|---|
| aspirin | **8.45 / 18.3** | 10.6 / 19.5 | FACE |
| ethanol (3 seeds, mean) | 3.29 / 14.5 | **3.16 / 13.7** | tie (within seed spread: FACE 2.87–3.93 / 13.3–16.3, MACE 3.02–3.38 / 13.0–14.5) |
| malonaldehyde | **6.28 / 16.9** | 6.91 / 18.1 | FACE |
| uracil | **1.95 / 11.1** | 2.40 / 12.8 | FACE |
| toluene | **1.46 / 6.82** | 1.72 / 7.85 | FACE |
(E in meV, F in meV/Å, test sets of 1000 configurations.)

Long training (400 epochs) of the same flagship: aspirin **3.13 / 9.08** — identical to the slower B32/K_dir32
configuration (3.04 / 9.09), i.e. the speed-optimised architecture loses nothing at convergence.

Robustness: one of three ethanol seeds diverged at lr 0.04 (epoch 9, LayerNorm double-backward overflow). Added
automatic recovery (after 20 consecutive non-finite steps: restore EMA weights, reset optimiser, halve lr) and
seeded the batch order. Weight decay (1e-3) as an alternative fix hurt accuracy (4.56 / 17.1 vs 3.08 / 13.9) →
rejected.

### 4.16 Node kernel on symmetric storage
The fused node kernel now reads P2/Q2/S2 (6 comps) and P3/Q3 (10 comps) directly and expands in registers; its
backward folds full-tensor gradients onto symmetric components (chain rule of the expansion). Removes the
sym→full GEMMs and their backward (~7 % of inference). Flagship: 4.13× → **4.80×** MACE-small (interleaved).

### 4.17 MD of the final flagship (400-epoch checkpoint, fused kernels + spline-compiled radials)
NVE, T₀ = 1000 K, 10 ps, dt 0.5 fs: no bond broken, mean T 660 K, drift −0.004 meV/atom/ps, E_tot std 0.11 meV/atom
— the inference-only fused/compiled path (C¹ Hermite radials) conserves energy as well as the PyTorch path.

### 4.18 Long training of the accuracy option (v8 CP, C96 B16, r_c 5 Å, 400 epochs): aspirin **2.88 / 8.73**
(flagship 3.13 / 9.08; TensorNet 2.4 / 8.9, BOTNet 2.3 / 8.5, NequIP 2.3 / 8.2, Allegro 2.3 / 7.3, MACE 2.2 / 6.6).

## 5. Current operating points (clean, idle GPU: RTX 2000 Ada, float32, energy + forces; aspirin rMD17)

| operating point | config | E / F, 60 ep | E / F, 400 ep | µs/atom @1,344 | @2,625 | vs MACE-small | vs MACE-medium |
|---|---|---|---|---|---|---|---|
| MACE-medium | 128×0e+128×1o | 9.1 / 17.8 | – | 361.5 | OOM* | 0.53× | 1.0× |
| MACE-small | 64×0e+64×1o | 10.6 / 19.5 | – | 190.9 | 192.3 | 1.0× | 1.9× |
| **FACE flagship** | C96 B16 K32, r_c 4.5 | **8.5 / 18.3** | **3.13 / 9.08** | **33.3** | **35.3** | **5.5–5.7×** | **10.9×** |
| FACE fast | C64 B16 K32, r_c 4.5 | 11.6 / 21.5 | – | 30.4 | 32.2 | 6.0–6.3× | 11.9× |
| FACE accurate (CP) | v8 CP C96 B16, r_c 5 | 7.2 / 16.4 | **2.88 / 8.73** | 93.8 | 96.0 | 2.0× | 3.9× |
| FACE-L | C192 B32 K_dir32, r_c 5 | – | 2.90 / 8.72 | 102.6 | 105.2 | 1.9× | 3.5× |
*MACE-medium exceeds the 12 GB cap at 2.6k atoms in this interleaved setting.
Published references (long training): MACE 2.2 / 6.6, Allegro 2.3 / 7.3, NequIP 2.3 / 8.2, BOTNet 2.3 / 8.5,
TensorNet 2.4 / 8.9, linear ACE 6.1 / 17.9.

Physics operators (latent-charge PME electrostatics + learned dispersion + ZBL) add 23 % to the fused flagship at 4,536 atoms, 44 % at 1,344 and 81 % at 567 atoms (tools/bench_operators.py, interleaved; supersedes an earlier, untraceable ≈8 % estimate).
Small systems additionally gain up to 2.4× from CUDA-graph replay (launch-overhead bound).

**Open gap.** Accuracy saturates at ≈8.7 meV/Å for both the 6.3 M-parameter FACE-L and the CP model, i.e. it is
data-efficiency (inductive bias), not capacity; the published MACE number (6.6) remains ≈25 % better on forces.

### 4.19 Long-run seed spread and element-dependent product weights (flagship, 400 epochs, aspirin)
| run | E MAE (meV) | F MAE (meV/Å) |
|---|---|---|
| flagship seed 0 | 3.13 | 9.08 |
| flagship seed 1 | 3.23 | 8.83 |
| flagship + per-element P/Q/S gates (seed 0) | 2.97 | 8.82 |
Seed spread at 400 epochs ≈ 3 %. Per-element product gates (MACE-like centre-element-dependent coupling, zero
inference cost): −5 % energy for the same seed, forces within the seed spread → kept as an option, does not close the
data-efficiency gap to MACE (6.6 meV/Å).

### 4.20 Deep ensembles: accuracy + calibrated uncertainty, paid for by the speed margin
Members are independently trained 400-epoch checkpoints; prediction = mean, uncertainty = member spread
(`FACEEnsembleCalculator`). Aspirin test set (1000 configurations):

| model | E MAE (meV) | F MAE (meV/Å) | Spearman(spread, error) | inference µs/atom @1,344 |
|---|---|---|---|---|
| single flagship (range of 3) | 2.97–3.23 | 8.82–9.08 | – | 33 |
| **3 × flagship** | **2.73** | **7.35** | 0.82 | ≈100 (1.9× faster than MACE-small) |
| CP + L + B32 (3 mixed) | 2.52 | 7.36 | 0.83 | ≈300 |
| **5 mixed (3 flagship + CP + L)** | **2.47** | **6.86** | 0.84 | ≈296 (faster than MACE-medium, ≈5× faster than MACE-paper size) |
| published MACE (256 ch, L=2) | 2.2 | 6.6 | – | ≈1634 (measured at 168 atoms; OOM at 1.3k under 12 GB) |
| published Allegro / NequIP | 2.3 / 2.3 | 7.3 / 8.2 | – | – |
The ensemble also addresses the literature's uncertainty gap (§6.4 of the review) at no extra modelling cost.

### 4.21 Two more theory changes tested at 400 epochs (aspirin, seeds 0/1)
| variant | seed 0 E / F | seed 1 E / F | mean | verdict |
|---|---|---|---|---|
| flagship | 3.13 / 9.08 | 3.23 / 8.83 | 3.18 / 8.96 | reference (single-model spread over 5 runs: 8.8–9.6 meV/Å) |
| polynomial readout (linear per-pass readout, 16-unit final MLP) | 3.59 / 9.75 | – | – | ✗ worse — refutes "deep MLP readout overfits" |
| + rank-3 transport, sym(u⊗M_j) and sym(u⊗u⊗V_j) (one (90,38) constant map) | 2.91 / 8.30 | 3.24 / 9.17 | 3.07 / 8.73 | ≈ −3 % F, within seed noise; +25 % cost → optional |
The rank-3 transport map is verified against explicit per-edge tensors to 1e-16 (tests/test_transport_map.py); a
trained flagship checkpoint reproduces its test errors exactly with the option off.

### 4.22 Ensembles of fast models reach published-MACE force accuracy
| ensemble | E MAE (meV) | F MAE (meV/Å) | Spearman(spread, error) | cost (µs/atom, idle GPU, 1,344 atoms) |
|---|---|---|---|---|
| 5 × flagship | 2.73 | 7.09 | 0.84 | ≈167 (< single MACE-small, 191) |
| 7 flagship-class (5 + 2 rank-3) | 2.64 | 6.84 | 0.85 | ≈250 (< MACE-medium, 361) |
| 11 (all trained variants) | 2.50 | **6.67** | 0.85 | ≈700 |
| published MACE (256 ch, L=2) | 2.2 | 6.6 | – | ≈1,634 at 168 atoms; OOM at 1,344 atoms under 12 GB |

### 4.23 Ensemble → single-model distillation via uncertainty-gated teacher MD (no new DFT data)
Recipe (tools/make_distill_md.py, `--aug` / `--init` in face/train.py):
1. Teacher = 7 flagship-class models (ensemble 2.64 / 6.84).
2. All 950 *training* configurations propagated simultaneously by Langevin MD at 500 K with the ensemble-mean forces
   (BAOAB, dt 0.5 fs, 400 steps, snapshot every 20) → 19,000 thermally distributed configurations; the ensemble
   spread is used as an uncertainty gate (reject > 3× the test-set spread): 18,917 kept, 83 rejected; mean teacher
   spread on kept snapshots 6.9 meV/Å vs 5.5 on the real test set (in-distribution).
   (Negative control: random Gaussian rattles of 0.02–0.06 Å are out of distribution — teacher spread 9.5 → 150 meV/Å —
   because uncorrelated displacements excite stiff modes far beyond thermal amplitude.)
3. Student = single flagship trained on DFT data (×4) + teacher snapshots, same number of optimiser steps as 400 epochs.
4. Short DFT-only fine-tune (60 epochs, lr 0.004) to remove teacher energy noise.
| single model (33 µs/atom) | E MAE | F MAE |
|---|---|---|
| flagship, 5 seeds | 2.97–3.42 | 8.82–9.60 (mean ≈ 9.1) |
| distilled | 3.49 | 8.32 |
| **distilled + DFT fine-tune** | **3.13** | **8.04** |
| rank-3 transport, distilled | 3.62 | 8.46 |
Test configurations are never used for teacher MD, labels or model selection.

### 4.24 Distillation into the CP (accuracy) model
| single CP model (≈94 µs/atom) | E MAE (meV) | F MAE (meV/Å) |
|---|---|---|
| CP, 400 epochs DFT only | 2.88 | 8.73 |
| CP student (19k teacher snapshots) | 3.07 | 7.83 |
| **CP student + DFT fine-tune** | **2.71** | **7.46** |
| published Allegro / NequIP / MACE | 2.3 / 2.3 / 2.2 | 7.3 / 8.2 / 6.6 |
Additional teacher batches: 500 K seed 1 (18,899 kept, 101 rejected), 600 K (18,717 kept, 283 rejected — the gate
rejects more at higher T, as intended).

### 4.25 CP pooling as a single GEMM
Per-rank slicing of the (N, M, 4, C) edge tensor cost a full copy forward and a zero-fill + add in backward per rank
(≈32 % of CP inference). Replaced by one GEMM against all 20 monomials on the contiguous (N, M, 4C) tensor, keeping
the rank-diagonal blocks of the small (N, 20, 4, C) result; outputs identical (trained checkpoint reproduces
2.712 / 7.464 exactly). CP model: 2.0× → **2.9×** MACE-small (interleaved, same load).
2-batch flagship student (38k teacher snapshots, same step budget): 3.55 / 8.55 — no better than 1 batch (3.49 / 8.32):
the small flagship is capacity-limited as a student; the CP model is the better student.

3-batch CP student (56k teacher snapshots incl. 600 K, same step budget): 3.37 / 8.32 — worse than the 1-batch student
(3.07 / 7.83): at a fixed optimiser budget more (and partly off-temperature) teacher data dilutes rather than helps.

## 6. Updated operating points (aspirin; idle-GPU timings scaled from interleaved ratios where marked ≈)

| operating point | E / F (meV, meV/Å) | inference | vs MACE-small | vs MACE-medium |
|---|---|---|---|---|
| **FACE flagship** (C96 B16, r_c 4.5), distilled + fine-tuned | 3.13 / 8.04 | 33 µs/atom | 5.7× | 10.9× |
| **FACE-CP** (accuracy), distilled + fine-tuned | **2.71 / 7.46** | ≈66 µs/atom | ≈2.9× | ≈5.5× |
| 5 × flagship ensemble (+ uncertainty) | 2.73 / 7.09 | ≈167 µs/atom | 1.1× | 2.2× |
| 11-model ensemble (+ uncertainty) | 2.50 / 6.67 | ≈700 µs/atom | – | – |
| published: Allegro / NequIP / BOTNet / TensorNet / MACE | 2.3 / 7.3, 2.3 / 8.2, 2.3 / 8.5, 2.4 / 8.9, 2.2 / 6.6 | – | – | – |

Single-model force error went from 9.1 (flagship) / 8.73 (CP) to 8.04 / 7.46 through uncertainty-gated ensemble
distillation + DFT fine-tuning (no new DFT data). Remaining gaps to the best published model: forces 13 %
(single) / ≈1 % (11-ensemble); energy ≈20 % (single).

### 4.26 Iterated ("born-again") distillation
Generation-2 teacher = 9 models (7 originals + the two distilled-fine-tuned students): 2.60 / 6.68 (vs 2.64 / 6.84).
Existing in-distribution teacher-MD configurations are simply relabelled (tools/relabel.py; no new MD). CP student,
25 epochs (1.5× budget), then the same DFT-only fine-tune:
| single CP model | E MAE | F MAE |
|---|---|---|
| generation 1 (distilled + ft) | 2.71 | 7.46 |
| generation 2 student | 2.87 | 7.32 |
| **generation 2 student + DFT fine-tune** | **2.69** | **7.04** |
Single-model forces now better than published Allegro (7.3) and within 7 % of published MACE (6.6).

### 4.27 Generation 3 (teacher = 10 models incl. gen-2 student: 2.58 / 6.60 = published-MACE force level)
| single CP model | E MAE (meV) | F MAE (meV/Å) |
|---|---|---|
| gen-3 student | 2.78 | 7.29 |
| **gen-3 student + DFT fine-tune** | **2.50** | **6.97** |
Gains per generation in F: −0.42 (gen 1→2), −0.07 (gen 2→3) → converging near 7.0 meV/Å for this model size.
MD check (NVE, T₀ = 1000 K, 5 ps): no bond broken, drift −0.002 meV/atom/ps — distillation does not compromise
smoothness or energy conservation.

## 7. Final state (aspirin, rMD17, Ntrain = 950 DFT configurations; same GPU for all timings)
| model | E / F (meV, meV/Å) | inference | vs MACE-small | vs MACE-medium |
|---|---|---|---|---|
| **FACE-CP gen-3 (single model)** | **2.50 / 6.97** | ≈66 µs/atom | ≈2.9× | ≈5.5× |
| FACE flagship, distilled + ft (single) | 3.13 / 8.04 | 33 µs/atom | 5.7× | 10.9× |
| 10-model ensemble (+ calibrated uncertainty, Spearman 0.85) | 2.58 / 6.60 | – | – | – |
| published MACE (256 ch, L=2) | 2.2 / 6.6 | ≈1,634 µs/atom @168 atoms; OOM @1,344 | – | – |
| published Allegro / NequIP | 2.3 / 7.3, 2.3 / 8.2 | – | – | – |

### 4.28 Generation 4: wider student (CP C160) from an 11-model teacher (2.55 / 6.53, better than published MACE)
| single model | E MAE (meV) | F MAE (meV/Å) | speed vs MACE-small / MACE-medium |
|---|---|---|---|
| CP C96 gen-3 (distilled + ft) | 2.50 | 6.97 | ≈2.9× / ≈5.5× |
| CP C160 gen-4 student | 2.74 | 7.07 | – |
| **CP C160 gen-4 student + DFT fine-tune** | **2.48** | **6.70** | 2.0× / 3.7× (interleaved) |
| published MACE (256 ch, L=2, float64) | 2.2 | 6.6 | ≈0.12× MACE-small; OOM at 1,344 atoms (12 GB) |
Student capacity was the limiter: at fixed teacher quality, C96 → C160 lowered forces 6.97 → 6.70.

### 4.29 Generality check: ethanol (400 epochs, CP C96, 3 seeds)
Single: 0.74–0.78 / 4.69–4.90; 3-ensemble 0.61 / 3.86 (Spearman 0.85). Published: ACE 1.2 / 7.3, NequIP 0.4 / 2.8,
Allegro / MACE 0.4 / 2.1. On ethanol FACE trails the published best by ≈2.3× (single) — a much larger gap than on
aspirin; heavy overfitting (train force RMSE ≈0.8 vs test 4.9). Control (MACE-small, same 400-epoch protocol) and the
distillation pipeline on ethanol are running.

Control (same 400-epoch protocol, same split, same GPU): **MACE-small on ethanol 1.11 / 6.76** vs FACE-CP single
0.74–0.78 / 4.69–4.90 → FACE ≈30 % better in both E and F at equal training. The gap to the published 0.4 / 2.1 is
therefore attributable to the much larger float64 MACE configuration (256 ch, L=2), not an ethanol-specific
weakness of the FACE theory.

Ethanol distillation (3-model teacher, 0.61 / 3.86; teacher MD 500 K, 18,872 kept / 128 gated; CP C96 student + DFT ft):
**0.68 / 4.22** (single model) vs 0.74–0.78 / 4.69–4.90 undistilled and MACE-small (same protocol) 1.11 / 6.76.

## 8. Final state after this round
| model | aspirin E / F | ethanol E / F | speed vs MACE-small | vs MACE-medium |
|---|---|---|---|---|
| **FACE-CP C160 gen-4 (single, distilled + ft)** | **2.48 / 6.70** | – | 2.0× | 3.7× |
| FACE-CP C96 (single, distilled + ft) | 2.50 / 6.97 (gen 3) | **0.68 / 4.22** (gen 1) | 2.9× | 5.5× |
| FACE flagship (single, distilled + ft) | 3.13 / 8.04 | – | 5.7× | 10.9× |
| MACE-small, 400 epochs, same protocol | – | 1.11 / 6.76 | 1.0× | 1.9× |
| published MACE (256 ch, L=2, float64) | 2.2 / 6.6 | 0.4 / 2.1 | ≈0.12× | – |
| published Allegro | 2.3 / 7.3 | 0.4 / 2.1 | – | – |

### 4.30 Energy-weighted final fine-tune
The DFT-only fine-tune of §4.23 still used force-weighted loss for half its epochs. An energy-weighted variant
(w_E = 1000, w_F = 10 throughout, lr 0.002, 60 epochs) on the gen-4 C160 model:
| single model | E MAE (meV) | F MAE (meV/Å) |
|---|---|---|
| gen-4 C160 + standard fine-tune | 2.48 | 6.70 |
| **gen-4 C160 + energy-weighted fine-tune** | **2.26** | **6.74** |
| published MACE (256 ch, L=2, float64) | 2.2 | 6.6 |
Single-model gap to the best published model on aspirin: energy ≈3 %, forces ≈2 %, at ≈2× MACE-small speed
(the published configuration runs at ≈0.12× MACE-small on this GPU and OOMs at 1,344 atoms).

### 4.31 Ethanol, generation 2 (teacher = 3 CP models + gen-1 distilled: 0.60 / 3.73; wide CP C160 student)
Run under GPU-driver instability on a shared GPU (nvlddmkm Event 13/153 resets) and two RAM-watchdog aborts;
completed via periodic EMA checkpoints (`runs/<tag>_<mol>_last.pt`, added to face/train.py) and automatic resume.
| single model, ethanol | E MAE (meV) | F MAE (meV/Å) |
|---|---|---|
| MACE-small, 400 epochs, same protocol | 1.11 | 6.76 |
| CP C96 gen-1, distilled + ft (previous best) | 0.68 | 4.22 |
| CP C160 gen-2 student | 0.676 | 3.98 |
| **CP C160 gen-2 + DFT fine-tune** | **0.619** | **3.716** |
| CP C160 gen-2 + energy-weighted fine-tune | 0.636 | 3.680 |
| published MACE (256 ch, L=2, float64) | 0.4 | 2.1 |
Forces −12 % vs gen-1; vs MACE-small at equal protocol −45 % F, −44 % E. Unlike aspirin, the energy-weighted fine-tune
does not improve ethanol energy (within noise of the standard fine-tune). Gap to the large published MACE remains
≈1.75× in F. Logs: runs/eth_gen2_cp160_ethanol_{try*,r*,s*}.log, runs/eth_gen2_cp160_ft_ethanol.log,
runs/eth_gen2_eft_ethanol.log.
