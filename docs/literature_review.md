# Machine-Learning Interatomic Potentials: A Critical Literature Review (state as of 2026-10-13)

**Scope.** This review covers descriptor-based, invariant message-passing, equivariant (spherical and Cartesian), unconstrained, long-range and foundation-model MLIPs. For each family it gives the core idea, strengths, documented limitations with citations, and candidate fixes. It closes with an open-problems table, an rMD17 baseline table and a table of speed numbers.

**Conventions.**
- arXiv IDs below were checked against the arXiv API or the PDF itself during this review, unless a citation is marked *(ID not verified)*.
- Errors are MAE unless stated. Energies are in meV (total energy per molecule for rMD17) and forces in meV/Å, per component.
- 1 kcal/mol = 43.364 meV. This factor was used wherever a paper reported kcal/mol and the table shows meV.
- "Conservative" means forces are computed as F = −∇E. "Direct" means forces come from a separate output head.

---

## 0. Executive summary

1. **Accuracy on small-molecule benchmarks is saturated.** On rMD17 with about 1000 training configurations, every modern equivariant model is within about 1–3 meV/Å of the others. The spread is set by noise, splits and hyperparameters, not by architecture (Table R1). The meaningful differentiators are now:
   - low-data accuracy (N=50),
   - extrapolation (3BPA at 1200 K, dihedral slices),
   - MD stability,
   - smoothness (phonons, κ),
   - long-range physics,
   - cost.
2. **The universal-model frontier moved fast in 2025–2026.** On Matbench Discovery (snapshot 2026-10-13), the top entries all train on MPtrj+OMat24+sAlex:
   - Prophet-OAME-MBD (CPS 0.912)
   - TECE-OAM-RRA-1.0 (0.908)
   - EquFlashV2 (0.907)
   - EquiformerV3+DeNS-OAM (0.902)
   - GRACE-3L-OAM-L (0.900)
   - PET-OAM-XL (0.898)

   The best model trained on MPtrj only ("compliant") is DPA-4.0.1-Pro-MPtrj (CPS 0.840). κSRME, the thermal-conductivity error that probes smoothness, dropped from about 1.7 (Orb-v2, eqV2) to about 0.065–0.12 at the top.
3. **Several known failure modes remain only partly solved:**
   - systematic PES softening in universal models (Deng et al. 2024)
   - non-conservative force pathologies (Bigi et al. 2024/2025)
   - missing long-range electrostatics and charge/spin states (now patched by LES, MACE-POLAR-1, 4G-HDNNP, AIMNet2-style terms)
   - out-of-distribution failure under high pressure, temperature or reactive conditions
   - UQ that is not calibrated under distribution shift
   - poor strong scaling of message-passing models for large-scale MD
   - absence of magnetism, excited states and fidelity beyond GGA in most foundation models
4. **Cost.** The fastest local models (SNAP, MTP, ACE on CPU; Allegro on GPU) reach about 0.3 µs/(atom·step) per A100 for 10^5–10^6 atoms (Allegro). Large universal equivariant models reach roughly 1.7–30 steps/s for 1000 atoms on one H100, which is about 30–600 µs/(atom·step) (Table S1).

---

## 1. Descriptor + linear/kernel/NN-on-descriptor models

### 1.1 Behler–Parrinello NN (BPNN/HDNNP) and 4G-HDNNP
- **Core idea.** The energy is a sum of atomic energies. Each atomic energy is an element-specific feed-forward NN acting on fixed atom-centred symmetry functions (ACSFs: radial and angular two- and three-body terms) within a cutoff. Reference: Behler & Parrinello, PRL 98, 146401 (2007).
- **Strengths.** Strictly local, so it is linear-scaling and trivially parallel. It is mature in LAMMPS (n2p2, RuNNer) and fast on CPU. It is well suited to single-system production MD.
- **Limitations.**
  - *Descriptor incompleteness.* ACSFs are 2+3-body. Pozdnyakov et al. constructed pairs of distinct environments that 3-body descriptors cannot distinguish. Fits then show irreducible errors and poor learning curves. Reference: *On the Completeness of Atomic Structure Representations*, arXiv:2001.11696, PRL 125, 166001 (2020).
  - *Element scaling.* The number of ACSFs grows combinatorially with element count, and each element needs its own network.
  - *Locality.* There are no long-range or charge-transfer effects. This motivated the 3G and 4G variants.
  - *Manual descriptor tuning.*
- **4G-HDNNP.** Ko, Finkler, Goedecker & Behler, arXiv:2009.06484 (Nat. Commun. 12, 398, 2021) add environment-dependent electronegativities. A global charge equilibration (QEq) step then gives non-local charge transfer and electrostatics. This builds on CENT (Ghasemi et al., PRB 92, 045131, 2015).
  - *Limitations.* QEq requires a global linear solve, which is cubic for a dense solve and needs iterative or Ewald solvers in periodic systems. Training requires reference charges, such as Hirshfeld charges, which are themselves partitioning-dependent.
- **Fixes.**
  - Complete many-body descriptors (ACE, MTP, SOAP-power spectrum plus higher orders).
  - Learned chemical embeddings, as in DeepMD/DPA and MACE.
  - Learned latent charges trained without charge labels (LES, §4).

### 1.2 GAP/SOAP
- **Core idea.** Gaussian-process (kernel) regression on SOAP power-spectrum descriptors, with sparse GPR. References: Bartók et al. arXiv:0910.1019 (PRL 104, 136403, 2010) and *On representing chemical environments*, arXiv:1209.3140 (PRB 87, 184115, 2013).
- **Strengths.** Data efficient. Principled GP predictive variance for UQ and active learning. Widely used for materials (Si, C, W, phase-change materials).
- **Limitations.**
  - *Cost.* Zuo et al. found that "optimal" GAP models are about two orders of magnitude more expensive than MTP, NNP or SNAP at similar accuracy (single-core LAMMPS, Mo). Reference: *A Performance and Cost Assessment of Machine Learning Interatomic Potentials*, arXiv:1906.08888 (J. Phys. Chem. A 124, 731, 2020). Cost is linear in the number of sparse points.
  - *Body order.* The SOAP power spectrum is 3-body, which is incomplete (Pozdnyakov 2020).
  - *Multi-element scaling.* Descriptor size grows as the square of the number of elements unless compressed.
  - *Accuracy on rMD17.* About 18 meV and 45 meV/Å on aspirin (Table R1), much worse than equivariant NNs.
- **Fixes.** Compressed SOAP / tensor-reduced features, linear ACE, sparse kernels, or NN heads on SOAP.

### 1.3 SNAP / qSNAP
- **Core idea.** Energy is linear (SNAP) or quadratic (qSNAP) in bispectrum components of the neighbour density expanded in 4D hyperspherical harmonics. References: Thompson et al., arXiv:1409.3880 (J. Comput. Phys. 285, 316, 2015); Wood & Thompson, qSNAP, arXiv:1711.11131 (JCP 148, 241721, 2018).
- **Strengths.** Very fast, with a GPU Kokkos implementation. SNAP holds the "billion-atom" quantum-accurate MD records. Nguyen-Cong et al. (SC21 Gordon Bell finalist) report 20 billion atoms on all 27,900 Summit GPUs, and a peak of 6.21 M atom-steps/(node·s), roughly 1 M atom-steps/s per V100. The carbon potential is described in arXiv:2205.01209. Fitting is a linear least-squares problem, which makes it simple and reproducible.
- **Limitations.**
  - Limited expressivity: the bispectrum is effectively a 4-body descriptor, and the linear form is fixed.
  - Accuracy plateaus. qSNAP over-fits for J_max > 3 (Zuo et al. 2020).
  - Poor multi-element scaling.
  - No long-range physics.
- **Fixes.** ACE (a strict generalisation), NN-SNAP hybrids (FitSNAP NN), and explicit-multielement SNAP.

### 1.4 Moment Tensor Potentials (MTP)
- **Core idea.** Contractions of moment tensors of the neighbour density give a complete polynomial basis of increasing "level". The energy is linear in that basis, with radial functions fitted nonlinearly. References: Shapeev, arXiv:1512.06054 (MMS 14, 1153, 2016); MLIP package, Novikov et al., MLST 2, 025002 (2021).
- **Strengths.** Systematically improvable. Lies on or near the accuracy–cost Pareto front among classical MLIPs (Zuo et al. 2020). Has built-in D-optimality (MaxVol) active learning, which is cheap and effective. Magnetic variant: mMTP.
- **Limitations.**
  - Number of basis functions grows quickly with element count.
  - CPU-centric.
  - Local only.
  - Training is a nonconvex fit of radial parameters.
- **Fixes.** Element-embedding compression, GPU kernels, and coupling to long-range terms.

### 1.5 ACE (linear ACE, PACE, ML-PACE)
- **Core idea.** The atomic cluster expansion (Drautz, PRB 99, 014104, 2019) gives a complete, systematically convergent, body-ordered basis. It is built as products of a one-particle basis (radial × Y_lm), symmetrised by Clebsch–Gordan contraction. Density trick: the cost is linear in the number of neighbours for any body order. The energy can be linear in the basis (linear ACE) or a nonlinear function of several "densities" (PACE).
- **Strengths.**
  - Completeness. It unifies SOAP, SNAP, MTP and ACSF as special cases.
  - Very data-efficient. On 3BPA, ACE extrapolates well relative to sGDML and GAP (Kovács et al., JCTC 17, 7696, 2021).
  - Fast CPU and GPU implementations (PACE, Lysogorskiy et al., arXiv:2103.00814, npj Comput. Mater. 7, 97, 2021).
  - Basis extrapolation-grade UQ (D-optimality) is available.
- **Limitations.**
  - Basis size explodes with element count and body order.
  - Accuracy on flexible molecules is below that of equivariant MPNNs: aspirin 6.1 meV / 17.9 meV/Å vs MACE 2.2 / 6.6 (Table R1).
  - The PACE v1 preprint reports 0.32 ms/atom/step (Cu) and 0.80 ms/atom/step (Si) on one CPU core for those particular (large) bases.
  - Local only.
- **Fixes.**
  - Tensor-reduced / low-rank ACE.
  - Graph ACE (GRACE, §3.6), which brings in semilocal interactions.
  - MACE, which is ACE inside message passing.
  - Learned element embeddings.

### 1.6 sGDML
- **Core idea.** Kernel regression directly on forces, using a Hessian-of-kernel (gradient-domain) model. The energy is conservative by construction. Molecular permutation symmetries are symmetrised. Reference: Chmiela et al., arXiv:1802.09238 (Nat. Commun. 9, 3887, 2018).
- **Strengths.** Excellent data efficiency for a single molecule. Exact energy conservation.
- **Limitations.**
  - It is a *global* descriptor of one molecule (inverse distances), so it does not transfer across sizes or compositions.
  - Cost scales as O(N_train^2·N_atoms^2) in memory.
  - It extrapolates very poorly. On 3BPA, the 600 K force RMSE is 439 meV/Å, versus 64 for ACE and 26 for NequIP (Allegro and CACE papers, quoting Kovács et al. 2021).
  - Its rMD17 numbers are not reported in the primary sources examined here. Only original-MD17 values exist: aspirin 8.2 meV / 29.5 meV/Å, Table R2.
- **Fixes.** Local or global hybrids such as BIGDML (periodic), plus kernel approximations.

### 1.7 FCHL (FCHL18/FCHL19)
- **Core idea.** A smooth, alchemical two- and three-body representation used with KRR, including operator-quantum ML for forces. Reference: Christensen et al., *FCHL revisited*, arXiv:1909.01946 (JCP 152, 044107, 2020).
- **Strengths.** Strong for its era on QM9 and rMD17 (aspirin 6.2 / 20.9).
- **Limitations.** Kernel cost, a 3-body ceiling, and no materials-scale software.
- **Note.** The revised MD17 (rMD17) dataset itself was introduced by Christensen & von Lilienfeld, arXiv:2007.09593 (MLST 1, 045018, 2020).

### 1.8 POD (Proper Orthogonal Descriptors)
- **Core idea.** Many-body descriptors obtained by proper orthogonal decomposition of parametrised radial/angular basis functions, with a linear or quadratic energy. Implemented in LAMMPS ML-POD. Reference: Nguyen & Rohskopf, arXiv:2209.02362 (JCP 2023).
- **Strengths.** Linear-regression simplicity and speed comparable to SNAP and ACE.
- **Limitations.** Same as other linear local models: a body-order and expressivity ceiling, poor multi-element scaling, and no long-range physics.

---

## 2. Invariant (and early vector-equivariant) message passing

### 2.1 SchNet
- **Core idea.** Continuous-filter convolutions on interatomic distances, with learned element embeddings and a few interaction blocks. Reference: Schütt et al., arXiv:1706.08566 (NeurIPS 2017).
- **Strengths.** Simple. Learned chemical embeddings remove per-element descriptor blow-up.
- **Limitations.**
  - Distances only, so each layer is effectively 2-body. It cannot distinguish some geometries without many layers.
  - Accuracy is poor: original MD17 aspirin 16.0 meV / 58.5 meV/Å (Table R2).
  - No rMD17 results in its primary source.
- **Fixes.** Angular and directional information (DimeNet), equivariant features (PaiNN, NequIP).

### 2.2 PhysNet
- **Core idea.** A SchNet-like MPNN that also predicts atomic partial charges, with charge-neutrality correction and explicit electrostatics plus D3 dispersion. Reference: Unke & Meuwly, arXiv:1902.08408 (JCTC 15, 3678, 2019).
- **Strengths.** The first widely used NN with physically motivated long-range terms and dipole fitting.
- **Limitations.**
  - Charges are fitted to reference dipoles, which is non-unique.
  - Charges are local and do not equilibrate globally, so charge transfer is missed.
  - Invariant 2-body features.

### 2.3 DimeNet / DimeNet++
- **Core idea.** Directional (angle-aware) message passing over edges, using 2D spherical Fourier–Bessel bases of distance and angle. References: Gasteiger et al., arXiv:2003.03123 (ICLR 2020); DimeNet++, arXiv:2011.14115.
- **Strengths.** Explicit 3-body information at each layer, giving a big accuracy jump over SchNet.
- **Limitations.**
  - O(N·k²) triplet cost.
  - MD instability despite reasonable force errors (Fu et al., *Forces are not Enough*, arXiv:2210.07237, TMLR 2023).
  - Accuracy below equivariant models: original MD17 aspirin 8.8 / 21.6.

### 2.4 GemNet / GemNet-OC
- **Core idea.** Two-hop geometric message passing with dihedral (quadruplet) information, shown to be universal under certain conditions. GemNet-OC adapts it for the large OC20 dataset. References: Gasteiger et al., arXiv:2106.08903 (NeurIPS 2021); GemNet-OC, arXiv:2204.02782 (TMLR 2022).
- **Strengths.** High accuracy. Strong OC20 baseline.
- **Limitations.**
  - O(N·k³) quadruplet cost.
  - Heavy memory use.
  - rMD17 forces only (aspirin 9.5 meV/Å; no energies reported).
  - OC20 variants use direct forces, with the non-conservative issues of §6.1.

### 2.5 SpookyNet
- **Core idea.** An MPNN with explicit charge and spin embeddings and nonlocal self-attention. It also includes empirical physics terms: ZBL-like short-range repulsion, electrostatics and D4-like dispersion. Reference: Unke et al., arXiv:2105.00304 (Nat. Commun. 12, 7273, 2021).
- **Strengths.** One of the first models to handle different total charges and spin multiplicities, with physically correct asymptotics.
- **Limitations.**
  - Global attention is O(N²).
  - Charge and spin enter as global labels, so local charge localisation is learned only implicitly.
  - Expensive relative to local models.

### 2.6 PaiNN
- **Core idea.** Rotationally equivariant message passing with scalar and Cartesian-vector (l=1) features, at linear cost. Reference: Schütt, Unke & Gastegger, arXiv:2102.03150 (ICML 2021).
- **Strengths.** Cheap. Gives direct access to tensorial outputs (dipoles, polarisabilities).
- **Limitations.**
  - Only l ≤ 1. Its many-body expressivity is lower than that of higher-l or higher-body-order models.
  - Published results are on *original* MD17 (aspirin 6.9–7.2 meV / 14.7–16.1 meV/Å, depending on the variant).
  - rMD17 values attributed to PaiNN in some tables (for example the MACE paper, Table 1) are original-MD17 numbers. Do not use them as rMD17 baselines.

### 2.7 ViSNet
- **Core idea.** Vector–scalar interactive message passing with runtime geometry calculation (angles and dihedrals obtained cheaply from vector features). Includes an l=2 spherical-harmonic variant. Reference: Wang et al., arXiv:2210.16518 (Nat. Commun. 15, 313, 2024).
- **Strengths.**
  - Strong on rMD17: aspirin 1.93 meV / 6.59 meV/Å (950-config split), Table R1.
  - Strong on MD17 and MD22.
  - Reports 42.8% lower latency than MACE (L=2) in its own benchmarks.
- **Limitations.**
  - Limited body-order analysis.
  - Mostly benchmarked on single molecules; few materials or universal deployments.
  - Long-range only through an add-on (ViSNet-LSRM).

---

## 3. Equivariant models

### 3.1 NequIP
- **Core idea.** E(3)-equivariant message passing with irreducible-representation features up to l_max. Uses Clebsch–Gordan tensor-product convolutions (e3nn), a gated nonlinearity and conservative forces. Reference: Batzner et al., arXiv:2101.03164 (Nat. Commun. 13, 2453, 2022).
- **Strengths.**
  - Exceptional data efficiency: roughly 1000× less data than DeePMD for water at similar errors (paper §IV).
  - Strong extrapolation on 3BPA.
  - Mature LAMMPS interface.
  - The 2025 framework (Tan et al., arXiv:2504.16068) adds torch.compile/AOTI and a custom Triton TP kernel, giving 4–18× inference speedups over TorchScript.
- **Limitations.**
  - *Cost.* The CG tensor product is O(L^6) for full coupling. At l=3 it dominated runtime: MACE's re-implementation timed NequIP at 103.5 ms latency on 3BPA versus 24.3 ms for MACE L=2 (MACE paper, Table 2, A100).
  - *Parallel scaling.* With T layers, the receptive field is T × r_c. Spatial decomposition then needs either large halos or a communication step per layer. SevenNet (arXiv:2402.03789) solved this with per-layer communication, at a cost to strong scaling.
  - *Inconsistent reported rMD17 values.* arXiv v3 Table II gives aspirin 2.3 / 8.5. The Allegro and MACE papers quote 2.3 / 8.2, and naphthalene E 0.9 vs 0.2. Both versions are listed in Table R1.
- **Fixes.**
  - Strictly local equivariance (Allegro).
  - Cheaper tensor products: Gaunt TP, arXiv:2401.10216; eSCN SO(2) convolution, arXiv:2302.03655; cuEquivariance kernels.
  - Many-body messages (MACE) to reduce depth.

### 3.2 Allegro
- **Core idea.** A strictly local equivariant model. Features live on *pairs* (edges) within a single cutoff sphere. Equivariant tensor products with neighbour spherical harmonics are iterated without message passing between atoms, so the receptive field equals r_c. Reference: Musaelian et al., arXiv:2204.05249 (Nat. Commun. 14, 579, 2023).
- **Strengths.**
  - Best-in-class parallel scaling. Ag with 100.6 M atoms on 128 A100s ran at 1.54 ns/day.
  - Single-A100 cost of 0.289 µs/(atom·step) for 1.02 M atoms of Ag, and 0.552 µs/(atom·step) for 421,824 atoms of Li₃PO₄ (Table IV of the paper; see Table S1).
  - rMD17 accuracy is on par with MACE (aspirin 2.3 / 7.3).
- **Limitations.**
  - Strict locality, so no semilocal or long-range physics.
  - At small sizes the GPU is underused: 27.8 µs/(atom·step) for 192 atoms.
  - Extrapolation on 3BPA at 1200 K is about 30% worse than MACE (MACE paper Table 2: force RMSE 83.0 vs 62.0 meV/Å).
  - Allegro-OAM-L on Matbench Discovery has κSRME 0.319, worse than NequIP-OAM-L (0.166) with a similar number of parameters (leaderboard snapshot).
- **Fixes.**
  - Add explicit long-range modules (LES is reported to have been integrated with Allegro in the JCTC version of arXiv:2507.14302).
  - Larger cutoffs with efficient neighbour lists.

### 3.3 MACE (and BOTNet)
- **Core idea.** ACE-style many-body symmetric products of equivariant features are formed *inside* each message-passing layer, typically with correlation order ν=3 (4-body messages). Two layers are usually enough, with a receptive field of about 2 × r_c. BOTNet (*The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials*, arXiv:2205.06643) gives the unifying framework and a body-ordered NequIP-like model. MACE reference: Batatia et al., arXiv:2206.07697 (NeurIPS 2022).
- **Strengths.**
  - SOTA accuracy on rMD17 (aspirin 2.2 / 6.6).
  - Best 3BPA extrapolation in 2022: 1200 K F RMSE 62.0 vs 76–86 for NequIP or BOTNet and 83 for Allegro.
  - Only two layers, which is faster than NequIP: MACE L=0 at 10.5 ms vs NequIP-style at 103.5 ms latency, A100, 3BPA.
  - Strong low-data results: N=50 aspirin 17.0 / 43.9 vs NequIP 19.5 / 52.0.
  - Large ecosystem:
    - MACE-MP-0, arXiv:2401.00096
    - MACE-OFF, arXiv:2312.15211
    - MACE-OMAT / MPA
    - MACE-MH-1, arXiv:2510.25380 (multi-head replay across PBE, ωB97M-V, r2SCAN)
    - MACE-POLAR-1, arXiv:2602.19411
    - MACELES (LES long-range)
    - MACE-Field, arXiv:2508.17870 (electric response)
    - Magnetic MACE, arXiv:2604.08143
- **Limitations.**
  - *Softening.* MACE-MP-0 systematically under-predicts PES curvature; see §6.2 (Deng et al., arXiv:2405.07105).
  - *Memory.* GRACE authors report that on an 80 GB A100 in LAMMPS, a MACE model fitted only up to 2016 atoms (diamond), against 12,672 for GRACE. They also report 400 vs 200 µs/atom per step (arXiv:2311.16326, §V/App. D). These are early-implementation numbers that cuEquivariance has since improved.
  - *Locality.* The receptive field is 10–12 Å (two layers × 5–6 Å). Long-range electrostatics needs LES or POLAR.
  - *Torsional sampling.* A 2026 preprint (Bi et al., arXiv:2606.15892) reports that baseline MACE over-populates barrier regions in aspirin torsion sampling. This is a single, recent, unreplicated study.
- **Fixes.** Multi-head fine-tuning with replay (MACE-MH-1), LES/POLAR long-range modules, ZBL repulsion plus high-energy data, and cuEquivariance kernels.

### 3.4 Equiformer / EquiformerV2 / EquiformerV3, eSCN
- **Core idea.**
  - Equiformer (arXiv:2206.11990, ICLR 2023) is an equivariant graph-attention transformer with irreps features.
  - eSCN (arXiv:2302.03655, ICML 2023) rotates each edge to align with z. This reduces the SO(3) tensor product to SO(2) convolutions, lowering the cost from O(L^6) to O(L^3), so L=6–8 becomes feasible.
  - EquiformerV2 (arXiv:2306.12059, ICLR 2024) combines eSCN convolutions with S² activations, scaling to high L for OC20.
  - EquiformerV3 (arXiv:2604.09130, 2026) adds:
    - a 1.75× faster implementation
    - merged equivariant layer norm
    - SwiGLU-S² activations, which add many-body expressivity
    - smooth-cutoff attention, which makes the PES smooth enough for conservative MD and higher derivatives
- **Strengths.** Highest-capacity architecture family. EquiformerV3+DeNS-OAM reaches Matbench Discovery CPS 0.902 (F1 0.931). The compliant variant EquiformerV3+DeNS-MP reaches 0.830.
- **Limitations.**
  - eqV2 (OMat24 models) was trained with direct forces and was *non-smooth*: κSRME 1.77 for eqV2-M (Orb-v3 paper Table 1), against 0.17 for conservative eSEN.
  - Very expensive. eqV2-M ran out of memory for 1000 atoms on an H200 in the Orb-v3 benchmark.
  - S² grid sampling only approximately preserves equivariance; V3 addresses this.
- **Fixes.** Conservative fine-tuning (eSEN recipe), smooth cutoffs and envelopes (V3), and Gaunt or SO(2) tensor-product kernels.

### 3.5 eSEN and UMA (Meta FAIR)
- **eSEN** (Fu et al., arXiv:2502.12147).
  - Core idea: a smooth, expressive equivariant network designed for physical-property prediction. Choices are made for PES smoothness: conservative forces, no discretisation of the S² grid that breaks smoothness, and bounded max-neighbour effects.
  - Training recipe: pre-train with direct forces, then fine-tune conservatively.
  - Result: eSEN-30M-OAM has F1 0.925 and κSRME 0.170. eSEN-30M-MP was the top compliant model at the time (F1 0.831).
- **UMA** (Wood et al., arXiv:2506.23971; v2 updated March 2026).
  - Core idea: one multi-task model trained on OMat24, OMol25, OC20, ODAC and OMC.
  - Uses a Mixture of Linear Experts (MoLE). Total parameters are large, but active parameters are small: UMA-S has 150M total and 6M active. Experts can be pre-merged for a fixed system, which gives inference at the cost of the active parameters only.
- **Strengths.** Cross-domain transfer and a single API. UMA-S runs 1000 atoms at 16 steps/s, about 1.4 ns/day, and fits more than 100k atoms on an 80 GB H100 (paper Table 3).
- **Limitations.**
  - UMA-L (700M active) runs at only about 1.6 inferences/s for 1000 atoms and fits roughly 1k+ atoms.
  - Mixed DFT fidelities are handled through task heads. Cross-fidelity consistency is not guaranteed.
  - Long-range interactions are not explicit; a 6 Å cutoff with about 50 neighbours.
  - UMA results were not on the Matbench Discovery top-20 in the 2026-10 snapshot. Their status there should be checked before citing them.
  - Licence restrictions.
- **Fixes.** Explicit electrostatics (LES has been reported for UMA in the JCTC version of arXiv:2507.14302) and distillation into fast student models.

### 3.6 GRACE (graph ACE)
- **Core idea.** ACE generalised from star graphs to arbitrary graph topologies (tree-like cluster basis functions). This gives a complete basis for *semilocal* interactions. MPNNs such as MACE appear as a particular subset. References: Bochkarev et al., arXiv:2311.16326 (PRX 14, 021036, 2024); foundation models in arXiv:2508.17936 (2025).
- **Strengths.**
  - Accuracy–efficiency Pareto front among foundation models (GRACE-2L-OAM, GRACE-1L).
  - Lower memory.
  - Distillation and fine-tuning recipes that avoid catastrophic forgetting.
  - GRACE-3L-OAM-L reaches CPS 0.900 (κSRME 0.121).
- **Limitations.**
  - Large basis.
  - The 1-layer models are local only.
  - GRACE-2L-OAM has κSRME 0.294, so the smaller models are still not smooth enough for κ.

### 3.7 SevenNet (incl. SevenNet-MF, SevenNet-Omni)
- **Core idea.** A NequIP-based architecture with a spatial-decomposition parallel algorithm for message passing in LAMMPS (arXiv:2402.03789). SevenNet-MF (arXiv:2409.07947) adds multi-fidelity training via fidelity-dependent embeddings, for example PBE+U mixed with r2SCAN.
- **Strengths.** Scalable multi-GPU MD for message passing. A principled multi-fidelity approach.
- **Limitations.**
  - The tensor-product cost is inherited from NequIP. SevenNet-MF-ompa ran at about 3.5 steps/s for 1000 atoms on an H200 (Orb-v3 Table 1).
  - κSRME 0.317 (MF-ompa).
- **Recent additions.** SevenNet-Omni-i12 (CPS 0.873, trained on "COSMOS" data) and SevenNet-Polar (rank-2 Born-charge tensors; alphaXiv 2607.14827, *ID not verified on arXiv*).

### 3.8 So3krates
- **Core idea.** A sparse equivariant transformer. Atomic features are separated from "spherical harmonic coordinates" (SPHCs), and attention runs in SPHC space, allowing non-local geometric attention at low parameter count. References: Frank et al., arXiv:2205.14276 (NeurIPS 2022); *A Euclidean transformer for fast and stable MLFFs*, arXiv:2309.15126 (Nat. Commun. 15, 6539, 2024).
- **Strengths.** Small and fast. Emphasis on MD stability. Long-range options via Euclidean fast attention (§4).
- **Limitations.**
  - Reported on *original* MD17 only (aspirin 6.0 meV / 10.2 meV/Å), with no rMD17 numbers in the primary sources examined.
  - Accuracy below MACE or NequIP at equal data.

### 3.9 Cartesian equivariant models (TensorNet, CACE, HotPP, ICTP, TACE, LATTE)
- **Core ideas.**
  - **TensorNet** (Simeon & De Fabritiis, arXiv:2306.06482, NeurIPS 2023) uses rank-2 Cartesian tensor embeddings, decomposed into scalar, vector and traceless-symmetric parts and mixed by matrix products. 770k parameters reach about NequIP accuracy on rMD17 (aspirin 2.4 / 8.9).
  - **CACE** (Cheng, arXiv:2402.07472, npj 2024) is a Cartesian atomic density expansion with polynomially independent features and low-dimensional element embeddings. It is the backbone for the first LES demonstrations.
  - **HotPP** (Wang et al., arXiv:2402.15286) passes arbitrary-order Cartesian tensors.
  - **ICTP** (Zaverkin et al., arXiv:2405.14253, NeurIPS 2024) uses higher-rank *irreducible* Cartesian tensors and outperforms MACE in the N=50 rMD17 regime (aspirin 14.8 / 40.2 vs 17.0 / 43.9).
  - **TACE** (arXiv:2509.14961) is a unified irreducible-Cartesian ACE with universal embeddings for fidelity, charge, magnetic moment and field. TACE-OAM-L reaches CPS 0.889.
  - **LATTE** (arXiv:2405.08137) is a Cartesian-contraction descriptor.
- **Strengths.** No CG coefficients. Simple operations that suit GPU matmuls. Cheaper at low rank.
- **Limitations.**
  - Reducible Cartesian tensors carry redundant components, so they must be projected to irreducible parts to avoid parameter waste. Cost grows as 3^rank, which is unfavourable at high rank compared with the O(L^3) cost of SO(2) methods.
  - TensorNet is limited to rank 2. With one layer its force errors fall outside the range of spherical models (TensorNet paper §4).
  - "CarNet / Cartesian MACE": I could not verify a paper by this exact name. A 2026 preprint (arXiv:2606.29584, geometric algebra plus Cartesian tensors) is related.

### 3.10 HDNNP-style universal graph models: M3GNet, CHGNet, MatterSim
- **M3GNet** (Chen & Ong, arXiv:2202.02450, Nat. Comput. Sci. 2022): an invariant graph model with three-body interactions, trained on MPF. It was the first widely used universal potential.
- **CHGNet** (Deng et al., arXiv:2302.14231, Nat. Mach. Intell. 2023): M3GNet-like, adding magnetic-moment (charge-state proxy) regularisation, trained on MPtrj.
- **MatterSim** (Yang et al., arXiv:2405.04967): M3GNet/Graphormer-style models trained on actively sampled data spanning 0–5000 K and 0–1000 GPa.
- **Limitations.**
  - Strong PES softening for M3GNet, CHGNet and MACE-MP-0 (Deng et al. 2024).
  - Invariant 3-body features.
  - κSRME of MatterSim-v1-5M is 0.574 (Orb-v3 Table 1).

### 3.11 DeePMD family (DeepPot-SE, DeePMD-kit, DPA-1/2/3/4)
- **Core ideas.**
  - DeePMD (Zhang et al., arXiv:1707.09571, PRL 2018) and DeepPot-SE use smooth local frames and embedding-net descriptors.
  - DeePMD-kit v2 (arXiv:2304.09409) is the software stack.
  - DPA-1 (arXiv:2208.08236) adds attention-based type embedding for pre-training.
  - DPA-2 (arXiv:2312.15492) is a multi-task "large atomic model".
  - DPA-3 (arXiv:2506.01686) uses a line-graph-series GNN for the "era of large atomistic models" (OpenLAM).
  - DPA4 (arXiv:2606.02419, 2026) uses SE(3)-equivariant "EMFA SO(2) convolution" in an edge-local frame, a Wigner bilinear nonlinearity, and purely conservative training.
- **Strengths.**
  - HPC record holder:
    - 127 M-atom Cu at 2.7–8.1 × 10⁻¹⁰ s/(step·atom) on full Summit (SC20 Gordon Bell; Jia et al., arXiv:2005.00223). This is about 7.4–22 V100·µs per atom-step.
    - 10 billion atoms (Guo et al., arXiv:2201.01446).
  - Model compression (tabulated embedding nets).
  - DPA4-Pro is the best *compliant* Matbench model: CPS 0.840, κSRME 0.211. Its paper claims several-fold to order-of-magnitude higher throughput than equally accurate models.
- **Limitations.**
  - Classic DeepPot-SE needs about 10³× more data than NequIP for water (NequIP paper).
  - The descriptor is 3-body-like (embedding of the s(r)·r̂ matrix), so it is less data-efficient.
  - DPA-3 (OpenLAM) has κSRME 0.687, so its PES is not smooth enough for phonons (Orb-v3 Table 1).
  - Magnetic DeePMD (DeePSPIN) and its non-magnetic counterpart differ unpredictably. On Fe–Cr–C, the non-magnetic model better reproduced viscosity and melting (arXiv:2507.18935).

### 3.12 Unconstrained / non-equivariant and local-frame models: PET, PET-MAD, Orb, EScAIP, AlphaNet, Facet
- **PET** (Pozdnyakov & Ceriotti, *Smooth, exact rotational symmetrization for deep learning on point clouds*, arXiv:2305.19302, NeurIPS 2023).
  - A point-edge transformer with no built-in rotational equivariance.
  - Symmetry is learned with rotational augmentation. Exact invariance can be restored by an ECSE symmetrisation protocol.
  - **PET-MAD** (arXiv:2503.14118) is a lightweight universal model trained on the MAD dataset (about 100k structures, consistent PBEsol settings).
  - **PET-OAM-XL** (730M parameters) reaches CPS 0.898.
  - *Pushing the limits of unconstrained MLIPs* (arXiv:2601.16195, 2026) studies how symmetry-breaking behaves as models scale.
- **Orb (v1/v2)** (arXiv:2410.22570) and **Orb-v3** (arXiv:2504.06231).
  - Graph-network-simulator-style models with no equivariance constraints.
  - Orb-v3 adds conservative variants, "equigrad" rotational-gradient regularisation and neighbour-limit variants (-20 / -inf).
  - Orb-v3-direct-20 runs at 216.5 forward passes/s for 1k atoms on an H200, against 28.1 for conservative-inf.
- **EScAIP** (arXiv:2410.24169, *The Importance of Being Scalable*).
  - An attention-based model that drops explicit tensor products in favour of scalable attention.
  - Direct forces. Argues that scaling (data, parameters, compute) beats hard-coded symmetry.
- **AlphaNet** (arXiv:2501.07155): equivariant *local frames* with learnable geometric transitions.
- **Facet** (arXiv:2509.08418): splines replace radial MLPs, and spherical-grid projection with MLPs replaces iterative tensor products, for efficient equivariant training on MPtrj.
- **Strengths.** Speed, simplicity and scaling with data. Frames and attention are cheaper than CG products.
- **Limitations.**
  - *Non-conservative forces.* Bigi, Langer & Ceriotti (*The dark side of the forces*, arXiv:2412.11569, ICML 2025) show unstable NVE dynamics and ill-defined geometry optimisation.
    - Drift equals spurious heating of about 7,000 billion K/s for a custom PET-NC model, and about 10× larger for ORB.
    - Strong thermostatting hides the drift but reduces diffusion by roughly 1.5–5×, which negates the speed advantage.
  - *Approximate symmetry.* Violations are small but measurable. They matter for phonon degeneracies, forces at high-symmetry points and long MD.
  - *Local-frame discontinuities.* Frames are ill-defined at degenerate geometries, such as collinear neighbours, unless they are smoothed.
- **Fixes.**
  - Conservative fine-tuning, or multiple-time-step integration with occasional conservative force evaluations. Bigi et al. show an MTS scheme with M=8 nearly recovers correct behaviour.
  - Symmetrisation at inference (ECSE).
  - Equigrad regularisation (Orb-v3).

### 3.13 Additional 2025–2026 architectures worth noting
- **Nequix** (arXiv:2508.16067): a 700k-parameter simplified NequIP with equivariant RMS layer norm and the Muon optimiser. Trained in 100 A100-hours, it was ranked third on Matbench Discovery plus MDR phonons at the time of the paper.
- **NequIP-OAM-L/XL and Allegro-OAM-L** (Matbench, 2025) show the 2025 NequIP framework at foundation scale.
- **GotenNet** (ICLR 2025; Aykent & Xia): efficient equivariant attention. It holds the lowest rMD17 numbers found here (aspirin 1.55 / 5.66, Table R1).
- **QuinNet** (NeurIPS 2023): adds quintuple (five-body) interactions.
- **MARA** (arXiv:2602.02671): continuous SE(3) attention over MACE, reporting 3–10% lower force errors than MACE.
- **Leaderboard models without paper IDs found in this review:** TECE-OAM-RRA-1.0, Prophet-OAME-MBD, EquFlash/EquFlashV2, MatRIS-10M-OAM.

---

## 4. Long-range interactions, electrostatics, charge and fields

| Approach | Mechanism | Pros | Documented limits |
|---|---|---|---|
| 4G-HDNNP (arXiv:2009.06484) / CENT (PRB 92, 045131, 2015) | NN electronegativities, global QEq, Ewald | Non-local charge transfer, correct physics for ionic systems and clusters | Needs reference charges; global solve; QEq's known over-delocalisation in dissociation limits |
| PhysNet / SpookyNet | Predicted charges plus Coulomb, D3/D4, charge and spin embeddings | Correct asymptotics; total charge and spin | Charges fitted to dipoles; no self-consistent polarisation; O(N²) attention (SpookyNet) |
| Ewald message passing (Kosmala et al., arXiv:2303.04791, ICML 2023) | Message passing in reciprocal space with learned frequency filters | Plug-in for any GNN; systematic gains on OC20 and OE62 energies | Periodic systems need an Ewald sum; cost grows with the number of k-points; not physically interpretable charges |
| LES (Cheng, arXiv:2408.15165, npj Comput. Mater. 2025; King et al., arXiv:2412.15455, Nat. Commun. 2025; library arXiv:2507.14302, JCTC) | Latent per-atom charges from local features, plus Ewald sum, trained only on E/F | Needs no charge labels; about 2× cost of short-range models; plugs into MACE, NequIP, CACE, CHGNet (Allegro and UMA in the JCTC version); fixes water and dimer artefacts; extended to polarisation and BECs (arXiv:2504.05169) and polarisable multipoles (arXiv:2605.05746) | Charges are identifiable only up to gauge or scale without constraints; no explicit polarisation in the base form; CC BY-NC licence |
| MACE-POLAR-1 (arXiv:2602.19411, 2026) | MACE plus explicit long-range electrostatics with non-self-consistent induction; trained on OMol25 | Sub-kcal/mol X23-DMC lattice energies; about 4× better protein–ligand interactions than short-range models; ions and redox | Reported to omit long-range dispersion; molecular data only (OMol25 has no crystals), so solid-state use is unvalidated (noted by arXiv:2605.28905) |
| AIMNet2 (Anstine, Zubatyuk & Isayev, Chem. Sci. 2025, doi:10.1039/d4sc08572h) | ML short-range plus physics long-range Coulomb and D3 dispersion; neutral and charged; 14 elements | Beats GFN2-xTB and approaches DFT on organics and element-organics | Limited element coverage; non-periodic focus |
| Euclidean Fast Attention (Frank et al., arXiv:2412.08541) | Linear-cost global attention with Euclidean rotary positional encodings | Global information at O(N) | Not a physical electrostatics model; long-range asymptotics are learned only implicitly |
| Field-dependent models: Allegro-pol (*Unified differentiable learning of electric response*, arXiv:2403.17207), MACE-Field (arXiv:2508.17870), LES-BEC (arXiv:2504.05169), SevenNet-Polar | Electric enthalpy F(R, E) with derivatives giving P, Z*, α; or latent charges giving BECs | IR and Raman spectra, ferroelectric switching, MD under a field | Polarisation is multivalued in periodic systems (Berry-phase branch issue); training data on Z* and α is scarce (MP-Dielectric) |
| Dispersion add-ons (D3/D4) | Fixed C6 terms beyond the cutoff | Cheap, standard (MACE-MP-0+D3, MACE-MH-1-D3) | Double counting when the training functional already includes dispersion; no many-body dispersion. The current Matbench #1 (Prophet-OAME-MBD) suggests an MBD term |

**Consensus evidence that locality fails.** Cheng (arXiv:2408.15165) shows that short-range MLIPs, *even with message passing*, give unphysical results for bulk water interfaces and molecular dimers. MACE-POLAR-1 shows about a 4× error reduction for protein–ligand interactions. Kosmala et al. show consistent OC20 gains. There is still no foundation model for *materials* with validated long-range electrostatics plus charge equilibration across metals and insulators.

---

## 5. Foundation / universal models, datasets and benchmarks

### 5.1 Datasets
| Dataset | Domain / size | Level of theory | Reference |
|---|---|---|---|
| MPtrj | About 1.58M frames from Materials Project relaxations | PBE/PBE+U | CHGNet paper, arXiv:2302.14231 |
| Alexandria (sAlex subset used with OMat24) | Millions of inorganic structures | PBE (also PBEsol, SCAN subsets) | Schmidt et al. (*ID not verified*) |
| OMat24 | About 110M non-equilibrium inorganic structures (rattled, AIMD, high-T) | PBE/PBE+U (PBE54 pseudopotentials, so not fully compatible with MP/WBM) | arXiv:2410.12771 |
| OMol25 | About 100M molecular calculations, 83 elements, charges and spins | ωB97M-V/def2-TZVPD | arXiv:2505.08762 |
| OC20 / OC22 | Catalysis adsorbate–slab systems (OC20 about 260M DFT single points; OC22 oxides) | RPBE (OC20), PBE+U (OC22) | Chanussot et al. 2021; Tran et al. 2023 (*IDs not verified here*) |
| SPICE (v1/v2) | Drug-like molecules, peptides, ions | ωB97M-D3(BJ)/def2-TZVPPD | arXiv:2209.10702 |
| MatPES | About 400k structures sampled from MD for PES coverage | PBE and r2SCAN | arXiv:2503.04070 |
| MAD | About 100k consistent, diverse structures | PBEsol | PET-MAD, arXiv:2503.14118 |
| rMD17 / MD22 / 3BPA | Single molecules | PBE/def2-SVP (rMD17); PBE+MBD (MD22); ωB97X/6-31G(d) (3BPA) | arXiv:2007.09593; Chmiela et al. 2023; Kovács et al. 2021 |

### 5.2 Matbench Discovery state (snapshot fetched 2026-10-13)

Discovery metric shown on unique prototypes. The site lists 43 of 53 eligible models.

| # | Model | CPS | F1 | κSRME ↓ | RMSD | Params | Training data |
|---|---|---|---|---|---|---|---|
| 1 | Prophet-OAME-MBD | 0.912 | 0.928 | 0.065 | 0.059 | 62.3M | MPtrj+OMat24+sAlex+ELEMENTA |
| 2 | TECE-OAM-RRA-1.0 | 0.908 | 0.929 | 0.093 | 0.058 | 222M | MPtrj+OMat24+sAlex |
| 3 | EquFlashV2 | 0.907 | 0.929 | 0.094 | 0.058 | 44.9M | same |
| 4 | EquiformerV3+DeNS-OAM | 0.902 | 0.931 | 0.118 | 0.059 | 30.3M | same |
| 5 | GRACE-3L-OAM-L | 0.900 | 0.925 | 0.121 | 0.058 | 42.1M | same |
| 6 | PET-OAM-XL | 0.898 | 0.924 | 0.119 | 0.060 | 730M | same |
| 7 | TACE-OAM-L | 0.889 | 0.910 | 0.126 | 0.061 | 82.9M | same |
| 8 | eSEN-30M-OAM | 0.888 | 0.925 | 0.170 | 0.061 | 30.2M | same |
| 10 | NequIP-OAM-XL | 0.886 | 0.906 | 0.125 | 0.063 | 32.1M | same |
| 15 | ORB v3 | 0.860 | 0.905 | 0.210 | 0.075 | 25.5M | MPtrj+Alex+OMat24 |
| 16 | **DPA-4.0.1-Pro-MPtrj** (best compliant) | 0.840 | 0.857 | 0.211 | 0.069 | 22.8M | MPtrj only |
| 17 | Allegro-OAM-L | 0.840 | 0.895 | 0.319 | 0.065 | 9.7M | OAM |
| 19 | EquiformerV3+DeNS-MP (compliant) | 0.830 | 0.863 | 0.275 | 0.070 | 30.3M | MPtrj only |

**Observations.**
- F1 is saturating at about 0.93.
- The ranking is now driven by κSRME, which requires smooth second and third derivatives (Póta et al., *Thermal Conductivity Predictions with Foundation Atomistic Models*, arXiv:2408.00755).
- Phonon accuracy is benchmarked separately on the MDR database (Loew et al., *Universal MLIPs are Ready for Phonons*, arXiv:2412.16551).
- MLIP Arena (arXiv:2509.20630, NeurIPS 2025) adds physics-awareness tasks: diatomics, EOS, stability under extreme conditions, combustion and reactivity. It exposes failures that error metrics against DFT hide.

### 5.3 Fine-tuning, multi-fidelity, distillation
- **Fine-tuning.**
  - Deng et al. (arXiv:2405.07105) show that softening can be corrected by fine-tuning on a *single* additional data point, because the error is systematic.
  - Multi-head replay fine-tuning (MACE-MH-1, arXiv:2510.25380) avoids catastrophic forgetting across PBE, r2SCAN and ωB97M-V.
- **Multi-fidelity.**
  - SevenNet-MF (arXiv:2409.07947) uses fidelity embeddings.
  - TACE uses "computational level" embeddings.
  - UMA uses task heads.
  - Open issue: there is no guarantee that energy differences are consistent across fidelities, or that a high-level head is physically better outside the data-rich low-level domain.
- **Distillation.** Teacher-to-student distillation across architectures and domains (Gardner et al., arXiv:2506.10956; GRACE, arXiv:2508.17936) produces fast, specialised students, for example ACE or PACE students of MACE teachers.

---

## 6. Cross-cutting issues

### 6.1 Smoothness and energy conservation
- **Direct-force models are non-conservative.** The Jacobian asymmetry metric is λ=0 for conservative models; Bigi et al. measured λ≈0.004 for PET-NC on water.
  - Consequences: energy drift, failed or ill-defined geometry optimisation (FIRE oscillating below 0.1 eV/Å for ORB), and temperature errors.
  - Reference: arXiv:2412.11569.
- **Non-smooth architectural choices also hurt.** Max-neighbour truncation, discrete S² grids and attention without smooth cutoffs give κSRME around 1.7 even when F1 is high (eqV2-M F1 0.917 vs κSRME 1.771).
  - Fixes: conservative fine-tuning (eSEN), smooth-cutoff attention (EquiformerV3), and MTS with occasional conservative forces.

### 6.2 PES softening in universal models
- Deng et al. (arXiv:2405.07105) found consistent energy and force *under*-prediction for M3GNet, CHGNet and MACE-MP-0. It shows up in surfaces, defects, solid-solution energies, phonons, migration barriers and high-energy states.
- The root cause is that pre-training data (relaxation trajectories) concentrate near minima, so curvature is under-predicted.
- Fixes: off-equilibrium data (OMat24 rattled and AIMD data, MatPES MD sampling), fine-tuning, and curvature or Hessian losses. Softening is reduced but not eliminated in OMat24-trained models (κSRME still 0.07–0.2).

### 6.3 Out-of-distribution failure and extrapolation
- **3BPA.** All models degrade by about 7× in force RMSE from 300 K to 1200 K (MACE 8.8 → 62.0 meV/Å).
- **High pressure and temperature.** MatterSim deliberately samples 0–1000 GPa and 0–5000 K. Most other universal models are trained near ambient conditions and fail under compression (MLIP Arena EOS and stability tasks).
- **Short range.** Without a ZBL or repulsive prior, models can collapse in high-energy collisions (radiation damage, shocks). ZBL is built into SpookyNet and is an option in MACE and NequIP.
- **Surfaces and defects** follow the softening pattern above. Adsorption on surfaces from foundation models varies widely: on the MACE-MH-1 model card, S24 adsorption MAE is 0.095 eV for MACE-MH-1-OMAT-D3, 0.174 eV for ORB-v3-D3 and 0.329 eV for UMA-S-1.1-OMAT-D3 (vendor-reported).

### 6.4 Uncertainty quantification and active learning
- **Ensembles.** The gold standard, but they cost M× and are often over-confident out of distribution.
- **LLPR / last-layer prediction rigidity.** Bigi et al., arXiv:2403.02251: cheap, needs one model, and calibratable.
- **GP variance (GAP), D-optimality (MTP, ACE).** Cheap extrapolation grades for active learning.
- **MC dropout.** Generally poorly calibrated for MLIPs.
- **Evidential regression.** Equivariant evidential deep learning, arXiv:2602.10419 (2026).
- **Documented gap.** Most UQ methods are calibrated in distribution only. No foundation model ships with validated OOD-calibrated uncertainty.
- **Active learning.** Effective for local models (MTP MaxVol, ACE). For foundation models it is mostly replaced by large-scale sampling (MatterSim active sampling, OMat24).

### 6.5 Data efficiency
- Equivariance and many-body messages improve learning-curve slopes:
  - NequIP vs DeepMD: about 1000× less data for water.
  - MACE, ICTP and NequIP at N=50 on rMD17 (Table R1b).
- Foundation-model fine-tuning is now the most data-efficient route in practice: the single-point softening correction, and few-shot MACE-MP-0 fine-tuning.

### 6.6 Body order and completeness
- 2- and 3-body descriptors (ACSF, SOAP power spectrum, distance-only MPNNs) are incomplete. Pozdnyakov et al. (arXiv:2001.11696, PRL 2020) give degenerate environment pairs.
- Message passing with only 2-body messages does not fully fix this. Equivariant features or explicit higher body order are needed (BOTNet design-space analysis, arXiv:2205.06643; MACE).
- ICTP cites results that rank-L tensors are needed to resolve L-fold symmetric environments, and higher correlation order is needed for degeneracies at lower body order.
- GRACE gives a complete *semilocal* basis.

### 6.7 Receptive field, locality and parallel scaling
- With T message-passing layers the receptive field is T·r_c. Spatial decomposition then needs either ghost atoms within T·r_c, with a halo volume that grows roughly as (T·r_c)³, or communication at every layer (SevenNet).
- Allegro removes this by design (receptive field = r_c) and scales to 10⁸ atoms.
- Strong scaling of message-passing models is limited. The NequIP 2025 framework reached 205 steps/s for a 23,558-atom protein using 256 Perlmutter nodes (1024 A100s). That is about 4.7k atom-steps/s per GPU, far below single-GPU throughput for large systems.
- The trade-off between locality, which helps parallelism, and long range, which needs Ewald or LES, is an open design axis.

### 6.8 Tensor-product cost
- A full Clebsch–Gordan tensor product between irreps up to L costs O(L⁶).
- eSCN SO(2) convolution (arXiv:2302.03655) reduces this to O(L³) by edge alignment.
- The Gaunt tensor product (Luo et al., arXiv:2401.10216, ICLR 2024) reduces it to O(L³) via a 2D Fourier basis.
- Cartesian methods avoid CG coefficients but cost 3^rank.
- ICTP shows irreducible Cartesian tensors can be faster than spherical ones for small L and ν.
- Practical kernels (cuEquivariance, OpenEquivariance, the NequIP Triton TP) give 4–18× speedups.

### 6.9 Magnetism and spin
- Non-collinear magnetic ACE for Fe (arXiv:2305.15137, npj 2024).
- mMTP.
- Spin-lattice NN (Chapman & Ma, arXiv:2205.04732).
- Magnetic MACE (arXiv:2604.08143, 2026), reporting 3–5× lower force and stress errors on FeAl and CrN than previous magnetic MLIPs, and finding the non-collinear ground state of Mn₃Pt.
- Foundation models treat magnetism implicitly at best (CHGNet's magnetic-moment proxy). Spin-polarised training can even *hurt* dynamic properties (Fe–Cr–C DeePMD, arXiv:2507.18935).

### 6.10 Excited states, nuclear quantum effects, reactive chemistry
- **Excited states.** Multi-state NNs (SchNarc and SPaiNN families) exist. Non-adiabatic couplings and conical intersections remain hard. No foundation-scale excited-state MLIP was identified.
- **NQEs.** MLIPs make PIMD affordable (water, high-pressure hydrogen). The bottleneck is accuracy at the reference level, CCSD(T) or QMC rather than DFT, plus the extra cost of the ring-polymer beads.
- **Reactive chemistry.**
  - OMol25 and Transition1x/RGD1-type data (MACE-MH-1 has an RGD1 head).
  - MLIP Arena includes combustion and reactivity tasks.
  - Bond breaking with spin and charge changes requires charge and spin embeddings (SpookyNet, UMA-OMol, AIMNet2).

### 6.11 Multi-element scaling
- Descriptor models scale badly with element count:
  - SOAP grows as the square of the number of elements.
  - ACE and MTP grow combinatorially.
- Learned embeddings (SchNet onwards; CACE low-dimensional embeddings; DPA type embeddings) make 89–97 elements feasible (see the 97-element uMLIP preprint, arXiv:2603.03223).
- Chemical channel compression trades some accuracy for this. Rare elements remain data-starved.

### 6.12 Stress and virial
- Conservative models obtain stress from strain derivatives.
- Direct-stress heads share the non-conservative issues above.
- Stress is often under-weighted in training, which causes EOS and elastic-constant errors.
- Orb-v3 and the MDR benchmarks show bulk and shear moduli errors of roughly 3–20 GPa across models (Orb-v3 §5 table).

### 6.13 Transferability across fidelity
- PBE (MPtrj, OMat24) and r2SCAN (MatPES) are inconsistent: GGA errors are systematic.
- OMat24 uses PBE54 pseudopotentials, which differ from the MP settings. The Orb-v3 authors therefore exclude OMat-only models from WBM discovery scoring.
- CCSD(T)-level MLIPs exist only for small systems (NequIP Table III; sGDML CCSD(T)).
- Δ-learning and multi-head approaches (MACE-MH-1, SevenNet-MF, UMA) are the current answer. Their cross-fidelity consistency has not been systematically validated.

### 6.14 Hessians, phonons, κ
- κSRME (arXiv:2408.00755) and MDR phonons (arXiv:2412.16551) are the de facto smoothness tests.
- Conservative, smooth models dominate. Direct-force models are clearly worse: Orb-v3-direct-inf has κSRME 0.348 against 0.210 for conservative-inf.

### 6.15 MD stability benchmarks
- *Forces are not Enough* (arXiv:2210.07237) showed that force MAE correlates poorly with MD stability and observables. Models with similar MAE differ widely in stable simulation time on MD17 and water.
- MLIP Arena and ml-peg extend this to foundation models.

### 6.16 Interpretability and physical priors
- Body-ordered models (ACE, MACE, BOTNet) allow decomposition into many-body terms.
- LES latent charges correlate with physical charges.
- Physical priors are cheap and broadly beneficial: ZBL repulsion, D3/D4 dispersion, Ewald electrostatics, smooth cutoff envelopes. The Allegro paper notes that the envelope exponent matters for extrapolation.

---

## 7. Consolidated open-problems table

| Problem | Methods most affected | Evidence | Candidate fixes |
|---|---|---|---|
| Non-conservative forces: energy drift, failed relaxations | Orb (direct), eqV2 (direct), EScAIP, PET-NC, GemNet-OC direct | Bigi et al. arXiv:2412.11569 (drift about 7×10¹² K/s for PET-NC, 10× more for ORB); Orb-v3 direct vs conservative κSRME 0.348 vs 0.210 | Conservative fine-tuning (eSEN), MTS with conservative correction, Jacobian-symmetry loss |
| Non-smooth PES (κ, phonons) | eqV2 (κSRME 1.77), Orb-v2 (1.73), DPA-3 (0.69), MatterSim (0.57) | Orb-v3 Table 1; κSRME arXiv:2408.00755 | Smooth cutoffs and attention (EquiformerV3), no max-neighbour truncation, Hessian-aware training |
| Systematic PES softening | M3GNet, CHGNet, MACE-MP-0; reduced but present in OMat24 models | Deng et al. arXiv:2405.07105 | Off-equilibrium data (OMat24, MatPES), single-point fine-tuning, curvature losses |
| Missing long-range electrostatics and charge transfer | All local models (Allegro, MACE, NequIP, ACE, MTP, SNAP, most foundation models) | Cheng arXiv:2408.15165 (water, dimers); MACE-POLAR-1 about 4× on protein–ligand; Ewald-MP arXiv:2303.04791 | LES, 4G-HDNNP QEq, MACE-POLAR, Ewald MP, AIMNet2-style physics terms |
| No charge or spin state input | Most materials foundation models | SpookyNet, AIMNet2, UMA-OMol needed explicit embeddings | Charge and spin embeddings plus LES charge-neutrality constraints (arXiv:2412.15455) |
| Field response (P, Z*, α) absent | Nearly all | Allegro-pol arXiv:2403.17207; MACE-Field arXiv:2508.17870 | Electric-enthalpy models, LES-BEC; larger dielectric datasets |
| OOD failure under high P/T, short-range collisions | Most uMLIPs except MatterSim-type sampling | MLIP Arena arXiv:2509.20630; 3BPA 1200 K degradation | ZBL priors, high-P/T active sampling, UQ-triggered fallback to DFT |
| Uncalibrated OOD uncertainty | All, especially foundation models | LLPR arXiv:2403.02251 (in-distribution calibration); ensembles over-confident | LLPR plus recalibration, ensembles of heads, evidential (arXiv:2602.10419), conformal methods |
| Poor strong scaling of MPNNs | NequIP, MACE, SevenNet, eSEN, UMA | NequIP framework: 205 steps/s for 23.5k atoms on 1024 A100s; memory OOM above 10k–50k atoms for eSEN and MACE-OFF-L (UMA Table 3) | Strictly local (Allegro), distillation into local students, graph parallelism (UMA), activation checkpointing |
| Tensor-product cost and memory | NequIP, MACE (high L), Equiformer | O(L⁶) CG; MACE memory limit of 2016 atoms (GRACE paper) | eSCN SO(2), Gaunt TP, Cartesian irreducible tensors (ICTP, TACE), custom kernels |
| Symmetry only approximate | PET, Orb, EScAIP, local-frame models | arXiv:2601.16195 and arXiv:2412.11569 discuss failures | ECSE symmetrisation, equigrad, augmentation at scale |
| Body-order incompleteness | ACSF-BPNN, SOAP-GAP, SchNet, M3GNet-like 3-body models | Pozdnyakov et al. arXiv:2001.11696 | ACE, MTP, MACE, GRACE; higher correlation order |
| Fidelity inconsistency (PBE vs r2SCAN vs hybrid vs CC) | All multi-dataset foundation models | OMat24 vs MP pseudopotential incompatibility (Orb-v3); MatPES r2SCAN | Multi-head replay (MACE-MH-1), fidelity embeddings (SevenNet-MF, TACE), Δ-learning to CC |
| Magnetism | Nearly all universal models | Magnetic ACE arXiv:2305.15137; mMACE arXiv:2604.08143; Fe–Cr–C arXiv:2507.18935 | Explicit spin degrees of freedom (non-collinear ACE, mMACE), constrained-moment DFT data |
| Excited states and NQE-quality references | All | No foundation-scale excited-state MLIP identified | Multi-state heads, Δ-learning to high-level references, PIMD-aware training |
| Benchmark saturation and leakage | rMD17, MD17, Matbench F1 | rMD17 spread within about 1–2 meV/Å (Table R1); F1 about 0.93 at the top; MLIP Arena on leakage | N=50 rMD17, 3BPA, MD22, κSRME, MDR, MLIP Arena, MD-stability metrics |

---

## 8. rMD17 baselines (Ntrain ≈ 1000)

**Protocol notes. Read these before using the numbers.**
- rMD17 is from Christensen & von Lilienfeld, arXiv:2007.09593 (PBE/def2-SVP, five fixed 1000-configuration splits).
- Most models train on 950 and validate on 50: MACE, ViSNet, QuinNet, GotenNet, ICTP. NequIP, Allegro and ACE report "1000 training configurations".
- Energies are total-energy MAE in meV. Forces are per-component MAE in meV/Å.
- ViSNet, QuinNet and GotenNet report kcal/mol. They were converted here at 43.364 meV per kcal/mol, which introduces rounding of about ±0.01.
- GemNet reports forces only.
- **Do not mix in original-MD17 numbers.** PaiNN, DimeNet, NewtonNet, SchNet, sGDML, So3krates and SpookyNet are original MD17 only (Table R2). The MACE paper places PaiNN, DimeNet and NewtonNet columns in its rMD17 table, but explicitly as original-MD17 values.

### Table R1a — rMD17, classical and early neural models (E meV / F meV/Å)

| Molecule | GAP¹ | ANI-pre¹ | FCHL19¹ | ACE¹ | UNiTE² | GemNet-T/Q³ (F only) |
|---|---|---|---|---|---|---|
| Aspirin | 17.7 / 44.9 | 16.6 / 40.6 | 6.2 / 20.9 | 6.1 / 17.9 | 2.4 / 7.6 | – / 9.5 |
| Azobenzene | 8.5 / 24.5 | 15.9 / 35.4 | 2.8 / 10.8 | 3.6 / 10.9 | 1.1 / 4.2 | – |
| Benzene | 0.75 / 6.0 | 3.3 / 10.0 | 0.3 / 2.6 | 0.04 / 0.5 | 0.07 / 0.73 | – / 0.5 |
| Ethanol | 3.5 / 18.1 | 2.5 / 13.4 | 0.9 / 6.2 | 1.2 / 7.3 | 0.62 / 3.7 | – / 3.6 |
| Malonaldehyde | 4.8 / 26.4 | 4.6 / 24.5 | 1.5 / 10.2 | 1.7 / 11.1 | 1.1 / 6.6 | – / 6.6 |
| Naphthalene | 3.8 / 16.5 | 11.3 / 29.2 | 1.2 / 6.5 | 0.9 / 5.1 | 0.46 / 2.6 | – / 1.9 |
| Paracetamol | 8.5 / 28.9 | 11.5 / 30.4 | 2.9 / 12.2 | 4.0 / 12.7 | 1.9 / 7.1 | – |
| Salicylic acid | 5.6 / 24.7 | 9.2 / 29.7 | 1.8 / 9.5 | 1.8 / 9.3 | 0.73 / 3.8 | – / 5.3 |
| Toluene | 4.0 / 17.8 | 7.7 / 24.3 | 1.6 / 8.8 | 1.1 / 6.5 | 0.45 / 2.5 | – / 2.2 |
| Uracil | 3.0 / 17.6 | 5.1 / 21.4 | 0.6 / 4.2 | 1.1 / 6.6 | 0.58 / 3.8 | – / 3.8 |

¹ GAP, ANI and ACE values are as reported by Kovács et al. (JCTC 2021) and reproduced in the NequIP (arXiv:2101.03164 v3, Table II), Allegro (arXiv:2204.05249, Table I) and MACE (arXiv:2206.07697, Table 1) papers. FCHL19 is the best of its training modes. The MACE paper lists FCHL19 malonaldehyde F as 10.3, paracetamol F as 12.3 and toluene E as 1.7; the NequIP and Allegro papers give 10.2, 12.2 and 1.6.
² UNiTE "direct-learning" values, as quoted in the NequIP and Allegro papers.
³ GemNet best of T/Q, as quoted in the NequIP, Allegro and MACE papers.

### Table R1b — rMD17, equivariant and modern models (E meV / F meV/Å)

| Molecule | NequIP l=3 (a)⁴ | NequIP l=3 (b)⁴ | Allegro | BOTNet | MACE | TensorNet 2L | ViSNet⁵ | QuinNet⁵ | GotenNet⁵ | ICTP-sym |
|---|---|---|---|---|---|---|---|---|---|---|
| Aspirin | 2.3 / 8.2 | 2.3 / 8.5 | 2.3 / 7.3 | 2.3 / 8.5 | 2.2 / 6.6 | 2.4 / 8.9 | 1.93 / 6.59 | 2.11 / 6.20 | **1.55 / 5.66** | 2.27 / 6.67 |
| Azobenzene | 0.7 / 2.9 | 0.7 / 3.6 | 1.2 / 2.6 | 0.7 / 3.3 | 1.2 / 3.0 | 0.7 / 3.1 | 0.68 / 2.54 | 1.71 / 2.22 | **0.52 / 2.09** | 1.20 / 2.92 |
| Benzene | 0.04 / 0.3 | 0.04 / 0.3 | 0.3 / 0.2 | 0.03 / 0.3 | 0.4 / 0.3 | **0.02** / 0.3 | 0.03 / 0.24 | 0.42 / **0.20** | **0.02 / 0.20** | 0.26 / 0.34 |
| Ethanol | 0.4 / 2.8 | 0.4 / 3.4 | 0.4 / 2.1 | 0.4 / 3.2 | 0.4 / 2.1 | 0.5 / 3.5 | 0.34 / 2.26 | 0.42 / 2.24 | **0.31 / 2.09** | 0.43 / 2.63 |
| Malonaldehyde | 0.8 / 5.1 | 0.8 / 5.2 | 0.6 / 3.6 | 0.8 / 5.8 | 0.8 / 4.1 | 0.8 / 5.4 | 0.57 / 3.87 | 0.73 / 3.79 | **0.56 / 3.60** | 0.82 / 4.96 |
| Naphthalene | 0.9 / 1.3 | 0.2 / 1.2 | 0.2 / **0.9** | 0.2 / 1.8 | 0.5 / 1.6 | 0.2 / 1.6 | 0.25 / 1.26 | 0.75 / 1.05 | **0.17** / 1.04 | 0.56 / 1.45 |
| Paracetamol | 1.4 / 5.9 | 1.4 / 6.9 | 1.5 / 4.9 | 1.3 / 5.8 | 1.3 / 4.8 | 1.3 / 5.9 | 1.12 / 4.46 | 1.57 / 4.25 | **0.92 / 4.02** | 1.44 / 4.89 |
| Salicylic acid | 0.7 / 4.0 | 0.7 / 4.0 | 0.9 / **2.9** | 0.8 / 4.3 | 0.9 / 3.1 | 0.8 / 4.6 | 0.70 / 3.45 | 1.43 / 3.34 | **0.61** / 3.05 | 0.97 / 3.66 |
| Toluene | 0.3 / 1.6 | 0.3 / 1.6 | 0.4 / 1.8 | 0.3 / 1.9 | 0.5 / 1.5 | 0.3 / 1.7 | 0.26 / 1.14 | 0.60 / **1.06** | **0.19** / 1.13 | 0.46 / 1.61 |
| Uracil | 0.4 / 3.1 | 0.4 / 3.2 | 0.6 / **1.8** | 0.4 / 3.2 | 0.5 / 2.1 | 0.4 / 3.1 | 0.30 / 2.15 | 0.65 / 2.11 | **0.28** / 1.81 | 0.57 / 2.64 |

Sources:
- NequIP (a): as quoted in the Allegro paper (arXiv:2204.05249, Table I) and re-quoted by MACE, TensorNet and ICTP.
- NequIP (b): NequIP arXiv:2101.03164 v3, Table II.
- Allegro: arXiv:2204.05249, Table I.
- BOTNet and MACE: arXiv:2206.07697, Table 1.
- TensorNet: arXiv:2306.06482, Table 2. Several force values carry ±0.1 seeds.
- ViSNet: arXiv:2210.16518, Table 2.
- QuinNet: NeurIPS 2023 paper, Table 2.
- GotenNet: ICLR 2025 paper, Table 4.
- ICTP-sym: arXiv:2405.14253, Table 1 (5 seeds, ± about 0.01–0.2).

⁴ The two NequIP rows differ, and the (a) naphthalene E value of 0.9 looks anomalous, since (b) gives 0.2 and ViSNet's table, which quotes NequIP, gives 0.2. Use (a) for comparability with the MACE and Allegro literature, and cite both.

⁵ Converted from kcal/mol. QuinNet's own table contains some inconsistent baseline entries, for example MACE naphthalene E listed as 0.1153 kcal/mol. Only QuinNet's own column is used here.

**Additional notes.**
- *Best model per row.* Bold marks the lowest value per row among the models shown. Differences below about 0.1 meV or 0.2 meV/Å are within seed noise for most entries.
- *Not shown.* A 2026 preprint (arXiv:2606.15892) reports MACE+PAN+PGS at 1.91 / 5.59 on aspirin under a matched in-house protocol (5 seeds; its baseline MACE is 2.21±0.09 / 6.62±0.20). This is promising but unreplicated. Some other per-molecule values in that preprint's tables appear mis-aligned.
- *Missing from rMD17.* No primary-source rMD17 numbers were found for SchNet, PaiNN, sGDML, DimeNet, So3krates, SpookyNet, Equiformer or PET. If you need them, run them yourself with the standard splits.

### Table R1c — rMD17 low-data regime (Ntrain = 50), E meV / F meV/Å

| Molecule | ACE | NequIP | MACE | ICTP-sym |
|---|---|---|---|---|
| Aspirin | 26.2 / 63.8 | 19.5 / 52.0 | 17.0 / 43.9 | 14.84 / 40.19 |
| Azobenzene | 9.0 / 28.8 | 6.0 / 20.0 | 5.4 / 17.7 | 5.47 / 17.25 |
| Benzene | 0.2 / 2.7 | 0.6 / 2.9 | 0.7 / 2.7 | 0.38 / 2.45 |
| Ethanol | 8.6 / 43.0 | 8.7 / 40.2 | 6.7 / 32.6 | 6.15 / 29.53 |
| Malonaldehyde | 12.8 / 63.5 | 12.7 / 52.5 | 10.0 / 43.3 | 9.72 / 42.88 |
| Naphthalene | 3.8 / 19.7 | 2.1 / 10.0 | 2.1 / 9.2 | 2.06 / 9.43 |
| Paracetamol | 13.6 / 45.7 | 14.3 / 39.7 | 9.7 / 31.5 | 8.94 / 30.13 |
| Salicylic acid | 8.9 / 41.7 | 8.0 / 35.0* | 6.5 / 28.4 | 5.95 / 27.78 |
| Toluene | 5.3 / 27.1 | 3.3 / 15.1 | 3.1 / 12.1 | 2.45 / 11.24 |
| Uracil | 6.5 / 36.2 | 7.3 / 40.1 | 4.4 / 25.9 | 4.66 / 25.97 |

Sources: MACE paper Table 1 (ACE, NequIP and MACE at N=50) and the ICTP paper, Table 1.

\*The ICTP paper quotes NequIP salicylic F as 35.9. The MACE PDF text extraction gives 35.0. Check the MACE PDF directly before using this value.

The PDF tables were extracted from multi-column layouts. The N=50 ACE/NequIP column alignment in the MACE paper was cross-checked against ICTP's re-quoted NequIP and MACE values, which agree except for that one entry.

### Table R2 — original MD17 only (NOT rMD17), for reference: aspirin and ethanol, E meV / F meV/Å, N=1000 (950/50)

| Model | Aspirin | Ethanol | Source |
|---|---|---|---|
| SchNet | 16.0 / 58.5 | 3.5 / 16.9 | QuinNet Table 1; NequIP Table I |
| sGDML | 8.2 / 29.5 | 3.0 / 14.3 | NequIP Table I |
| DimeNet | 8.8 / 21.6 | 2.8 / 10.0 | QuinNet and NequIP tables |
| PaiNN | 7.2 / 14.7 (variant-dependent; 6.9 / 16.1 in MACE and So3krates tables) | 2.8 / 9.7 | QuinNet Table 1 |
| SpookyNet | 6.5 / 11.2 | 2.3 / 4.1 | QuinNet Table 1 |
| ET (TorchMD-NET) | 5.3 / 11.0 | 2.3 / 4.7 | QuinNet Table 1 |
| GemNet (F only) | – / 9.4 | – / 3.7 | QuinNet Table 1 |
| NequIP l=3 | 5.7 / 8.0 | 2.2 / 3.1 | QuinNet Table 1 (the So3krates table gives 0.13 / 0.19 kcal/mol for aspirin, about 5.6 / 8.2) |
| So3krates | 6.0 / 10.2 | 2.3 / 4.2 | arXiv:2205.14276, Table 1 |
| ViSNet | 5.0 / 6.7 | 2.2 / 2.6 | QuinNet Table 1 |
| QuinNet | 5.2 / 6.3 | 2.2 / 2.6 | QuinNet Table 1 |

Original-MD17 energies are noisy (the reason rMD17 was created), so the energy columns are not meaningful comparisons.

---

## 9. Reported speed numbers

### Table S1 — speed and cost
Numbers are as reported. Hardware, precision and system matter, so cross-row comparison is approximate. µs/(atom·step) = 10⁶ / (atoms × steps/s).

| Model | System / size | Hardware | Reported speed | ≈ µs/(atom·step) | Source |
|---|---|---|---|---|---|
| SNAP (carbon) | 20 billion atoms | Summit, 27,900 V100 (6 per node) | Peak 6.21 M atom-steps/(node·s) | About 1 µs per V100 (aggregate 0.16 µs per node) | Nguyen-Cong et al., SC21 |
| DeePMD-kit (Cu) | 127 M atoms | Full Summit, 27,360 V100 | 8.1 / 4.6 / 2.7 × 10⁻¹⁰ s/(step·atom) (double, mixed-single, mixed-half); 0.8–2.5 ns/day | 2.7–8.1×10⁻⁴ machine-wide; about 7–22 µs per GPU | arXiv:2005.00223 (SC20 Gordon Bell) |
| GAP vs MTP/NNP/SNAP | Mo, 23,328 atoms | 1 core, i7-6850K | "Optimal" GAP about 100× costlier than MTP, NNP or SNAP (figure only) | Read from Fig. 2 | arXiv:1906.08888 |
| PACE (ACE) | Cu / Si bulk | 1 CPU core | 0.32 / 0.80 ms/(atom·step) | 320 / 800 | arXiv:2103.00814 v1 |
| Allegro | Ag, 1,022,400 atoms (5 fs step) | 1 A100 | 1.46 ns/day | 0.289 | arXiv:2204.05249, Table IV |
| Allegro | Ag, 100.6 M atoms | 128 A100 | 1.54 ns/day | 0.003 machine-wide (about 0.38 per GPU) | same |
| Allegro | Li₃PO₄, 421,824 atoms (2 fs) | 1 / 64 A100 | 0.518 / 15.5 ns/day | 0.552 / 0.018 machine-wide | same |
| Allegro | Li₃PO₄, 192 atoms | 1 A100 | 32.4 ns/day | 27.8 (GPU underused) | same |
| NequIP-framework Allegro (2025) | DHFR, 23,558 atoms | 256 Perlmutter nodes (1024 A100-40GB) | 205 steps/s max; AOTI plus custom TP gives 4–18× over TorchScript | About 0.21 machine-wide (about 212 per GPU) | arXiv:2504.16068 |
| MACE vs NequIP/BOTNet | 3BPA (27 atoms), latency | A100 | MACE L=0 / 1 / 2: 10.5 / 17.5 / 24.3 ms; BOTNet 101.2 ms; NequIP (mace code) 103.5 ms | n/a (latency-bound) | arXiv:2206.07697, Table 2 |
| GRACE vs MACE | Diamond, 2016 atoms (LAMMPS) | A100-80GB | GRACE 200 vs MACE 400 µs/atom; max fit 12,672 vs 2016 atoms | 200 / 400 | arXiv:2311.16326 |
| UMA-S / UMA-S-1.2 / UMA-M | Periodic, 1000 atoms, about 50 neighbours | 1 H100, TF32, compiled | 16 / 24 / 3 steps/s (UMA-S about 1.4 ns/day); UMA-S fits 100k+ atoms | 62 / 42 / 333 | arXiv:2506.23971 v2, Table 3 |
| eSEN-30M-OAM | Same | 1 H100 | 1.7 steps/s; out of memory at 10k atoms | About 590 | same |
| Orb-v3-conservative-inf-omat | Same | 1 H100 | 30 steps/s | About 33 | same |
| MACE-MPA-0 / MACE-OFF23-L | Same | 1 H100 | 24 / 20 steps/s | 42 / 50 | same |
| Orb-v3 direct-20 / direct-inf / cons-20 / cons-inf | Dense periodic, 1k atoms, forward passes/s | H200 | 216.5 / 125.0 / 41.2 / 28.1 | 4.6 / 8.0 / 24 / 36 | arXiv:2504.06231, Table 1 |
| Orb-v2 / MACE-MPA-0 / MatterSim-5M / SevenNet-MF-ompa / eqV2-M | Same | H200 | 88.3 / 21.2 / 18.8 / 3.5 / out of memory | 11 / 47 / 53 / 286 / – | same |
| Nequix (700k parameters) | – | – | Claims "two orders of magnitude faster inference" than the then-top Matbench model; trained in 100 A100-hours | n/a | arXiv:2508.16067 |
| DPA4 | – | – | Claims several-fold to 10× higher throughput than equally accurate models; 3× faster compiled training | n/a | arXiv:2606.02419 |

**Speed notes.**
- **Orb-v3 table alignment.** The pdftotext layout extraction appears shifted by one row. The values above follow the raw text order, which is consistent with the UMA paper's independent measurements: MACE-MPA-0 at 24 steps/s and Orb-v3 conservative at 30 steps/s on an H100. Verify against the PDF before quoting.
- **UMA Table 3.** Only the 1000-atom row is confirmed by the text ("UMA-S can simulate 1000 atoms at 16 steps per second"). The other rows were reconstructed from the extraction.
- **Practical takeaways.**
  - Allegro at large N on one A100 runs at about 0.3–0.55 µs/(atom·step).
  - SNAP on V100 runs at about 1 µs.
  - Universal equivariant models run at roughly 30–600 µs/(atom·step) at 1000 atoms on an H100. That is two to three orders of magnitude slower than specialised local models.
  - Direct-force Orb is about 5–8× faster than its conservative variant.
  - A competitive new model should report µs/(atom·step) at about 10³, 10⁴ and 10⁵ atoms, peak memory, and single- and multi-GPU scaling, with compile settings stated.

---

## 10. Key references (verified arXiv IDs unless marked)
- **Descriptors, kernels and linear models**
  - BPNN: PRL 98, 146401 (2007)
  - 4G-HDNNP: 2009.06484
  - CENT: PRB 92, 045131 (2015)
  - GAP: 0910.1019
  - SOAP: 1209.3140
  - SNAP: 1409.3880
  - qSNAP: 1711.11131
  - MTP: 1512.06054
  - ACE: PRB 99, 014104 (2019)
  - PACE: 2103.00814
  - sGDML: 1802.09238
  - FCHL19: 1909.01946
  - POD: 2209.02362
  - Zuo et al. cost benchmark: 1906.08888
  - Completeness (Pozdnyakov): 2001.11696
  - rMD17: 2007.09593
- **Invariant message passing**
  - SchNet: 1706.08566
  - PhysNet: 1902.08408
  - DimeNet: 2003.03123
  - DimeNet++: 2011.14115
  - GemNet: 2106.08903
  - GemNet-OC: 2204.02782
  - SpookyNet: 2105.00304
  - PaiNN: 2102.03150
  - ViSNet: 2210.16518
- **Equivariant**
  - NequIP: 2101.03164
  - Allegro: 2204.05249
  - BOTNet / design space: 2205.06643
  - MACE: 2206.07697
  - MACE-MP-0: 2401.00096
  - MACE-OFF: 2312.15211
  - MACE-MH-1: 2510.25380
  - MACE-POLAR-1: 2602.19411
  - MACE-Field: 2508.17870
  - mMACE: 2604.08143
  - Equiformer: 2206.11990
  - EquiformerV2: 2306.12059
  - EquiformerV3: 2604.09130
  - eSCN: 2302.03655
  - Gaunt TP: 2401.10216
  - eSEN: 2502.12147
  - UMA: 2506.23971
  - SevenNet: 2402.03789
  - SevenNet-MF: 2409.07947
  - GRACE: 2311.16326
  - GRACE foundation: 2508.17936
  - So3krates: 2205.14276
  - So3krates (Euclidean transformer): 2309.15126
  - Euclidean fast attention: 2412.08541
  - TensorNet: 2306.06482
  - CACE: 2402.07472
  - HotPP: 2402.15286
  - ICTP: 2405.14253
  - TACE: 2509.14961
  - LATTE: 2405.08137
  - Nequix: 2508.16067
  - NequIP framework: 2504.16068
  - Allegro at scale: 2304.10061
- **Unconstrained, local-frame and attention**
  - PET: 2305.19302
  - PET-MAD: 2503.14118
  - Unconstrained MLIPs at scale: 2601.16195
  - Orb: 2410.22570
  - Orb-v3: 2504.06231
  - EScAIP: 2410.24169
  - AlphaNet: 2501.07155
  - Facet: 2509.08418
  - MARA: 2602.02671
  - GotenNet: ICLR 2025 proceedings
  - QuinNet: NeurIPS 2023 proceedings
- **Universal graph models and DeePMD family**
  - M3GNet: 2202.02450
  - CHGNet: 2302.14231
  - MatterSim: 2405.04967
  - DeePMD: 1707.09571
  - DeePMD-kit v2: 2304.09409
  - DPA-1: 2208.08236
  - DPA-2: 2312.15492
  - DPA-3: 2506.01686
  - DPA4: 2606.02419
  - DeePMD 100M atoms: 2005.00223
  - DeePMD 10B atoms: 2201.01446
- **Long-range, charge and fields**
  - Ewald MP: 2303.04791
  - LES: 2408.15165
  - LES charges: 2412.15455
  - LES library: 2507.14302
  - LES-BEC: 2504.05169
  - Polarizable multipoles: 2605.05746
  - Allegro-pol: 2403.17207
  - AIMNet2: doi:10.1039/d4sc08572h
- **Datasets**
  - OMat24: 2410.12771
  - OMol25: 2505.08762
  - MatPES: 2503.04070
  - SPICE: 2209.10702
- **Cross-cutting and benchmarks**
  - Dark side of the forces: 2412.11569
  - Softening: 2405.07105
  - κSRME: 2408.00755
  - MDR phonons: 2412.16551
  - MLIP Arena: 2509.20630
  - Forces are not Enough: 2210.07237
  - LLPR: 2403.02251
  - Distillation: 2506.10956
  - Magnetic ACE: 2305.15137
  - Fe–Cr–C magnetic DeePMD: 2507.18935
  - Evidential UQ: 2602.10419
- **Not verified in this review**
  - Alexandria, OC20/OC22 and MD22 arXiv IDs
  - The SC21 SNAP paper (no arXiv found)
  - SevenNet-Polar (alphaXiv 2607.14827 only)
  - Papers for TECE, Prophet-OAME, EquFlash and MatRIS (leaderboard entries only)
