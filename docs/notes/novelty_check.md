# Novelty check: four candidate MLIP foundations (as of 2026-10-15)

Method: web searches (standard + extended) over arXiv, Nature/npj, ACS, AIP, APS, NeurIPS/ICLR pages, plus direct fetches of key arXiv abstracts. Citations below are ones actually found in this search; items marked **(unverified)** are from memory or secondary listings only and should be checked before citing.

---

## Summary table

| # | Candidate | Verdict | Closest prior art |
|---|-----------|---------|-------------------|
| 1 | Wave-interference (complex Helmholtz/Yukawa Green's-function fields) | **PARTIALLY NOVEL**, but the angular mechanism as stated is flawed (see 1.3) | Euclidean Fast Attention / ERoPE (complex plane-wave phases, sinc kernels); LODE and generalized 1/r^p LODE; Ewald MP; NEMORA (learned FMM with Yukawa kernels); PSWF-LR; SOG-Net |
| 2 | Variational latent-state potential (E = min_s of a learned nonlinear functional) | **PARTIALLY NOVEL**. The principle is published; the specific generalization is open | Xie, Persson & Small (2020, BpopNN: NN energy minimized over atomic populations); M-OFDFT (learned nonlinear functional minimized over per-atom density coefficients); magnetic MTP (energy minimized over moments); QEq family (CENT, 4G-HDNNP, Shaidu et al. 2024 variational QEq without charge labels; Baldwin et al. 2026 design space); DEQuify (fixed point, not variational) |
| 3 | Optimal-transport potential (Sinkhorn distances to learned prototypes) | **PARTIALLY NOVEL**. Near-identical construction exists for molecular properties, and OT kernels exist for atomistic energies | OT-GNN (Wasserstein distances to learned prototype point clouds, molecular properties); REMatch-SOAP (entropy-regularized OT kernel, atomization energies); Wasserstein-WL kernels; FGW for materials |
| 4 | Differentiable persistent-homology potential (energies + forces) | **PARTIALLY NOVEL** in a narrow sense (no PH potential trained on forces or used in MD was found). The approach is physically weak | Minamitani et al. JCP 2023 (PH descriptor for MLP energies of amorphous C, energies only); PH+GNN defect energies; topology-based ML for Li cluster structure search |

**Recommendation:** pursue **Candidate 2** in its specific open form: equivariant *vector* latents with learned *nonlinear, provably convex* on-site energies, latents never supervised, trained only on E/F/stress, solved with warm-started Newton steps, and with forces from the envelope theorem. This is the only candidate that has a clean physical story, exact energy conservation, a precise delta over prior work, and a credible route to speed (warm starts across MD steps).

---

## 1. Wave-interference potential

### 1.1 Closest prior work found
- **Euclidean Fast Attention (EFA) / ERoPE.** J. T. Frank, S. Chmiela, K.-R. Müller, O. T. Unke, "Euclidean Fast Attention: Machine Learning Global Atomic Representations at Linear Cost", arXiv:2412.08541 (2024; v2 2025; journal version 2026). Each atom's features are multiplied by complex phases exp(i ω u·r). These phases are coherently summed over all atoms (a structure-factor-like field) and contracted. Integrating over directions u on S² (Lebedev grid) gives sinc(ω r) kernels, and an equivariant version uses Y_lm(u). **This is the closest published relative.** It is coherent superposition of complex waves emitted by atoms, with learned frequencies, inside an MLFF.
- **LODE.** A. Grisafi, M. Ceriotti, "Incorporating long-range physics in atomic-scale machine learning", J. Chem. Phys. 2019 **(details unverified)**. Atom densities are convolved with a Green's function (Coulomb) to form a potential field, which is expanded at each atom. It was generalized to arbitrary 1/r^p kernels in K. Huguenin-Dumittan, P. Loche, N. Haoran, M. Ceriotti, "Physics-inspired equivariant descriptors of non-bonded interactions", J. Phys. Chem. Lett. 2023, arXiv:2308.13208. This is the Green's-function-field-at-the-nucleus idea with real, non-oscillatory kernels.
- **Ewald message passing.** A. Kosmala, J. Gasteiger, N. Gao, S. Günnemann, "Ewald-based Long-Range Message Passing for Molecular Graphs", ICML 2023, arXiv:2303.04791. Uses learned reciprocal-space (plane-wave) filters, so it is effectively a learned Fourier-space kernel.
- **NEMORA.** J. L. Kaplan, S. Varner, R. Willett, J. J. de Pablo, "NEMORA: Neural Equivariant Multipole Operators for Long-Range Atomistic Learning", arXiv:2610.10776 (Oct 2026). A learned equivariant FMM that keeps the analytical Green's-function radial factors. Its appendix treats the Yukawa kernel e^{-κr}/r explicitly, according to a search snippet.
- **PSWF-LR.** J. Liang, L. Lu, Y. Ji, S. Jiang, "Prolate spheroidal wave functions enable fast and exponent-aware long-range machine learning interatomic potentials", arXiv:2606.06617 (Jun 2026). Spectral/Fourier long-range kernels for 1/r^p.
- **SOG-Net.** Y. Ji, J. Liang, Z. Xu, Phys. Rev. Lett. 135, 178001 (2025). A sum-of-Gaussians learned long-range kernel.
- **Latent Ewald Summation (LES).** B. Cheng, "Machine learning of charges and long-range interactions from energies and forces", Nat. Commun. 2025 (s41467-025-63852-x). Uses latent multi-channel charges with an Ewald (Coulomb Green's function) sum.
- Learned Green's functions for the Helmholtz equation exist in seismic and wave modelling, but not in atomistics.

**Not found:** an MLIP whose primitive is a *complex, oscillatory, damped* Helmholtz Green's function exp(ikr − κr)/r with learned complex wavenumbers and element-dependent complex amplitudes, followed by a nonlinear readout of |ψ|² and cross-channel products at the nuclei.

### 1.2 Are Ewald / LES / Yukawa kernels "the same idea"?
They are not identical, but they are the same *family*: linear field from atom sources, then Green's-function convolution, then a read-out at the atoms. The candidate differs in three ways: (a) the kernel is oscillatory and complex rather than monotone and real; (b) it uses short-range screened kernels as the main many-body engine rather than as a long-range add-on; (c) the readout is nonlinear in the field (|ψ|²). EFA already covers (a) and (c) in Fourier space. Bessel/sine radial bases (DimeNet, NequIP) are not coherent superpositions and are not prior art for the interference claim.

### 1.3 Technical problem with the stated mechanism (important)
A point source's Green's function is **isotropic**. The field at nucleus i is ψ_i = Σ_j A_{Z_j} G(r_ij). This depends only on the *multiset of distances* {r_ij}. Then |ψ_i|² = Σ_{j,k} A_j A_k* G(r_ij) G*(r_ik) is a sum over neighbor pairs of a function of (r_ij, r_ik) only. It contains **no bond-angle information**: it equals products of l=0 density moments, the l=0 slice of an ACE/power-spectrum basis. "Angular dependence from interference" appears only if one of these holds:
- the field is evaluated *off* the nucleus (bond midpoints, a grid, or neighbour sites, which turns it into message passing; neighbour-site evaluation uses r_jk and hence angles);
- field gradients or Hessians are used (these are Cartesian tensors, which brings back MTP/TensorNet-style features); or
- sources are anisotropic (multipoles, which brings back spherical harmonics).

The pure single-site version is therefore strictly *less* expressive than radial-only ACE. Novelty would have to rest on the multi-site variant, which overlaps heavily with EFA and LODE.

**Verdict: PARTIALLY NOVEL.** The specific damped-oscillatory complex Green's-function parameterization is not published. The conceptual core (coherent complex-phase superposition with a nonlinear readout) is in EFA, and the single-site angular claim is mathematically incorrect.

---

## 2. Variational latent-state potential

### 2.1 Closest prior work found
- **X. Xie, K. A. Persson, D. W. Small**, "Incorporating Electronic Information into Machine Learning Potential Energy Surfaces via Approaching the Ground-State Electronic Energy as a Function of Atom-Based Electronic Populations", J. Chem. Theory Comput. 16, 4256–4270 (2020), doi:10.1021/acs.jctc.0c00217, arXiv:2003.01893. **This is the closest prior work.** A neural network gives E(R, atomic populations), a *nonlinear learned* function, and the energy is *minimized over populations* at each geometry. The differences: the latents are physical scalars (charge and spin populations), and they are supervised with constrained-DFT labels.
- **M-OFDFT.** H. Zhang et al., "Overcoming the barrier of orbital-free density functional theory for molecular systems using deep learning", arXiv:2309.16578 (Nat. Comput. Sci. 2024; journal details unverified). A learned *nonlinear* (deep) kinetic functional of per-atom density-coefficient *vectors* p, with E = min_p E_θ(p, M) solved by gradient descent. The structure is very close (per-atom latent vectors, nonlinear learned functional, variational minimum). It differs in that p is a physical density supervised on KS-DFT data, the functional is fully non-local, and the method is not an MLIP-speed potential. Follow-up: Remme et al., "Stable and Accurate Orbital-Free DFT Powered by Machine Learning", arXiv:2503.00443.
- **Magnetic Moment Tensor Potentials.** I. Novikov et al., npj Comput. Mater. 2022, doi:10.1038/s41524-022-00696-9, arXiv:2012.12763; follow-up "Fitting to magnetic forces...", arXiv:2405.07069. The energy is a learned (polynomial) function of per-atom magnetic moments, and the moments are minimized at fixed geometry. The latents are physical scalar moments and are supervised with constrained DFT.
- **QEq family (quadratic on-site, scalar charges).** CENT: S. A. Ghasemi et al., PRB 2015, arXiv:1501.07344. 4G-HDNNP: T. W. Ko, J. A. Finkler, S. Goedecker, J. Behler, Nat. Commun. 2021, arXiv:2009.06484. Variational QEq trained *with or without* charge labels: Y. Shaidu, F. Pellegrini, E. Küçükbenli, R. Lot, S. de Gironcoli, npj Comput. Mater. 10 (2024), doi:10.1038/s41524-024-01225-6. Shell/polarizable QEq in a foundation model: "A foundation machine learning potential with polarizable long-range interactions for materials modelling", arXiv:2410.13820 (Nat. Commun. 2025; authors unverified). Design-space study: W. J. Baldwin, I. Batatia, M. Vondrák, J. T. Margraf, G. Csányi, "Design Space of Self-Consistent Electrostatic Machine Learning Interatomic Potentials", arXiv:2603.14700 (2026). It notes that Hellmann–Feynman gives parameter gradients without implicit differentiation for variational functionals, and that "more expressive self-consistent models are needed". kQEq (Staacke et al. 2021) **(unverified)**.
- **DEQuify.** Burger, Thiede, Aspuru-Guzik, Vijaykumar, "DEQuify your force field: More efficient simulations using deep equilibrium models", arXiv:2509.08734 (ICLR 2025 AI4MAT workshop). Fixed-point implicit layers with feature recycling across MD steps. These are *not* variational: the energy is not a minimum over the latents, so forces still require implicit differentiation.
- Non-variational latents: LES (latent charges, no minimization); SpookyNet and AIMNet2 (charge/spin embeddings, iterative but non-variational).

### 2.2 What is and is not new
The general principle "E(R) = min over per-atom auxiliary variables of a learned nonlinear functional, with forces from the envelope theorem" **is published**: Xie et al. 2020 (scalar, nonlinear), M-OFDFT (vector, nonlinear, physical density), mMTP (moments), and Shaidu et al. 2024 (unsupervised latents, but quadratic).

The following combination was **not found**:
1. *General, non-physical, multi-channel vector latents* (possibly equivariant l=0 ⊕ l=1);
2. *Learned nonlinear on-site self-energies* f_θ(s_i; env_i) with *learned pair couplings* J_θ(r_ij) g(s_i, s_j);
3. latents **never supervised**, with training on E/F/stress only and forces from the envelope theorem;
4. designed as a fast local MLIP (warm-started Newton per MD step, DEQuify-style recycling).

### 2.3 Technical requirements for the claim to work
- **Smoothness.** If the inner problem has several minima, the argmin can jump and E(R) loses C¹ smoothness, so forces become discontinuous. Enforce strict convexity in s: an input-convex network for f, plus a coupling term whose Hessian is dominated by the on-site curvature (e.g., spectral-norm control of J). Then the implicit function theorem gives a smooth, unique s*(R).
- **Training.** The parameter gradient of E at the optimum needs no implicit differentiation (envelope theorem). Force-loss gradients *do* need implicit differentiation, through second derivatives: dF/dθ involves ∂s*/∂R, which requires the inner Hessian solve. The Baldwin et al. 2026 discussion is relevant here.
- **Expressivity argument.** The minimum of a sum of local terms over coupled latents produces non-local, many-body, non-additive energies (Schur-complement-like) from strictly local ingredients. This is the reason to use the model, and it should be demonstrated on charge-transfer and polarization benchmarks.

**Verdict: PARTIALLY NOVEL.** The precise formulation above is open. It must be positioned explicitly against Xie et al. 2020, M-OFDFT, mMTP and Shaidu et al. 2024.

---

## 3. Optimal-transport potential

### 3.1 Closest prior work found
- **OT-GNN.** B. Chen, G. Bécigneul, O.-E. Ganea, R. Barzilay, T. Jaakkola, "Optimal Transport Graph Neural Networks", arXiv:2006.04804 (2020). The representation is built from Wasserstein distances between a structure's point cloud of node embeddings and **learned prototype point clouds**, with a universality proof, applied to molecular property prediction. This is essentially candidate 3's construction, minus geometry, forces and local atom-centred environments.
- **REMatch.** S. De, A. P. Bartók, G. Csányi, M. Ceriotti, "Comparing molecules and solids across structural and alchemical space", PCCP 18, 13754 (2016). An entropy-regularized OT (Sinkhorn) match between sets of SOAP environments, used in KRR for atomization energies (MAE below 1 kcal/mol). The reviews arXiv:2101.04673 and arXiv:1706.00179 describe it.
- **Wasserstein Weisfeiler–Lehman graph kernels.** M. Togninalli et al., NeurIPS 2019, arXiv:1906.01277.
- **MROT.** OT for molecular regression and domain adaptation, arXiv:2202.06208.
- Fused Gromov–Wasserstein for materials similarity: "Discovering new photovoltaics using optimal transport theory", arXiv:2602.22036 (2026). Compositional EMD for materials: Sci. Rep. 2024 (s41598-024-79126-3).
- React-OT (arXiv:2404.13430) uses OT for transition-state generation, which is a different use.

**Not found:** an MLIP whose atomic energy is a learned function of Sinkhorn distances between an atom's *neighbour measure* (in 3D) and learned 3D prototype environments, trained on forces.

### 3.2 Technical problems
- **Invariance.** A Euclidean-cost OT between a neighbour cloud and a fixed prototype is not rotation-invariant. Fixes are (a) Gromov–Wasserstein on internal distances (non-convex, expensive, gradients only via an approximate local optimum); (b) minimizing over rotations (a non-smooth max over SO(3)); or (c) OT over invariant features, which reduces to REMatch or OT-GNN applied to SOAP/ACE features.
- **Smoothness and cutoffs.** Atoms crossing the cutoff change total mass. This requires unbalanced OT with cutoff-weighted masses.
- **Cost.** Each atom needs P prototypes × n² × (Sinkhorn iterations). This is far slower than ACE/MACE.

**Verdict: PARTIALLY NOVEL**, and only in its geometric MLIP instantiation. The core idea (OT distances to learned prototypes) is OT-GNN, and OT kernels for energies are REMatch. Weak candidate.

---

## 4. Differentiable persistent-homology potential

### 4.1 Closest prior work found
- **E. Minamitani, T. Shiga, M. Kashiwagi, I. Obayashi**, "Persistent homology-based descriptor for machine-learning potential of amorphous structures", J. Chem. Phys. 159, 084101 (2023), arXiv:2206.13727. Persistence-diagram histograms with linear models or CNNs predict the *mean energy per atom* of amorphous carbon and graphene. Forces are not used and no MD is reported. Co-author list partially unverified. Follow-up talks (JSAP 2024) and a JST PRESTO report cover elastic constants. No force-trained follow-up was found.
- **Fang & Yan**, "Leveraging Persistent Homology Features for Accurate Defect Formation Energy Predictions via Graph Neural Networks", Chem. Mater. 37(4), 1531 (2025) (arXiv July 2024). PH features inside GNNs, energies only.
- **X. Chen, D. Chen, M. Weng, Y. Jiang, G.-W. Wei, F. Pan**, J. Phys. Chem. Lett. 2020. Topology-based ML for structure search of Li clusters. Not a force field.
- Mototake, "Neural Reduced Potential via Persistent Homology", NeurIPS 2025. PH with HNN-style reduced potentials for magnetic-domain dynamics; continuum, not atomistic.
- Element-specific PH for binding affinities (Wei group) and multiparameter PH for molecular properties (Demir & Kiziltan, ICLR 2023 workshop). Both are energies or properties only.

**Not found:** a PH-based potential trained on forces, energy-conserving, used in MD.

### 4.2 Technical problems (why the gap probably exists)
- Persistence diagrams are Lipschitz-stable but only **piecewise smooth** in coordinates. Gradients jump whenever the simplex pairing changes. For Rips 0-dim PH, this happens whenever the MST topology changes. The energy can be continuous while the forces are **discontinuous**, which breaks NVE energy conservation and phonon/Hessian quality.
- In 3D, H0 is essentially MST edge lengths and H1/H2 carry ring and cavity information. Most of this is already captured by radial, angular and ring features, so accuracy gains are unlikely to justify the cost (Rips complexes are expensive) and non-smoothness.

**Verdict: PARTIALLY NOVEL** (force-trained PH MLIP not published), but scientifically the weakest candidate.

---

## 5. Additional foundations that appear absent (proposals)

Several other ideas were checked and **found to be already covered**:
- Tensor-sketched / random-feature equivariant potentials: Darby et al., "Tensor-reduced atomic density representations", arXiv:2210.01705; random-feature MLIPs (Dhaliwal et al., npj 2022); franken TensorSketch heads.
- Learned-Hamiltonian / spectral potentials: deep Hamiltonian models and HELM (arXiv:2510.00224); semi-empirical Hamiltonian learning from energies (HIP-NN-SEQM, Zhou et al. PNAS 2022, **unverified**).
- Implicit/DEQ force fields: DEQuify.

Two directions returned no MLIP hits in these searches:

**P1. Latent free-energy (marginalized) potential.** E(R) = −τ log ∫ exp(−H_θ(s; R)/τ) ds, with per-atom latent vectors s and H_θ built as in candidate 2. A Laplace or Gaussian-variational inner solve gives E ≈ H(s*) + (τ/2) log det ∇²_s H. The τ→0 limit recovers candidate 2. For τ>0 the energy stays smooth even when the inner landscape has several minima, which removes the main C¹ risk of a min-based model without forcing convexity. Forces are still exact gradients (envelope theorem plus a log-det term, computable with a sparse Hessian). Search evidence: searches for latent/implicit/variational MLIPs returned only min-based models (QEq, BpopNN, mMTP, M-OFDFT) and fixed-point models (DEQuify). No model that *marginalizes* unsupervised latents was found. The paramagnetic MTP work averages over physical spins at the DFT level, which is different. **Strictly, this is a refinement of candidate 2, not an independent foundation.**

**P2. Soft-registration template potential.** E_i = −τ log Σ_t w_t(Z_i) exp(−D_t(env_i)/τ), where D_t is a differentiable Kabsch/Procrustes alignment distance after a Sinkhorn soft assignment of neighbours to the points of a learned template t (a "learned polyhedral template matching"). Rotations are removed exactly by SVD alignment, so the energy is invariant and the forces equivariant without spherical harmonics. Search evidence: no MLIP hits for learned templates with Procrustes or soft assignment; only generic point-set registration and EM-Procrustes appeared. Caveats: SVD gradients become ill-conditioned near degenerate singular values, the method shares the cost issues of candidate 3, and it is closely related to OT-GNN. **Lower priority than P1.**

---

## 6. Recommendation
Pursue **Candidate 2**, ideally with the P1 extension (finite τ) as a fallback for smoothness. Proposed framing: "Unsupervised variational latent fields: an MLIP whose energy is the minimum (or free energy) of a learned convex local functional over equivariant latent vectors". Explicit novelty deltas:
- vs Xie/Persson/Small 2020: non-physical vector latents, no constrained-DFT labels, local equivariant architecture;
- vs M-OFDFT: no density supervision, local, MLIP-speed;
- vs QEq / Shaidu 2024 / Baldwin 2026: nonlinear learned on-site energies and vector channels instead of quadratic scalar charges;
- vs DEQuify: a variational rather than fixed-point formulation, so the envelope theorem gives cheap exact forces and convexity guarantees uniqueness.

Required ablations: quadratic vs nonlinear on-site energy, scalar vs vector latents, and inner iterations vs accuracy and speed. Required benchmarks: charge-transfer and polarization cases (e.g., the Ko et al. 4G test systems), plus standard local benchmarks to show there is no regression.
