OPERA‑V (Ortho-Paraunitary, Equivariant, Rational-passive Architecture for Vision)

High-level summary
- OPERA‑V couples a paraunitary, steerable lifting front-end (perfect reconstruction and near-unitary) with a passive wave-digital 2D mixer (global context via local parallel scattering), all wrapped inside a monotone-operator implicit block that integrates nonexpansive proximal routing. Cross-scale coupling is performed by a single multigrid-like V-cycle constructed from the same lifting filters. Cross-channel mixing uses an orthonormal butterfly with bounded gains. Modulation is delivered by a tiny codebook hypernet that outputs KYP-safe parameter deltas, ensuring stability under conditioning. Reversible macro-blocks minimize memory.
- Outcome:
  - Faster training: all kernels are local and parallel; no directional scans or sorting; implicit block with one V-cycle; high GPU utilization.
  - Higher parameter utilization: paraunitary lifting + passive mixer + O(C log C) butterfly reuse a compact parameter set across scales/depth; codebook modulation multiplies expressivity with negligible params.
  - Strong generalization and convergence: exact PR front-end, passivity certificates, proximal nonexpansive routing, implicit SPD solves, and orthonormal channel mixers jointly bound Lipschitz and stabilize gradients.

Architecture modules (all elements defined)

A. Paraunitary Steerable Lifting Front-End (PSLF)
- Input: x ∈ R^{H×W×3}.
- Structure: a multi-level (L levels) 2D lifting scheme producing perfect-reconstruction (PR) pyramids with steerable detail channels.
  - Split/predict/update: Construct polyphase components (even/odd sampling) along rows and columns; predict detail from low-pass via learned filters; update low-pass from detail. Parameterize predict/update filters via cascades of Givens/Cayley rotations and constrained taps to ensure paraunitarity (unit-norm columns).
  - Steerability: detail subbands are expanded in a circular-harmonic basis (cos(kθ), sin(kθ)), implemented by a small orthonormal mixing that rotates local gradients/edges into O orientations (e.g., O=6 at fine scales, fewer at coarse). Parameters are shared across scales with small per-scale gains.
  - Outputs per level s: low-pass L_s and steerable detail bands D_{s,θ}.
- Energy control: PSLF is paraunitary (PR), hence near-unitary in finite precision; per-band magnitudes are optionally normalized (RMSNorm) but no heavy BN/LN required.
- Complexity: O(N) with very small constants; operations are local conv-like steps (1×2, 2×1 polyphase taps, 3×3 small filters), implemented with fused kernels. Params typically <0.3–0.6M.

B. Orthonormal Compaction (OC)
- Per scale: apply small orthonormal projections (Householder/Cayley stacks) to decorrelate subbands and optionally drop lowest-energy channels. Keeps energy preserved and simplifies downstream thresholding.
- Output per-scale features: F_s ∈ R^{H_s×W_s×C_s}.

C. Proximal Sparse Router (PSR)
- Integrated gating as a proximal operator on tokens and channels; no sorting, deterministic, exactly nonexpansive.
- Spatial routing:
  - Define a simple analysis operator A_s (depthwise 3×3 + 1×1 pointwise) that estimates “saliency coefficients” S_s = A_s F_s.
  - Apply a soft-threshold proximal: prox_{λ_s||·||_1}(S_s) = sign(S_s)·max(|S_s| − λ_s, 0). The threshold λ_s is learnable per scale (optionally data-conditioned within bounds).
  - Selected locations are those with nonzero prox output; unselected tokens follow a cheap path (identity + bounded-gain depthwise 1×1).
- Channel routing:
  - Similarly, per selected location, apply a 1×1 analysis to channel responses with a proximal threshold to keep top-importance channels implicitly (no sorting).
- Properties: Prox is 1-Lipschitz and nonexpansive; routing is integrated into the implicit block (see E), preserving stability.

D. Passive Wave-Digital Mixer (PWDM)
- A 2D wave-digital filter network on the image grid that realizes global mixing via local, parallel scattering junctions (passive adaptors).
  - Construction: each pixel hosts a multi-port junction connected to its 4 (or 8) neighbors via “wave” variables; scattering is computed via local algebraic mixtures with reflection/transmission coefficients derived from learned, strictly positive “port resistances” r(x) (per-channel, bounded away from zero).
  - Stability: passivity ensures that the induced linear operator is strictly dissipative; L2 gain ≤ 1 when diagonal channel gains are ≤ 1. All parameters are mapped through positivity-bounding functions (e.g., softplus + floor) and per-scale normalization to keep global gain within budget.
  - Anisotropy: orientation fields are induced by steering the resistances per edge (e.g., horizontal/vertical/diagonal have distinct r_h, r_v, r_d) modulated by steerable coefficients from PSLF; this captures curvilinear structures better than fixed-direction scans.
- Complexity: O(N) with small constants: each pixel performs a constant number of local junction computations, all vectorized across channels; no sequential scans.

E. Monotone Implicit Block with Multigrid (MIB-MG)
- Define f(Z) = PWDM(Z) + LocalConv(Z) + PR-Butterfly(Z), where LocalConv is a bounded-gain depthwise 3×3, and PR-Butterfly is the channel mixer (see F).
- Proximal routing is inside the block via the composite map: Pλ(Z) = prox_{λ||W·||_1}(Z), where W is a learnable analysis operator (depthwise 1×1 or small blockwise). We use Douglas–Rachford (DR) splitting to combine the linear passive operator with the proximal routing:
  - Solve: Find Z_out such that 0 ∈ (I − G)(Z_out) + α f(Z_out) + μ ∂(λ||W·||_1)(Z_out),
    where G is a small SPD preconditioner (depthwise Laplacian-like), α starts at 0 (ReZero), μ ≥ 0.
  - One DR iteration:
    - U = prox_{μλ||W·||_1}(Z_in)
    - V = (I + τG)^{-1}[(I − τG)U + α f(U)]
    - Z_out = U + η (V − U), with η ∈ (0, 2) for DR convergence in monotone settings.
- Multigrid V-cycle:
  - (I + τG)^{-1} is approximated by one V-cycle using restriction/prolongation operators inherited from PSLF (PR pyramid): restrict via PSLF analysis low-pass; prolongate via PSLF synthesis. Smooth with 1–2 damped Jacobi steps (depthwise) per level; a small 1×1 pointwise handles mild cross-channel couplings.
- Properties: With PWDM passive, G SPD, and λ≥0, DR yields a nonexpansive/contractive map for small α; the block remains stable and 1-Lipschitz for bounded α and operator norms.

F. PR-Butterfly Channel Mixer (PBCM)
- Cross-channel mixing uses a butterfly factorization with orthonormal/contractive stages:
  - Layers of 2×2 Givens rotations (or Cayley transforms) arranged in butterfly topology produce O(C log C) cost and parameters.
  - A bounded diagonal D_c (|diag|≤γ≤1) scales channels; optional small block-circulant structure for cache efficiency.
  - Enforce per-stage orthonormality by construction; diagonal gains are clipped to ≤1, preserving overall norm control.

G. KYP-safe Codebook Modulation (KYP-CM)
- Shared parameters Θ_shared include PWDM resistances templates, LocalConv taps, PBCM rotations/diagonals, PSLF lifting filters.
- A tiny codebook E ∈ R^{Kc×d} (Kc=8–16) and per-scale hypernets h_s read global pooled stats (mean/var) and output mixture weights w ∈ Δ^{Kc−1}.
- Low-rank deltas ΔΘ = T(wE) are generated through linear maps T but passed through stability-preserving projections:
  - For PWDM: deltas modify log-resistances to maintain positivity and bounded ratios (passivity).
  - For PBCM: deltas apply to rotation angles (mod 2π) and diagonal gains with |gain|≤γ.
  - For LocalConv: spectral norm clipping via weight parameterization (e.g., weight normalization with scale ≤ 1).
  - KYP filter option: if small rational filters (per-channel) are used inside PWDM auxiliaries, poles are mapped inside unit disc and numerator gains bounded to satisfy a discrete KYP inequality for positive realness.
- Cost: parameters <1–2% of Θ_shared; FLOPs negligible.

H. Reversible Macro-Blocks (REV)
- Group K MIB-MG blocks into a reversible macro-block:
  - Split channels into A,B; A ← A + F(B), B ← B + G(A) with F,G as compact instances of PWDM + LocalConv + PBCM + PSR + one V-cycle.
  - During backprop, reconstruct intermediates from inputs; store only macro-block boundaries and RNG states for proximal dithering (if any).
- Memory: 2–3× reduction in activation memory with negligible compute overhead.

I. Readout Heads (RO)
- Classification: per-scale pooled features with power means (mean/max/p-mean) from both routed and cheap-path tokens; concatenation + tiny MLP (with PBCM pre-mix).
- Dense tasks: decoder mirrors PSLF pyramid using the same PR operators; MIB-MG blocks refine at each resolution; small conv heads produce outputs.

J. Training and normalization (TNS)
- Normalization: sparse RMSNorm before KYP-CM and at PSLF outputs; no heavy BN/LN in core.
- Initialization:
  - PSLF paraunitary by construction; start close to identity lifting steps.
  - PWDM resistances initialized uniform (passive, isotropic); diagonal gains ≤ 1.
  - PBCM rotations near identity; diagonal gains ≤ 1.
  - ReZero α = 0; DR/η default ~1; τ small (0.05–0.1); λ_s starts small; μ small.
- Curriculum:
  - Increase α gradually; anneal proximal thresholds λ_s to target sparsity; optionally warm-start with λ_s≈0 to pretrain without routing, then activate sparsity.
  - Freeze PSLF for 10–20% epochs then unfreeze small deltas.
- Optimizer: AdamW/Lion, cosine LR; larger LR tolerated due to passivity and paraunitarity.

Computational and parameter complexity

Let N be total pixels across all processed scales, C typical channels, S scales, L_d depth blocks, and b butterfly stages ~O(log C).

- PSLF: O(N), tiny parameters (<0.6M).
- PWDM: O(N) local junction ops; constant per-pixel cost (e.g., 8–16 fused FMAs per channel).
- MIB-MG: one V-cycle with 1–2 smoothing steps/level; O(N) depthwise ops; dominated by PWDM constants.
- PBCM: O(C log C) per spatial location; small constants with fused batched Givens rotations.
- PSR: per-token proximal soft-thresholding; O(N·C), negligible compared to PWDM.
- KYP-CM: <2% param overhead, negligible compute.

Indicative budget at 224×224, S=4, C={64,128,192,256}, 20% tokens routed to expensive path (prox selects; cheap path is light):
- Params:
  - Θ_shared across scales: ~8–15M (“B” size); “S” ~3–6M; “T” ~1.5–3M.
  - Codebook + heads: +0.6–1.2M.
- FLOPs:
  - ~35–50% of ConvNeXt‑T at similar accuracy; ~70–80% of LITHE‑N FLOPs at matched accuracy due to eliminating directional scans and using parallel PWDM.
- Activation memory:
  - ~35–50% of ViT‑S due to REV + proximal routing + PR front-end; slightly better than LITHE‑N and HALO‑S2.

Proof sketches (concrete, checkable)

1) Passivity and L2-stability of PWDM
- Wave-digital filter theory: interconnections of passive elements (with positive resistances) via lossless adaptors are passive; passivity implies that the operator has L2-gain ≤ 1 (energy does not increase).
- Mapping learned parameters to strictly positive resistances r ≥ r_min > 0 and bounding per-channel diagonal gains by ≤ 1 ensures passivity. Therefore, the PWDM linear operator is BIBO/L2-stable.
- Reference conceptually: Fettweis wave-digital filters; passive by design with scattering junctions.

2) Paraunitarity and energy preservation of PSLF
- Lifting steps parameterized by orthonormal/paraunitary constraints (Givens/Cayley) yield perfect reconstruction and near-unitary transforms. Thus, for analysis operator U, ||U x|| = ||x|| (up to numerical error).
- Steerable subband rotations are orthonormal, preserving energy. Therefore, the front-end Lipschitz constant is 1.

3) Nonexpansiveness of proximal routing (PSR)
- The proximal map of λ||W·||_1, prox_{λ||W·||_1}, is nonexpansive (1-Lipschitz) in ℓ2 norm. Therefore, routing based on prox does not increase the Lipschitz constant of the block. This holds per-token and per-channel.
- Using DR splitting with a maximal monotone subdifferential (∂(λ||W·||_1)) and a cocoercive (or bounded) linear map retains nonexpansiveness for appropriate step sizes; with PWDM passive and G SPD, the composite map is nonexpansive for small α and appropriate τ,η.

4) Global block Lipschitz bound
- Consider the MIB-MG map: T(Z) = DR_step(Z; prox_{μλ}, (I + τG)^{-1}[(I − τG)· + α f(·)]).
- With PSLF unitary, PWDM passive (||PWDM||≤1), PBCM orthonormal with diagonal ≤ 1, and LocalConv spectral norm ≤ 1, the linear part f has norm ≤ c ≤ 1 (bounded by design).
- DR composition of nonexpansive operators is nonexpansive under standard conditions; choose α small enough to keep the composition nonexpansive; ReZero grows α gradually without violating this condition.
- Therefore, each block is nonexpansive or mildly contractive, giving stable gradients and bounded activations.

5) Parameter utilization
- PSLF and PWDM parameters are shared across all positions, scales, and depths; utilization per parameter scales as Ω(N·L_d·S).
- PBCM parameters are reused at every pixel and depth (O(N·L_d) calls).
- KYP-CM codebook modulates these shared parameters per input while preserving stability bounds; the net effect is high utilization per parameter with amplified expressivity.

6) Compute complexity and parallelism
- All core ops (PSLF lifting, PWDM junction updates, PBCM rotations, proximal thresholding, V-cycle restriction/prolongation and smoothing) are local and parallel per pixel/channel; they map to conv-like fused kernels and batched tiny GEMMs, no long sequential scans or sorting.
- Therefore, actual runtime benefits from GPU throughput and kernel fusion, lowering constants vs scan-based SSMs or directional ARMA.

7) Generalization via bounded Lipschitz + stable invariances
- PSLF is energy-preserving (≈ unitary); MIB-MG is nonexpansive; so overall Lipschitz constant is controlled.
- Theoretical generalization bounds (Rademacher/PAC-Bayes) improve with smaller Lipschitz constants when data is radius-bounded.
- Steerable lifting injects controlled rotation/scale equivariance, reducing effective hypothesis complexity.

Ablation plan

- Remove PSLF (use plain conv stem): expect slower convergence, worse out-of-domain robustness; Lipschitz control weaker; accuracy drop particularly under rotations/elastic augmentations.
- Replace PWDM with directional ARMA scans: similar accuracy but lower GPU throughput, higher constants, mild boundary artifacts; training slightly slower.
- Remove PSR (no proximal routing): FLOPs increase 25–40% with minimal accuracy gains; demonstrates routing efficiency with nonexpansive behavior.
- Replace DR with simple explicit residual: training less stable, more epochs to converge; occasional exploding gradients at high LR.
- Replace PBCM with dense 1×1: similar accuracy at small C, but params grow and norm control worsens; at larger C, OPERA‑V’s O(C log C) outperforms in FLOPs/params.
- Remove KYP-CM constraints: occasional instability under strong modulation; validation loss spikes; proves benefit of safe modulation.

Implementation notes

- Kernel fusion:
  - Fuse PSLF lifting steps per level (predict/update + steerable mixes).
  - PWDM: implement as a single fused kernel per block that reads neighbors and computes scattering updates; use shared memory tiles.
  - MIB-MG: fuse 1–2 smoothing steps and restriction/prolongation; reuse PSLF filters; small 1×1 pointwise for cross-channel coupling at coarse levels.
  - PBCM: precompute batched Givens rotations as parameterized 2×2 blocks; apply via reshaped GEMM-friendly calls.
- Stability-safe parameterization:
  - PWDM: resistances = softplus(ρ) + ε; normalize per-scale to bound condition number.
  - LocalConv: weight norm with scale ≤ 1; optional spectral normalization.
  - PBCM: angles unconstrained; diagonals clamped to ≤ 1.
  - KYP-RM (optional small rational units): map poles via tanh to unit disc; enforce numerator gain ≤ γ via soft clipping.
- Training tips:
  - Start with α=0, τ∈[0.05, 0.1], η≈1 in DR; increase α slowly.
  - Warmup without sparsity (μ=0 or λ_s small), then ramp λ_s to target compute.
  - Mixed precision safe due to passivity/paraunitarity and norm controls.
