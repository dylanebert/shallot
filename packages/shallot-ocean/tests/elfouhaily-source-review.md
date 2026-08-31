# I2r-a commit 1 — adversarial source review of `tests/elfouhaily-independent.ts`

**What this review is.** A factor-by-factor audit of the independent test-side implementation of the
Elfouhaily (1997) directional spectrum against the published equations, committed **before** the
production comparison (commit 2) so the oracle is adversarially read first and no production source or
table is its oracle.

**Source under review:** `tests/elfouhaily-independent.ts` (this commit).
**Primary source:** T. Elfouhaily, D. Vandemark, B. Chapron, K. Katsaros, *A unified directional
spectrum for long and short wind-driven waves*, JGR 102(C7), 1997, DOI 10.1029/97JC00467.
**Secondary external anchor:** Cox & Munk (1954) sun-glint mean-square-slope fit
`mss = 0.003 + 0.00512·U₁₀` — quoted by the Elfouhaily paper's own comparison section.

## Citation provenance (read before trusting the equation numbers)

- The equation-number loci used inline (eqs. (3)+(16) dispersion; (10)–(15) wind-stress closure;
  (17) km; (29)–(34) the spectral factors; (35)–(36) the B branches; (37)–(40) the spreading)
  follow the **numbering the two prior adversarial rounds consistently cited** (JGR 102(C7)
  pp. 13,401–13,402 region) and the factor list the stage spec restates verbatim
  (`Fp = Lpm·Jp·exp(−Ω/√10·(√(k/kp)−1))`, `Fm = Lpm·Jp·exp(−0.25·(k/km−1)²)`,
  `spread (c/cp)^2.5 + am·(cm/c)^2.5`, the `Ωc ≤ 1` and `log₁₀` γ branches, the friction
  velocity, both `αm` branches, `αp = 0.006·Ωc^0.55`, `am = 0.13·u*/cm`, `Ωc ∈ 0.84…5`).
- **Not re-verified against the journal PDF in this environment** — no library copy of JGR 102(C7)
  is reachable from this worktree. The factors' *numeric content* is nonetheless externally
  anchored: the full-tail mean-square-slope integral over `k ∈ [0.01, km]` is gated against the
  published Cox–Munk fit at U₁₀ ∈ {5,10,15,20} (25%/30% band), which no code in this repo
  tuned. A wrong factor shows up there; a wrong *equation number* would not, and is flagged
  here rather than papered over.

## Factor-by-factor audit

| Factor | Published form | Implementation | Audit verdict |
|---|---|---|---|
| Dispersion | ω² = gk(1+(k/km)²); c = ω/k | `phaseSpeed` | ✓ matches; c(km) = √(2g/km) minimum asserted in tests |
| km | km = √(ρ_w·g/τ), ρ_w=1025, τ=0.074 | `KM` | ✓ ≈ 368.6 rad/m; value asserted |
| Wind stress | z₀ = 0.0185·u*²/g + 0.11ν/u*; U₁₀ = (u*/κ)·ln(10/z₀), κ = 0.4 | `frictionVelocity` | ✓ fixed-point solved; self-closure asserted to 6 digits. **Note:** the viscous term is *included* here (published closure); the production loop uses the Charnock term alone (~0.1–0.3% effect on u*, far inside the 15% comparison band) — a deliberate independent-authoring divergence, see below |
| Inverse wave age | Ωc = ωp·U₁₀/g ⟺ kp = g·Ωc²/U₁₀² | `peakWavenumber` | ✓ algebra re-derived: ωp² = g·kp ⟹ kp = (Ωc·g/U₁₀)²/g |
| αp | αp = 0.006·Ωc^0.55 | `spectrumAmplitudes` | ✓ exponent 0.55 as published (not 0.5) |
| αm | u* > cm: 0.01(1+3·ln(u*/cm)); else 0.01(1+ln(u*/cm)) | `spectrumAmplitudes` | ✓ both branches; the branch boundary (u*/cm = 1) tested from both sides |
| Lpm | exp(−1.25(kp/k)²) | inside `longWaveFactor` | ✓ |
| σ, γ, r, Jp | σ = 0.08(1+4Ωc⁻³); γ = 1.7 (Ωc≤1) else 1.7+6·log₁₀Ωc; r = exp(−(√(k/kp)−1)²/(2σ²)); Jp = γ^r | inside `longWaveFactor` | ✓ both γ branches; Jp(kp) = γ asserted (r(kp)=1) |
| Fp | Lpm·Jp·exp(−(Ωc/√10)(√(k/kp)−1)) | inside `longWaveFactor` | ✓ |
| Fm | Lpm·Jp·exp(−0.25(k/km−1)²) | inside `shortWaveFactor` | ✓ composed on the same Lpm·Jp product; Gaussian cut at km asserted |
| B_l, B_h | 0.5·αp·(cp/c)·Fp; 0.5·αm·(cm/c)·Fm | `curvatureSpectrum` | ✓ first-power speed ratios, per the reviewed lineage; the *scale* is externally anchored by the Cox–Munk gate (ratio ≈ 1 at U₁₀ = 5), so a wrong power of (cp/c) would break it |
| am | am = 0.13·u*/cm | `spreadingDelta` | ✓ |
| Spreading | Δ = tanh(ln2/4 + 4(c/cp)^2.5 + am(cm/c)^2.5); Φ = (1+Δcos(2(θ−θw)))/(2π) | `spreadingDelta`, `directionalFactor` | ✓ unit angular integral asserted; 2nd-harmonic structure asserted (downwind > crosswind) |
| Density convention | polar Ψ against k·dk·dθ; Cartesian against dk_x·dk_z ⟹ Ψ_cart = Ψ_pol/k = B·Φ/k⁴ | `directionalDensity` | ✓ the only polar→Cartesian step; the k⁴ convention is what makes the discrete realization (√(Ψ·s·dk²)) variance-carrying; see "conventions" below |
| Cox–Munk | mss = 0.003 + 0.00512·U₁₀ | `coxMunkMeanSquareSlope` | ✓ external published fit, exposed unmodified |

## Conventions audit (where silent wrongness hides)

1. **Polar vs Cartesian measure.** The paper's Ψ is a polar density (variance = ∬Ψ k dk dθ). A
   2-D FFT realization needs a Cartesian density (variance = ∬Ψ_cart dk_x dk_z = ∬Ψ k dk dθ
   ⟹ Ψ_cart = Ψ_pol/k). With the curvature-spectrum definition above this is B·Φ/k⁴. Both
   `radialMoment` and `directionalDensity` agree with this bookkeeping, and the omnidirectional
   moment's independence from Δ and θw (the second harmonic integrates to zero over the full
   circle) is asserted.
2. **Full-tail integration band.** The Cox–Munk comparison integrates k ∈ [0.01, km] — the paper's
   own comparison band — in log space (Simpson over ln k), which matches the integrand's smoothness;
   panel count 256 keeps the quadrature error orders below the 25% external band.
3. **Friction closure divergence (declared, deliberate).** The independent implementation includes
   the viscous roughness term (0.11ν/u*); production uses Charnock alone. Both are published
   forms of the closure family; the divergence is ~0.1% on u* at the declared U₁₀ and is *carried*
   by the 15% production-comparison gate rather than hidden. This is exactly the kind of
   independent-authoring difference the two-implementation design is supposed to surface and
   bound, not suppress.

## What this review did NOT verify

- The equation numbers themselves against the journal text (see provenance above).
- The Cox–Munk constants against Cox & Munk (1954) directly; they enter only through the
  published fit as quoted by the Elfouhaily paper, which is the point of the anchor.
- Nothing here was read out of `src/spectrum.ts` or any production table; the production
  comparison is commit 2's business and is gated at 15%.

**Review verdict: the independent implementation is a faithful, independently structured
expression of the published equations as restated by the stage spec and the reviewed lineage,
externally anchored by Cox–Munk, with its one deliberate divergence (viscous roughness term)
declared above. Cleared to serve as the commit-2 oracle.**
