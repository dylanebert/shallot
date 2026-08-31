# Independent Elfouhaily source review

This review freezes the test-side implementation before it is used as the production oracle. The
reviewed source is `tests/elfouhaily-independent.ts`; it imports no production spectrum code.

**Primary source:** T. Elfouhaily, D. Vandemark, B. Chapron, K. Katsaros, *A unified directional
spectrum for long and short wind-driven waves*, JGR 102(C7), 1997,
DOI `10.1029/97JC00467`. The reviewed journal PDF is the stable
[Archimer copy](https://archimer.ifremer.fr/doc/00091/20226/17877.pdf), SHA-256
`50fe069ada389d3367590a2c99d9d5cfee474646948413012d95e2539fcff6a2` as retrieved 2026-08-31.

**External moment anchor:** the clean-surface Cox–Munk total mean-square-slope fit
`0.003 + 0.00512 U₁₀`, which the paper compares with its integrated spectrum in Figure 7b.

## Verified source loci

| Quantity | Journal locus | Independent implementation |
|---|---|---|
| directional density `Ψ = SΦ/k` | eq. (45) | `directionalDensity` converts `S = B/k³` to `BΦ/k⁴` |
| normalized angular factor | eqs. (46), (49) | `directionalFactor` |
| `B = B_l + B_h` | eq. (30) | `curvatureSpectrum` |
| long branch and side-effect factor | eqs. (31), (32) | `longWaveFactor` |
| `αp = 0.006 Ωc^0.55` | eq. (34) | `spectrumAmplitudes` |
| short branch and side-effect factor | eqs. (40), (41) | `shortWaveFactor` |
| two-regime `αm` | eq. (44) | `spectrumAmplitudes` |
| spreading `Δ` | eq. (57) | `spreadingDelta` |
| `a0 = ln(2)/4`, `ap = 4`, `am = 0.13 u*/cm` | grouped eq. (59) | `spreadingDelta` |
| neutral wind profile and `u* = √Cd10N U10` | eqs. (60), (61) | source boundary described below |
| spectral measures and dimensions | appendix eqs. (A2)–(A6), Table A1 | `directionalDensity`, `radialIntegral` |

The stage shorthand named the two speed-ratio terms in eq. (57) but omitted the additive `a0` and
coefficient `ap`. The oracle keeps all three grouped eq. (59) definitions explicit. Earlier notes that
assigned `ap` and `am` to eqs. (60) and (61) were refuted by the PDF: those numbers belong to the wind
profile and drag-coefficient relation.

## Reproducing the source anchors

Run `bun test ./packages/shallot-ocean/tests/elfouhaily-independent.test.ts` from the Shallot root.
That test derives, without production imports, the paper's `km = √(ρg/τ)` (about 368.6 rad/m),
`cm = √(2g/km)` (about 0.23 m/s), `Lpm(kp) = exp(−1.25)`, `Jp(kp) = γ = 1.7` at `Ωc = 0.9`, angular
normalization to one, both `αm` regimes, and the full-tail slope moments at four winds. These
source-derived anchors make the equation-locus review repeatable without vendoring the paper; the
production comparison imports the independent implementation only from its test module.

## Factor audit

- `KM` and `CM` implement the paper's gravity-capillary minimum-speed definitions.
- `peakEnvelope`, `longWaveFactor`, and `shortWaveFactor` keep `Lpm·Jp` in both branches while
  keeping the long-branch exponential out of `Fm`.
- `spectrumAmplitudes` carries both branches of eq. (44), and `curvatureSpectrum` applies the
  first-power `cp/c` and `cm/c` ratios from eqs. (31) and (40).
- `spreadingDelta` carries both the long- and short-wave directionality terms, including `am`.
- `directionalFactor` integrates to one; therefore full-circle height and total-slope moments are
  independent of the second harmonic. Directional mutation witnesses must read an angular row rather
  than the omnidirectional Cox–Munk integral.
- `directionalDensity` is a Cartesian density. The realization must multiply it by the Fourier-cell
  area `Δk²`; appendix (A2)–(A6) is the dimensional source for that conversion.

## Declared auxiliary closure

The paper specifies `u* = √Cd10N U10` in eq. (61) and refers neutral `Cd10N` to Garratt/Wu; it does
not prescribe the Charnock-plus-viscous fixed-point closure used here. The oracle declares that
closure rather than attributing it to the paper. Production is compared against the resulting values,
so any independently chosen friction closure remains visible at the 15% density gate and at the
wind-dependent Cox–Munk moment gate.

## Adversarial conclusions

The oracle is source-independent and dimensionally coherent. Its own tests pin both `Ωc` branches,
both `αm` branches, angular normalization, wind-axis anisotropy, the full-tail Cox–Munk moment, and
finite positive height variance. The production comparison must add directional rows: an
omnidirectional integral cannot witness removal of `am` or inversion of the spreading ratios because
the cosine harmonic integrates to zero. It must also include `Ωc > 1` rows so replacing `log10` with
`ln` cannot survive behind the `Ωc ≤ 1` branch.
