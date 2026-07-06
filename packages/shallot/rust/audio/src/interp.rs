/// 4-point, 3rd-order cubic Hermite interpolation, x-form (Niemitalo,
/// "Polynomial Interpolators for High-Quality Resampling of Oversampled Audio").
/// `t` is the fractional position in [0, 1] between `y0` and `y1`; `ym1` and
/// `y2` are the flanking samples. Catmull-Rom tangents (central differences):
/// interpolates `y0` at t=0, `y1` at t=1, and reproduces any polynomial up to
/// degree 2 exactly. Replaces linear interpolation for sample playback and
/// wavetable reads — the extra two taps band-limit the pitch-shift aliasing
/// linear interpolation leaves behind.
pub fn hermite4(ym1: f32, y0: f32, y1: f32, y2: f32, t: f32) -> f32 {
    let c0 = y0;
    let c1 = 0.5 * (y1 - ym1);
    let c2 = ym1 - 2.5 * y0 + 2.0 * y1 - 0.5 * y2;
    let c3 = 0.5 * (y2 - ym1) + 1.5 * (y0 - y1);
    ((c3 * t + c2) * t + c1) * t + c0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolates_endpoints() {
        // t=0 returns y0, t=1 returns y1 — the defining property of an
        // interpolant (vs. an approximant). Exact, not toleranced.
        for &(ym1, y0, y1, y2) in &[(0.3, 1.0, -2.0, 0.5), (-1.0, 0.0, 4.0, 2.0)] {
            assert_eq!(hermite4(ym1, y0, y1, y2, 0.0), y0);
            assert!((hermite4(ym1, y0, y1, y2, 1.0) - y1).abs() < 1e-6);
        }
    }

    #[test]
    fn reproduces_polynomials_up_to_quadratic() {
        // The analytic parity: a 4-point 3rd-order Hermite reproduces any
        // polynomial of degree <= 2 sampled on its stencil (indices -1,0,1,2),
        // so it returns the exact polynomial value at the fractional position.
        // Tolerance is f32 roundoff over ~10 flops at magnitudes up to ~14
        // (a + 2b + 4c with coeffs ~2): ~14 * 10 * 1.2e-7 ≈ 2e-5, rounded up.
        let tol = 5e-5;
        for &(a, b, c) in &[(0.5, 1.3, 0.0), (-0.7, 0.4, 1.1), (2.0, -1.5, 0.8)] {
            let p = |x: f32| a + b * x + c * x * x;
            let (ym1, y0, y1, y2) = (p(-1.0), p(0.0), p(1.0), p(2.0));
            for step in 0..=10 {
                let t = step as f32 / 10.0;
                let got = hermite4(ym1, y0, y1, y2, t);
                assert!(
                    (got - p(t)).abs() < tol,
                    "quadratic ({a},{b},{c}) at t={t}: got {got}, want {}",
                    p(t)
                );
            }
        }
    }

    #[test]
    fn hermite_golden() {
        // Bit-check against captured Niemitalo x-form output. Production uses the same
        // canonical arrangement, so agreement is f32 roundoff over ~8 flops at the
        // stencil magnitudes (≤ 8): 8·8·2⁻²³ ≈ 4e-6, rounded to 1e-5.
        let tol = 1e-5;
        for &(ym1, y0, y1, y2, t, want) in crate::golden::HERMITE {
            let got = hermite4(ym1, y0, y1, y2, t);
            assert!(
                (got - want).abs() <= tol,
                "hermite4({ym1}, {y0}, {y1}, {y2}, {t}) = {got}, golden {want}, Δ{}",
                (got - want).abs()
            );
        }
    }

    #[test]
    fn flat_input_is_flat() {
        // Constant signal interpolates to the constant — no overshoot.
        for step in 0..=10 {
            let t = step as f32 / 10.0;
            assert!((hermite4(0.7, 0.7, 0.7, 0.7, t) - 0.7).abs() < 1e-6);
        }
    }
}
