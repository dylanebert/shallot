use crate::interp::hermite4;
use core::f32::consts::TAU;
use std::sync::OnceLock;

pub const WAVETABLE_FRAMES: usize = 64;
pub const WAVETABLE_SAMPLES: usize = 2048;

/// Full-cycle sine lookup table for the [`Waveform::Sine`] oscillator. Phase is
/// already in `[0,1)`, so the read is a free argument reduction + one linear
/// interpolation — no `libm` software `sin` per sample (the shipped wasm has no
/// hardware sine). A full cycle (no quarter-wave folding) keeps the read
/// branchless; 1024 segments interpolated linearly is transparent (THD below the
/// f32 numerical floor, measured in `sine_thd_below_floor`). The +1 guard sample
/// (`table[N] == table[0]`) lets the lerp read `i+1` without a wrap.
const SIN_LUT_N: usize = 1024;

pub(crate) fn sin_lut() -> &'static [f32; SIN_LUT_N + 1] {
    static LUT: OnceLock<[f32; SIN_LUT_N + 1]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0.0f32; SIN_LUT_N + 1];
        for (i, v) in t.iter_mut().enumerate() {
            *v = (i as f32 / SIN_LUT_N as f32 * TAU).sin();
        }
        t
    })
}

#[inline(always)]
fn sin_lut_read(table: &[f32], phase: f32) -> f32 {
    let x = phase * SIN_LUT_N as f32;
    let i = x as usize;
    let frac = x - i as f32;
    table[i] + (table[i + 1] - table[i]) * frac
}

#[derive(Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum Waveform {
    Sine = 0,
    Saw = 1,
    Square = 2,
    Triangle = 3,
    Wavetable = 4,
}

impl Waveform {
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => Self::Saw,
            2 => Self::Square,
            3 => Self::Triangle,
            4 => Self::Wavetable,
            _ => Self::Sine,
        }
    }
}

pub fn poly_blep(t: f32, dt: f32) -> f32 {
    if t < dt {
        let t = t / dt;
        2.0 * t - t * t - 1.0
    } else if t > 1.0 - dt {
        let t = (t - 1.0) / dt;
        t * t + 2.0 * t + 1.0
    } else {
        0.0
    }
}

/// Band-limited square (pw = 0.5, ±1 amplitude) via PolyBLEP at both edges.
/// Shared by the square waveform and the triangle's leaky integrator.
fn poly_blep_square(phase: f32, dt: f32) -> f32 {
    let naive = if phase < 0.5 { 1.0 } else { -1.0 };
    naive + poly_blep(phase, dt) - poly_blep((phase + 0.5) % 1.0, dt)
}

/// `tri` is the triangle oscillator's leaky-integrator state (per oscillator
/// node); unused for the other waveforms.
pub fn oscillator(
    waveform: Waveform,
    phase: f32,
    dt: f32,
    sin_table: &[f32],
    wavetable: &[f32],
    wavetable_pos: f32,
    tri: &mut f32,
) -> f32 {
    match waveform {
        Waveform::Sine => sin_lut_read(sin_table, phase),
        Waveform::Saw => {
            let naive = 2.0 * phase - 1.0;
            naive - poly_blep(phase, dt)
        }
        Waveform::Square => poly_blep_square(phase, dt),
        Waveform::Triangle => {
            // Band-limited triangle = leaky-integrated band-limited square
            // (DaisySP WAVE_POLYBLEP_TRI): y[n] = dt*x[n] + (1-dt)*y[n-1],
            // scaled by 4 to recover unit amplitude after integration.
            *tri = dt * poly_blep_square(phase, dt) + (1.0 - dt) * *tri;
            *tri * 4.0
        }
        Waveform::Wavetable => wavetable_read(wavetable, wavetable_pos, phase),
    }
}

pub fn wavetable_read(table: &[f32], position: f32, phase: f32) -> f32 {
    let pos = position.clamp(0.0, 1.0) * (WAVETABLE_FRAMES - 1) as f32;
    let frame_lo = pos as usize;
    let frame_hi = (frame_lo + 1).min(WAVETABLE_FRAMES - 1);
    let frame_frac = pos - frame_lo as f32;

    let sample_pos = phase * (WAVETABLE_SAMPLES - 1) as f32;
    let idx = sample_pos as usize;
    let sample_frac = sample_pos - idx as f32;

    // Hermite in the phase dimension (the audio-rate read, where linear interp
    // aliases on pitch shift); the table is one cycle, so taps wrap. The frame
    // dimension stays a linear crossfade — a slow morph, not an audio-rate read.
    let s0 = wavetable_frame(table, frame_lo, idx, sample_frac);
    let s1 = wavetable_frame(table, frame_hi, idx, sample_frac);
    s0 + (s1 - s0) * frame_frac
}

fn wavetable_frame(table: &[f32], frame: usize, idx: usize, frac: f32) -> f32 {
    let base = frame * WAVETABLE_SAMPLES;
    let n = WAVETABLE_SAMPLES;
    let ym1 = table[base + (idx + n - 1) % n];
    let y0 = table[base + idx % n];
    let y1 = table[base + (idx + 1) % n];
    let y2 = table[base + (idx + 2) % n];
    hermite4(ym1, y0, y1, y2, frac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_thd_below_floor() {
        // The gate Stage 2 demands: a dirtier sine approximation must be caught.
        // Drive the oscillator at exactly k0 cycles over the FFT window
        // (dt = k0/FFT_SIZE) so the fundamental and every harmonic land on exact
        // bins — no leakage, no window — then sum harmonic energy vs the
        // fundamental. The full-cycle LUT reads -141 dB here, which is the f32
        // 256-pt FFT's own numerical floor: the LUT adds no resolvable harmonics.
        // The bound is -120 dB: >20 dB of headroom over the LUT for cross-platform
        // FFT-floor jitter, while rejecting any real approximation (the degree-7
        // minimax polynomial weighed against the LUT measured -74 dB on a clean
        // high-resolution DFT — it would fail this by ~46 dB).
        use crate::fft::{FftPlan, FFT_SIZE, SPECTRUM_SIZE};
        let plan = FftPlan::new();
        let empty = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];
        let table = sin_lut();
        let k0 = 3usize;
        let dt = k0 as f32 / FFT_SIZE as f32;
        let mut samples = [0.0f32; FFT_SIZE];
        let mut phase = 0.0f32;
        for s in samples.iter_mut() {
            *s = oscillator(Waveform::Sine, phase, dt, table, &empty, 0.0, &mut 0.0);
            phase += dt;
            if phase >= 1.0 {
                phase -= 1.0;
            }
        }
        let mut re = [0.0f32; SPECTRUM_SIZE];
        let mut im = [0.0f32; SPECTRUM_SIZE];
        plan.rfft(&samples, &mut re, &mut im);
        let energy = |k: usize| re[k] * re[k] + im[k] * im[k];
        let fund = energy(k0);
        let mut harm = 0.0f32;
        let mut k = 2 * k0;
        while k < SPECTRUM_SIZE {
            harm += energy(k);
            k += k0;
        }
        let thd_db = 10.0 * (harm / fund).log10();
        assert!(
            thd_db < -120.0,
            "sine THD {thd_db:.1} dB exceeds -120 dB — the sine approximation added audible harmonics"
        );
    }

    #[test]
    fn sine_in_range() {
        let empty = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];
        for i in 0..100 {
            let phase = i as f32 / 100.0;
            let out = oscillator(
                Waveform::Sine,
                phase,
                0.01,
                sin_lut(),
                &empty,
                0.0,
                &mut 0.0,
            );
            assert!(
                out >= -1.0 && out <= 1.0,
                "sine out of range: {out} at phase {phase}"
            );
        }
    }

    #[test]
    fn saw_in_range() {
        let empty = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];
        for i in 0..100 {
            let phase = i as f32 / 100.0;
            let out = oscillator(Waveform::Saw, phase, 0.01, sin_lut(), &empty, 0.0, &mut 0.0);
            assert!(
                out >= -1.1 && out <= 1.1,
                "saw out of range: {out} at phase {phase}"
            );
        }
    }

    #[test]
    fn square_in_range() {
        let empty = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];
        for i in 0..100 {
            let phase = i as f32 / 100.0;
            let out = oscillator(
                Waveform::Square,
                phase,
                0.01,
                sin_lut(),
                &empty,
                0.0,
                &mut 0.0,
            );
            assert!(
                out >= -1.1 && out <= 1.1,
                "square out of range: {out} at phase {phase}"
            );
        }
    }

    #[test]
    fn triangle_settled_amplitude() {
        // The leaky integrator carries state across samples, so the triangle
        // is only meaningful over a continuous sweep. Its fundamental gain is
        // 1/2π regardless of dt, which the ×4 normalizes to ~unit peak at any
        // pitch (DaisySP's constant). Check the settled peak, past the startup
        // transient of the integrator's ~70 Hz pole.
        let empty = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];
        let dt = 440.0 / 48000.0;
        let mut phase = 0.0f32;
        let mut tri = 0.0f32;
        let mut peak = 0.0f32;
        for i in 0..4096 {
            let out = oscillator(
                Waveform::Triangle,
                phase,
                dt,
                sin_lut(),
                &empty,
                0.0,
                &mut tri,
            );
            if i > 2048 {
                peak = peak.max(out.abs());
            }
            phase += dt;
            if phase >= 1.0 {
                phase -= 1.0;
            }
        }
        assert!(
            (0.85..=1.15).contains(&peak),
            "settled triangle peak should be ~unit, got {peak}"
        );
    }

    #[test]
    fn poly_blep_zero_in_middle() {
        assert!((poly_blep(0.5, 0.01) - 0.0).abs() < 1e-6);
        assert!((poly_blep(0.3, 0.01) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn poly_blep_nonzero_at_boundaries() {
        assert!(poly_blep(0.005, 0.01).abs() > 0.0);
        assert!(poly_blep(0.995, 0.01).abs() > 0.0);
    }

    #[test]
    fn poly_blep_formula_parity() {
        let dt = 0.01f32;
        for i in 0..1000 {
            let t = i as f32 / 1000.0;
            let ours = poly_blep(t, dt);
            let daisysp = if t < dt {
                let tn = t / dt;
                tn + tn - tn * tn - 1.0
            } else if t > 1.0 - dt {
                let tn = (t - 1.0) / dt;
                tn * tn + tn + tn + 1.0
            } else {
                0.0
            };
            assert!(
                (ours - daisysp).abs() < 1e-10,
                "poly_blep mismatch at t={t}: ours={ours}, daisysp={daisysp}"
            );
        }
    }

    #[test]
    fn poly_blep_golden() {
        // Bit-check the band-limit kernel against captured DaisySP `Polyblep` output
        // (oscillator.cpp). Production substitutes `2*t` for `t+t` (IEEE-identical)
        // and reassociates the additive terms, so it agrees to ~1 ulp. There are no
        // transcendentals on this path, so the tolerance is pure arrangement roundoff:
        // f32 ulp over ~5 flops at |v| ≤ 1 (≈ 5·2⁻²³), rounded up to 1e-6.
        let tol = 1e-6;
        for &(t, dt, want) in crate::golden::POLY_BLEP {
            let got = poly_blep(t, dt);
            assert!(
                (got - want).abs() <= tol,
                "poly_blep({t}, {dt}) = {got}, golden {want}, Δ{}",
                (got - want).abs()
            );
        }
    }

    #[test]
    fn saw_spectral_rolloff() {
        use crate::fft::{FftPlan, FFT_SIZE, SPECTRUM_SIZE};
        let plan = FftPlan::new();
        let empty = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];
        let dt = 440.0 / 48000.0;
        let mut samples = [0.0f32; FFT_SIZE];
        let mut phase = 0.0f32;
        for i in 0..FFT_SIZE {
            samples[i] = oscillator(Waveform::Saw, phase, dt, sin_lut(), &empty, 0.0, &mut 0.0);
            phase += dt;
            if phase >= 1.0 {
                phase -= 1.0;
            }
        }
        let mut re = [0.0f32; SPECTRUM_SIZE];
        let mut im = [0.0f32; SPECTRUM_SIZE];
        plan.rfft(&samples, &mut re, &mut im);

        let fund_bin = (440.0 / 48000.0 * FFT_SIZE as f32).round() as usize;
        let fund_energy = re[fund_bin] * re[fund_bin] + im[fund_bin] * im[fund_bin];

        let nyquist_half_freq = 48000.0 / 4.0;
        let nyquist_half_bin = (nyquist_half_freq / 48000.0 * FFT_SIZE as f32).round() as usize;
        let mut high_energy = 0.0f32;
        for k in nyquist_half_bin..SPECTRUM_SIZE {
            high_energy += re[k] * re[k] + im[k] * im[k];
        }
        let ratio_db = 10.0 * (fund_energy / high_energy.max(1e-30)).log10();
        assert!(ratio_db > 18.0, "PolyBLEP saw should attenuate above Nyquist/2 by >18dB vs fundamental, got {ratio_db:.1}dB");
    }

    #[test]
    fn saw_sign_convention() {
        let empty = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];
        let dt = 0.01;
        let phase = 0.5;
        let out = oscillator(Waveform::Saw, phase, dt, sin_lut(), &empty, 0.0, &mut 0.0);
        let naive = 2.0 * phase - 1.0;
        assert!(
            (out - naive).abs() < 1e-6,
            "at mid-cycle, saw={out} should equal naive={naive}"
        );
    }

    #[test]
    fn square_blep_at_transitions() {
        let empty = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];
        let dt = 0.01;

        let near_zero = oscillator(
            Waveform::Square,
            0.005,
            dt,
            sin_lut(),
            &empty,
            0.0,
            &mut 0.0,
        );
        let naive_zero = 1.0;
        assert!(
            (near_zero - naive_zero).abs() > 0.01,
            "PolyBLEP should modify square near phase=0"
        );

        let near_half = oscillator(
            Waveform::Square,
            0.505,
            dt,
            sin_lut(),
            &empty,
            0.0,
            &mut 0.0,
        );
        let naive_half = -1.0;
        assert!(
            (near_half - naive_half).abs() > 0.01,
            "PolyBLEP should modify square near phase=0.5"
        );

        let mid = oscillator(Waveform::Square, 0.25, dt, sin_lut(), &empty, 0.0, &mut 0.0);
        assert!(
            (mid - 1.0).abs() < 1e-6,
            "PolyBLEP should be zero at phase=0.25, got {mid}"
        );
    }

    #[test]
    fn square_daisysp_scale_comparison() {
        let empty = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];
        let dt = 440.0 / 48000.0;
        let mut rms_sum = 0.0f32;
        let n = 2048;
        let mut phase = 0.0f32;
        for _ in 0..n {
            let s = oscillator(
                Waveform::Square,
                phase,
                dt,
                sin_lut(),
                &empty,
                0.0,
                &mut 0.0,
            );
            rms_sum += s * s;
            phase += dt;
            if phase >= 1.0 {
                phase -= 1.0;
            }
        }
        let rms = (rms_sum / n as f32).sqrt();
        assert!(
            (rms - 1.0).abs() < 0.1,
            "our square RMS should be ~1.0 (no 0.707 scale), got {rms}"
        );
    }

    #[test]
    fn triangle_odd_harmonics() {
        // A triangle's Fourier series is odd harmonics only, amplitude ∝ 1/n².
        // The leaky-integrated band-limited square reproduces that structure:
        // the even harmonic vanishes and H3/H1 ≈ 1/9. dt = 375/48000 gives a
        // 128-sample period, so the 256-point window holds exactly 2 cycles
        // (fundamental in bin 2) — no spectral leakage, clean ratios.
        use crate::fft::{FftPlan, FFT_SIZE, SPECTRUM_SIZE};
        let plan = FftPlan::new();
        let empty = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];
        let dt = 375.0 / 48000.0;
        let mut phase = 0.0f32;
        let mut tri = 0.0f32;
        for _ in 0..2048 {
            oscillator(
                Waveform::Triangle,
                phase,
                dt,
                sin_lut(),
                &empty,
                0.0,
                &mut tri,
            );
            phase += dt;
            if phase >= 1.0 {
                phase -= 1.0;
            }
        }
        let mut samples = [0.0f32; FFT_SIZE];
        for s in samples.iter_mut() {
            *s = oscillator(
                Waveform::Triangle,
                phase,
                dt,
                sin_lut(),
                &empty,
                0.0,
                &mut tri,
            );
            phase += dt;
            if phase >= 1.0 {
                phase -= 1.0;
            }
        }
        let mut re = [0.0f32; SPECTRUM_SIZE];
        let mut im = [0.0f32; SPECTRUM_SIZE];
        plan.rfft(&samples, &mut re, &mut im);
        let mag = |k: usize| (re[k] * re[k] + im[k] * im[k]).sqrt();
        let fund_bin = (dt * FFT_SIZE as f32).round() as usize;
        let h1 = mag(fund_bin);
        let h2 = mag(2 * fund_bin);
        let h3 = mag(3 * fund_bin);
        assert!(
            h2 / h1 < 0.05,
            "triangle has no even harmonics: H2/H1={}",
            h2 / h1
        );
        let r = h3 / h1;
        assert!(
            (0.08..0.14).contains(&r),
            "triangle H3/H1 should be ~1/9: {r}"
        );
    }

    #[test]
    fn wavetable_interpolation() {
        let mut table = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];
        for i in 0..WAVETABLE_SAMPLES {
            table[i] = 1.0;
        }
        let out = wavetable_read(&table, 0.0, 0.5);
        assert!((out - 1.0).abs() < 1e-6);

        let out_interp = wavetable_read(&table, 0.5, 0.5);
        assert!(out_interp >= 0.0 && out_interp <= 1.0);
    }
}
