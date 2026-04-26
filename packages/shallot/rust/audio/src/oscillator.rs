use core::f32::consts::TAU;

pub const WAVETABLE_FRAMES: usize = 64;
pub const WAVETABLE_SAMPLES: usize = 2048;

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

pub fn oscillator(
    waveform: Waveform,
    phase: f32,
    dt: f32,
    wavetable: &[f32],
    wavetable_pos: f32,
) -> f32 {
    match waveform {
        Waveform::Sine => (phase * TAU).sin(),
        Waveform::Saw => {
            let naive = 2.0 * phase - 1.0;
            naive - poly_blep(phase, dt)
        }
        Waveform::Square => {
            let naive = if phase < 0.5 { 1.0 } else { -1.0 };
            let mut out = naive;
            out += poly_blep(phase, dt);
            out -= poly_blep((phase + 0.5) % 1.0, dt);
            out
        }
        Waveform::Triangle => 4.0 * (phase - 0.5).abs() - 1.0,
        Waveform::Wavetable => wavetable_read(wavetable, wavetable_pos, phase),
    }
}

pub fn wavetable_read(table: &[f32], position: f32, phase: f32) -> f32 {
    let pos = position.clamp(0.0, 1.0) * (WAVETABLE_FRAMES - 1) as f32;
    let frame_lo = pos as usize;
    let frame_hi = (frame_lo + 1).min(WAVETABLE_FRAMES - 1);
    let frame_frac = pos - frame_lo as f32;

    let sample_pos = phase * (WAVETABLE_SAMPLES - 1) as f32;
    let idx_lo = sample_pos as usize;
    let idx_hi = (idx_lo + 1) % WAVETABLE_SAMPLES;
    let sample_frac = sample_pos - idx_lo as f32;

    let offset_lo = frame_lo * WAVETABLE_SAMPLES;
    let offset_hi = frame_hi * WAVETABLE_SAMPLES;

    let s0 = table[offset_lo + idx_lo]
        + (table[offset_lo + idx_hi] - table[offset_lo + idx_lo]) * sample_frac;
    let s1 = table[offset_hi + idx_lo]
        + (table[offset_hi + idx_hi] - table[offset_hi + idx_lo]) * sample_frac;

    s0 + (s1 - s0) * frame_frac
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_in_range() {
        let empty = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];
        for i in 0..100 {
            let phase = i as f32 / 100.0;
            let out = oscillator(Waveform::Sine, phase, 0.01, &empty, 0.0);
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
            let out = oscillator(Waveform::Saw, phase, 0.01, &empty, 0.0);
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
            let out = oscillator(Waveform::Square, phase, 0.01, &empty, 0.0);
            assert!(
                out >= -1.1 && out <= 1.1,
                "square out of range: {out} at phase {phase}"
            );
        }
    }

    #[test]
    fn triangle_in_range() {
        let empty = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];
        for i in 0..100 {
            let phase = i as f32 / 100.0;
            let out = oscillator(Waveform::Triangle, phase, 0.01, &empty, 0.0);
            assert!(
                out >= -1.0 && out <= 1.0,
                "triangle out of range: {out} at phase {phase}"
            );
        }
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
    fn saw_spectral_rolloff() {
        use crate::fft::{FftPlan, FFT_SIZE, SPECTRUM_SIZE};
        let plan = FftPlan::new();
        let empty = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];
        let dt = 440.0 / 48000.0;
        let mut samples = [0.0f32; FFT_SIZE];
        let mut phase = 0.0f32;
        for i in 0..FFT_SIZE {
            samples[i] = oscillator(Waveform::Saw, phase, dt, &empty, 0.0);
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
        let out = oscillator(Waveform::Saw, phase, dt, &empty, 0.0);
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

        let near_zero = oscillator(Waveform::Square, 0.005, dt, &empty, 0.0);
        let naive_zero = 1.0;
        assert!(
            (near_zero - naive_zero).abs() > 0.01,
            "PolyBLEP should modify square near phase=0"
        );

        let near_half = oscillator(Waveform::Square, 0.505, dt, &empty, 0.0);
        let naive_half = -1.0;
        assert!(
            (near_half - naive_half).abs() > 0.01,
            "PolyBLEP should modify square near phase=0.5"
        );

        let mid = oscillator(Waveform::Square, 0.25, dt, &empty, 0.0);
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
            let s = oscillator(Waveform::Square, phase, dt, &empty, 0.0);
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
    fn triangle_no_blep() {
        let empty = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];
        let dt = 440.0 / 48000.0;
        let mut phase = 0.0f32;
        let mut prev = oscillator(Waveform::Triangle, phase, dt, &empty, 0.0);
        phase += dt;
        let max_diff = 4.0 * dt;
        for _ in 1..2048 {
            let cur = oscillator(Waveform::Triangle, phase, dt, &empty, 0.0);
            let diff = (cur - prev).abs();
            assert!(
                diff <= max_diff + 1e-6,
                "triangle consecutive diff {diff} exceeds 4*dt={max_diff}"
            );
            prev = cur;
            phase += dt;
            if phase >= 1.0 {
                phase -= 1.0;
            }
        }
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
