use core::f32::consts::TAU;

pub const FFT_SIZE: usize = 256;
pub const SPECTRUM_SIZE: usize = FFT_SIZE / 2 + 1;

const HALF: usize = FFT_SIZE / 2;
const LOG2_HALF: usize = 7;

pub struct FftPlan {
    twiddle_re: [f32; HALF / 2],
    twiddle_im: [f32; HALF / 2],
    bit_rev: [u8; HALF],
    unscramble_re: [f32; SPECTRUM_SIZE],
    unscramble_im: [f32; SPECTRUM_SIZE],
}

fn bit_reverse(mut x: u32, bits: u32) -> u32 {
    let mut result = 0u32;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

impl FftPlan {
    pub fn new() -> Self {
        let mut twiddle_re = [0.0f32; HALF / 2];
        let mut twiddle_im = [0.0f32; HALF / 2];
        for k in 0..HALF / 2 {
            let angle = -TAU * k as f32 / HALF as f32;
            twiddle_re[k] = angle.cos();
            twiddle_im[k] = angle.sin();
        }

        let mut bit_rev = [0u8; HALF];
        for i in 0..HALF {
            bit_rev[i] = bit_reverse(i as u32, LOG2_HALF as u32) as u8;
        }

        let mut unscramble_re = [0.0f32; SPECTRUM_SIZE];
        let mut unscramble_im = [0.0f32; SPECTRUM_SIZE];
        for k in 0..SPECTRUM_SIZE {
            let angle = -TAU * k as f32 / FFT_SIZE as f32;
            unscramble_re[k] = angle.cos();
            unscramble_im[k] = angle.sin();
        }

        FftPlan {
            twiddle_re,
            twiddle_im,
            bit_rev,
            unscramble_re,
            unscramble_im,
        }
    }

    fn fft_core(&self, zr: &mut [f32; HALF], zi: &mut [f32; HALF], inverse: bool) {
        for i in 0..HALF {
            let j = self.bit_rev[i] as usize;
            if i < j {
                zr.swap(i, j);
                zi.swap(i, j);
            }
        }

        let mut len = 2usize;
        for _ in 0..LOG2_HALF {
            let half_len = len / 2;
            let step = HALF / len;
            for start in (0..HALF).step_by(len) {
                for k in 0..half_len {
                    let tw = k * step;
                    let wr = self.twiddle_re[tw];
                    let wi = if inverse {
                        -self.twiddle_im[tw]
                    } else {
                        self.twiddle_im[tw]
                    };
                    let a = start + k;
                    let b = a + half_len;
                    let tr = wr * zr[b] - wi * zi[b];
                    let ti = wr * zi[b] + wi * zr[b];
                    zr[b] = zr[a] - tr;
                    zi[b] = zi[a] - ti;
                    zr[a] += tr;
                    zi[a] += ti;
                }
            }
            len *= 2;
        }
    }

    pub fn rfft(
        &self,
        input: &[f32; FFT_SIZE],
        out_re: &mut [f32; SPECTRUM_SIZE],
        out_im: &mut [f32; SPECTRUM_SIZE],
    ) {
        let mut zr = [0.0f32; HALF];
        let mut zi = [0.0f32; HALF];
        for n in 0..HALF {
            zr[n] = input[2 * n];
            zi[n] = input[2 * n + 1];
        }

        self.fft_core(&mut zr, &mut zi, false);

        for k in 0..=HALF {
            let ak = if k < HALF { k } else { 0 };
            let m = if k == 0 || k == HALF { 0 } else { HALF - k };
            let (ar, ai) = (zr[ak], zi[ak]);
            let (mr, mi) = (zr[m], zi[m]);

            let xe_r = 0.5 * (ar + mr);
            let xe_i = 0.5 * (ai - mi);
            let xo_r = 0.5 * (ai + mi);
            let xo_i = 0.5 * (mr - ar);

            let wr = self.unscramble_re[k];
            let wi = self.unscramble_im[k];
            out_re[k] = xe_r + wr * xo_r - wi * xo_i;
            out_im[k] = xe_i + wr * xo_i + wi * xo_r;
        }
    }

    pub fn irfft(
        &self,
        in_re: &[f32; SPECTRUM_SIZE],
        in_im: &[f32; SPECTRUM_SIZE],
        output: &mut [f32; FFT_SIZE],
    ) {
        let mut zr = [0.0f32; HALF];
        let mut zi = [0.0f32; HALF];

        for k in 0..HALF {
            let (xm_re, xm_im) = if k == 0 {
                (in_re[HALF], in_im[HALF])
            } else {
                (in_re[HALF - k], -in_im[HALF - k])
            };

            let xe_r = 0.5 * (in_re[k] + xm_re);
            let xe_i = 0.5 * (in_im[k] + xm_im);
            let diff_r = 0.5 * (in_re[k] - xm_re);
            let diff_i = 0.5 * (in_im[k] - xm_im);

            let wr = self.unscramble_re[k];
            let wi = self.unscramble_im[k];
            let xo_r = diff_r * wr + diff_i * wi;
            let xo_i = diff_i * wr - diff_r * wi;

            zr[k] = xe_r - xo_i;
            zi[k] = xe_i + xo_r;
        }

        self.fft_core(&mut zr, &mut zi, true);

        let inv = 1.0 / HALF as f32;
        for n in 0..HALF {
            output[2 * n] = zr[n] * inv;
            output[2 * n + 1] = zi[n] * inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let plan = FftPlan::new();
        let mut input = [0.0f32; FFT_SIZE];
        for i in 0..FFT_SIZE {
            input[i] = (i as f32 * 0.1).sin() + (i as f32 * 0.037).cos();
        }
        let mut re = [0.0f32; SPECTRUM_SIZE];
        let mut im = [0.0f32; SPECTRUM_SIZE];
        plan.rfft(&input, &mut re, &mut im);
        let mut output = [0.0f32; FFT_SIZE];
        plan.irfft(&re, &im, &mut output);

        for i in 0..FFT_SIZE {
            assert!(
                (input[i] - output[i]).abs() < 1e-5,
                "sample {i}: expected {}, got {}",
                input[i],
                output[i]
            );
        }
    }

    #[test]
    fn dc_signal() {
        let plan = FftPlan::new();
        let input = [1.0f32; FFT_SIZE];
        let mut re = [0.0f32; SPECTRUM_SIZE];
        let mut im = [0.0f32; SPECTRUM_SIZE];
        plan.rfft(&input, &mut re, &mut im);

        assert!((re[0] - FFT_SIZE as f32).abs() < 1e-3);
        assert!(im[0].abs() < 1e-3);
        for k in 1..SPECTRUM_SIZE {
            assert!(re[k].abs() < 1e-3, "bin {k}: re={}", re[k]);
            assert!(im[k].abs() < 1e-3, "bin {k}: im={}", im[k]);
        }
    }

    #[test]
    fn pure_tone_single_bin() {
        let plan = FftPlan::new();
        let mut input = [0.0f32; FFT_SIZE];
        let k = 16;
        for n in 0..FFT_SIZE {
            input[n] = (TAU * k as f32 * n as f32 / FFT_SIZE as f32).sin();
        }
        let mut re = [0.0f32; SPECTRUM_SIZE];
        let mut im = [0.0f32; SPECTRUM_SIZE];
        plan.rfft(&input, &mut re, &mut im);

        let peak_energy = re[k] * re[k] + im[k] * im[k];
        let mut other_energy = 0.0f32;
        for i in 0..SPECTRUM_SIZE {
            if i != k {
                other_energy += re[i] * re[i] + im[i] * im[i];
            }
        }
        assert!(
            peak_energy > other_energy * 1000.0,
            "energy should concentrate at bin {k}: peak={peak_energy}, other={other_energy}"
        );
    }

    #[test]
    fn pure_tone_phase() {
        let plan = FftPlan::new();
        let k = 10;

        let mut cos_input = [0.0f32; FFT_SIZE];
        for n in 0..FFT_SIZE {
            cos_input[n] = (TAU * k as f32 * n as f32 / FFT_SIZE as f32).cos();
        }
        let mut re = [0.0f32; SPECTRUM_SIZE];
        let mut im = [0.0f32; SPECTRUM_SIZE];
        plan.rfft(&cos_input, &mut re, &mut im);

        let expected_re = FFT_SIZE as f32 / 2.0;
        assert!(
            (re[k] - expected_re).abs() < 1.0,
            "cos: re[{k}]={}, expected ~{expected_re}",
            re[k]
        );
        assert!(im[k].abs() < 1.0, "cos: im[{k}]={}, expected ~0", im[k]);

        let mut sin_input = [0.0f32; FFT_SIZE];
        for n in 0..FFT_SIZE {
            sin_input[n] = (TAU * k as f32 * n as f32 / FFT_SIZE as f32).sin();
        }
        plan.rfft(&sin_input, &mut re, &mut im);
        assert!(re[k].abs() < 1.0, "sin: re[{k}]={}, expected ~0", re[k]);
        let expected_im = -(FFT_SIZE as f32) / 2.0;
        assert!(
            (im[k] - expected_im).abs() < 1.0,
            "sin: im[{k}]={}, expected ~{expected_im}",
            im[k]
        );
    }

    #[test]
    fn linearity() {
        let plan = FftPlan::new();
        let mut x = [0.0f32; FFT_SIZE];
        let mut y = [0.0f32; FFT_SIZE];
        let mut combo = [0.0f32; FFT_SIZE];
        let a = 2.5f32;
        let b = -1.3f32;
        for n in 0..FFT_SIZE {
            x[n] = (n as f32 * 0.1).sin();
            y[n] = (n as f32 * 0.037).cos();
            combo[n] = a * x[n] + b * y[n];
        }
        let mut xr = [0.0f32; SPECTRUM_SIZE];
        let mut xi = [0.0f32; SPECTRUM_SIZE];
        let mut yr = [0.0f32; SPECTRUM_SIZE];
        let mut yi = [0.0f32; SPECTRUM_SIZE];
        let mut cr = [0.0f32; SPECTRUM_SIZE];
        let mut ci = [0.0f32; SPECTRUM_SIZE];
        plan.rfft(&x, &mut xr, &mut xi);
        plan.rfft(&y, &mut yr, &mut yi);
        plan.rfft(&combo, &mut cr, &mut ci);

        for k in 0..SPECTRUM_SIZE {
            let expected_re = a * xr[k] + b * yr[k];
            let expected_im = a * xi[k] + b * yi[k];
            assert!(
                (cr[k] - expected_re).abs() < 1e-3,
                "linearity re[{k}]: {} vs {expected_re}",
                cr[k]
            );
            assert!(
                (ci[k] - expected_im).abs() < 1e-3,
                "linearity im[{k}]: {} vs {expected_im}",
                ci[k]
            );
        }
    }

    #[test]
    fn dc_and_nyquist_real() {
        let plan = FftPlan::new();
        let mut input = [0.0f32; FFT_SIZE];
        for n in 0..FFT_SIZE {
            input[n] = (n as f32 * 0.3).sin() + (n as f32 * 0.7).cos() + 0.5;
        }
        let mut re = [0.0f32; SPECTRUM_SIZE];
        let mut im = [0.0f32; SPECTRUM_SIZE];
        plan.rfft(&input, &mut re, &mut im);

        assert!(
            im[0].abs() < 1e-4,
            "DC imaginary should be 0, got {}",
            im[0]
        );
        assert!(
            im[HALF].abs() < 1e-4,
            "Nyquist imaginary should be 0, got {}",
            im[HALF]
        );
    }

    #[test]
    fn known_transform_two_cosines() {
        let plan = FftPlan::new();
        let k1 = 5usize;
        let k2 = 30usize;
        let mut input = [0.0f32; FFT_SIZE];
        for n in 0..FFT_SIZE {
            input[n] = (TAU * k1 as f32 * n as f32 / FFT_SIZE as f32).cos()
                + (TAU * k2 as f32 * n as f32 / FFT_SIZE as f32).cos();
        }
        let mut re = [0.0f32; SPECTRUM_SIZE];
        let mut im = [0.0f32; SPECTRUM_SIZE];
        plan.rfft(&input, &mut re, &mut im);

        let energy_k1 = re[k1] * re[k1] + im[k1] * im[k1];
        let energy_k2 = re[k2] * re[k2] + im[k2] * im[k2];
        let mut other_energy = 0.0f32;
        for i in 0..SPECTRUM_SIZE {
            if i != k1 && i != k2 {
                other_energy += re[i] * re[i] + im[i] * im[i];
            }
        }
        assert!(energy_k1 > other_energy * 100.0, "bin {k1} should dominate");
        assert!(energy_k2 > other_energy * 100.0, "bin {k2} should dominate");
        assert!(
            (energy_k1 - energy_k2).abs() / energy_k1 < 0.01,
            "two cosines should have equal energy"
        );
    }

    #[test]
    fn inverse_scaling() {
        let plan = FftPlan::new();
        let mut re = [0.0f32; SPECTRUM_SIZE];
        let im = [0.0f32; SPECTRUM_SIZE];
        let k = 20;
        let amplitude = 3.0f32;
        re[k] = amplitude * (FFT_SIZE as f32 / 2.0);
        let mut output = [0.0f32; FFT_SIZE];
        plan.irfft(&re, &im, &mut output);

        for n in 0..FFT_SIZE {
            let expected = amplitude * (TAU * k as f32 * n as f32 / FFT_SIZE as f32).cos();
            assert!(
                (output[n] - expected).abs() < 0.05,
                "sample {n}: expected {expected}, got {}",
                output[n]
            );
        }
    }

    #[test]
    fn parseval() {
        let plan = FftPlan::new();
        let mut input = [0.0f32; FFT_SIZE];
        for i in 0..FFT_SIZE {
            input[i] = (i as f32 * 0.3).sin();
        }
        let mut re = [0.0f32; SPECTRUM_SIZE];
        let mut im = [0.0f32; SPECTRUM_SIZE];
        plan.rfft(&input, &mut re, &mut im);

        let time_energy: f32 = input.iter().map(|x| x * x).sum();
        let mut freq_energy = re[0] * re[0] + im[0] * im[0];
        for k in 1..HALF {
            freq_energy += 2.0 * (re[k] * re[k] + im[k] * im[k]);
        }
        freq_energy += re[HALF] * re[HALF] + im[HALF] * im[HALF];

        let expected = time_energy * FFT_SIZE as f32;
        assert!(
            (freq_energy - expected).abs() / expected < 1e-4,
            "Parseval: freq_energy={freq_energy}, expected={expected}"
        );
    }
}
