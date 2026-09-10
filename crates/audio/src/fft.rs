use core::f32::consts::TAU;

pub const FFT_SIZE: usize = 256;
pub const SPECTRUM_SIZE: usize = FFT_SIZE / 2 + 1;

const HALF: usize = FFT_SIZE / 2;
const LOG2_HALF: usize = 7;

// Per-stage twiddles, destrided so each radix-2 stage reads a contiguous run
// instead of the strided `k * step` a single shared table forces. Stage `s`
// (butterfly span `len = 2^(s+1)`, half-span `half_len = 2^s`) owns the
// `half_len` slots at offset `2^s - 1`; the slot count over the 7 stages sums to
// `HALF - 1`. Contiguity is what lets the butterfly loop pull 4 twiddles with one
// `v128_load` (Stage 1.4) — a strided table would need a gather wasm SIMD lacks.
const STAGE_TW_LEN: usize = HALF - 1;

fn stage_offset(s: usize) -> usize {
    (1 << s) - 1
}

pub struct FftPlan {
    stage_re: [f32; STAGE_TW_LEN],
    stage_im: [f32; STAGE_TW_LEN],
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

/// One radix-2 butterfly: `z[b]` is rotated by the twiddle `(wr, wi)` and
/// combined with `z[a]` in place. The scalar form — shared by the native build
/// and the SIMD build's sub-4 stages — so all paths agree bit-for-bit.
#[inline(always)]
fn bfly1(zr: &mut [f32; HALF], zi: &mut [f32; HALF], a: usize, b: usize, wr: f32, wi: f32) {
    let tr = wr * zr[b] - wi * zi[b];
    let ti = wr * zi[b] + wi * zr[b];
    zr[b] = zr[a] - tr;
    zi[b] = zi[a] - ti;
    zr[a] += tr;
    zi[a] += ti;
}

/// One butterfly group of a radix-2 DIT stage: `half_len` butterflies pairing
/// `start + k` with `start + half_len + k`, twiddle `tw_re[k] + sign·i·tw_im[k]`
/// (`sign = -1` for the inverse transform). Both halves and the twiddle run are
/// contiguous, so the `half_len >= 4` stages process 4 butterflies per `f32x4`;
/// the sub-4 stages (half_len 1, 2) fall to the scalar [`bfly1`].
///
/// Hand-vectorized (Stage 1.4): the autovectorizer leaves `fft_core` scalar (the
/// bit-reversed strides of the enclosing loops defeat it), but each group's inner
/// loop is contiguous once the twiddles are destrided. The math is elementwise —
/// no cross-lane reduction — so each SIMD lane equals [`bfly1`] value-for-value,
/// and native (scalar) vs wasm (SIMD) stay inside the FMA floor the native-vs-wasm
/// parity differential already budgets. Reference: kissfft `kf_bfly2`.
#[cfg(target_feature = "simd128")]
#[inline]
fn butterfly(
    zr: &mut [f32; HALF],
    zi: &mut [f32; HALF],
    start: usize,
    half_len: usize,
    tw_re: &[f32],
    tw_im: &[f32],
    sign: f32,
) {
    use core::arch::wasm32::{f32x4_add, f32x4_mul, f32x4_splat, f32x4_sub, v128_load, v128_store};
    if half_len < 4 {
        for k in 0..half_len {
            bfly1(
                zr,
                zi,
                start + k,
                start + half_len + k,
                tw_re[k],
                sign * tw_im[k],
            );
        }
        return;
    }
    let signv = f32x4_splat(sign);
    // SAFETY: a in [start, start+half_len), b in [start+half_len, start+2·half_len),
    // both inside [0, HALF); k steps by 4 and half_len is a multiple of 4, so every
    // 4-wide load/store stays in bounds. tw_re/tw_im have len == half_len.
    // simd128 is the shipped wasm target feature (`.cargo/config.toml`); v128_load
    // is unaligned-safe.
    unsafe {
        let (zrp, zip) = (zr.as_mut_ptr(), zi.as_mut_ptr());
        let (trp, tip) = (tw_re.as_ptr(), tw_im.as_ptr());
        let mut k = 0;
        while k < half_len {
            let a = start + k;
            let b = start + half_len + k;
            let wr = v128_load(trp.add(k).cast());
            let wi = f32x4_mul(signv, v128_load(tip.add(k).cast()));
            let zbr = v128_load(zrp.add(b).cast());
            let zbi = v128_load(zip.add(b).cast());
            let tr = f32x4_sub(f32x4_mul(wr, zbr), f32x4_mul(wi, zbi));
            let ti = f32x4_add(f32x4_mul(wr, zbi), f32x4_mul(wi, zbr));
            let zar = v128_load(zrp.add(a).cast());
            let zai = v128_load(zip.add(a).cast());
            v128_store(zrp.add(b).cast(), f32x4_sub(zar, tr));
            v128_store(zip.add(b).cast(), f32x4_sub(zai, ti));
            v128_store(zrp.add(a).cast(), f32x4_add(zar, tr));
            v128_store(zip.add(a).cast(), f32x4_add(zai, ti));
            k += 4;
        }
    }
}

#[cfg(not(target_feature = "simd128"))]
#[inline]
fn butterfly(
    zr: &mut [f32; HALF],
    zi: &mut [f32; HALF],
    start: usize,
    half_len: usize,
    tw_re: &[f32],
    tw_im: &[f32],
    sign: f32,
) {
    for k in 0..half_len {
        bfly1(
            zr,
            zi,
            start + k,
            start + half_len + k,
            tw_re[k],
            sign * tw_im[k],
        );
    }
}

impl FftPlan {
    pub fn new() -> Self {
        let mut stage_re = [0.0f32; STAGE_TW_LEN];
        let mut stage_im = [0.0f32; STAGE_TW_LEN];
        let mut len = 2usize;
        for s in 0..LOG2_HALF {
            let half_len = len / 2;
            let off = stage_offset(s);
            for k in 0..half_len {
                let angle = -TAU * k as f32 / len as f32;
                stage_re[off + k] = angle.cos();
                stage_im[off + k] = angle.sin();
            }
            len *= 2;
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
            stage_re,
            stage_im,
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

        let sign = if inverse { -1.0 } else { 1.0 };
        let mut len = 2usize;
        for s in 0..LOG2_HALF {
            let half_len = len / 2;
            let off = stage_offset(s);
            let tw_re = &self.stage_re[off..off + half_len];
            let tw_im = &self.stage_im[off..off + half_len];
            for start in (0..HALF).step_by(len) {
                butterfly(zr, zi, start, half_len, tw_re, tw_im, sign);
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
