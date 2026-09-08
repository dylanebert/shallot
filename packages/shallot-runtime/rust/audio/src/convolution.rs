use crate::fft::{FftPlan, FFT_SIZE, SPECTRUM_SIZE};
use crate::BLOCK_SIZE;

pub const MAX_IR_SAMPLES: usize = 3528;
pub const MAX_IR_BLOCKS: usize = (MAX_IR_SAMPLES + BLOCK_SIZE - 1) / BLOCK_SIZE;

fn zeroed_spectra() -> Box<[[f32; SPECTRUM_SIZE]; MAX_IR_BLOCKS]> {
    vec![[0.0f32; SPECTRUM_SIZE]; MAX_IR_BLOCKS]
        .into_boxed_slice()
        .try_into()
        .ok()
        .unwrap()
}

pub struct Convolver {
    ir_re: Box<[[f32; SPECTRUM_SIZE]; MAX_IR_BLOCKS]>,
    ir_im: Box<[[f32; SPECTRUM_SIZE]; MAX_IR_BLOCKS]>,
    ir_prev_re: Box<[[f32; SPECTRUM_SIZE]; MAX_IR_BLOCKS]>,
    ir_prev_im: Box<[[f32; SPECTRUM_SIZE]; MAX_IR_BLOCKS]>,
    dry_re: Box<[[f32; SPECTRUM_SIZE]; MAX_IR_BLOCKS]>,
    dry_im: Box<[[f32; SPECTRUM_SIZE]; MAX_IR_BLOCKS]>,
    dry_index: usize,
    prev_tail: [f32; BLOCK_SIZE],
    num_blocks: usize,
    prev_num_blocks: usize,
    crossfade_active: bool,
}

fn accumulate(
    dry_re: &[[f32; SPECTRUM_SIZE]; MAX_IR_BLOCKS],
    dry_im: &[[f32; SPECTRUM_SIZE]; MAX_IR_BLOCKS],
    dry_index: usize,
    ir_re: &[[f32; SPECTRUM_SIZE]; MAX_IR_BLOCKS],
    ir_im: &[[f32; SPECTRUM_SIZE]; MAX_IR_BLOCKS],
    num_blocks: usize,
    accum_re: &mut [f32; SPECTRUM_SIZE],
    accum_im: &mut [f32; SPECTRUM_SIZE],
) {
    accum_re.fill(0.0);
    accum_im.fill(0.0);
    for j in 0..num_blocks {
        let di = (dry_index + j) % MAX_IR_BLOCKS;
        for i in 0..SPECTRUM_SIZE {
            accum_re[i] += dry_re[di][i] * ir_re[j][i] - dry_im[di][i] * ir_im[j][i];
            accum_im[i] += dry_re[di][i] * ir_im[j][i] + dry_im[di][i] * ir_re[j][i];
        }
    }
}

impl Convolver {
    pub fn new() -> Self {
        Convolver {
            ir_re: zeroed_spectra(),
            ir_im: zeroed_spectra(),
            ir_prev_re: zeroed_spectra(),
            ir_prev_im: zeroed_spectra(),
            dry_re: zeroed_spectra(),
            dry_im: zeroed_spectra(),
            dry_index: 0,
            prev_tail: [0.0; BLOCK_SIZE],
            num_blocks: 0,
            prev_num_blocks: 0,
            crossfade_active: false,
        }
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn reset(&mut self) {
        for block in self.dry_re.iter_mut() {
            block.fill(0.0);
        }
        for block in self.dry_im.iter_mut() {
            block.fill(0.0);
        }
        self.dry_index = 0;
        self.prev_tail.fill(0.0);
        self.crossfade_active = false;
    }

    pub fn update_ir(&mut self, ir: &[f32], plan: &FftPlan) {
        self.prev_num_blocks = self.num_blocks;
        core::mem::swap(&mut self.ir_re, &mut self.ir_prev_re);
        core::mem::swap(&mut self.ir_im, &mut self.ir_prev_im);

        let len = ir.len().min(MAX_IR_SAMPLES);
        self.num_blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

        let mut padded = [0.0f32; FFT_SIZE];
        for j in 0..self.num_blocks {
            padded.fill(0.0);
            let start = j * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(len);
            padded[..end - start].copy_from_slice(&ir[start..end]);
            let mut re = [0.0f32; SPECTRUM_SIZE];
            let mut im = [0.0f32; SPECTRUM_SIZE];
            plan.rfft(&padded, &mut re, &mut im);
            self.ir_re[j] = re;
            self.ir_im[j] = im;
        }
        for j in self.num_blocks..MAX_IR_BLOCKS {
            self.ir_re[j].fill(0.0);
            self.ir_im[j].fill(0.0);
        }

        self.crossfade_active = true;
    }

    pub fn process(
        &mut self,
        input: &[f32; BLOCK_SIZE],
        output: &mut [f32; BLOCK_SIZE],
        plan: &FftPlan,
    ) {
        if self.num_blocks == 0 && !self.crossfade_active {
            output.fill(0.0);
            return;
        }

        let mut input_256 = [0.0f32; FFT_SIZE];
        input_256[..BLOCK_SIZE].copy_from_slice(&self.prev_tail);
        input_256[BLOCK_SIZE..].copy_from_slice(input);
        self.prev_tail.copy_from_slice(input);

        self.dry_index = (self.dry_index + MAX_IR_BLOCKS - 1) % MAX_IR_BLOCKS;

        let mut dry_re_tmp = [0.0f32; SPECTRUM_SIZE];
        let mut dry_im_tmp = [0.0f32; SPECTRUM_SIZE];
        plan.rfft(&input_256, &mut dry_re_tmp, &mut dry_im_tmp);
        let di = self.dry_index;
        self.dry_re[di] = dry_re_tmp;
        self.dry_im[di] = dry_im_tmp;

        let mut accum_re = [0.0f32; SPECTRUM_SIZE];
        let mut accum_im = [0.0f32; SPECTRUM_SIZE];
        accumulate(
            &self.dry_re,
            &self.dry_im,
            self.dry_index,
            &self.ir_re,
            &self.ir_im,
            self.num_blocks,
            &mut accum_re,
            &mut accum_im,
        );
        let mut time_buf = [0.0f32; FFT_SIZE];
        plan.irfft(&accum_re, &accum_im, &mut time_buf);

        if self.crossfade_active {
            let mut prev_accum_re = [0.0f32; SPECTRUM_SIZE];
            let mut prev_accum_im = [0.0f32; SPECTRUM_SIZE];
            accumulate(
                &self.dry_re,
                &self.dry_im,
                self.dry_index,
                &self.ir_prev_re,
                &self.ir_prev_im,
                self.prev_num_blocks,
                &mut prev_accum_re,
                &mut prev_accum_im,
            );
            let mut prev_time = [0.0f32; FFT_SIZE];
            plan.irfft(&prev_accum_re, &prev_accum_im, &mut prev_time);

            for i in 0..BLOCK_SIZE {
                let t = i as f32 / BLOCK_SIZE as f32;
                output[i] = prev_time[BLOCK_SIZE + i] * (1.0 - t) + time_buf[BLOCK_SIZE + i] * t;
            }
            self.crossfade_active = false;
        } else {
            for i in 0..BLOCK_SIZE {
                output[i] = time_buf[BLOCK_SIZE + i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_ir_passthrough() {
        let plan = FftPlan::new();
        let mut conv = Convolver::new();
        let mut ir = [0.0f32; BLOCK_SIZE];
        ir[0] = 1.0;
        conv.update_ir(&ir, &plan);
        conv.crossfade_active = false;

        let mut input = [0.0f32; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            input[i] = (i as f32 * 0.1).sin();
        }
        let mut out = [0.0f32; BLOCK_SIZE];
        conv.process(&input, &mut out, &plan);

        for i in 0..BLOCK_SIZE {
            assert!(
                (input[i] - out[i]).abs() < 1e-4,
                "sample {i}: expected {}, got {}",
                input[i],
                out[i],
            );
        }
    }

    #[test]
    fn delayed_delta() {
        let plan = FftPlan::new();
        let mut conv = Convolver::new();

        let delay = 64usize;
        let mut ir = vec![0.0f32; delay + 1];
        ir[delay] = 1.0;
        conv.update_ir(&ir, &plan);
        conv.crossfade_active = false;

        let mut input = [0.0f32; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            input[i] = (i as f32 * 0.1).sin();
        }
        let zeros = [0.0f32; BLOCK_SIZE];
        let mut out1 = [0.0f32; BLOCK_SIZE];
        conv.process(&input, &mut out1, &plan);
        let mut out2 = [0.0f32; BLOCK_SIZE];
        conv.process(&zeros, &mut out2, &plan);

        for i in delay..BLOCK_SIZE {
            assert!(
                (out1[i] - input[i - delay]).abs() < 1e-4,
                "out1[{i}]: expected {}, got {}",
                input[i - delay],
                out1[i],
            );
        }
        for i in 0..delay {
            assert!(
                (out2[i] - input[BLOCK_SIZE - delay + i]).abs() < 1e-4,
                "out2[{i}]: expected {}, got {}",
                input[BLOCK_SIZE - delay + i],
                out2[i],
            );
        }
    }

    #[test]
    fn reset_clears_stale_history() {
        let plan = FftPlan::new();
        let mut conv = Convolver::new();
        let mut ir = [0.0f32; BLOCK_SIZE];
        ir[0] = 1.0;
        conv.update_ir(&ir, &plan);
        conv.crossfade_active = false;

        let loud = [1.0f32; BLOCK_SIZE];
        let mut out = [0.0f32; BLOCK_SIZE];
        for _ in 0..4 {
            conv.process(&loud, &mut out, &plan);
        }

        conv.reset();

        let silent = [0.0f32; BLOCK_SIZE];
        conv.process(&silent, &mut out, &plan);
        let energy: f32 = out.iter().map(|s| s * s).sum();
        assert!(
            energy < 1e-8,
            "output after reset should be silent, got energy {energy}",
        );
    }

    #[test]
    fn reset_then_fresh_input_clean() {
        let plan = FftPlan::new();
        let mut conv = Convolver::new();
        let mut ir = [0.0f32; BLOCK_SIZE];
        ir[0] = 1.0;
        conv.update_ir(&ir, &plan);
        conv.crossfade_active = false;

        let loud = [1.0f32; BLOCK_SIZE];
        let mut out = [0.0f32; BLOCK_SIZE];
        for _ in 0..4 {
            conv.process(&loud, &mut out, &plan);
        }

        conv.reset();

        let mut input = [0.0f32; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            input[i] = (i as f32 * 0.1).sin();
        }
        conv.process(&input, &mut out, &plan);

        for i in 0..BLOCK_SIZE {
            assert!(
                (input[i] - out[i]).abs() < 1e-4,
                "sample {i}: expected {}, got {} — stale history leaked through",
                input[i],
                out[i],
            );
        }
    }

    #[test]
    fn reset_during_crossfade() {
        let plan = FftPlan::new();
        let mut conv = Convolver::new();
        let mut ir1 = [0.0f32; BLOCK_SIZE];
        ir1[0] = 1.0;
        conv.update_ir(&ir1, &plan);
        conv.crossfade_active = false;

        let input = [0.5f32; BLOCK_SIZE];
        let mut out = [0.0f32; BLOCK_SIZE];
        for _ in 0..4 {
            conv.process(&input, &mut out, &plan);
        }

        let mut ir2 = [0.0f32; BLOCK_SIZE];
        ir2[0] = 0.5;
        conv.update_ir(&ir2, &plan);
        assert!(conv.crossfade_active);

        conv.reset();
        assert!(!conv.crossfade_active);

        let silent = [0.0f32; BLOCK_SIZE];
        conv.process(&silent, &mut out, &plan);
        let energy: f32 = out.iter().map(|s| s * s).sum();
        assert!(
            energy < 1e-6,
            "reset during crossfade should produce silence, got energy {energy}",
        );
    }

    #[test]
    fn multi_block_ir_clean_after_reset() {
        let plan = FftPlan::new();
        let mut conv = Convolver::new();

        let mut ir = vec![0.0f32; BLOCK_SIZE * 3];
        ir[0] = 0.5;
        ir[BLOCK_SIZE] = 0.3;
        ir[BLOCK_SIZE * 2] = 0.1;
        conv.update_ir(&ir, &plan);
        conv.crossfade_active = false;

        let loud = [1.0f32; BLOCK_SIZE];
        let mut out = [0.0f32; BLOCK_SIZE];
        for _ in 0..8 {
            conv.process(&loud, &mut out, &plan);
        }

        conv.reset();

        let silent = [0.0f32; BLOCK_SIZE];
        for _ in 0..4 {
            conv.process(&silent, &mut out, &plan);
            let energy: f32 = out.iter().map(|s| s * s).sum();
            assert!(
                energy < 1e-8,
                "multi-block IR: stale data after reset, energy {energy}",
            );
        }
    }

    #[test]
    fn output_bounded_by_ir_and_input() {
        let plan = FftPlan::new();
        let mut conv = Convolver::new();
        let mut ir = [0.0f32; BLOCK_SIZE];
        ir[0] = 1.0;
        conv.update_ir(&ir, &plan);
        conv.crossfade_active = false;

        let input = [0.5f32; BLOCK_SIZE];
        let mut out = [0.0f32; BLOCK_SIZE];
        for _ in 0..10 {
            conv.process(&input, &mut out, &plan);
            for i in 0..BLOCK_SIZE {
                assert!(
                    out[i].abs() <= 1.01,
                    "output {:.4} exceeds input×IR bound at sample {i}",
                    out[i],
                );
            }
        }
    }

    #[test]
    fn crossfade_no_clicks() {
        let plan = FftPlan::new();
        let mut conv = Convolver::new();

        let mut ir1 = [0.0f32; BLOCK_SIZE];
        ir1[0] = 1.0;
        conv.update_ir(&ir1, &plan);
        conv.crossfade_active = false;

        let input = [0.5f32; BLOCK_SIZE];
        let mut out = [0.0f32; BLOCK_SIZE];
        for _ in 0..4 {
            conv.process(&input, &mut out, &plan);
        }

        let mut ir2 = [0.0f32; BLOCK_SIZE];
        ir2[0] = 0.5;
        conv.update_ir(&ir2, &plan);
        conv.process(&input, &mut out, &plan);

        let mut max_diff = 0.0f32;
        for i in 1..BLOCK_SIZE {
            let diff = (out[i] - out[i - 1]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(
            max_diff < 0.1,
            "max consecutive diff {max_diff} exceeds click threshold",
        );
    }

    // Module 9: energy preservation and IR update tests

    #[test]
    fn energy_preservation_broadband() {
        let plan = FftPlan::new();
        let mut conv = Convolver::new();

        // Use a simple IR: delta at 0 with gain 0.8
        let mut ir = [0.0f32; BLOCK_SIZE];
        ir[0] = 0.8;
        conv.update_ir(&ir, &plan);
        conv.crossfade_active = false;

        // Broadband input (noise-like)
        let mut input = [0.0f32; BLOCK_SIZE];
        let mut seed = 0x12345678u32;
        for i in 0..BLOCK_SIZE {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            input[i] = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }

        let input_energy: f32 = input.iter().map(|s| s * s).sum();
        let ir_energy: f32 = ir.iter().map(|s| s * s).sum();

        let mut out = [0.0f32; BLOCK_SIZE];
        conv.process(&input, &mut out, &plan);
        let output_energy: f32 = out.iter().map(|s| s * s).sum();

        // For a delta IR with gain g, output energy ≈ input_energy * g²
        let expected = input_energy * ir_energy;
        let ratio = output_energy / expected;
        assert!(
            (ratio - 1.0).abs() < 0.1,
            "energy ratio {ratio}: output={output_energy}, expected={expected}",
        );
    }

    fn brute_force_convolve(signal: &[f32], ir: &[f32], output: &mut [f32]) {
        for i in 0..output.len() {
            let mut sum = 0.0f32;
            for j in 0..ir.len() {
                if i >= j {
                    sum += signal[i - j] * ir[j];
                }
            }
            output[i] = sum;
        }
    }

    #[test]
    fn multi_block_noise_ir_matches_brute_force() {
        let plan = FftPlan::new();
        let mut conv = Convolver::new();

        let ir_len = BLOCK_SIZE * 3 + 50;
        let mut ir = vec![0.0f32; ir_len];
        let mut seed = 0xDEADBEEFu32;
        for i in 0..ir_len {
            seed ^= seed.wrapping_shl(13);
            seed ^= seed >> 17;
            seed ^= seed.wrapping_shl(5);
            ir[i] = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0;
            ir[i] *= (-2.0 * i as f32 / ir_len as f32).exp();
        }

        conv.update_ir(&ir, &plan);
        conv.crossfade_active = false;

        let num_blocks = 8;
        let total_samples = num_blocks * BLOCK_SIZE;
        let mut full_input = vec![0.0f32; total_samples];
        seed = 0x12345678;
        for i in 0..total_samples {
            seed ^= seed.wrapping_shl(13);
            seed ^= seed >> 17;
            seed ^= seed.wrapping_shl(5);
            full_input[i] = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }

        let mut fft_output = vec![0.0f32; total_samples];
        for b in 0..num_blocks {
            let start = b * BLOCK_SIZE;
            let mut input_block = [0.0f32; BLOCK_SIZE];
            input_block.copy_from_slice(&full_input[start..start + BLOCK_SIZE]);
            let mut out_block = [0.0f32; BLOCK_SIZE];
            conv.process(&input_block, &mut out_block, &plan);
            fft_output[start..start + BLOCK_SIZE].copy_from_slice(&out_block);
        }

        let mut brute_output = vec![0.0f32; total_samples];
        brute_force_convolve(&full_input, &ir, &mut brute_output);

        let mut max_err = 0.0f32;
        let mut max_err_idx = 0;
        for i in 0..total_samples {
            let err = (fft_output[i] - brute_output[i]).abs();
            if err > max_err {
                max_err = err;
                max_err_idx = i;
            }
        }
        assert!(
            max_err < 0.01,
            "FFT convolution differs from brute force at sample {max_err_idx}: \
             fft={:.6} brute={:.6} err={max_err:.6}",
            fft_output[max_err_idx],
            brute_output[max_err_idx],
        );
    }

    #[test]
    fn kick_transient_through_noise_ir_no_spikes() {
        let plan = FftPlan::new();
        let mut conv = Convolver::new();

        let ir_len = MAX_IR_SAMPLES;
        let mut ir = vec![0.0f32; ir_len];
        let mut seed = 0xCAFEBABEu32;
        for i in 0..ir_len {
            seed ^= seed.wrapping_shl(13);
            seed ^= seed >> 17;
            seed ^= seed.wrapping_shl(5);
            ir[i] = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0;
            ir[i] *= (-3.0 * i as f32 / ir_len as f32).exp();
        }
        let energy: f32 = ir.iter().map(|s| s * s).sum();
        let scale = 1.0 / energy.sqrt();
        for s in ir.iter_mut() {
            *s *= scale;
        }

        conv.update_ir(&ir, &plan);
        conv.crossfade_active = false;

        let num_blocks = 60;
        let mut prev_peak = 0.0f32;
        for b in 0..num_blocks {
            let mut input = [0.0f32; BLOCK_SIZE];
            if b < 5 {
                let env = if b == 0 {
                    0.5
                } else {
                    0.7 * (-3.0 * b as f32 / 5.0).exp()
                };
                for i in 0..BLOCK_SIZE {
                    let phase = (b * BLOCK_SIZE + i) as f32 * 440.0 / 44100.0;
                    input[i] = (core::f32::consts::TAU * phase).sin() * env;
                }
            }
            let mut out = [0.0f32; BLOCK_SIZE];
            conv.process(&input, &mut out, &plan);

            let peak: f32 = out.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
            let jump = peak - prev_peak;
            assert!(
                jump < 0.5,
                "block {b}: output peak jumped {prev_peak:.4} → {peak:.4} (Δ{jump:.4}) — possible artifact",
            );
            prev_peak = peak;
        }
    }

    #[test]
    fn ir_update_during_active_convolution_no_clicks() {
        let plan = FftPlan::new();
        let mut conv = Convolver::new();

        let mut ir1 = [0.0f32; BLOCK_SIZE];
        ir1[0] = 1.0;
        conv.update_ir(&ir1, &plan);
        conv.crossfade_active = false;

        let input = [0.3f32; BLOCK_SIZE];
        let mut out = [0.0f32; BLOCK_SIZE];

        // Warm up
        for _ in 0..4 {
            conv.process(&input, &mut out, &plan);
        }

        // Update IR to very different value during active convolution
        let mut ir2 = [0.0f32; BLOCK_SIZE];
        ir2[0] = 0.1;
        conv.update_ir(&ir2, &plan);

        // Process the crossfade block
        conv.process(&input, &mut out, &plan);

        let mut max_diff = 0.0f32;
        for i in 1..BLOCK_SIZE {
            let diff = (out[i] - out[i - 1]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(
            max_diff < 0.15,
            "IR update caused click: max consecutive diff {max_diff}",
        );
    }
}
