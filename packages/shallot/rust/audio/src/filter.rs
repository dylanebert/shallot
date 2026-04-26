#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u32)]
pub enum FilterMode {
    LowPass = 0,
    HighPass = 1,
    BandPass = 2,
    Notch = 3,
}

impl FilterMode {
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => Self::HighPass,
            2 => Self::BandPass,
            3 => Self::Notch,
            _ => Self::LowPass,
        }
    }
}

#[derive(Clone, Copy)]
pub struct SvfFilter {
    pub ic1eq: f32,
    pub ic2eq: f32,
    pub mode: FilterMode,
    pub q: f32,
    pub a1: f32,
    pub a2: f32,
    pub a3: f32,
    pub k: f32,
}

impl Default for SvfFilter {
    fn default() -> Self {
        Self {
            ic1eq: 0.0,
            ic2eq: 0.0,
            mode: FilterMode::LowPass,
            q: 0.707,
            a1: 0.0,
            a2: 0.0,
            a3: 0.0,
            k: 0.0,
        }
    }
}

impl SvfFilter {
    pub fn update_coefficients(&mut self, sample_rate: f32, cutoff: f32) {
        let freq = cutoff.clamp(20.0, sample_rate * 0.49);
        let g = (core::f32::consts::PI * freq / sample_rate).tan();
        self.k = 1.0 / self.q;
        self.a1 = 1.0 / (1.0 + g * (g + self.k));
        self.a2 = g * self.a1;
        self.a3 = g * self.a2;
    }

    pub fn tick(&mut self, v0: f32) -> f32 {
        let v3 = v0 - self.ic2eq;
        let v1 = self.a1 * self.ic1eq + self.a2 * v3;
        let v2 = self.ic2eq + self.a2 * self.ic1eq + self.a3 * v3;
        self.ic1eq = 2.0 * v1 - self.ic1eq;
        self.ic2eq = 2.0 * v2 - self.ic2eq;

        match self.mode {
            FilterMode::LowPass => v2,
            FilterMode::HighPass => v0 - self.k * v1 - v2,
            FilterMode::BandPass => v1,
            FilterMode::Notch => v0 - self.k * v1,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Biquad {
    pub(crate) b0: f32,
    pub(crate) b1: f32,
    pub(crate) b2: f32,
    pub(crate) a1: f32,
    pub(crate) a2: f32,
    xm1: f32,
    xm2: f32,
    ym1: f32,
    ym2: f32,
}

impl Biquad {
    pub fn passthrough() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            xm1: 0.0,
            xm2: 0.0,
            ym1: 0.0,
            ym2: 0.0,
        }
    }

    pub fn low_shelf(cutoff: f32, gain: f32, sr: f32) -> Self {
        let q = 0.707f32;
        let w0 = core::f32::consts::TAU * cutoff / sr;
        let cw0 = w0.cos();
        let sw0 = w0.sin();
        let alpha = sw0 / (2.0 * q);
        let a = gain.sqrt();
        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

        let a0 = (a + 1.0) + (a - 1.0) * cw0 + two_sqrt_a_alpha;
        let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cw0);
        let a2 = (a + 1.0) + (a - 1.0) * cw0 - two_sqrt_a_alpha;
        let b0 = a * ((a + 1.0) - (a - 1.0) * cw0 + two_sqrt_a_alpha);
        let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cw0);
        let b2 = a * ((a + 1.0) - (a - 1.0) * cw0 - two_sqrt_a_alpha);

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            xm1: 0.0,
            xm2: 0.0,
            ym1: 0.0,
            ym2: 0.0,
        }
    }

    pub fn high_shelf(cutoff: f32, gain: f32, sr: f32) -> Self {
        // Steam Audio computes high shelf in f64 (iir.cpp:188-206), then stores as f32.
        let q = 0.707f64;
        let w0 = core::f64::consts::TAU * cutoff as f64 / sr as f64;
        let cw0 = w0.cos();
        let sw0 = w0.sin();
        let alpha = sw0 / (2.0 * q);
        let a = (gain as f64).sqrt();
        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

        let a0 = (a + 1.0) - (a - 1.0) * cw0 + two_sqrt_a_alpha;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cw0);
        let a2 = (a + 1.0) - (a - 1.0) * cw0 - two_sqrt_a_alpha;
        let b0 = a * ((a + 1.0) + (a - 1.0) * cw0 + two_sqrt_a_alpha);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cw0);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cw0 - two_sqrt_a_alpha);

        Self {
            b0: (b0 / a0) as f32,
            b1: (b1 / a0) as f32,
            b2: (b2 / a0) as f32,
            a1: (a1 / a0) as f32,
            a2: (a2 / a0) as f32,
            xm1: 0.0,
            xm2: 0.0,
            ym1: 0.0,
            ym2: 0.0,
        }
    }

    pub fn peaking(low_cutoff: f32, high_cutoff: f32, gain: f32, sr: f32) -> Self {
        let center = (low_cutoff * high_cutoff).sqrt();
        let q_inverse = (high_cutoff - low_cutoff) / center;
        let w0 = core::f32::consts::TAU * center / sr;
        let cw0 = w0.cos();
        let sw0 = w0.sin();
        let alpha = sw0 * q_inverse / 2.0;
        let a = gain.sqrt();

        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * cw0;
        let a2 = 1.0 - alpha / a;
        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cw0;
        let b2 = 1.0 - alpha * a;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            xm1: 0.0,
            xm2: 0.0,
            ym1: 0.0,
            ym2: 0.0,
        }
    }

    pub fn tick(&mut self, input: f32) -> f32 {
        let x = input + 1e-9;
        let y = self.b0 * x + self.b1 * self.xm1 + self.b2 * self.xm2
            - self.a1 * self.ym1
            - self.a2 * self.ym2;
        self.xm2 = self.xm1;
        self.xm1 = x;
        self.ym2 = self.ym1;
        self.ym1 = y;
        y
    }

    pub fn set_coeffs(&mut self, other: &Biquad) {
        self.b0 = other.b0;
        self.b1 = other.b1;
        self.b2 = other.b2;
        self.a1 = other.a1;
        self.a2 = other.a2;
    }

    pub fn reset(&mut self) {
        self.xm1 = 0.0;
        self.xm2 = 0.0;
        self.ym1 = 0.0;
        self.ym2 = 0.0;
    }
}

#[derive(Clone, Copy)]
pub struct AllpassFilter {
    buffer: [f32; 1024],
    write_pos: usize,
    delay: usize,
    feedback: f32,
}

impl AllpassFilter {
    pub fn new(delay: usize, feedback: f32) -> Self {
        Self {
            buffer: [0.0; 1024],
            write_pos: 0,
            delay: delay.min(1024),
            feedback,
        }
    }

    pub fn tick(&mut self, input: f32) -> f32 {
        let read_pos = (self.write_pos + 1024 - self.delay) % 1024;
        let delayed = self.buffer[read_pos];
        let v = input - self.feedback * delayed;
        self.buffer[self.write_pos] = v;
        self.write_pos = (self.write_pos + 1) % 1024;
        delayed + self.feedback * v
    }

    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coefficient_update_sets_nonzero() {
        let mut f = SvfFilter::default();
        f.update_coefficients(44100.0, 1000.0);
        assert!(f.a1 != 0.0);
        assert!(f.a2 != 0.0);
        assert!(f.a3 != 0.0);
        assert!(f.k != 0.0);
    }

    #[test]
    fn tick_produces_output() {
        let mut f = SvfFilter::default();
        f.update_coefficients(44100.0, 1000.0);
        let out = f.tick(1.0);
        assert!(out != 0.0);
    }

    #[test]
    fn lowpass_attenuates_high_frequency() {
        let mut f = SvfFilter::default();
        f.mode = FilterMode::LowPass;
        f.update_coefficients(44100.0, 200.0);

        let mut energy = 0.0f32;
        let dt = 1.0 / 44100.0;
        for i in 0..4410 {
            let input = (2.0 * std::f32::consts::PI * 10000.0 * i as f32 * dt).sin();
            let out = f.tick(input);
            energy += out * out;
        }
        let rms = (energy / 4410.0).sqrt();
        assert!(
            rms < 0.1,
            "high-freq signal should be attenuated, got rms={rms}"
        );
    }

    #[test]
    fn mode_selection() {
        assert_eq!(FilterMode::from_u32(0), FilterMode::LowPass);
        assert_eq!(FilterMode::from_u32(1), FilterMode::HighPass);
        assert_eq!(FilterMode::from_u32(2), FilterMode::BandPass);
        assert_eq!(FilterMode::from_u32(3), FilterMode::Notch);
        assert_eq!(FilterMode::from_u32(99), FilterMode::LowPass);
    }

    #[test]
    fn highpass_attenuates_low_frequency() {
        let mut f = SvfFilter::default();
        f.mode = FilterMode::HighPass;
        f.update_coefficients(48000.0, 5000.0);

        let dt = 1.0 / 48000.0;
        let mut low_energy = 0.0f32;
        let mut high_energy = 0.0f32;

        for i in 0..48000 {
            let input = (2.0 * std::f32::consts::PI * 200.0 * i as f32 * dt).sin();
            let out = f.tick(input);
            low_energy += out * out;
        }
        f.ic1eq = 0.0;
        f.ic2eq = 0.0;
        f.update_coefficients(48000.0, 5000.0);
        for i in 0..48000 {
            let input = (2.0 * std::f32::consts::PI * 15000.0 * i as f32 * dt).sin();
            let out = f.tick(input);
            high_energy += out * out;
        }
        let ratio_db = 10.0 * (high_energy / low_energy.max(1e-30)).log10();
        assert!(
            ratio_db > 20.0,
            "HP@5kHz: 15kHz should pass, 200Hz attenuated >20dB, got {ratio_db:.1}dB"
        );
    }

    #[test]
    fn bandpass_passes_center() {
        let dt = 1.0 / 48000.0;
        let mut center_energy = 0.0f32;
        let mut off_energy = 0.0f32;

        for &(freq, is_center) in &[(2000.0f32, true), (500.0, false)] {
            let mut f = SvfFilter::default();
            f.mode = FilterMode::BandPass;
            f.q = 2.0;
            f.update_coefficients(48000.0, 2000.0);
            let mut energy = 0.0f32;
            for i in 0..48000 {
                let input = (2.0 * std::f32::consts::PI * freq * i as f32 * dt).sin();
                let out = f.tick(input);
                if i >= 4800 {
                    energy += out * out;
                }
            }
            if is_center {
                center_energy = energy;
            } else {
                off_energy = energy;
            }
        }
        assert!(
            center_energy > off_energy * 4.0,
            "BP@2kHz Q=2: center energy {center_energy} should be >4x off-center {off_energy}"
        );
    }

    #[test]
    fn notch_rejects_center() {
        let dt = 1.0 / 48000.0;
        let mut center_energy = 0.0f32;
        let mut flank_energy = 0.0f32;

        for &(freq, is_center) in &[(2000.0f32, true), (500.0, false)] {
            let mut f = SvfFilter::default();
            f.mode = FilterMode::Notch;
            f.q = 2.0;
            f.update_coefficients(48000.0, 2000.0);
            let mut energy = 0.0f32;
            for i in 0..48000 {
                let input = (2.0 * std::f32::consts::PI * freq * i as f32 * dt).sin();
                let out = f.tick(input);
                if i >= 4800 {
                    energy += out * out;
                }
            }
            if is_center {
                center_energy = energy;
            } else {
                flank_energy = energy;
            }
        }
        assert!(
            flank_energy > center_energy * 4.0,
            "Notch@2kHz: flank energy {flank_energy} should dominate center {center_energy}"
        );
    }

    #[test]
    fn resonance_peak_at_cutoff() {
        let dt = 1.0 / 48000.0;
        let mut cutoff_energy = 0.0f32;
        let mut passband_energy = 0.0f32;

        for &(freq, is_cutoff) in &[(1000.0f32, true), (100.0, false)] {
            let mut f = SvfFilter::default();
            f.mode = FilterMode::LowPass;
            f.q = 10.0;
            f.update_coefficients(48000.0, 1000.0);
            let mut energy = 0.0f32;
            for i in 0..48000 {
                let input = (2.0 * std::f32::consts::PI * freq * i as f32 * dt).sin();
                let out = f.tick(input);
                if i >= 4800 {
                    energy += out * out;
                }
            }
            if is_cutoff {
                cutoff_energy = energy;
            } else {
                passband_energy = energy;
            }
        }
        assert!(cutoff_energy > passband_energy * 3.0, "LP Q=10: resonance at cutoff ({cutoff_energy}) should be >3x passband ({passband_energy})");
    }

    #[test]
    fn svf_coefficient_stability_at_nyquist() {
        let mut f = SvfFilter::default();
        f.mode = FilterMode::LowPass;
        f.update_coefficients(48000.0, 23000.0);
        let dt = 1.0 / 48000.0;
        for i in 0..4800 {
            let input = (2.0 * std::f32::consts::PI * 1000.0 * i as f32 * dt).sin();
            let out = f.tick(input);
            assert!(
                out.is_finite(),
                "filter blew up at sample {i} with cutoff near Nyquist"
            );
        }
    }

    #[test]
    fn svf_coefficient_stability_low_freq() {
        let mut f = SvfFilter::default();
        f.mode = FilterMode::LowPass;
        f.update_coefficients(48000.0, 20.0);
        let dt = 1.0 / 48000.0;
        let mut energy = 0.0f32;
        for i in 0..48000 {
            let input = (2.0 * std::f32::consts::PI * 10000.0 * i as f32 * dt).sin();
            let out = f.tick(input);
            assert!(
                out.is_finite(),
                "filter blew up at sample {i} with 20Hz cutoff"
            );
            if i >= 4800 {
                energy += out * out;
            }
        }
        let rms = (energy / 43200.0).sqrt();
        assert!(
            rms < 0.01,
            "LP@20Hz should nearly zero broadband input, rms={rms}"
        );
    }

    #[test]
    fn allpass_phase_shift_only() {
        let dt = 1.0 / 48000.0;
        let mut ratios = Vec::new();
        for &freq in &[500.0f32, 3000.0] {
            let mut ap = AllpassFilter::new(225, 0.5);
            let mut in_energy = 0.0f32;
            let mut out_energy = 0.0f32;
            for i in 0..48000 {
                let input = (2.0 * std::f32::consts::PI * freq * i as f32 * dt).sin();
                let output = ap.tick(input);
                if i >= 4800 {
                    in_energy += input * input;
                    out_energy += output * output;
                }
            }
            ratios.push(out_energy / in_energy);
        }
        let diff = (ratios[0] - ratios[1]).abs();
        assert!(
            diff < 0.05,
            "allpass energy ratio should be same for different freqs: {:.4} vs {:.4}",
            ratios[0],
            ratios[1]
        );
    }

    #[test]
    fn allpass_preserves_energy() {
        let mut ap = AllpassFilter::new(225, 0.5);
        let dt = 1.0 / 44100.0;
        let mut in_energy = 0.0f32;
        let mut out_energy = 0.0f32;
        for i in 0..44100 {
            let input = (2.0 * std::f32::consts::PI * 1000.0 * i as f32 * dt).sin();
            let output = ap.tick(input);
            in_energy += input * input;
            out_energy += output * output;
        }
        let ratio = out_energy / in_energy;
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "allpass should preserve energy, got ratio {ratio}",
        );
    }

    #[test]
    fn four_allpass_cascade_energy() {
        let mut allpasses = [
            AllpassFilter::new(225, 0.5),
            AllpassFilter::new(341, 0.5),
            AllpassFilter::new(441, 0.5),
            AllpassFilter::new(556, 0.5),
        ];
        let dt = 1.0 / 44100.0;
        let mut in_energy = 0.0f32;
        let mut out_energy = 0.0f32;
        for i in 0..44100 {
            let input = (2.0 * std::f32::consts::PI * 1000.0 * i as f32 * dt).sin();
            let mut signal = input;
            for ap in allpasses.iter_mut() {
                signal = ap.tick(signal);
            }
            in_energy += input * input;
            out_energy += signal * signal;
        }
        let ratio = out_energy / in_energy;
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "4-allpass cascade should preserve energy, got ratio {ratio}",
        );
    }

    fn measure_biquad_energy(bq: &mut Biquad, freq: f32, sr: f32, samples: usize) -> f32 {
        let dt = 1.0 / sr;
        let warmup = (sr * 0.1) as usize;
        for i in 0..warmup {
            let input = (std::f32::consts::TAU * freq * i as f32 * dt).sin();
            bq.tick(input);
        }
        let mut energy = 0.0f32;
        for i in 0..samples {
            let input = (std::f32::consts::TAU * freq * (warmup + i) as f32 * dt).sin();
            let out = bq.tick(input);
            energy += out * out;
        }
        energy / samples as f32
    }

    #[test]
    fn biquad_passthrough() {
        for constructor in [
            |_: f32| Biquad::low_shelf(800.0, 1.0, 48000.0),
            |_: f32| Biquad::high_shelf(8000.0, 1.0, 48000.0),
            |_: f32| Biquad::peaking(800.0, 8000.0, 1.0, 48000.0),
            |_: f32| Biquad::passthrough(),
        ] {
            let mut bq = constructor(0.0);
            let dt = 1.0 / 48000.0;
            let mut in_energy = 0.0f32;
            let mut out_energy = 0.0f32;
            for i in 0..48000 {
                let input = (std::f32::consts::TAU * 1000.0 * i as f32 * dt).sin();
                let out = bq.tick(input);
                if i >= 4800 {
                    in_energy += input * input;
                    out_energy += out * out;
                }
            }
            let ratio = out_energy / in_energy;
            assert!(
                (ratio - 1.0).abs() < 0.02,
                "gain=1.0 should pass through, ratio={ratio}",
            );
        }
    }

    #[test]
    fn biquad_lowshelf_attenuates_low() {
        let sr = 48000.0;
        let mut bq_low = Biquad::low_shelf(800.0, 0.1, sr);
        let mut bq_high = Biquad::low_shelf(800.0, 0.1, sr);
        let low_energy = measure_biquad_energy(&mut bq_low, 200.0, sr, 48000);
        let high_energy = measure_biquad_energy(&mut bq_high, 12000.0, sr, 48000);
        assert!(
            low_energy < high_energy * 0.5,
            "lowShelf(0.1) should attenuate 200Hz more than 12kHz: low={low_energy}, high={high_energy}",
        );
    }

    #[test]
    fn biquad_highshelf_attenuates_high() {
        let sr = 48000.0;
        let mut bq_low = Biquad::high_shelf(8000.0, 0.1, sr);
        let mut bq_high = Biquad::high_shelf(8000.0, 0.1, sr);
        let low_energy = measure_biquad_energy(&mut bq_low, 200.0, sr, 48000);
        let high_energy = measure_biquad_energy(&mut bq_high, 12000.0, sr, 48000);
        assert!(
            high_energy < low_energy * 0.5,
            "highShelf(0.1) should attenuate 12kHz more than 200Hz: high={high_energy}, low={low_energy}",
        );
    }

    #[test]
    fn biquad_peaking_attenuates_mid() {
        let sr = 48000.0;
        let mut bq_mid = Biquad::peaking(800.0, 8000.0, 0.1, sr);
        let mut bq_low = Biquad::peaking(800.0, 8000.0, 0.1, sr);
        let mut bq_high = Biquad::peaking(800.0, 8000.0, 0.1, sr);
        let mid_energy = measure_biquad_energy(&mut bq_mid, 2000.0, sr, 48000);
        let low_energy = measure_biquad_energy(&mut bq_low, 200.0, sr, 48000);
        let high_energy = measure_biquad_energy(&mut bq_high, 12000.0, sr, 48000);
        assert!(
            mid_energy < low_energy * 0.5,
            "peaking(0.1) should attenuate 2kHz more than 200Hz: mid={mid_energy}, low={low_energy}",
        );
        assert!(
            mid_energy < high_energy * 0.5,
            "peaking(0.1) should attenuate 2kHz more than 12kHz: mid={mid_energy}, high={high_energy}",
        );
    }

    #[test]
    fn biquad_stability() {
        let sr = 48000.0;
        for &gain in &[0.001f32, 10.0] {
            let filters = [
                Biquad::low_shelf(800.0, gain, sr),
                Biquad::high_shelf(8000.0, gain, sr),
                Biquad::peaking(800.0, 8000.0, gain, sr),
            ];
            for mut bq in filters {
                for i in 0..48000 {
                    let input = (std::f32::consts::TAU * 1000.0 * i as f32 / sr).sin();
                    let out = bq.tick(input);
                    assert!(
                        out.is_finite(),
                        "biquad blew up at sample {i} with gain={gain}",
                    );
                }
            }
        }
    }
}
