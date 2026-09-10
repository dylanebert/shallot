use crate::interp::hermite4;

/// Buffer capacity in samples — 2s at 48kHz (or ~1s at 96kHz), the standard delay-effect
/// range. A voice's `Delay` node allocates one of these lazily in `set_voice_instrument`
/// (`Voice.delay_lines`), so the cap only costs memory for voices that use it.
pub const DELAY_MAX_SAMPLES: usize = 96_000;

/// Forward-write ring buffer + 4-point Hermite fractional read (DaisySP
/// `Utility/delayline.h` `Write`/`ReadHermite`, reindexed to a forward write pointer —
/// shallot's other ring reads (`sample::sample_read_loop`) are forward, so this stays
/// consistent instead of mirroring DaisySP's backward `write_ptr_`; the read/write
/// recurrence and the interpolation formula are unchanged) composed with a one-pole
/// damping filter in the feedback loop — standard delay-effect design, not itself in
/// delayline.h (DaisySP's own delay effect is the LGPL Soundpipe port, excluded as a
/// transcription source).
pub struct DelayLine {
    buffer: Box<[f32; DELAY_MAX_SAMPLES]>,
    write_pos: usize,
    damp_state: f32,
}

impl DelayLine {
    pub fn new() -> Self {
        DelayLine {
            buffer: vec![0.0f32; DELAY_MAX_SAMPLES]
                .into_boxed_slice()
                .try_into()
                .ok()
                .unwrap(),
            write_pos: 0,
            damp_state: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
        self.damp_state = 0.0;
    }

    /// Hermite-interpolated read `delay_samples` behind the write head. `read_pos`
    /// unwraps the ring into a virtual forward timeline anchored at `write_pos` (the
    /// same fractional-index-then-taps shape as `sample::sample_read_loop`), so
    /// increasing index means more recent — taps ym1/y0/y1/y2 walk oldest to newest.
    fn read(&self, delay_samples: f32) -> f32 {
        let n = DELAY_MAX_SAMPLES;
        let read_pos = self.write_pos as f64 - delay_samples as f64;
        let pos = read_pos.rem_euclid(n as f64);
        let i = pos as usize;
        let frac = (pos - i as f64) as f32;
        let ym1 = self.buffer[(i + n - 1) % n];
        let y0 = self.buffer[i];
        let y1 = self.buffer[(i + 1) % n];
        let y2 = self.buffer[(i + 2) % n];
        hermite4(ym1, y0, y1, y2, frac)
    }

    /// One sample: read the delayed (wet) signal, damp it with a one-pole lowpass,
    /// feed `input + feedback * damped` back into the line, advance the write
    /// pointer. Returns the wet signal pre-mix — the node crossfades it with dry.
    pub fn process(
        &mut self,
        input: f32,
        delay_samples: f32,
        feedback: f32,
        damp_coeff: f32,
    ) -> f32 {
        let wet = self.read(delay_samples);
        self.damp_state += damp_coeff * (wet - self.damp_state);
        self.buffer[self.write_pos] = input + feedback * self.damp_state;
        self.write_pos = (self.write_pos + 1) % DELAY_MAX_SAMPLES;
        wet
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_golden() {
        // Production's read/write recurrence bit-checked against the independent
        // gen_vectors.rs transcription (ring buffer + Hermite + one-pole feedback
        // damping) at f32 roundoff over the composed chain (~30 flops/sample across
        // DELAY_TEST_LEN=32 samples of feedback accumulation).
        let tol = 5e-4;
        for &(delay, feedback, damp, want) in crate::golden::DELAY {
            let mut dl = DelayLine::new();
            for n in 0..crate::golden::DELAY_TEST_LEN {
                let input = if n == 0 { 1.0 } else { 0.0 };
                let got = dl.process(input, delay, feedback, damp);
                assert!(
                    (got - want[n]).abs() <= tol,
                    "delay({delay},{feedback},{damp}) sample {n}: got {got}, want {}, Δ{}",
                    want[n],
                    (got - want[n]).abs()
                );
            }
        }
    }

    #[test]
    fn zero_feedback_is_pure_delay() {
        // feedback=0 silences the feedback path entirely, so the output is exactly a
        // delayed copy of the input — isolates the ring-buffer read/write mechanics
        // from the damping filter.
        let mut dl = DelayLine::new();
        let delay = 10.0;
        let mut input = [0.0f32; 64];
        for (i, s) in input.iter_mut().enumerate() {
            *s = (i as f32 * 0.3).sin();
        }
        let mut output = [0.0f32; 64];
        for i in 0..64 {
            output[i] = dl.process(input[i], delay, 0.0, 1.0);
        }
        for i in 10..64 {
            assert!(
                (output[i] - input[i - 10]).abs() < 1e-4,
                "sample {i}: expected {}, got {}",
                input[i - 10],
                output[i]
            );
        }
    }

    #[test]
    fn reset_clears_history() {
        let mut dl = DelayLine::new();
        for _ in 0..100 {
            dl.process(1.0, 10.0, 0.5, 0.5);
        }
        dl.reset();
        let out = dl.process(0.0, 10.0, 0.5, 0.5);
        assert_eq!(out, 0.0, "reset should clear delay-line history");
    }

    #[test]
    fn feedback_below_one_decays() {
        // A stable (<1) feedback coefficient must decay an impulse response toward
        // zero, not sustain or grow it.
        let mut dl = DelayLine::new();
        let mut peak_first = 0.0f32;
        let mut peak_last = 0.0f32;
        for n in 0..2000 {
            let input = if n == 0 { 1.0 } else { 0.0 };
            let out = dl.process(input, 20.0, 0.9, 0.5);
            if (100..120).contains(&n) {
                peak_first = peak_first.max(out.abs());
            }
            if (1900..1920).contains(&n) {
                peak_last = peak_last.max(out.abs());
            }
        }
        assert!(
            peak_last < peak_first * 0.1,
            "echo train should decay: first-window peak {peak_first}, late-window peak {peak_last}"
        );
    }
}
