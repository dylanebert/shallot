//! Modulation-family DSP primitives: the triangle LFO shared by chorus/flanger/
//! phaser and the one-pole allpass the phaser cascades.
//!
//! The per-node process loops (smoothing + mix) live in `graph.rs` alongside the
//! other `*_node` wrappers, exactly as `delay_node` wraps `delay::DelayLine`; this
//! file holds only the pinned, reference-transcribed cores.
//!
//! - **Triangle LFO** — DaisySP `Effects/chorus.cpp`/`flanger.cpp`/`phaser.cpp`
//!   `ProcessLfo`: a phase bouncing in [-1, 1] with a direction that flips at each
//!   fold. The DaisySP source folds the direction into a signed `lfo_freq_`; we keep
//!   the magnitude (`inc`, recomputed per block from the rate param) separate from
//!   the sign (`dir`, the persisted bounce state), which is the same recurrence.
//! - **One-pole allpass** — cycfi/q `fx/allpass.hpp` `one_pole_allpass` (Boost-1.0):
//!   coefficient `tan(π·freq/sr − π/4)` (90° shift at `freq`), recurrence
//!   `out = y + a·s; y = s − a·out` with a single float of state. A cascade of these
//!   is the phaser (DaisySP's own `phaser.cpp` is a delay-line Schroeder allpass whose
//!   top-level `Phaser::Process` sums identical parallel engines — degenerate to a
//!   scaled single notch — so the standard first-order allpass *chain* is transcribed
//!   from q instead; see `specs/audio-effect-nodes.md`).

use core::f32::consts::PI;

/// Max phaser allpass stages (DaisySP `Phaser::SetPoles` caps at 8). The per-stage
/// state is one float, held inline in `NodeState::Phaser`.
pub const MAX_PHASER_STAGES: usize = 8;

/// Chorus base delay in ms — the longer delay (thickening, no resonant comb) that,
/// with `FLANGER_BASE_MS`, distinguishes the two nodes over one shared engine.
pub const CHORUS_BASE_MS: f32 = 15.0;

/// Flanger base delay in ms — short enough for the swept comb notches.
pub const FLANGER_BASE_MS: f32 = 1.5;

/// Phaser allpass sweep range (Hz): the notch centers travel `PHASER_FMIN`..`FMAX`.
pub const PHASER_FMIN: f32 = 200.0;
pub const PHASER_FMAX: f32 = 2000.0;

/// Triangle-LFO increment per sample for a `rate`-Hz LFO. DaisySP `SetLfoFreq`:
/// `4·freq/sr`, magnitude-clamped to 0.25 (= ±0.125·sr). A full bounce cycle spans
/// `4/inc` samples, so the LFO frequency is exactly `rate`.
pub fn lfo_inc(rate: f32, sample_rate: f32) -> f32 {
    (4.0 * rate / sample_rate).clamp(0.0, 0.25)
}

/// One step of the DaisySP triangle LFO: advance `phase` by `dir·inc`, folding at ±1
/// and flipping `dir`. Returns the new phase in [-1, 1]; the caller scales it by the
/// modulation amplitude.
pub fn tri_step(phase: &mut f32, dir: &mut f32, inc: f32) -> f32 {
    *phase += *dir * inc;
    if *phase > 1.0 {
        *phase = 1.0 - (*phase - 1.0);
        *dir = -*dir;
    } else if *phase < -1.0 {
        *phase = -1.0 - (*phase + 1.0);
        *dir = -*dir;
    }
    *phase
}

/// q `one_pole_allpass` coefficient: `tan(π·freq/sr − π/4)`, the pole location in
/// (-1, 1) giving a 90° phase shift at `freq`. q's own `fasttan` is a fast
/// approximation; production transcribes the accurate `tan`.
pub fn allpass_coeff(freq: f32, sample_rate: f32) -> f32 {
    (PI * freq / sample_rate - 0.25 * PI).tan()
}

/// One sample of q `one_pole_allpass::operator()`: `out = y + a·s; y = s − a·out`.
/// `y` is the single-float per-stage state, mutated in place.
pub fn allpass_tick(y: &mut f32, a: f32, s: f32) -> f32 {
    let out = *y + a * s;
    *y = s - a * out;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tri_lfo_golden() {
        // Production's `tri_step` bounce bit-checked against the independent
        // gen_vectors transcription (DaisySP ProcessLfo) — exact, no arithmetic loss
        // beyond the shared f32 adds/folds.
        let tol = 1e-6;
        for &(inc, want) in crate::golden::TRI_LFO {
            let mut phase = 0.0f32;
            let mut dir = 1.0f32;
            for (n, w) in want.iter().enumerate() {
                let got = tri_step(&mut phase, &mut dir, inc);
                assert!(
                    (got - w).abs() <= tol,
                    "tri_lfo(inc={inc}) sample {n}: got {got}, want {w}, Δ{}",
                    (got - w).abs()
                );
            }
        }
    }

    #[test]
    fn allpass_golden() {
        // Production's coefficient formula + allpass recurrence bit-checked against the
        // gen_vectors transcription (q one_pole_allpass) over an impulse response.
        let tol = 1e-5;
        for &(freq, sr, want_a, want) in crate::golden::ALLPASS_1P {
            let a = allpass_coeff(freq, sr);
            assert!(
                (a - want_a).abs() <= tol,
                "allpass_coeff({freq},{sr}): got {a}, want {want_a}"
            );
            let mut y = 0.0f32;
            for (n, w) in want.iter().enumerate() {
                let s = if n == 0 { 1.0 } else { 0.0 };
                let got = allpass_tick(&mut y, a, s);
                assert!(
                    (got - w).abs() <= tol,
                    "allpass({freq},{sr}) sample {n}: got {got}, want {w}, Δ{}",
                    (got - w).abs()
                );
            }
        }
    }

    #[test]
    fn tri_lfo_stays_bounded_and_periodic() {
        // The bounce must stay within [-1, 1] and return near its start after one full
        // 4/inc-sample cycle — the defining triangle property, independent of the
        // golden bit-check.
        let inc = 0.02_f32;
        let period = (4.0 / inc).round() as usize;
        let mut phase = 0.0f32;
        let mut dir = 1.0f32;
        for _ in 0..period {
            let p = tri_step(&mut phase, &mut dir, inc);
            assert!(p.abs() <= 1.0 + 1e-6, "triangle LFO left [-1,1]: {p}");
        }
        assert!(
            phase.abs() < 2.0 * inc,
            "triangle LFO should return to ~0 after one cycle, got {phase}"
        );
    }

    #[test]
    fn allpass_preserves_energy() {
        // A first-order allpass has unit magnitude at every frequency, so a tone's
        // energy passes unchanged — the property that makes a cascade a phase-only
        // sweep summed against dry.
        let sr = 48000.0_f32;
        let a = allpass_coeff(1000.0, sr);
        let mut y = 0.0f32;
        let mut ein = 0.0f32;
        let mut eout = 0.0f32;
        for n in 0..4000 {
            let s = (n as f32 * 2.0 * PI * 440.0 / sr).sin();
            let out = allpass_tick(&mut y, a, s);
            if n >= 2000 {
                ein += s * s;
                eout += out * out;
            }
        }
        let ratio = eout / ein;
        assert!(
            (ratio - 1.0).abs() < 0.02,
            "allpass should preserve energy, got ratio {ratio}"
        );
    }
}
