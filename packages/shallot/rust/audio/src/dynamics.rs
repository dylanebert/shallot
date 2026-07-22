/// Standard compressor/limiter/expander/gate. cycfi/q `fx/dynamic.hpp`'s
/// `soft_knee_compressor` (compressor/limiter) and `expander` (expander/gate) static
/// curves, fed by `fx/envelope.hpp`'s `ar_envelope_follower` — per q's own doc comment,
/// the envelope follower runs BEFORE the curve (smooths the signal level, not the
/// gain-reduction output), so "attack = rising level" needs no per-family branch
/// inversion: both curves are monotonic increasing in the smoothed level. The
/// soft-knee formula is cross-checked against dasp-pytorch's independently-derived
/// quadratic knee in `examples/gen_vectors.rs`.
#[derive(Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum DynamicsMode {
    Compressor = 0,
    Limiter = 1,
    Expander = 2,
    Gate = 3,
}

impl DynamicsMode {
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => Self::Limiter,
            2 => Self::Expander,
            3 => Self::Gate,
            _ => Self::Compressor,
        }
    }
}

// -120dB — q's own fast_rms_envelope_follower floor convention (support/decibel.hpp
// callers), reused here so `log10` never sees zero.
const DB_FLOOR_LIN: f32 = 1e-6;

/// q `ar_envelope_follower` coefficient: `exp(-2/(sps*seconds))`, the accurate `exp`
/// (q's own `fast_exp3` is a fast approximation production doesn't transcribe).
pub fn ar_coeff(seconds: f32, sample_rate: f32) -> f32 {
    (-2.0 / (sample_rate * seconds.max(1e-9))).exp()
}

// q `soft_knee_compressor`: quadratic soft knee. `ratio` in conventional n:1 form
// (e.g. 4.0 for "4:1"); q's own 1/n convention is `1.0 / ratio`.
fn compressor_curve(env_db: f32, threshold: f32, knee: f32, ratio: f32) -> f32 {
    let slope = 1.0 - 1.0 / ratio;
    let lower = threshold - knee * 0.5;
    let upper = threshold + knee * 0.5;
    if env_db <= lower {
        0.0
    } else if env_db <= upper {
        let soft_slope = slope * ((env_db - lower) / knee) * 0.5;
        soft_slope * (lower - env_db)
    } else {
        slope * (threshold - env_db)
    }
}

// q `expander`: hard knee, ratio applied directly (no 1/n inversion).
fn expander_curve(env_db: f32, threshold: f32, ratio: f32) -> f32 {
    if env_db >= threshold {
        0.0
    } else {
        ratio * (env_db - threshold)
    }
}

/// One sample: smooth `input`'s level into `env` (the voice's persisted follower
/// state), map through the mode's static curve + makeup gain, mix dry/wet (parallel
/// compression at `mix < 1`).
pub fn process(
    env: &mut f32,
    input: f32,
    mode: DynamicsMode,
    threshold: f32,
    ratio: f32,
    knee: f32,
    attack_coeff: f32,
    release_coeff: f32,
    makeup: f32,
    mix: f32,
) -> f32 {
    let level = input.abs();
    let coeff = if level > *env {
        attack_coeff
    } else {
        release_coeff
    };
    *env = level + coeff * (*env - level);

    // ratio/knee are floored at the call site (graph.rs's dynamics_node) — trusted here,
    // matching delay.rs's DelayLine::process (the pure DSP core takes valid ranges as given).
    let env_db = 20.0 * (*env).max(DB_FLOOR_LIN).log10();
    let gc_db = match mode {
        DynamicsMode::Compressor | DynamicsMode::Limiter => {
            compressor_curve(env_db, threshold, knee, ratio)
        }
        DynamicsMode::Expander | DynamicsMode::Gate => expander_curve(env_db, threshold, ratio),
    };

    let gain = 10f32.powf((gc_db + makeup) / 20.0);
    let mix = mix.clamp(0.0, 1.0);
    input * (1.0 - mix) + input * gain * mix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamics_golden() {
        // Production bit-checked against the independent gen_vectors.rs transcription
        // (ar_envelope_follower + soft_knee_compressor/expander, cross-checked there
        // against dasp-pytorch's quadratic knee) at f32 roundoff over the composed
        // per-sample recurrence.
        let tol = 5e-4;
        for &(mode, threshold, ratio, knee, attack, release, makeup, mix, sr, quiet, loud, want) in
            crate::golden::DYNAMICS
        {
            let dyn_mode = DynamicsMode::from_u32(mode);
            let attack_coeff = ar_coeff(attack, sr);
            let release_coeff = ar_coeff(release, sr);
            let mut env = 0.0f32;
            for n in 0..crate::golden::DYNAMICS_TEST_LEN {
                let input = if (16..48).contains(&n) { loud } else { quiet };
                let got = process(
                    &mut env,
                    input,
                    dyn_mode,
                    threshold,
                    ratio,
                    knee,
                    attack_coeff,
                    release_coeff,
                    makeup,
                    mix,
                );
                assert!(
                    (got - want[n]).abs() <= tol,
                    "dynamics(mode={mode},thr={threshold},ratio={ratio}) sample {n}: got {got}, want {}, Δ{}",
                    want[n],
                    (got - want[n]).abs()
                );
            }
        }
    }

    #[test]
    fn compressor_reduces_gain_above_threshold() {
        // A steady tone well above threshold must settle to LESS than unity gain —
        // isolates the compress-family curve from the envelope-follower ballistics.
        let mut env = 0.0f32;
        let attack_coeff = ar_coeff(0.0002, 48000.0);
        let release_coeff = ar_coeff(0.0002, 48000.0);
        let mut out = 0.0;
        for _ in 0..2000 {
            out = process(
                &mut env,
                0.8,
                DynamicsMode::Compressor,
                -12.0,
                4.0,
                6.0,
                attack_coeff,
                release_coeff,
                0.0,
                1.0,
            );
        }
        assert!(
            out < 0.8,
            "compressor should reduce gain above threshold, got {out}"
        );
    }

    #[test]
    fn expander_passes_signal_above_threshold() {
        // A steady tone above the expander's threshold must settle to unity gain —
        // the expand-family curve is a no-op once the envelope clears the threshold.
        let mut env = 0.0f32;
        let attack_coeff = ar_coeff(0.0002, 48000.0);
        let release_coeff = ar_coeff(0.0002, 48000.0);
        let mut out = 0.0;
        for _ in 0..2000 {
            out = process(
                &mut env,
                0.5,
                DynamicsMode::Expander,
                -30.0,
                4.0,
                0.0,
                attack_coeff,
                release_coeff,
                0.0,
                1.0,
            );
        }
        assert!(
            (out - 0.5).abs() < 1e-3,
            "expander above threshold should pass unity gain, got {out}"
        );
    }
}
