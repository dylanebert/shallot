//! Golden-vector generator for the bit-portable DSP formulas.
//!
//! Emits `src/golden.rs` — the captured reference outputs `cargo test` bit-checks
//! the production kernel against. This file is the *firewall*: every formula here
//! is transcribed independently from its named reference, NOT from the production
//! code in `src/`, so a production drift breaks the gate and can't be papered over
//! by regenerating. Run after deliberately changing a reference transcription:
//!
//!   cargo run --example gen_vectors        # rewrite src/golden.rs
//!   cargo run --example gen_vectors -- --check   # fail if it would change
//!
//! References (the spec — a mismatch is a transcription bug until the vector rules
//! it out):
//!   - PolyBLEP:  DaisySP `Source/Synthesis/oscillator.cpp` `Polyblep` (the band-
//!     limit correction kernel; production's saw/square sign convention is its own
//!     and stays behaviorally tested).
//!   - SVF:       Andrew Simper / Cytomic, "Solving the continuous SVF equations
//!     using trapezoidal integration and equivalent currents" (2013), the TPT SVF
//!     listing. f32, the production sample precision.
//!   - Biquads:   RBJ Audio-EQ cookbook as transcribed in Steam Audio
//!     `core/src/core/iir.cpp` (`lowShelf`/`highShelf`/`peaking`). Computed in f64
//!     and rounded to f32 — the accurate closed-form coefficient (Steam Audio's own
//!     highShelf is f64). Production's f32 paths are gated to it within derived f32
//!     roundoff, so the vector measures production's fidelity to the true formula.
//!   - Hermite:   Olli Niemitalo, "Polynomial Interpolators for High-Quality
//!     Resampling of Oversampled Audio", the 4-point 3rd-order x-form.
//!   - Delay:     DaisySP `Utility/delayline.h` (`Write`/`ReadHermite`) composed with a
//!     one-pole feedback damping filter (standard delay-effect design).
//!   - Dynamics:  cycfi/q `fx/envelope.hpp` `ar_envelope_follower` (branching attack/
//!     release one-pole, coefficient `exp(-2/(sps*seconds))`) feeding `fx/dynamic.hpp`'s
//!     `soft_knee_compressor` (compressor/limiter) or `expander` (expander/gate) static
//!     curve, cross-checked against dasp-pytorch `functional.py compressor`'s
//!     independently-derived quadratic soft-knee formula.
//!   - Waveshaper: DaisySP `Effects/overdrive.cpp` (`SetDrive` drive law + `SoftClip`)
//!     for the soft mode + shared drive pre-gain, `Effects/wavefolder.cpp` for fold,
//!     a plain clip for hard; the SoftClip curve is cross-checked against `tanh`
//!     (dasp-pytorch `distortion`). The tail is DaisySP `Utility/dcblock.cpp`.

use std::f32::consts::PI as PI32;
use std::f64::consts::TAU as TAU64;
use std::fmt::Write as _;

const SVF_STEPS: usize = 16;

// --- reference transcriptions (independent of src/) -----------------------------

// DaisySP `Polyblep` verbatim (oscillator.cpp).
fn poly_blep_ref(phase_inc: f32, mut t: f32) -> f32 {
    let dt = phase_inc;
    if t < dt {
        t /= dt;
        t + t - t * t - 1.0
    } else if t > 1.0 - dt {
        t = (t - 1.0) / dt;
        t * t + t + t + 1.0
    } else {
        0.0
    }
}

// Cytomic TPT SVF (Simper 2013). Mode codes match `FilterMode`: 0 LP, 1 HP, 2 BP, 3 Notch.
struct SvfRef {
    ic1eq: f32,
    ic2eq: f32,
    a1: f32,
    a2: f32,
    a3: f32,
    k: f32,
}

impl SvfRef {
    fn new(sr: f32, cutoff: f32, q: f32) -> Self {
        let g = (PI32 * cutoff / sr).tan();
        let k = 1.0 / q;
        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;
        SvfRef {
            ic1eq: 0.0,
            ic2eq: 0.0,
            a1,
            a2,
            a3,
            k,
        }
    }

    fn tick(&mut self, v0: f32, mode: u32) -> f32 {
        let v3 = v0 - self.ic2eq;
        let v1 = self.a1 * self.ic1eq + self.a2 * v3;
        let v2 = self.ic2eq + self.a2 * self.ic1eq + self.a3 * v3;
        self.ic1eq = 2.0 * v1 - self.ic1eq;
        self.ic2eq = 2.0 * v2 - self.ic2eq;
        match mode {
            1 => v0 - self.k * v1 - v2,
            2 => v1,
            3 => v0 - self.k * v1,
            _ => v2,
        }
    }
}

// RBJ shelving/peaking coefficients (Steam Audio iir.cpp), computed in f64 then
// rounded to f32. Returns [b0, b1, b2, a1, a2] (a0 normalized to 1).
fn low_shelf_ref(cutoff: f64, gain: f64, sr: f64) -> [f32; 5] {
    let q = 0.707_f64;
    let w0 = TAU64 * cutoff / sr;
    let cw0 = w0.cos();
    let sw0 = w0.sin();
    let alpha = sw0 / (2.0 * q);
    let a = gain.sqrt();
    let tsaa = 2.0 * a.sqrt() * alpha;
    let a0 = (a + 1.0) + (a - 1.0) * cw0 + tsaa;
    let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cw0);
    let a2 = (a + 1.0) + (a - 1.0) * cw0 - tsaa;
    let b0 = a * ((a + 1.0) - (a - 1.0) * cw0 + tsaa);
    let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cw0);
    let b2 = a * ((a + 1.0) - (a - 1.0) * cw0 - tsaa);
    norm5(b0, b1, b2, a1, a2, a0)
}

fn high_shelf_ref(cutoff: f64, gain: f64, sr: f64) -> [f32; 5] {
    let q = 0.707_f64;
    let w0 = TAU64 * cutoff / sr;
    let cw0 = w0.cos();
    let sw0 = w0.sin();
    let alpha = sw0 / (2.0 * q);
    let a = gain.sqrt();
    let tsaa = 2.0 * a.sqrt() * alpha;
    let a0 = (a + 1.0) - (a - 1.0) * cw0 + tsaa;
    let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cw0);
    let a2 = (a + 1.0) - (a - 1.0) * cw0 - tsaa;
    let b0 = a * ((a + 1.0) + (a - 1.0) * cw0 + tsaa);
    let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cw0);
    let b2 = a * ((a + 1.0) + (a - 1.0) * cw0 - tsaa);
    norm5(b0, b1, b2, a1, a2, a0)
}

fn peaking_ref(low: f64, high: f64, gain: f64, sr: f64) -> [f32; 5] {
    let center = (low * high).sqrt();
    let q_inverse = (high - low) / center;
    let w0 = TAU64 * center / sr;
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
    norm5(b0, b1, b2, a1, a2, a0)
}

fn norm5(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64, a0: f64) -> [f32; 5] {
    [
        (b0 / a0) as f32,
        (b1 / a0) as f32,
        (b2 / a0) as f32,
        (a1 / a0) as f32,
        (a2 / a0) as f32,
    ]
}

// Niemitalo 4-point 3rd-order Hermite (x-form) verbatim.
fn hermite_ref(ym1: f32, y0: f32, y1: f32, y2: f32, t: f32) -> f32 {
    let c0 = y0;
    let c1 = 0.5 * (y1 - ym1);
    let c2 = ym1 - 2.5 * y0 + 2.0 * y1 - 0.5 * y2;
    let c3 = 0.5 * (y2 - ym1) + 1.5 * (y0 - y1);
    ((c3 * t + c2) * t + c1) * t + c0
}

// DaisySP `Utility/delayline.h` Write/ReadHermite (forward-write ring, Hermite
// fractional read via hermite_ref) composed with a one-pole feedback damping filter —
// the delay node's full per-sample recurrence, independently re-derived here (own
// small ring buffer, own loop) rather than calling production's src/delay.rs.
const DELAY_RING_SIZE: usize = 64;

struct DelayLineRef {
    buf: [f32; DELAY_RING_SIZE],
    write_pos: usize,
    damp_state: f32,
}

impl DelayLineRef {
    fn new() -> Self {
        DelayLineRef {
            buf: [0.0; DELAY_RING_SIZE],
            write_pos: 0,
            damp_state: 0.0,
        }
    }

    fn process(&mut self, input: f32, delay: f32, feedback: f32, damp_coeff: f32) -> f32 {
        let n = DELAY_RING_SIZE;
        let read_pos = self.write_pos as f64 - delay as f64;
        let pos = read_pos.rem_euclid(n as f64);
        let i = pos as usize;
        let frac = (pos - i as f64) as f32;
        let ym1 = self.buf[(i + n - 1) % n];
        let y0 = self.buf[i];
        let y1 = self.buf[(i + 1) % n];
        let y2 = self.buf[(i + 2) % n];
        let wet = hermite_ref(ym1, y0, y1, y2, frac);
        self.damp_state += damp_coeff * (wet - self.damp_state);
        self.buf[self.write_pos] = input + feedback * self.damp_state;
        self.write_pos = (self.write_pos + 1) % n;
        wet
    }
}

const DELAY_TEST_LEN: usize = 32;

fn delay_impulse_response(delay: f32, feedback: f32, damp_coeff: f32) -> [f32; DELAY_TEST_LEN] {
    let mut dl = DelayLineRef::new();
    let mut out = [0.0f32; DELAY_TEST_LEN];
    for (n, o) in out.iter_mut().enumerate() {
        let input = if n == 0 { 1.0 } else { 0.0 };
        *o = dl.process(input, delay, feedback, damp_coeff);
    }
    out
}

// q `ar_envelope_follower` (fx/envelope.hpp): branching one-pole, coefficient
// exp(-2/(sps*seconds)) — the accurate `exp`, not q's own `fast_exp3` approximation
// (production is gated to the true formula, per the biquad precedent).
fn ar_coeff(seconds: f32, sr: f32) -> f32 {
    (-2.0_f32 / (sr * seconds.max(1e-9))).exp()
}

struct ArEnvelopeRef {
    y: f32,
}

impl ArEnvelopeRef {
    fn new() -> Self {
        ArEnvelopeRef { y: 0.0 }
    }

    fn tick(&mut self, s: f32, attack_coeff: f32, release_coeff: f32) -> f32 {
        let coeff = if s > self.y {
            attack_coeff
        } else {
            release_coeff
        };
        self.y = s + coeff * (self.y - s);
        self.y
    }
}

// q `soft_knee_compressor` (fx/dynamic.hpp): quadratic soft knee. `ratio` in
// conventional n:1 form (e.g. 4.0 for "4:1"), so q's own 1/n convention is
// `ratio_q = 1.0 / ratio`.
fn compressor_curve_ref(env_db: f32, threshold: f32, knee: f32, ratio: f32) -> f32 {
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

// q `expander` (fx/dynamic.hpp): hard knee, ratio applied directly (no 1/n inversion).
fn expander_curve_ref(env_db: f32, threshold: f32, ratio: f32) -> f32 {
    if env_db >= threshold {
        0.0
    } else {
        ratio * (env_db - threshold)
    }
}

// dasp-pytorch `functional.py compressor`'s quadratic soft-knee (Giannoulis 2012
// static curve), independently re-derived in f64 to cross-check `compressor_curve_ref`
// above — both compute the gain-computer output `x_sc - x_db`, from unrelated source
// code. gen_vectors asserts they agree before emission (see `main`).
fn dasp_knee_ref(env_db: f64, threshold: f64, knee: f64, ratio: f64) -> f64 {
    let lower = threshold - knee * 0.5;
    let upper = threshold + knee * 0.5;
    if env_db <= lower {
        0.0
    } else if env_db <= upper {
        ((1.0 / ratio) - 1.0) * (env_db - threshold + knee * 0.5).powi(2) / (2.0 * knee)
    } else {
        (env_db - threshold) * ((1.0 / ratio) - 1.0)
    }
}

// The dynamics node's full per-sample recurrence: q's envelope follower (level
// domain) feeding the mode's static curve (dB domain) — smoothing runs BEFORE the
// curve, per q's own dynamic.hpp doc comment ("env is the envelope of the signal in
// decibels obtained (e.g) using the envelope_follower"), not after on the gain-
// reduction output. Mode 0/1 (compressor/limiter) share `compressor_curve_ref`; mode
// 2/3 (expander/gate) share `expander_curve_ref` — "limiter"/"gate" are parameter
// presets, not distinct formulas.
const DB_FLOOR_LIN: f32 = 1e-6; // -120dB, q's own fast_rms_envelope_follower floor convention

struct DynamicsRef {
    env: ArEnvelopeRef,
}

impl DynamicsRef {
    fn new() -> Self {
        DynamicsRef {
            env: ArEnvelopeRef::new(),
        }
    }

    fn process(
        &mut self,
        input: f32,
        mode: u32,
        threshold: f32,
        ratio: f32,
        knee: f32,
        attack_coeff: f32,
        release_coeff: f32,
        makeup: f32,
        mix: f32,
    ) -> f32 {
        let level = input.abs();
        let env = self.env.tick(level, attack_coeff, release_coeff);
        let env_db = 20.0 * env.max(DB_FLOOR_LIN).log10();
        let gc_db = match mode {
            0 | 1 => compressor_curve_ref(env_db, threshold, knee, ratio),
            _ => expander_curve_ref(env_db, threshold, ratio),
        };
        let gain = 10f32.powf((gc_db + makeup) / 20.0);
        input * (1.0 - mix) + input * gain * mix
    }
}

const DYNAMICS_TEST_LEN: usize = 64;

fn dynamics_step_response(
    mode: u32,
    threshold: f32,
    ratio: f32,
    knee: f32,
    attack: f32,
    release: f32,
    makeup: f32,
    mix: f32,
    sr: f32,
    quiet: f32,
    loud: f32,
) -> [f32; DYNAMICS_TEST_LEN] {
    let attack_coeff = ar_coeff(attack, sr);
    let release_coeff = ar_coeff(release, sr);
    let mut dyn_ref = DynamicsRef::new();
    let mut out = [0.0f32; DYNAMICS_TEST_LEN];
    for (n, o) in out.iter_mut().enumerate() {
        let input = if (16..48).contains(&n) { loud } else { quiet };
        *o = dyn_ref.process(
            input,
            mode,
            threshold,
            ratio,
            knee,
            attack_coeff,
            release_coeff,
            makeup,
            mix,
        );
    }
    out
}

// DaisySP `SoftClip`/`SoftLimit` (dsp.h): rational-tanh saturator, hard-limited ±3.
fn soft_clip_ref(x: f32) -> f32 {
    if x < -3.0 {
        -1.0
    } else if x > 3.0 {
        1.0
    } else {
        x * (27.0 + x * x) / (27.0 + 9.0 * x * x)
    }
}

// DaisySP `Wavefolder::Process` (wavefolder.cpp), unit gain + zero offset.
fn wavefold_ref(x: f32) -> f32 {
    let ft = ((x + 1.0) * 0.5).floor();
    let sgn = if (ft as i32) % 2 == 0 { 1.0 } else { -1.0 };
    sgn * (x - 2.0 * ft)
}

// DaisySP `Overdrive::SetDrive` (overdrive.cpp): drive [0,1] → (pre_gain, post_gain).
fn drive_gains_ref(drive: f32) -> (f32, f32) {
    let drive = drive.clamp(0.0, 1.0);
    let d = 2.0 * drive;
    let d2 = d * d;
    let pre_a = d * 0.5;
    let pre_b = d2 * d2 * d * 24.0;
    let pre = pre_a + (pre_b - pre_a) * d2;
    let squashed = d * (2.0 - d);
    let post = 1.0 / soft_clip_ref(0.33 + squashed * (pre - 0.33));
    (pre, post)
}

// Waveshaper memoryless transfer: shared drive law + the mode's nonlinearity
// (0 soft = DaisySP overdrive, 1 hard = clip, 2 fold = wavefolder). Independently
// re-derived here, not calling src/waveshaper.rs.
fn waveshaper_ref(mode: u32, drive: f32, x: f32) -> f32 {
    let (pre, post) = drive_gains_ref(drive);
    match mode {
        1 => (pre * x).clamp(-1.0, 1.0),
        2 => wavefold_ref(pre * x),
        _ => soft_clip_ref(pre * x) * post,
    }
}

// DaisySP `DcBlock` (dcblock.cpp): one-pole DC-removal, gain = 1 - 10/sr.
const DC_BLOCK_TEST_LEN: usize = 32;

fn dc_block_response(sr: f32) -> [f32; DC_BLOCK_TEST_LEN] {
    let gain = 1.0 - 10.0 / sr;
    let (mut x1, mut y1) = (0.0f32, 0.0f32);
    let mut out = [0.0f32; DC_BLOCK_TEST_LEN];
    for (n, o) in out.iter_mut().enumerate() {
        // Two-level probe: a unit step, then a drop at n=16 to exercise `in - x1`.
        let input = if n < 16 { 1.0 } else { 0.5 };
        let y = input - x1 + gain * y1;
        y1 = y;
        x1 = input;
        *o = y;
    }
    out
}

const TRI_LFO_LEN: usize = 48;

// DaisySP chorus/flanger/phaser `ProcessLfo` triangle bounce (verbatim; the signed
// `lfo_freq_` split into magnitude `inc` + sign `dir`).
fn tri_lfo_sequence(inc: f32) -> [f32; TRI_LFO_LEN] {
    let (mut phase, mut dir) = (0.0f32, 1.0f32);
    let mut out = [0.0f32; TRI_LFO_LEN];
    for o in out.iter_mut() {
        phase += dir * inc;
        if phase > 1.0 {
            phase = 1.0 - (phase - 1.0);
            dir = -dir;
        } else if phase < -1.0 {
            phase = -1.0 - (phase + 1.0);
            dir = -dir;
        }
        *o = phase;
    }
    out
}

const ALLPASS_LEN: usize = 32;

// q `one_pole_allpass`: coefficient `tan(π·f/sr − π/4)`, recurrence
// `out = y + a·s; y = s − a·out`.
fn allpass_coeff_ref(freq: f32, sr: f32) -> f32 {
    (PI32 * freq / sr - 0.25 * PI32).tan()
}

fn allpass_impulse(freq: f32, sr: f32) -> (f32, [f32; ALLPASS_LEN]) {
    let a = allpass_coeff_ref(freq, sr);
    let mut y = 0.0f32;
    let mut out = [0.0f32; ALLPASS_LEN];
    for (n, o) in out.iter_mut().enumerate() {
        let s = if n == 0 { 1.0 } else { 0.0 };
        let ap = y + a * s;
        y = s - a * ap;
        *o = ap;
    }
    (a, out)
}

// Cross-check `soft_clip_ref` (DaisySP's rational-tanh) against the true `tanh`
// (dasp-pytorch `distortion` is `tanh(x·gain)`) over the saturator's active range.
// SoftClip is a designed tanh approximation, so a typo in its constants diverges
// past the ~0.024 max approximation error — fail loud before emitting.
fn check_softclip_tanh_cross_reference() {
    for step in -300..=300 {
        let x = step as f32 * 0.01;
        let diff = (soft_clip_ref(x) - x.tanh()).abs();
        assert!(
            diff <= 0.03,
            "softclip/tanh cross-check at x={x}: softclip={}, tanh={}, Δ{diff}",
            soft_clip_ref(x),
            x.tanh()
        );
    }
}

// --- emission -------------------------------------------------------------------

fn lit(x: f32) -> String {
    // Debug emits the shortest representation that re-parses to the identical bits,
    // always with a `.` or exponent for finite f32 — a valid Rust float literal.
    format!("{x:?}")
}

fn arr5(v: [f32; 5]) -> String {
    format!(
        "[{}, {}, {}, {}, {}]",
        lit(v[0]),
        lit(v[1]),
        lit(v[2]),
        lit(v[3]),
        lit(v[4])
    )
}

// Cross-check `compressor_curve_ref` (q's soft_knee_compressor) against
// `dasp_knee_ref` (dasp-pytorch's independently-derived quadratic knee) over a sweep
// of env/threshold/knee/ratio combinations. A mismatch means one of the two
// transcriptions has a bug — fail loud before emitting anything.
fn check_knee_cross_reference() {
    for &threshold in &[-24.0_f32, -12.0, -6.0] {
        for &knee in &[0.0_f32, 3.0, 6.0, 12.0] {
            for &ratio in &[2.0_f32, 4.0, 10.0, 20.0] {
                for step in 0..=80 {
                    let env_db = threshold - 20.0 + step as f32 * 0.5;
                    let q = compressor_curve_ref(env_db, threshold, knee, ratio);
                    let dasp =
                        dasp_knee_ref(env_db as f64, threshold as f64, knee as f64, ratio as f64);
                    let diff = (q as f64 - dasp).abs();
                    assert!(
                        diff <= 1e-4,
                        "knee cross-check mismatch at env_db={env_db} threshold={threshold} \
                         knee={knee} ratio={ratio}: q={q}, dasp={dasp}, Δ{diff}"
                    );
                }
            }
        }
    }
}

fn main() {
    check_knee_cross_reference();
    check_softclip_tanh_cross_reference();

    let mut out = String::new();
    out.push_str(
        "// GENERATED by `cargo run --example gen_vectors` — do not edit.\n\
         // Golden vectors captured from the named DSP references (see examples/gen_vectors.rs).\n\
         // The production kernel is bit-checked against these at a derived f32 tolerance.\n\
         #![allow(dead_code)]\n\
         #![cfg_attr(rustfmt, rustfmt::skip)]\n\n",
    );

    let _ = writeln!(out, "pub(crate) const SVF_STEPS: usize = {SVF_STEPS};\n");

    // PolyBLEP: (t, dt, value). dt spans typical pitches; t sweeps both edge regions.
    out.push_str(
        "/// DaisySP `Polyblep`: (t, dt, value). dt = phase increment (freq/sr).\n\
         pub(crate) const POLY_BLEP: &[(f32, f32, f32)] = &[\n",
    );
    let dts = [0.005_f32, 0.02, 0.1];
    for &dt in &dts {
        for i in 0..=40 {
            let t = i as f32 / 40.0;
            let v = poly_blep_ref(dt, t);
            let _ = writeln!(out, "    ({}, {}, {}),", lit(t), lit(dt), lit(v));
        }
    }
    out.push_str("];\n\n");

    // Hermite: (ym1, y0, y1, y2, t, value). Mix of polynomial and arbitrary stencils.
    out.push_str(
        "/// Niemitalo Hermite x-form: (ym1, y0, y1, y2, t, value).\n\
         pub(crate) const HERMITE: &[(f32, f32, f32, f32, f32, f32)] = &[\n",
    );
    let stencils = [
        (0.3_f32, 1.0, -2.0, 0.5),
        (-1.0, 0.0, 4.0, 2.0),
        (0.7, 0.7, 0.7, 0.7),
        (-0.2, 0.9, -0.4, 1.3),
        (2.0, -1.5, 0.8, 0.1),
        (1.0, 2.0, 4.0, 8.0),
    ];
    for &(ym1, y0, y1, y2) in &stencils {
        for step in 0..=4 {
            let t = step as f32 / 4.0;
            let v = hermite_ref(ym1, y0, y1, y2, t);
            let _ = writeln!(
                out,
                "    ({}, {}, {}, {}, {}, {}),",
                lit(ym1),
                lit(y0),
                lit(y1),
                lit(y2),
                lit(t),
                lit(v)
            );
        }
    }
    out.push_str("];\n\n");

    // SVF: (mode, cutoff, q, sr, [a1,a2,a3,k], step_response[SVF_STEPS]).
    out.push_str(
        "/// Cytomic TPT SVF: (mode, cutoff, q, sr, [a1,a2,a3,k], unit-step response).\n\
         pub(crate) const SVF: &[(u32, f32, f32, f32, [f32; 4], [f32; SVF_STEPS])] = &[\n",
    );
    let svf_cases = [
        (0u32, 1000.0_f32, 0.707_f32),
        (0, 1000.0, 10.0),
        (1, 5000.0, 0.707),
        (2, 2000.0, 2.0),
        (3, 2000.0, 2.0),
    ];
    let sr = 48000.0_f32;
    for &(mode, cutoff, q) in &svf_cases {
        let mut f = SvfRef::new(sr, cutoff, q);
        let coeffs = [f.a1, f.a2, f.a3, f.k];
        let mut steps = [0.0f32; SVF_STEPS];
        for s in steps.iter_mut() {
            *s = f.tick(1.0, mode);
        }
        let coeff_s = format!(
            "[{}, {}, {}, {}]",
            lit(coeffs[0]),
            lit(coeffs[1]),
            lit(coeffs[2]),
            lit(coeffs[3])
        );
        let steps_s = steps.iter().map(|&x| lit(x)).collect::<Vec<_>>().join(", ");
        let _ = writeln!(
            out,
            "    ({}, {}, {}, {}, {}, [{}]),",
            mode,
            lit(cutoff),
            lit(q),
            lit(sr),
            coeff_s,
            steps_s
        );
    }
    out.push_str("];\n\n");

    // Biquad coefficients: [b0, b1, b2, a1, a2], computed in f64 (the accurate RBJ form).
    out.push_str(
        "/// Steam Audio / RBJ low-shelf: (cutoff, gain, sr, [b0,b1,b2,a1,a2]).\n\
         pub(crate) const LOW_SHELF: &[(f32, f32, f32, [f32; 5])] = &[\n",
    );
    for &(cutoff, gain) in &[(800.0_f64, 0.5_f64), (800.0, 2.0), (200.0, 0.1)] {
        let v = low_shelf_ref(cutoff, gain, 48000.0);
        let _ = writeln!(
            out,
            "    ({}, {}, {}, {}),",
            lit(cutoff as f32),
            lit(gain as f32),
            lit(48000.0),
            arr5(v)
        );
    }
    out.push_str("];\n\n");

    out.push_str(
        "/// Steam Audio / RBJ high-shelf: (cutoff, gain, sr, [b0,b1,b2,a1,a2]).\n\
         pub(crate) const HIGH_SHELF: &[(f32, f32, f32, [f32; 5])] = &[\n",
    );
    for &(cutoff, gain) in &[(8000.0_f64, 0.5_f64), (8000.0, 2.0), (12000.0, 0.1)] {
        let v = high_shelf_ref(cutoff, gain, 48000.0);
        let _ = writeln!(
            out,
            "    ({}, {}, {}, {}),",
            lit(cutoff as f32),
            lit(gain as f32),
            lit(48000.0),
            arr5(v)
        );
    }
    out.push_str("];\n\n");

    out.push_str(
        "/// Steam Audio / RBJ peaking: (low, high, gain, sr, [b0,b1,b2,a1,a2]).\n\
         pub(crate) const PEAKING: &[(f32, f32, f32, f32, [f32; 5])] = &[\n",
    );
    for &(low, high, gain) in &[(800.0_f64, 8000.0_f64, 0.5_f64), (800.0, 8000.0, 2.0)] {
        let v = peaking_ref(low, high, gain, 48000.0);
        let _ = writeln!(
            out,
            "    ({}, {}, {}, {}, {}),",
            lit(low as f32),
            lit(high as f32),
            lit(gain as f32),
            lit(48000.0),
            arr5(v)
        );
    }
    out.push_str("];\n\n");

    // Delay: (delay_samples, feedback, damp_coeff, impulse_response[DELAY_TEST_LEN]).
    let _ = writeln!(
        out,
        "pub(crate) const DELAY_TEST_LEN: usize = {DELAY_TEST_LEN};\n"
    );
    out.push_str(
        "/// DaisySP delayline.h Write/ReadHermite + one-pole feedback damping: \
         (delay_samples, feedback, damp_coeff, impulse_response).\n\
         pub(crate) const DELAY: &[(f32, f32, f32, [f32; DELAY_TEST_LEN])] = &[\n",
    );
    let delay_cases = [
        (5.5_f32, 0.5_f32, 0.3_f32),
        (1.0, 0.7, 0.0),
        (10.25, 0.0, 0.8),
        (3.0, 0.9, 0.5),
        (20.0, 0.6, 1.0),
    ];
    for &(delay, feedback, damp) in &delay_cases {
        let resp = delay_impulse_response(delay, feedback, damp);
        let resp_s = resp.iter().map(|&x| lit(x)).collect::<Vec<_>>().join(", ");
        let _ = writeln!(
            out,
            "    ({}, {}, {}, [{}]),",
            lit(delay),
            lit(feedback),
            lit(damp),
            resp_s
        );
    }
    out.push_str("];\n\n");

    // Dynamics: (mode, threshold, ratio, knee, attack, release, makeup, mix, sr,
    // quiet, loud, step_response[DYNAMICS_TEST_LEN]). mode: 0 compressor, 1 limiter,
    // 2 expander, 3 gate — limiter/gate reuse the compressor/expander curve, just at
    // more extreme ratio/attack presets.
    let _ = writeln!(
        out,
        "pub(crate) const DYNAMICS_TEST_LEN: usize = {DYNAMICS_TEST_LEN};\n"
    );
    out.push_str(
        "/// q ar_envelope_follower + soft_knee_compressor/expander, cross-checked against \
         dasp-pytorch's quadratic knee: (mode, threshold, ratio, knee, attack, release, makeup, \
         mix, sr, quiet, loud, step_response).\n\
         pub(crate) const DYNAMICS: &[(u32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, [f32; DYNAMICS_TEST_LEN])] = &[\n",
    );
    let sr = 48000.0_f32;
    // Ballistics deliberately fast relative to the 32/16-sample step windows below —
    // these constants pin the envelope-follower + curve math, not musically realistic
    // timing (that's a TS `NODE_DEFAULTS` concern, exercised at the ear-validation gate).
    let dynamics_cases = [
        // compressor: moderate threshold/ratio/knee.
        (
            0u32, -12.0_f32, 4.0_f32, 6.0_f32, 0.0002_f32, 0.0002_f32, 0.0_f32, 1.0_f32,
            0.0001_f32, 0.7_f32,
        ),
        // limiter: same curve family, hard knee + higher ratio.
        (1, -3.0, 20.0, 0.0, 0.0002, 0.0002, 0.0, 1.0, 0.0001, 0.9),
        // expander: quiet floor below threshold, loud burst above it.
        (2, -30.0, 2.0, 0.0, 0.0002, 0.0002, 0.0, 1.0, 0.0005, 0.2),
        // gate: very quiet floor, steep ratio.
        (3, -40.0, 50.0, 0.0, 0.0002, 0.0002, 0.0, 1.0, 0.00003, 0.3),
        // compressor with makeup gain + parallel (mix<1) blend.
        (0, -18.0, 8.0, 3.0, 0.0002, 0.0002, 6.0, 0.6, 0.0001, 0.5),
    ];
    for &(mode, threshold, ratio, knee, attack, release, makeup, mix, quiet, loud) in
        &dynamics_cases
    {
        let resp = dynamics_step_response(
            mode, threshold, ratio, knee, attack, release, makeup, mix, sr, quiet, loud,
        );
        let resp_s = resp.iter().map(|&x| lit(x)).collect::<Vec<_>>().join(", ");
        let _ = writeln!(
            out,
            "    ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, [{}]),",
            mode,
            lit(threshold),
            lit(ratio),
            lit(knee),
            lit(attack),
            lit(release),
            lit(makeup),
            lit(mix),
            lit(sr),
            lit(quiet),
            lit(loud),
            resp_s
        );
    }
    out.push_str("];\n\n");

    // Waveshaper: (mode, drive, input, output). Memoryless transfer — mode 0 soft
    // (DaisySP overdrive), 1 hard (clip), 2 fold (wavefolder); the DC-blocker tail is
    // stateful, pinned by DC_BLOCK below.
    out.push_str(
        "/// DaisySP overdrive drive law + SoftClip/clip/wavefold: (mode, drive, input, output).\n\
         pub(crate) const WAVESHAPER: &[(u32, f32, f32, f32)] = &[\n",
    );
    let waveshaper_inputs = [-1.5_f32, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 1.5];
    for &mode in &[0u32, 1, 2] {
        for &drive in &[0.2_f32, 0.5, 0.8] {
            for &x in &waveshaper_inputs {
                let v = waveshaper_ref(mode, drive, x);
                let _ = writeln!(
                    out,
                    "    ({}, {}, {}, {}),",
                    mode,
                    lit(drive),
                    lit(x),
                    lit(v)
                );
            }
        }
    }
    out.push_str("];\n\n");

    // DC blocker: (sr, step_response) over the fixed two-level probe (unit step, then
    // a drop at n=16).
    let _ = writeln!(
        out,
        "pub(crate) const DC_BLOCK_TEST_LEN: usize = {DC_BLOCK_TEST_LEN};\n"
    );
    out.push_str(
        "/// DaisySP DcBlock (gain = 1 - 10/sr): (sr, step_response over a two-level probe).\n\
         pub(crate) const DC_BLOCK: &[(f32, [f32; DC_BLOCK_TEST_LEN])] = &[\n",
    );
    for &sr in &[48000.0_f32, 44100.0] {
        let resp = dc_block_response(sr);
        let resp_s = resp.iter().map(|&x| lit(x)).collect::<Vec<_>>().join(", ");
        let _ = writeln!(out, "    ({}, [{}]),", lit(sr), resp_s);
    }
    out.push_str("];\n\n");

    // Triangle LFO: (inc, phase_sequence[TRI_LFO_LEN]). inc values bounce within the
    // window; the production `tri_step` shares this recurrence across chorus/flanger/phaser.
    let _ = writeln!(
        out,
        "pub(crate) const TRI_LFO_LEN: usize = {TRI_LFO_LEN};\n"
    );
    out.push_str(
        "/// DaisySP chorus/flanger/phaser ProcessLfo triangle bounce: (inc, phase_sequence).\n\
         pub(crate) const TRI_LFO: &[(f32, [f32; TRI_LFO_LEN])] = &[\n",
    );
    for &inc in &[0.05_f32, 0.13, 0.25] {
        let seq = tri_lfo_sequence(inc);
        let seq_s = seq.iter().map(|&x| lit(x)).collect::<Vec<_>>().join(", ");
        let _ = writeln!(out, "    ({}, [{}]),", lit(inc), seq_s);
    }
    out.push_str("];\n\n");

    // One-pole allpass: (freq, sr, coefficient, impulse_response[ALLPASS_LEN]). Pins
    // both the coefficient formula and the recurrence the phaser cascades.
    let _ = writeln!(
        out,
        "pub(crate) const ALLPASS_LEN: usize = {ALLPASS_LEN};\n"
    );
    out.push_str(
        "/// q one_pole_allpass: (freq, sr, coefficient a, impulse_response).\n\
         pub(crate) const ALLPASS_1P: &[(f32, f32, f32, [f32; ALLPASS_LEN])] = &[\n",
    );
    for &(freq, sr) in &[
        (500.0_f32, 48000.0_f32),
        (2000.0, 48000.0),
        (1000.0, 44100.0),
    ] {
        let (a, resp) = allpass_impulse(freq, sr);
        let resp_s = resp.iter().map(|&x| lit(x)).collect::<Vec<_>>().join(", ");
        let _ = writeln!(
            out,
            "    ({}, {}, {}, [{}]),",
            lit(freq),
            lit(sr),
            lit(a),
            resp_s
        );
    }
    out.push_str("];\n");

    let path = format!("{}/src/golden.rs", env!("CARGO_MANIFEST_DIR"));
    let check = std::env::args().any(|a| a == "--check");
    if check {
        let existing = std::fs::read_to_string(&path).unwrap_or_default();
        if existing != out {
            eprintln!("golden.rs is stale — run `cargo run --example gen_vectors`");
            std::process::exit(1);
        }
        println!("golden.rs up to date");
    } else {
        std::fs::write(&path, out).expect("write golden.rs");
        println!("wrote {path}");
    }
}
