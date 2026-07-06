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

fn main() {
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
