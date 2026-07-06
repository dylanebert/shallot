//! Native reference vector for the wasm parity differential.
//!
//! Runs a fixed, deterministic scene (osc → filter → env, two spatial voices
//! with reflection + modest reverb) for `BLOCKS` blocks and writes the stereo
//! output as little-endian f32 to the path in argv[1]. The native build is the
//! golden-trusted reference (`cargo test` bit-checks it against the captured
//! references); `scripts/audio-wasm-bench.ts` replays the *identical* call
//! sequence on the shipped `.wasm` and asserts a bounded diff, so wasm-specific
//! reassociation (the `core::arch::wasm32` intrinsics the native golden gate
//! can't see — they're `cfg(wasm32)`-gated) is caught.
//!
//! THE SCENE MUST STAY BYTE-IDENTICAL to `buildDiffScene` in
//! `scripts/audio-wasm-bench.ts`. Any drift between the two surfaces as a diff
//! failure — which is the point of the gate, not a bug in it.
//!
//! Run: `cargo run --release --example diff_vector -- <out.bin>` from rust/audio.

use shallot_audio::{AudioEngine, BLOCK_SIZE};
use std::io::Write;

const SAMPLE_RATE: f32 = 48_000.0;
const BLOCKS: usize = 24;
const REFLECT_IR_LEN: usize = 512;

// NodeType discriminants (mirror graph::NodeType — see benches/process.rs).
const OSC: u32 = 1;
const FILTER: u32 = 2;
const ENV: u32 = 3;
const NO_BUF: u32 = 0xFF;

/// Build the fixed scene and run it. Returns the captured stereo output,
/// `BLOCKS` blocks of `[left[128], right[128]]` back to back.
fn render() -> Vec<f32> {
    let mut e = AudioEngine::new(SAMPLE_RATE);
    e.set_real_voice_budget(64);

    // instrument 0: osc(buf0,off0) -> filter(buf1,off4) -> env(buf2,off8)
    e.set_instrument(0, 3, 2, NO_BUF);
    e.set_instrument_node(0, 0, OSC, NO_BUF, NO_BUF, 0, 0);
    e.set_instrument_node(0, 1, FILTER, 0, NO_BUF, 1, 4);
    e.set_instrument_node(0, 2, ENV, 1, NO_BUF, 2, 8);

    // a short decaying-noise reflection IR, deterministic (same LCG as the bench)
    let ptr = e.ir_staging_ptr();
    let mut seed: u32 = 0x9E3779B9;
    // SAFETY: the staging buffer holds MAX_IR_SAMPLES (>= REFLECT_IR_LEN), so
    // writes over [0, REFLECT_IR_LEN) are in bounds.
    unsafe {
        for i in 0..REFLECT_IR_LEN {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (seed >> 9) as f32 / (1u32 << 23) as f32 * 2.0 - 1.0;
            let decay = (-(i as f32) / (REFLECT_IR_LEN as f32 * 0.3)).exp();
            *ptr.add(i) = noise * decay * 0.3;
        }
    }

    // modest reverb — present but short, so the FDN tail doesn't dominate the
    // diff with chaotic near-unity-feedback divergence (a stability artifact,
    // not a correctness one). Keeps the gate measuring the synthesis path.
    e.set_reverb(0.4, 0.4, 0.4, 0.3);

    for vi in 0..2u32 {
        e.set_voice_instrument(vi, 0);
        e.voice_active(vi, 1);
        e.set_param(vi, 0, 110.0 + vi as f32 * 30.0); // freq
        e.set_param(vi, 1, 0.0); // waveform: sine
        e.set_param(vi, 2, 0.0); // wavetable pos
        e.set_param(vi, 3, 0.5); // volume
        e.set_param(vi, 4, 2000.0); // filter cutoff
        e.set_param(vi, 5, 0.707); // filter Q
        e.set_param(vi, 6, 0.0); // filter mode: lowpass
        e.set_param(vi, 7, 1.0); // filter mix
        e.set_param(vi, 8, 0.005); // env attack
        e.set_param(vi, 9, 0.05); // env decay
        e.set_param(vi, 10, 0.7); // env sustain
        e.set_param(vi, 11, 0.2); // env release
        e.set_param(vi, 12, 0.0); // attack curve
        e.set_param(vi, 13, 0.0); // decay curve
        e.set_param(vi, 14, 0.0); // release curve

        e.set_voice_spatial(vi, 1);
        e.set_spatial(vi, vi as f32 * 0.3, 0.0, 3.0, 1.0, 100.0, 1.0);
        e.set_reflection_ir(vi, REFLECT_IR_LEN as u32);
        e.set_reflection_gain(vi, 0.5);

        e.set_gate(vi, 1);
    }

    let mut out = Vec::with_capacity(BLOCKS * BLOCK_SIZE * 2);
    for _ in 0..BLOCKS {
        let ptr = e.process();
        // SAFETY: process() returns &self.output ([f32; BLOCK_SIZE*2]) as a
        // pointer, valid and live until the next process() call.
        let block = unsafe { std::slice::from_raw_parts(ptr, BLOCK_SIZE * 2) };
        out.extend_from_slice(block);
    }
    out
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: diff_vector <out.bin>");

    // Construct on a wide stack — AudioEngine::new builds a large value before
    // boxing its members (same reason the unit tests spawn a 4 MB thread).
    let out = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(render)
        .unwrap()
        .join()
        .unwrap();

    let mut bytes = Vec::with_capacity(out.len() * 4);
    for s in &out {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    let mut f = std::fs::File::create(&path).expect("create out file");
    f.write_all(&bytes).expect("write out file");
    eprintln!("diff_vector: wrote {} samples to {}", out.len(), path);
}
