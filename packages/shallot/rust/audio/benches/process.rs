//! Audio kernel perf + scaling harness.
//!
//! `process()` renders one `BLOCK_SIZE` (128-sample) block — the unit the
//! AudioWorklet calls on its realtime thread. The worklet has a hard deadline:
//! one block must render in 128 / sample_rate seconds (~2.67 ms at 48 kHz), and
//! the floor is Steam Deck / mobile, not desktop. This sweeps `process()` over
//! voice count, instrument complexity, and spatialization and reports µs/block
//! against that budget, with the cost split into two regimes the kernel runs:
//!
//!   - **fixed** — the FDN reverb (always) + binaural HRTF render (when any
//!     spatial voice is live). Constant in voice count. Measured as the
//!     intercept of a least-squares fit over the voice sweep.
//!   - **per-voice** — graph synthesis + per-voice spatial encode + the
//!     reflection convolver. Scales with voice count. Measured as the slope.
//!
//! The regimes are separable so a later change is attributable: a binaural/FDN
//! change moves the intercept, a per-sample-synthesis change moves the slope.
//! The harness is self-validating — `osc-filter-env` vs `const-filter-env` is a
//! sine-source flipped to a constant, so their slope delta is the per-sample
//! oscillator cost, which must be positive.
//!
//! Native, not WASM: the relative scaling + regime split is portable, and the
//! crate's `lib` target builds natively. Absolute on-floor numbers ride the
//! Playwright harness periodically (the way render validates 4090 vs Deck).
//!
//! Run: `cargo bench` (or `cargo bench --bench process`) from `rust/audio`.
//!
//! Why a hand-rolled timer over criterion's harness: the deliverable is a
//! slope/intercept fit across the voice sweep plus a budget-% table, which
//! needs every point's mean back in-process to fit and print. Criterion's model
//! benches each point independently and writes its stats to disk, so the fit
//! would have to re-measure or scrape JSON regardless. A focused median-of-
//! batches timer keeps the kernel dependency-free (it has zero deps today) and
//! prints the decomposition directly.

use shallot_audio::{AudioEngine, BLOCK_SIZE, MAX_VOICES};
use std::hint::black_box;
use std::time::Instant;

const SAMPLE_RATE: f32 = 48_000.0;

// NodeType discriminants (mirror graph::NodeType — passed as raw u32 to the
// instrument-build ABI, so the bench doesn't need the private enum).
const OSC: u32 = 1;
const FILTER: u32 = 2;
const ENV: u32 = 3;
const CONST: u32 = 6;
const SAMPLE: u32 = 7;
const DELAY: u32 = 8;
const DYNAMICS: u32 = 9;
const WAVESHAPER: u32 = 10;
const EQ: u32 = 11;
const CHORUS: u32 = 12;
const NO_BUF: u32 = 0xFF;

const WARMUP_BLOCKS: usize = 256;
const SAMPLES: usize = 25;
const BATCH_BLOCKS: usize = 32;
const REFLECT_IR_LEN: usize = 512;
const COUNTS: [usize; 10] = [0, 1, 2, 4, 8, 16, 24, 32, 48, 64];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Complexity {
    /// One Sample node reading a looping buffer — the cheap reference, no
    /// per-sample transcendental, just smoothing + Hermite interpolation.
    Sample,
    /// Sine osc → SVF filter → envelope. The heavy realistic voice: per-sample
    /// `sin` (osc) + per-sample `tan` (SVF coefficient recompute).
    OscFilterEnv,
    /// Constant source → filter → envelope. `osc-filter-env` with the sine
    /// flipped to a constant — isolates the oscillator's per-sample cost.
    ConstFilterEnv,
    /// Constant source → dynamics (compressor mode). Isolates the dynamics
    /// node's per-sample `log10`/`powf` cost against `const-filter-env`'s
    /// per-sample `exp` (the envelope curve) baseline — the stage-2 provisional
    /// flag's bench (audio-effect-nodes.md).
    ConstDynamics,
    /// The maximal standard insert rack over a sample source: 3×EQ biquads →
    /// dynamics → waveshaper → delay → chorus → envelope (9 nodes, the heaviest
    /// realistic voice a game authors). Records the full effect-chain per-block
    /// cost for the audio-effect-nodes close-out; the delta over `sample` is the
    /// whole rack's per-voice price.
    EffectChain,
}

impl Complexity {
    fn label(self) -> &'static str {
        match self {
            Complexity::Sample => "sample",
            Complexity::OscFilterEnv => "osc-filter-env",
            Complexity::ConstFilterEnv => "const-filter-env",
            Complexity::ConstDynamics => "const-dynamics",
            Complexity::EffectChain => "effect-chain",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Spatial {
    /// Stereo passthrough — no HRTF, no FOA encode, no convolver.
    Dry,
    /// Per-voice air absorption + occlusion + FOA encode, mixed to the binaural
    /// bus rendered through the fixed HRTF.
    Spatial,
    /// Spatial plus the per-voice reflection convolver (partitioned FFT).
    Reflect,
}

impl Spatial {
    fn label(self) -> &'static str {
        match self {
            Spatial::Dry => "dry",
            Spatial::Spatial => "spatial",
            Spatial::Reflect => "reflect",
        }
    }
}

/// Construct on a wide stack — `AudioEngine::new` builds a large value before
/// boxing its members, the same reason the unit tests spawn a 4 MB thread.
fn new_engine() -> Box<AudioEngine> {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| Box::new(AudioEngine::new(SAMPLE_RATE)))
        .unwrap()
        .join()
        .unwrap()
}

/// Flush denormals to zero. Browsers run WASM with FTZ/DAZ, so this matches the
/// worklet's arithmetic and keeps the silent-FDN-tail at zero voices from
/// dropping into denormal-slow math and inflating the fixed-cost intercept.
#[cfg(target_arch = "x86_64")]
#[allow(deprecated)]
fn enable_ftz() {
    use core::arch::x86_64::{_mm_getcsr, _mm_setcsr};
    // MXCSR bit 15 = Flush-To-Zero (outputs), bit 6 = Denormals-Are-Zero (inputs)
    unsafe {
        _mm_setcsr(_mm_getcsr() | 0x8040);
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn enable_ftz() {}

/// Fill sample buffer 0 of instrument 0 with a looping one-period sine, so a
/// Sample node's read does real Hermite-interpolation work.
fn alloc_loop_sample(e: &mut AudioEngine) {
    let len = 4800usize;
    let ptr = e.sample_alloc(0, 0, 1, len as u32);
    unsafe {
        for i in 0..len {
            let phase = i as f32 / len as f32 * std::f32::consts::TAU;
            *ptr.add(i) = phase.sin() * 0.5;
        }
    }
}

/// Build instrument 0 for the config, and (for `Sample`) populate buffer 0 and
/// (for `Reflect`) stage the reflection IR — anything the voices read.
fn setup_config(e: &mut AudioEngine, complexity: Complexity, spatial: Spatial) {
    match complexity {
        Complexity::Sample => {
            e.set_instrument(0, 1, 0, NO_BUF);
            e.set_instrument_node(0, 0, SAMPLE, NO_BUF, NO_BUF, 0, 0);
            alloc_loop_sample(e);
        }
        Complexity::OscFilterEnv | Complexity::ConstFilterEnv => {
            // source(buf0,off0) -> filter(buf1,off4) -> envelope(buf2,off8)
            let source = if complexity == Complexity::OscFilterEnv {
                OSC
            } else {
                CONST
            };
            e.set_instrument(0, 3, 2, NO_BUF);
            e.set_instrument_node(0, 0, source, NO_BUF, NO_BUF, 0, 0);
            e.set_instrument_node(0, 1, FILTER, 0, NO_BUF, 1, 4);
            e.set_instrument_node(0, 2, ENV, 1, NO_BUF, 2, 8);
        }
        Complexity::ConstDynamics => {
            // source(buf0,off0) -> dynamics(buf1,off1)
            e.set_instrument(0, 2, 1, NO_BUF);
            e.set_instrument_node(0, 0, CONST, NO_BUF, NO_BUF, 0, 0);
            e.set_instrument_node(0, 1, DYNAMICS, 0, NO_BUF, 1, 1);
        }
        Complexity::EffectChain => {
            // A linear 9-node rack ping-ponging graph buffers 0/1: the sample node
            // writes buffer 0, each effect reads the previous buffer and writes the
            // other, the envelope's output (buffer 0) is the instrument output. 43
            // params (<=64), 2 graph buffers (<=8). The delay + chorus DelayLine
            // side-state allocates in set_voice_instrument, so no extra wiring here.
            e.set_instrument(0, 9, 0, NO_BUF);
            e.set_instrument_node(0, 0, SAMPLE, NO_BUF, NO_BUF, 0, 0); // 5 params
            e.set_instrument_node(0, 1, EQ, 0, NO_BUF, 1, 5); // 4
            e.set_instrument_node(0, 2, EQ, 1, NO_BUF, 0, 9); // 4
            e.set_instrument_node(0, 3, EQ, 0, NO_BUF, 1, 13); // 4
            e.set_instrument_node(0, 4, DYNAMICS, 1, NO_BUF, 0, 17); // 8
            e.set_instrument_node(0, 5, WAVESHAPER, 0, NO_BUF, 1, 25); // 3
            e.set_instrument_node(0, 6, DELAY, 1, NO_BUF, 0, 28); // 4
            e.set_instrument_node(0, 7, CHORUS, 0, NO_BUF, 1, 32); // 4
            e.set_instrument_node(0, 8, ENV, 1, NO_BUF, 0, 36); // 7 -> 43
            alloc_loop_sample(e);
        }
    }

    if spatial == Spatial::Reflect {
        // a short decaying-noise reflection IR, deterministic per run
        let ptr = e.ir_staging_ptr();
        let mut seed: u32 = 0x9E3779B9;
        unsafe {
            for i in 0..REFLECT_IR_LEN {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let noise = (seed >> 9) as f32 / (1u32 << 23) as f32 * 2.0 - 1.0;
                let decay = (-(i as f32) / (REFLECT_IR_LEN as f32 * 0.3)).exp();
                *ptr.add(i) = noise * decay * 0.3;
            }
        }
    }

    if spatial != Spatial::Dry {
        e.set_reverb(0.6, 0.5, 0.4, 0.4);
    }
}

/// Activate voice `vi`, set its params, position it, and gate it on. Voices
/// hold at envelope sustain (no gate-off) so they synthesize for the whole run.
fn arm_voice(e: &mut AudioEngine, vi: u32, complexity: Complexity, spatial: Spatial) {
    e.set_voice_instrument(vi, 0);
    e.voice_active(vi, 1);

    match complexity {
        Complexity::Sample => {
            e.set_param(vi, 0, 0.0); // buffer id
            e.set_param(vi, 1, 1.0 + vi as f32 * 0.001); // rate (vary slightly)
            e.set_param(vi, 2, 1.0); // looping
            e.set_param(vi, 3, 0.4); // volume
        }
        Complexity::OscFilterEnv => {
            e.set_param(vi, 0, 110.0 + vi as f32 * 3.0); // freq (vary per voice)
            e.set_param(vi, 1, 0.0); // waveform: sine
            e.set_param(vi, 2, 0.0); // wavetable pos
            e.set_param(vi, 3, 0.5); // volume
            set_filter_env_params(e, vi);
        }
        Complexity::ConstFilterEnv => {
            e.set_param(vi, 0, 0.5); // constant source value
            set_filter_env_params(e, vi);
        }
        Complexity::ConstDynamics => {
            e.set_param(vi, 0, 0.5); // constant source value — above the default threshold
            set_dynamics_params(e, vi);
        }
        Complexity::EffectChain => set_effect_chain_params(e, vi),
    }

    if spatial != Spatial::Dry {
        e.set_voice_spatial(vi, 1);
        let az = vi as f32 * 0.31; // spread directions across the sphere
        e.set_spatial(vi, az, 0.0, 3.0, 1.0, 100.0, 1.0);
        if spatial == Spatial::Reflect {
            e.set_reflection_ir(vi, REFLECT_IR_LEN as u32);
            e.set_reflection_gain(vi, 0.5);
        }
    }

    e.set_gate(vi, 1);
}

fn set_filter_env_params(e: &mut AudioEngine, vi: u32) {
    // filter (offset 4): cutoff, Q, mode, mix=1 (mix=0 bypasses the filter)
    e.set_param(vi, 4, 2000.0);
    e.set_param(vi, 5, 0.707);
    e.set_param(vi, 6, 0.0);
    e.set_param(vi, 7, 1.0);
    // envelope (offset 8): attack, decay, sustain>0 (holds steady), release, 3 curves
    e.set_param(vi, 8, 0.005);
    e.set_param(vi, 9, 0.05);
    e.set_param(vi, 10, 0.7);
    e.set_param(vi, 11, 0.2);
    e.set_param(vi, 12, 0.0);
    e.set_param(vi, 13, 0.0);
    e.set_param(vi, 14, 0.0);
}

fn set_dynamics_params(e: &mut AudioEngine, vi: u32) {
    // dynamics (offset 1): mode=compressor, threshold, ratio, knee, attack, release, makeup, mix
    e.set_param(vi, 1, 0.0);
    e.set_param(vi, 2, -18.0);
    e.set_param(vi, 3, 4.0);
    e.set_param(vi, 4, 6.0);
    e.set_param(vi, 5, 0.005);
    e.set_param(vi, 6, 0.05);
    e.set_param(vi, 7, 0.0);
    e.set_param(vi, 8, 1.0);
}

/// Set every node in the effect-chain rack so nothing bypasses — the mix-driven
/// nodes (waveshaper/delay/chorus) run at mix>0, the EQ biquads at non-unit gain,
/// the sample source loops. Offsets follow the setup_config chain order.
fn set_effect_chain_params(e: &mut AudioEngine, vi: u32) {
    // sample (offset 0): bufferId, rate, loop, volume, channel
    e.set_param(vi, 0, 0.0);
    e.set_param(vi, 1, 1.0 + vi as f32 * 0.001);
    e.set_param(vi, 2, 1.0);
    e.set_param(vi, 3, 0.4);
    e.set_param(vi, 4, 0.0);
    // eq1/2/3 (offsets 5/9/13): mode, freq, gain, q — peak boost, low shelf, high cut
    e.set_param(vi, 5, 1.0);
    e.set_param(vi, 6, 800.0);
    e.set_param(vi, 7, 2.0);
    e.set_param(vi, 8, 1.0);
    e.set_param(vi, 9, 0.0);
    e.set_param(vi, 10, 200.0);
    e.set_param(vi, 11, 1.5);
    e.set_param(vi, 12, 0.707);
    e.set_param(vi, 13, 2.0);
    e.set_param(vi, 14, 6000.0);
    e.set_param(vi, 15, 0.7);
    e.set_param(vi, 16, 0.707);
    // dynamics (offset 17): mode, threshold, ratio, knee, attack, release, makeup, mix
    e.set_param(vi, 17, 0.0);
    e.set_param(vi, 18, -18.0);
    e.set_param(vi, 19, 4.0);
    e.set_param(vi, 20, 6.0);
    e.set_param(vi, 21, 0.005);
    e.set_param(vi, 22, 0.05);
    e.set_param(vi, 23, 0.0);
    e.set_param(vi, 24, 1.0);
    // waveshaper (offset 25): mode=soft, drive, mix
    e.set_param(vi, 25, 0.0);
    e.set_param(vi, 26, 0.5);
    e.set_param(vi, 27, 0.5);
    // delay (offset 28): time, feedback, damp, mix
    e.set_param(vi, 28, 0.15);
    e.set_param(vi, 29, 0.3);
    e.set_param(vi, 30, 0.3);
    e.set_param(vi, 31, 0.3);
    // chorus (offset 32): rate, depth, feedback, mix
    e.set_param(vi, 32, 1.5);
    e.set_param(vi, 33, 0.5);
    e.set_param(vi, 34, 0.2);
    e.set_param(vi, 35, 0.4);
    // envelope (offset 36): attack, decay, sustain>0, release, 3 curves
    e.set_param(vi, 36, 0.005);
    e.set_param(vi, 37, 0.05);
    e.set_param(vi, 38, 0.7);
    e.set_param(vi, 39, 0.2);
    e.set_param(vi, 40, 0.0);
    e.set_param(vi, 41, 0.0);
    e.set_param(vi, 42, 0.0);
}

/// Median µs/block for the engine in its current state. Warms to steady state
/// (envelopes at sustain, filters/FDN settled), then takes the median of
/// `SAMPLES` batch means — median resists scheduler hiccups.
fn measure_block_us(e: &mut AudioEngine) -> f64 {
    for _ in 0..WARMUP_BLOCKS {
        black_box(e.process());
    }
    let mut means = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let t0 = Instant::now();
        for _ in 0..BATCH_BLOCKS {
            black_box(e.process());
        }
        let ns = t0.elapsed().as_nanos() as f64;
        means.push(ns / BATCH_BLOCKS as f64 / 1000.0);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    means[SAMPLES / 2]
}

/// Ordinary least-squares fit of `us = intercept + slope * voices` over the
/// active points (voices >= 1). Intercept = fixed regime, slope = per-voice.
fn fit(points: &[(usize, f64)]) -> (f64, f64) {
    let active: Vec<(f64, f64)> = points
        .iter()
        .filter(|(n, _)| *n >= 1)
        .map(|(n, us)| (*n as f64, *us))
        .collect();
    let k = active.len() as f64;
    let mean_x = active.iter().map(|(x, _)| x).sum::<f64>() / k;
    let mean_y = active.iter().map(|(_, y)| y).sum::<f64>() / k;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for (x, y) in &active {
        sxy += (x - mean_x) * (y - mean_y);
        sxx += (x - mean_x) * (x - mean_x);
    }
    let slope = sxy / sxx;
    let intercept = mean_y - slope * mean_x;
    (intercept, slope)
}

struct Sweep {
    complexity: Complexity,
    spatial: Spatial,
    points: Vec<(usize, f64)>,
    fixed: f64,
    per_voice: f64,
    total_max: f64,
}

fn run_config(complexity: Complexity, spatial: Spatial) -> Sweep {
    let mut e = new_engine();
    e.set_real_voice_budget(MAX_VOICES as u32); // every armed voice synthesizes
    setup_config(&mut e, complexity, spatial);

    let mut points = Vec::with_capacity(COUNTS.len());
    let mut armed = 0u32;
    for &n in COUNTS.iter() {
        while (armed as usize) < n {
            arm_voice(&mut e, armed, complexity, spatial);
            armed += 1;
        }
        points.push((n, measure_block_us(&mut e)));
    }

    let (fixed, per_voice) = fit(&points);
    let total_max = points.last().unwrap().1;
    Sweep {
        complexity,
        spatial,
        points,
        fixed,
        per_voice,
        total_max,
    }
}

fn main() {
    enable_ftz();

    let budget_us = BLOCK_SIZE as f64 / SAMPLE_RATE as f64 * 1e6;
    let pct = |us: f64| us / budget_us * 100.0;

    println!("audio kernel perf + scaling harness");
    println!(
        "sample_rate={} Hz  block={}  worklet budget={:.1} us/block\n",
        SAMPLE_RATE as u32, BLOCK_SIZE, budget_us
    );

    let complexities = [
        Complexity::Sample,
        Complexity::OscFilterEnv,
        Complexity::ConstFilterEnv,
        Complexity::ConstDynamics,
        Complexity::EffectChain,
    ];
    let spatials = [Spatial::Dry, Spatial::Spatial, Spatial::Reflect];

    let mut results = Vec::new();
    for &c in complexities.iter() {
        for &s in spatials.iter() {
            let r = run_config(c, s);
            println!("== complexity={}  spatial={} ==", c.label(), s.label());
            println!("  voices   us/block   %budget");
            for (n, us) in &r.points {
                println!("  {:>5}   {:>8.3}   {:>6.2}%", n, us, pct(*us));
            }
            let max_n = *COUNTS.last().unwrap();
            println!(
                "  regimes: fixed={:.3} us ({:.2}%)   per-voice={:.4} us/voice ({:.3}%/voice)",
                r.fixed,
                pct(r.fixed),
                r.per_voice,
                pct(r.per_voice),
            );
            let headroom = 100.0 - pct(r.total_max);
            let capacity = if r.per_voice > 0.0 {
                ((budget_us - r.fixed) / r.per_voice).floor()
            } else {
                f64::INFINITY
            };
            println!(
                "  {}-voice total: {:.3} us ({:.2}% budget, {:.1}% headroom, capacity ~{:.0} voices)\n",
                max_n,
                r.total_max,
                pct(r.total_max),
                headroom,
                capacity,
            );
            results.push(r);
        }
    }

    let find = |c: Complexity, s: Spatial| {
        results
            .iter()
            .find(|r| r.complexity == c && r.spatial == s)
            .unwrap()
    };

    println!("SUMMARY (us/block)");
    println!(
        "  {:<18} {:<8} {:>8} {:>11} {:>8} {:>11}",
        "complexity", "spatial", "fixed", "per-voice", "t@64", "%budget@64"
    );
    for r in &results {
        println!(
            "  {:<18} {:<8} {:>8.3} {:>11.4} {:>8.2} {:>10.2}%",
            r.complexity.label(),
            r.spatial.label(),
            r.fixed,
            r.per_voice,
            r.total_max,
            pct(r.total_max),
        );
    }

    println!("\nregime attribution (us):");
    let bin_fixed = find(Complexity::OscFilterEnv, Spatial::Spatial).fixed
        - find(Complexity::OscFilterEnv, Spatial::Dry).fixed;
    println!(
        "  binaural fixed cost   = fixed(spatial) - fixed(dry)              = {:+.3} us",
        bin_fixed
    );
    let pv_spatial = find(Complexity::OscFilterEnv, Spatial::Spatial).per_voice
        - find(Complexity::OscFilterEnv, Spatial::Dry).per_voice;
    println!(
        "  per-voice spatial     = per-voice(spatial) - per-voice(dry)      = {:+.4} us/voice",
        pv_spatial
    );
    let pv_conv = find(Complexity::OscFilterEnv, Spatial::Reflect).per_voice
        - find(Complexity::OscFilterEnv, Spatial::Spatial).per_voice;
    println!(
        "  per-voice convolver   = per-voice(reflect) - per-voice(spatial)  = {:+.4} us/voice",
        pv_conv
    );
    let osc_cost = find(Complexity::OscFilterEnv, Spatial::Dry).per_voice
        - find(Complexity::ConstFilterEnv, Spatial::Dry).per_voice;
    println!(
        "  per-sample oscillator = per-voice(osc) - per-voice(const) [dry]   = {:+.4} us/voice  <- sine->constant validation (expect > 0)",
        osc_cost
    );
    let dynamics_cost = find(Complexity::ConstDynamics, Spatial::Dry).per_voice
        - find(Complexity::ConstFilterEnv, Spatial::Dry).per_voice;
    println!(
        "  dynamics transcendentals = per-voice(const-dynamics) - per-voice(const-filter-env) [dry] = {:+.4} us/voice  <- log10+powf vs the envelope's exp() baseline",
        dynamics_cost
    );
    let rack_cost = find(Complexity::EffectChain, Spatial::Dry).per_voice
        - find(Complexity::Sample, Spatial::Dry).per_voice;
    println!(
        "  effect rack           = per-voice(effect-chain) - per-voice(sample) [dry]     = {:+.4} us/voice  <- 3xEQ+dynamics+waveshaper+delay+chorus over the bare sample voice",
        rack_cost
    );

    if osc_cost <= 0.0 {
        eprintln!(
            "\nWARNING: sine->constant validation failed (oscillator cost {:.4} <= 0); \
             the harness is not resolving the per-sample synthesis cost.",
            osc_cost
        );
    }
}
