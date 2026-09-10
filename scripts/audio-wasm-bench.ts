// Audio kernel wasm perf + parity harness — the wasm twin of
// `rust/audio/benches/process.rs`.
//
// Why this exists and the native bench can't be it: the native `cargo bench`
// compiles x86 (already SSE-vectorized at -O3), so it can't see the wasm
// scalar→SIMD delta. This loads the *shipped* `.wasm` under bun and drives the
// real `audio_*` C ABI, so the regime fit (fixed + per-voice·N) reflects the
// codegen that actually ships. It builds two artifacts — scalar (current flags)
// and `+simd128` — and reports them A/B.
//
// It also runs the **parity differential**: the native build (golden-trusted)
// renders a fixed scene via `cargo run --example diff_vector`; this replays the
// identical scene on each wasm build and asserts a bounded diff. That gates the
// wasm SIMD path, which the native `golden.rs` gate can't see (the
// `core::arch::wasm32` intrinsics 1.3+ adds are `cfg(wasm32)`-gated).
//
// And a **vectorization inspector**: tallies `v128` ops per function in a
// non-stripped build, the "which loops vectorized" read 1.2's autovec audit
// consumes.
//
// Run: `bun run audio:wasm-bench` from packages/shallot.

import { existsSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { $ } from "bun";

const audioDir = resolve(import.meta.dir, "../rust/audio");
const rawWasm = resolve(audioDir, "target/wasm32-unknown-unknown/release/shallot_audio.wasm");
const OUT = "/tmp/shallot-audio-wasm";
if (!existsSync(OUT)) mkdirSync(OUT);
const scalarWasm = `${OUT}/audio_scalar.wasm`;
const simdWasm = `${OUT}/audio_simd.wasm`;
const namedWasm = `${OUT}/audio_simd_named.wasm`; // non-stripped, for the inspector
const nativeRef = `${OUT}/diff_native.bin`;

const WASM_OPT_BASE = ["-O3", "--enable-nontrapping-float-to-int", "--enable-bulk-memory"];

// --- kernel constants (mirror benches/process.rs) ---------------------------
const SAMPLE_RATE = 48_000;
const BLOCK_SIZE = 128;
const MAX_VOICES = 64;
const WARMUP_BLOCKS = 256;
const SAMPLES = 15;
const BATCH_BLOCKS = 48;
const NSWEEPS = 3; // per-point min across full sweeps rejects sustained interference
const REFLECT_IR_LEN = 512;
const COUNTS = [0, 1, 2, 4, 8, 16, 24, 32, 48, 64];
const BUDGET_US = (BLOCK_SIZE / SAMPLE_RATE) * 1e6; // 2.67 ms worklet block

// NodeType discriminants (mirror graph::NodeType)
const OSC = 1;
const FILTER = 2;
const ENV = 3;
const CONST = 6;
const SAMPLE = 7;
const NO_BUF = 0xff;

type Complexity = "sample" | "osc-filter-env" | "const-filter-env";
type Spatial = "dry" | "spatial" | "reflect";

// ---------------------------------------------------------------------------
// Engine: one wasm instance = one pristine engine (the global-singleton model,
// so a fresh instance per config mirrors the bench's `new_engine()`).
// ---------------------------------------------------------------------------
class Engine {
    ex: any;
    constructor(module: WebAssembly.Module) {
        this.ex = new WebAssembly.Instance(module, {}).exports;
        this.ex.audio_init(SAMPLE_RATE);
    }
    // memory.buffer detaches when wasm memory grows (alloc paths), so always
    // build a fresh view against the current buffer.
    f32(ptr: number, len: number): Float32Array {
        return new Float32Array(this.ex.memory.buffer, ptr, len);
    }
    process(): number {
        return this.ex.audio_process();
    }
    readBlock(): Float32Array {
        const ptr = this.ex.audio_process();
        return this.f32(ptr, BLOCK_SIZE * 2).slice(); // copy out before next call
    }
}

function setupConfig(e: Engine, complexity: Complexity, spatial: Spatial) {
    const ex = e.ex;
    if (complexity === "sample") {
        ex.audio_set_instrument(0, 1, 0);
        ex.audio_set_instrument_node(0, 0, SAMPLE, NO_BUF, NO_BUF, 0, 0);
        const len = 4800;
        const ptr = ex.audio_sample_alloc(0, len);
        const buf = e.f32(ptr, len);
        for (let i = 0; i < len; i++) buf[i] = Math.sin((i / len) * Math.PI * 2) * 0.5;
    } else {
        const source = complexity === "osc-filter-env" ? OSC : CONST;
        ex.audio_set_instrument(0, 3, 2);
        ex.audio_set_instrument_node(0, 0, source, NO_BUF, NO_BUF, 0, 0);
        ex.audio_set_instrument_node(0, 1, FILTER, 0, NO_BUF, 1, 4);
        ex.audio_set_instrument_node(0, 2, ENV, 1, NO_BUF, 2, 8);
    }

    if (spatial === "reflect") {
        const ptr = ex.audio_ir_staging_ptr();
        const buf = e.f32(ptr, REFLECT_IR_LEN);
        let seed = 0x9e3779b9 >>> 0;
        for (let i = 0; i < REFLECT_IR_LEN; i++) {
            seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
            const noise = ((seed >>> 9) / (1 << 23)) * 2 - 1;
            const decay = Math.exp(-i / (REFLECT_IR_LEN * 0.3));
            buf[i] = noise * decay * 0.3;
        }
    }
    if (spatial !== "dry") ex.audio_set_reverb(0.6, 0.5, 0.4, 0.4);
}

function filterEnvParams(ex: any, vi: number) {
    ex.audio_set_param(vi, 4, 2000.0);
    // biome-ignore lint/suspicious/noApproximativeNumericConstant: filter Q is the literal 0.707 the native reference (diff_vector.rs) uses — Math.SQRT1_2 would desync the parity differential.
    ex.audio_set_param(vi, 5, 0.707);
    ex.audio_set_param(vi, 6, 0.0);
    ex.audio_set_param(vi, 7, 1.0);
    ex.audio_set_param(vi, 8, 0.005);
    ex.audio_set_param(vi, 9, 0.05);
    ex.audio_set_param(vi, 10, 0.7);
    ex.audio_set_param(vi, 11, 0.2);
    ex.audio_set_param(vi, 12, 0.0);
    ex.audio_set_param(vi, 13, 0.0);
    ex.audio_set_param(vi, 14, 0.0);
}

function armVoice(e: Engine, vi: number, complexity: Complexity, spatial: Spatial) {
    const ex = e.ex;
    ex.audio_set_voice_instrument(vi, 0);
    ex.audio_voice_active(vi, 1);
    if (complexity === "sample") {
        ex.audio_set_param(vi, 0, 0.0);
        ex.audio_set_param(vi, 1, 1.0 + vi * 0.001);
        ex.audio_set_param(vi, 2, 1.0);
        ex.audio_set_param(vi, 3, 0.4);
    } else if (complexity === "osc-filter-env") {
        ex.audio_set_param(vi, 0, 110.0 + vi * 3.0);
        ex.audio_set_param(vi, 1, 0.0);
        ex.audio_set_param(vi, 2, 0.0);
        ex.audio_set_param(vi, 3, 0.5);
        filterEnvParams(ex, vi);
    } else {
        ex.audio_set_param(vi, 0, 0.5);
        filterEnvParams(ex, vi);
    }
    if (spatial !== "dry") {
        ex.audio_set_voice_spatial(vi, 1);
        ex.audio_set_spatial(vi, vi * 0.31, 0.0, 3.0, 1.0, 100.0, 1.0);
        if (spatial === "reflect") {
            ex.audio_set_reflection_ir(vi, REFLECT_IR_LEN);
            ex.audio_set_reflection_gain(vi, 0.5);
        }
    }
    ex.audio_set_gate(vi, 1);
}

let sink = 0; // accumulates process() returns so the engine can't DCE timed calls; read once in main()

// Tier the wasm Module up to JSC's top compiler before any config is timed —
// the compiled code is shared across instances of one Module, so warming any
// instance warms them all. Without this the first configs measure cold (BBQ
// tier) and the per-voice signal (~1us, smaller than the tier gap) inverts.
function warmModule(module: WebAssembly.Module) {
    const e = new Engine(module);
    e.ex.audio_set_real_voice_budget(MAX_VOICES);
    // reflect keeps the convolver in the (LTO-inlined) hot path warm too.
    setupConfig(e, "osc-filter-env", "reflect");
    for (let vi = 0; vi < 16; vi++) armVoice(e, vi, "osc-filter-env", "reflect");
    for (let i = 0; i < 8_000; i++) sink += e.process();
}

// Min-of-batches us/block. Min (not median) is the right estimator for a
// CPU-bound microbench: the noise is one-sided (scheduler preemption only adds
// time), so the least-disturbed batch is closest to the true cost. WARMUP per
// point still settles DSP state (envelopes to sustain, FDN/filters settled).
function measureBlockUs(e: Engine): number {
    for (let i = 0; i < WARMUP_BLOCKS; i++) sink += e.process();
    let best = Infinity;
    for (let s = 0; s < SAMPLES; s++) {
        const t0 = Bun.nanoseconds();
        for (let b = 0; b < BATCH_BLOCKS; b++) sink += e.process();
        best = Math.min(best, (Bun.nanoseconds() - t0) / BATCH_BLOCKS / 1000);
    }
    return best;
}

// OLS fit of us = intercept + slope*voices over the active points (n >= 1).
function fit(points: [number, number][]): { fixed: number; perVoice: number } {
    const active = points.filter(([n]) => n >= 1);
    const k = active.length;
    const mx = active.reduce((a, [x]) => a + x, 0) / k;
    const my = active.reduce((a, [, y]) => a + y, 0) / k;
    let sxy = 0;
    let sxx = 0;
    for (const [x, y] of active) {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx) * (x - mx);
    }
    const perVoice = sxy / sxx;
    return { fixed: my - perVoice * mx, perVoice };
}

interface Sweep {
    complexity: Complexity;
    spatial: Spatial;
    points: [number, number][];
    fixed: number;
    perVoice: number;
    totalMax: number;
}

function runConfig(module: WebAssembly.Module, complexity: Complexity, spatial: Spatial): Sweep {
    // Each sweep is a fresh instance armed incrementally (voice count only
    // grows). Repeating the whole sweep and taking the per-point min rejects a
    // point whose ~seconds-long window caught sustained interference — `min`
    // *within* a point can't, since the whole window is slow.
    const perPoint: number[][] = COUNTS.map(() => []);
    for (let sweep = 0; sweep < NSWEEPS; sweep++) {
        const e = new Engine(module);
        e.ex.audio_set_real_voice_budget(MAX_VOICES);
        setupConfig(e, complexity, spatial);
        let armed = 0;
        for (let ci = 0; ci < COUNTS.length; ci++) {
            while (armed < COUNTS[ci]) armVoice(e, armed++, complexity, spatial);
            perPoint[ci].push(measureBlockUs(e));
        }
    }
    const points = COUNTS.map((n, ci) => [n, Math.min(...perPoint[ci])] as [number, number]);
    const { fixed, perVoice } = fit(points);
    return { complexity, spatial, points, fixed, perVoice, totalMax: points.at(-1)![1] };
}

const COMPLEXITIES: Complexity[] = ["sample", "osc-filter-env", "const-filter-env"];
const SPATIALS: Spatial[] = ["dry", "spatial", "reflect"];

// Sweep both builds config-interleaved: each config's scalar and simd sweeps
// run back-to-back, so slow system drift between them can't corrupt the A/B
// delta (the quantity that matters).
function sweepBoth(scalarMod: WebAssembly.Module, simdMod: WebAssembly.Module) {
    const scalar: Sweep[] = [];
    const simd: Sweep[] = [];
    // spatial outer, complexity inner: the per-voice attribution pairs
    // (osc−const, both same spatial) land adjacent, so drift over the run can't
    // bias one against the other.
    for (const s of SPATIALS)
        for (const c of COMPLEXITIES) {
            scalar.push(runConfig(scalarMod, c, s));
            simd.push(runConfig(simdMod, c, s));
        }
    return { scalar, simd };
}

const find = (rs: Sweep[], c: Complexity, s: Spatial) =>
    rs.find((r) => r.complexity === c && r.spatial === s)!;

// ---------------------------------------------------------------------------
// Parity differential — replays diff_vector.rs's scene EXACTLY. Keep in sync.
// ---------------------------------------------------------------------------
const DIFF_BLOCKS = 24;
function runDiffScene(module: WebAssembly.Module): Float32Array {
    const e = new Engine(module);
    const ex = e.ex;
    ex.audio_set_real_voice_budget(64);
    ex.audio_set_instrument(0, 3, 2);
    ex.audio_set_instrument_node(0, 0, OSC, NO_BUF, NO_BUF, 0, 0);
    ex.audio_set_instrument_node(0, 1, FILTER, 0, NO_BUF, 1, 4);
    ex.audio_set_instrument_node(0, 2, ENV, 1, NO_BUF, 2, 8);

    const ptr = ex.audio_ir_staging_ptr();
    const ir = e.f32(ptr, REFLECT_IR_LEN);
    let seed = 0x9e3779b9 >>> 0;
    for (let i = 0; i < REFLECT_IR_LEN; i++) {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        const noise = ((seed >>> 9) / (1 << 23)) * 2 - 1;
        const decay = Math.exp(-i / (REFLECT_IR_LEN * 0.3));
        ir[i] = noise * decay * 0.3;
    }
    ex.audio_set_reverb(0.4, 0.4, 0.4, 0.3);

    for (let vi = 0; vi < 2; vi++) {
        ex.audio_set_voice_instrument(vi, 0);
        ex.audio_voice_active(vi, 1);
        ex.audio_set_param(vi, 0, 110.0 + vi * 30.0);
        ex.audio_set_param(vi, 1, 0.0);
        ex.audio_set_param(vi, 2, 0.0);
        ex.audio_set_param(vi, 3, 0.5);
        ex.audio_set_param(vi, 4, 2000.0);
        // biome-ignore lint/suspicious/noApproximativeNumericConstant: filter Q is the literal 0.707 the native reference (diff_vector.rs) uses — Math.SQRT1_2 would desync the parity differential.
        ex.audio_set_param(vi, 5, 0.707);
        ex.audio_set_param(vi, 6, 0.0);
        ex.audio_set_param(vi, 7, 1.0);
        ex.audio_set_param(vi, 8, 0.005);
        ex.audio_set_param(vi, 9, 0.05);
        ex.audio_set_param(vi, 10, 0.7);
        ex.audio_set_param(vi, 11, 0.2);
        ex.audio_set_param(vi, 12, 0.0);
        ex.audio_set_param(vi, 13, 0.0);
        ex.audio_set_param(vi, 14, 0.0);
        ex.audio_set_voice_spatial(vi, 1);
        ex.audio_set_spatial(vi, vi * 0.3, 0.0, 3.0, 1.0, 100.0, 1.0);
        ex.audio_set_reflection_ir(vi, REFLECT_IR_LEN);
        ex.audio_set_reflection_gain(vi, 0.5);
        ex.audio_set_gate(vi, 1);
    }

    const out = new Float32Array(DIFF_BLOCKS * BLOCK_SIZE * 2);
    for (let b = 0; b < DIFF_BLOCKS; b++) out.set(e.readBlock(), b * BLOCK_SIZE * 2);
    return out;
}

// Tolerance is DERIVED, not tuned. The two builds differ only by (a) x86 FMA
// contraction vs wasm's separate mul+add and (b) glibc libm vs the Rust `libm`
// crate's transcendentals — both ≈ 1 ulp/op. Propagated through the stable
// IIR/convolver path over 24 blocks that stays ~1e-5 (measured). A real port
// bug (a wrong SIMD reduction) produces O(0.1–1) error — orders of magnitude
// above. 1e-3 sits in that gap; the harness prints the measured floor so the
// 2-order separation is visible, not assumed.
const DIFF_TOL = 1e-3;

function diff(a: Float32Array, b: Float32Array): { max: number; rms: number } {
    let max = 0;
    let sse = 0;
    for (let i = 0; i < a.length; i++) {
        const d = Math.abs(a[i] - b[i]);
        if (d > max) max = d;
        sse += d * d;
    }
    return { max, rms: Math.sqrt(sse / a.length) };
}

async function loadNativeRef(): Promise<Float32Array> {
    return new Float32Array(await Bun.file(nativeRef).arrayBuffer());
}

// ---------------------------------------------------------------------------
// Vectorization inspector — v128 op tally per function (non-stripped build).
// ---------------------------------------------------------------------------
// Legacy `_ZN<len><seg>…17h<hash>E` → `seg::seg::…`; non-mangled names (the
// `#[no_mangle]` ABI exports) pass through. Turns the inspector's keys from raw
// mangle into readable paths like `shallot_audio::graph::synthesize_graph_voice`.
function demangle(s: string): string {
    if (!s.startsWith("_ZN")) return s;
    const parts: string[] = [];
    let i = 3;
    while (i < s.length) {
        let j = i;
        while (j < s.length && s[j] >= "0" && s[j] <= "9") j++;
        if (j === i) break;
        const seg = s.slice(j, j + Number(s.slice(i, j)));
        if (/^h[0-9a-f]+$/.test(seg)) break; // the trailing disambiguator hash
        parts.push(seg);
        i = j + seg.length;
    }
    return parts.join("::") || s;
}

async function v128PerFunction(
    wasmPath: string,
): Promise<{ total: number; top: [string, number][] }> {
    const text = await $`wasm-opt --print ${wasmPath}`.text();
    const perFn = new Map<string, number>();
    let cur = "(top-level)";
    let total = 0;
    for (const line of text.split("\n")) {
        const m = line.match(/^\s*\(func \$([^\s)]+)/);
        if (m) cur = demangle(m[1]);
        const hits = (line.match(/\bv128\.|f32x4\.|i32x4\.|f64x2\./g) || []).length;
        if (hits) {
            perFn.set(cur, (perFn.get(cur) || 0) + hits);
            total += hits;
        }
    }
    const top = [...perFn.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12);
    return { total, top };
}

// ---------------------------------------------------------------------------
// Builds
// ---------------------------------------------------------------------------
async function buildWasm(simd: boolean, out: string, strip = true) {
    const env: Record<string, string> = {};
    for (const [k, v] of Object.entries(process.env)) if (v !== undefined) env[k] = v;
    // Set RUSTFLAGS explicitly both ways. The shipped .cargo/config.toml now
    // sets +simd128 for wasm32, and RUSTFLAGS (even empty) overrides config —
    // so the scalar leg must clear it, else it would inherit +simd128 from the
    // file and the A/B would compare simd against simd.
    env.RUSTFLAGS = simd ? "-C target-feature=+simd128" : "";
    const stripArg = strip ? [] : ["--config", "profile.release.strip=false"];
    await $`cargo build --target wasm32-unknown-unknown --release ${stripArg}`
        .cwd(audioDir)
        .env(env);
    const optFlags = simd ? [...WASM_OPT_BASE, "--enable-simd"] : WASM_OPT_BASE;
    // -g keeps the name section through wasm-opt (else it strips what
    // strip=false preserved), so the inspector reads named functions not indices.
    const keepNames = strip ? [] : ["-g"];
    await $`wasm-opt ${optFlags} ${keepNames} ${rawWasm} -o ${out}`;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------
const pct = (us: number) => (us / BUDGET_US) * 100;
const fnum = (x: number, w = 8, d = 3) => x.toFixed(d).padStart(w);

function printAB(scalar: Sweep[], simd: Sweep[]) {
    console.log(
        "\nSUMMARY — fixed (intercept) + per-voice (slope), us/block, scalar vs +simd128\n",
    );
    console.log(
        `  ${"complexity".padEnd(16)} ${"spatial".padEnd(8)} ` +
            `${"fixed.s".padStart(8)} ${"fixed.simd".padStart(10)} ${"Δ%".padStart(6)}   ` +
            `${"pv.s".padStart(7)} ${"pv.simd".padStart(8)} ${"Δ%".padStart(6)}`,
    );
    for (let i = 0; i < scalar.length; i++) {
        const s = scalar[i];
        const v = simd[i];
        const dFixed = s.fixed !== 0 ? ((v.fixed - s.fixed) / s.fixed) * 100 : 0;
        const dPv = s.perVoice !== 0 ? ((v.perVoice - s.perVoice) / s.perVoice) * 100 : 0;
        console.log(
            `  ${s.complexity.padEnd(16)} ${s.spatial.padEnd(8)} ` +
                `${fnum(s.fixed)} ${fnum(v.fixed, 10)} ${fnum(dFixed, 6, 1)}   ` +
                `${fnum(s.perVoice, 7, 4)} ${fnum(v.perVoice, 8, 4)} ${fnum(dPv, 6, 1)}`,
        );
    }

    const attr = (rs: Sweep[]) => ({
        binaural:
            find(rs, "osc-filter-env", "spatial").fixed - find(rs, "osc-filter-env", "dry").fixed,
        osc:
            find(rs, "osc-filter-env", "dry").perVoice -
            find(rs, "const-filter-env", "dry").perVoice,
        conv:
            find(rs, "osc-filter-env", "reflect").perVoice -
            find(rs, "osc-filter-env", "spatial").perVoice,
    });
    const a = attr(scalar);
    const b = attr(simd);
    console.log("\nregime attribution (us) — the 1.3+ targets, scalar → simd:");
    console.log(`  binaural fixed (1.3)   ${fnum(a.binaural)} → ${fnum(b.binaural)}`);
    console.log(
        `  per-voice osc sin      ${fnum(a.osc, 8, 4)} → ${fnum(b.osc, 8, 4)}  (Stage 2 owns the sine)`,
    );
    console.log(`  per-voice convolver(1.4)${fnum(a.conv, 7, 4)} → ${fnum(b.conv, 8, 4)}`);

    // Self-validation: osc−const is the per-sample `sin` cost, so it MUST be
    // positive (the same gate the native bench asserts). If it inverts, the
    // measurement noise is swamping the per-voice signal — distrust the slopes.
    if (a.osc <= 0 || b.osc <= 0)
        console.log(
            `  WARNING: osc−const slope <= 0 (scalar ${a.osc.toFixed(4)}, simd ${b.osc.toFixed(4)}) ` +
                `— per-voice numbers are noise-dominated, not a trustworthy baseline.`,
        );

    const worst = (rs: Sweep[]) => find(rs, "osc-filter-env", "reflect").totalMax;
    console.log(
        `\n  heaviest config (osc-filter-env +reflect, 64 voices): ` +
            `scalar ${fnum(worst(scalar))} us (${pct(worst(scalar)).toFixed(1)}%)  →  ` +
            `simd ${fnum(worst(simd))} us (${pct(worst(simd)).toFixed(1)}%)  of the ${BUDGET_US.toFixed(0)}us budget`,
    );
}

// ---------------------------------------------------------------------------
async function main() {
    console.log("audio kernel wasm perf + parity harness");
    console.log(
        `sample_rate=${SAMPLE_RATE} block=${BLOCK_SIZE} budget=${BUDGET_US.toFixed(1)}us/block`,
    );

    console.log("\nbuilding native reference vector (golden-trusted)...");
    await $`cargo run --release --example diff_vector -- ${nativeRef}`.cwd(audioDir).quiet();

    console.log("building scalar wasm...");
    await buildWasm(false, scalarWasm);
    console.log("building +simd128 wasm...");
    await buildWasm(true, simdWasm);
    console.log("building +simd128 wasm (non-stripped, for the inspector)...");
    await buildWasm(true, namedWasm, false);

    const scalarMod = await WebAssembly.compile(await Bun.file(scalarWasm).arrayBuffer());
    const simdMod = await WebAssembly.compile(await Bun.file(simdWasm).arrayBuffer());

    // --- parity differential (gate) ---
    console.log("\n=== PARITY DIFFERENTIAL (native vs wasm) ===");
    const ref = await loadNativeRef();
    let parityOk = true;
    for (const [label, mod] of [
        ["scalar", scalarMod],
        ["+simd128", simdMod],
    ] as const) {
        const out = runDiffScene(mod);
        const { max, rms } = diff(ref, out);
        const ok = max < DIFF_TOL && Number.isFinite(max);
        parityOk &&= ok;
        console.log(
            `  ${label.padEnd(9)} max=${max.toExponential(3)} rms=${rms.toExponential(3)} ` +
                `tol=${DIFF_TOL.toExponential(0)}  ${ok ? "PASS" : "FAIL"}`,
        );
    }

    // --- A/B perf sweep ---
    console.log("\nwarming both modules to top JIT tier...");
    warmModule(scalarMod);
    warmModule(simdMod);
    console.log("running config-interleaved scalar/simd sweep...");
    const { scalar, simd } = sweepBoth(scalarMod, simdMod);
    printAB(scalar, simd);

    // --- vectorization inspector ---
    console.log(
        "\n=== VECTORIZATION INSPECTOR (+simd128 shipped, flag only, no hand-vectorization) ===",
    );
    const { total, top } = await v128PerFunction(namedWasm);
    console.log(`  total v128 ops: ${total}`);
    console.log("  top functions by v128 count (LTO folds the binaural/FDN/mix loops into");
    console.log("  audio_process; the per-voice graph path stays separate — finer per-loop");
    console.log("  attribution within audio_process needs #[inline(never)] probes in 1.2):");
    for (const [fn, n] of top) console.log(`    ${String(n).padStart(5)}  ${fn}`);

    // Observable read of the DCE-defeat accumulator (a process() pointer is
    // always a finite int, so this never throws — it just keeps `sink` live).
    if (!Number.isFinite(sink)) throw new Error(`sink ${sink}`);

    console.log(`\n${parityOk ? "PARITY PASS" : "PARITY FAIL"}`);
    if (!parityOk) process.exit(1);
}

await main();
