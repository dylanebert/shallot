import {
    AmbientLight,
    Camera,
    CameraMode,
    Color,
    Compute,
    checkTextureLimits,
    DirectionalLight,
    GlazePlugin,
    InputPlugin,
    MirrorPlugin,
    mirror,
    Orbit,
    OrbitPlugin,
    Part,
    PartPlugin,
    type Plugin,
    PointLight,
    RenderPlugin,
    run,
    Sear,
    SearPlugin,
    SlabPlugin,
    type State,
    type System,
    Time,
    Transform,
    TransformsPlugin,
} from "@dylanebert/shallot";
import { createBvh } from "@dylanebert/shallot/bvh/core";
import type { BenchmarkAPI, BenchmarkMeasurement } from "@dylanebert/shallot/extras";
import { ProfilePlugin } from "@dylanebert/shallot/extras";
import { arrayFromBitmaps } from "@dylanebert/shallot/render/core";
import {
    BANDWIDTH_SWEEP_BYTES,
    BandwidthPlugin,
    disposeBandwidth,
    getBandwidth,
    setBandwidth,
} from "../bandwidth";
import { type Check, type Params, register, type Scenario } from "../gym";
import { getLoad, LoadPlugin, setLoad } from "../load";
import {
    disposeSubmission,
    getSubmission,
    SUBMISSION_SYSTEM,
    SubmissionPlugin,
    setSubmission,
    setSubmissionPool,
    submissionPool,
} from "../submission";

// one allocating call site attributed by the harness's heap sampling profiler (the GC-pause hunt's "where").
interface TopAlloc {
    name: string;
    location: string;
    bytes: number;
}

declare global {
    interface Window {
        // the CDP allocation probe the harness exposes for the cpu-memory axis (harness/core/page.ts): a
        // collectGarbage-bracketed, no-forced-GC heap delta + GC activity + top allocators over windowMs.
        __probeAlloc?: (
            windowMs: number,
        ) => Promise<{ heapDelta: number; gcCount: number; gcPauseMs: number; top: TopAlloc[] }>;
    }
}

// stress — the bottleneck-saturation atom (Robustness pillar). One scene, several per-resource-axis knobs;
// the assert drives each axis and proves the engine degrades the way we CHOSE under it. The methodology, set
// by the compute axis (sub-stage 0) and copied by every even-pacing axis: induce one resource to its felt-
// lag wall, then prove the chosen response — even slowdown, no unchosen spike — reading the profiler's per-
// pass + per-system metrics (the source of truth, not Mirror). The axes:
//   • COMPUTE (`paceChecks` + the dt-clamp sim) — the shared GPU-ALU burn (../load); the template.
//   • BANDWIDTH (`paceChecks` + the readback-volume spike) — a DRAM streaming burn (../bandwidth), the same
//     even-slowdown response; plus the one bandwidth spike (a large Mirror readback must not sync-stall).
//   • SUBMISSION (`submissionAxis`) — the CPU-side per-frame residual GPU-driven rendering leaves: ECS
//     query iteration + slab writes (../submission). CPU-bound, so it can't ride the GPU-timeline paceChecks
//     — it reads the per-system CPU percentiles directly (the same even-slowdown response, the spike guard
//     promoted to the primary signal). Sub-stage 1, the submission axis.
//   • CPU-MEMORY (`memoryAxis`) — the GC-pause hunt. The chosen response is zero steady-state allocation
//     (no allocation → no GC → no pause), so there's no wall: the proof is allocPerFrame FLAT as work
//     scales (slope ≈ 0), measured via the harness CDP allocation probe (window.__probeAlloc).
//   • GPU-MEMORY (`gpuMemChecks`) — fail-loud, not even slowdown: a large allocation past a device limit
//     throws a named UnsupportedError (the gap-fill).
//
// Unlike a representative content scene (pile/render), a stressor is synthetic single-axis saturation — the
// opposite of triple-duty — so it earns its own atom. The scene is a small lit field of boxes: it exists only
// to give the renderer real sear/glaze/part/light spans to attribute the induced load against.

// the experiment's windows (rAF ticks). The per-pass GPU timing is per-engine-frame accurate regardless
// (busyPerFrameMs divides by Compute.frame), so a modest window suffices; under load fewer engine frames
// elapse per tick, so the base/hot windows are generous for stable per-pass means.
const RAMP_WARMUP = 15;
const RAMP_FRAMES = 80;
const MEASURE_WARMUP = 30;
const MEASURE_FRAMES = 200;

// the ramp ceiling — past the manual `compute` cap, so a fast GPU (a 4090 needs far more iters/lane than
// a Deck to saturate a small scene) still reaches dominance. Mirrors the character probe's LOAD_MAX.
const RAMP_MAX = 262144;
const RAMP_BASE = 2048;

// the bandwidth axis ramps the sweep count (../bandwidth): 1 sweep = 512 MB (read+write of the 256 MB
// buffer), doubling 1→256 until the GPU's per-frame bytes reach the same felt-lag wall. A 4090 (~1 TB/s)
// needs ~55 sweeps (≈28 GB/frame); a lower-bandwidth device fewer. Capped at 256 (128 GB/frame, far past
// any GPU's wall). The base/wall thresholds (WALL_MS / FELT_MS) are axis-shared — the wall is GPU work/frame.
const BW_RAMP_MAX = 256;
const BW_RAMP_BASE = 1;

// the submission axis ramps the per-frame query-and-write pass count (../submission): each pass iterates the
// POOL-entity churn query + writes every member's slab, so per-pass CPU ≈ POOL × (query step + vec4 write).
// A fast desktop CPU reaches the felt-lag wall (cpu work/frame) at ~1k passes (lovelace); capped at 4096
// (far past any CPU's wall, like the compute ceiling for a fast GPU). The wall here is CPU work/frame
// (cpu.total), not GPU busy — submission is the lone CPU-bound axis.
const SUB_RAMP_MAX = 4096;
const SUB_RAMP_BASE = 8;

// the CPU-MEMORY axis (sub-stage 1) — the GC-pause hunt. The chosen response is zero steady-state
// allocation, so there's no wall to ramp to: the proof is that allocation/frame is FLAT as work scales
// (slope ≈ 0). allocPerFrame is the CDP heap delta (window.__probeAlloc — collectGarbage-bracketed,
// no forced GC) over the engine frames Compute.frame advances during the window. Two levers on the same
// churn loop (../submission): the pool size (the firehose — per-entity slab writes) and the query-loop
// pass count (the for…of iterator, now pooled — query.ts).
const MEM_WINDOW_MS = 1000; // per-bracket sampling window (no forced GC) — hundreds of idle frames
const MEM_POOL_LO = 256;
const MEM_POOL_HI = 4096; // = the submission POOL (the warm default)
const MEM_PASS_LO = 16;
const MEM_PASS_HI = 512; // pre-pooled-iterator this is 512 iterators/frame (~20 KB), well above the noise
// the derived zero-allocation bar. The smallest V8 heap object is ≥16 B, and a real per-item allocation is
// ≥1 object/item (a for…of iterator + its IteratorResult is ~2 objects ≈ 32–48 B). So a slope below one
// object proves no per-item object is allocated — sub-object granularity is impossible for a real JS
// per-item allocation. Same bar for both slopes (each asks "is there a per-item object?"). Derived, not
// tuned: the post-fix slopes land ≈ 0 and are reported alongside.
const OBJECT_FLOOR_BYTES = 16;

// the readback-volume spike probe: a "large" GPU→CPU Mirror readback — 32 MB, the size of a 4K `view.tag`
// buffer (one u32/pixel), the realistic upper end of what Mirror reads back in a real scene.
const READBACK_BYTES = 32 << 20;
// the per-frame Mirror ENCODE (copyBufferToBuffer + mapAsync setup) is O(1) command records — independent
// of readback size, so its p99 lands at the ~0.1 ms CPU quantum. 0.5 ms is ~3 quanta: comfortably above
// the O(1) encode yet an order below a size-dependent sync map-and-wait (~5 ms for 32 MB), so it cleanly
// separates "async-staged encode" from "the readback stalled the frame". A derived separator, not a tuned bar.
const READBACK_ENCODE_MARGIN = 0.5;

// SATURATION: the compute span must own ≥80% of GPU busy time — "the compute axis is the bottleneck".
// Hardware-independent (a ratio, not an absolute ms), so it reads the same on every GPU.
const DOMINANCE = 0.8;
// ATTRIBUTION (empirical, the saturation gate's one tuned bar): the non-load spans' total drift between
// the clean and saturated runs must stay under this fraction of the induced load. The render spans are
// ~0.06 ms total against a multi-ms load, so the real ratio is ~0.01 — 0.25 is a wide, robust margin.
// If a future axis can't clear it, that's the roadmap's "→ clarification" signal, surfaced as a failing
// check, not a silent pass.
const ATTRIB = 0.25;

// ── the compute WALL + the even-pacing response (sub-stage 1, the template axis) ──────────────────
// The wall: the load level where the GPU's per-frame work (the present cadence the player feels) enters
// the low-end felt-lag regime — the same felt-lag target the character latency probe calibrates to
// (`scenarios/character.ts` CALIBRATE_MS). A 4090 needs far more iters/lane than a Deck to reach it, so
// the ramp finds it per-hardware and the recorded load is a hardware-specific reference, not a portable
// constant.
const WALL_MS = 28; // GPU work/frame ≈36 fps — felt-lag (the character CALIBRATE_MS precedent); ramp target
const FELT_MS = 22; // the wall-reached gate floor (character FELT_MS): the GPU is in felt-lag at/above it

// EVEN PACING — the chosen response on the compute axis is an even slowdown (a consistent lower framerate),
// NOT an adaptive-quality auto-scaler (roadmap guardrail). "Even" = the engine produces frames at a steady
// cadence and injects no per-frame work beyond the author's chosen GPU load. The proof reads the GPU
// timeline, NOT the raw wall-clock tick interval: under GPU-bound load the chosen 2-frames-in-flight loop
// (memory: "fence-gated loop = 2 GPU frames-in-flight … don't fix it") schedules its CPU ticks in a beat
// around the even GPU-completion cadence — a long tick paired with a short one — and the headless harness's
// free-running rAF (no vsync) samples that beat. The GPU queue absorbs it, so frames still *complete +
// present* at an even rate; the beat is in the CPU observation, not the presented framerate. (Traced
// 2026-06-17: at the wall the fence-completion interval is ~42 ms, stddev ~2 ms, while the CPU tick interval
// swings 4–79 ms.)
//   • frames produced evenly: the load pass's per-occurrence p99 ≈ its mean — the GPU does steady work each
//     frame, so it completes frames at an even cadence. Measured occP99/occMs ≈ 1.04 (sub-5% dispatch
//     jitter); a bursty/uneven load — the thing an even slowdown must NOT be — pushes it well past this.
const OCC_K = 1.25;
//   • no added-work spike: no system's CPU p99 climbs more than this above its clean baseline. This is the
//     precise spike detector (the raw tick interval can't be — the 2-in-flight beat is itself ≈2× the GPU
//     work, so it hides any small spike). A GC pause, a sync pipeline compile, a big sync readback all
//     surface on the CPU side (the per-system p99 the profiler reports); the steady systems here are
//     sub-0.1 ms (100 µs-quantized p99 ≤ ~0.3 ms), so a +1 ms climb is 10× the quantum yet an order below
//     any real spike. Baseline-relative, so it's self-calibrating across hardware.
const CPU_SPIKE_MARGIN = 1.0;
// The timebase fix (the `run` loop drives dt from the rAF presentation timestamp, not callback-time now() —
// Raph Levien, "Swapchains and frame pacing") is validated by measurement (clean idle frame-interval stddev
// 0.06 ms vs 0.31 ms sampling now(), a 5× tightening) but NOT gated: the headless harness's free-running rAF
// is itself irregular (occasional sub-frame double-fires + dropped frames), so the idle spread is too noisy
// for a robust tight bound. The reporter surfaces the idle min/median so a regression is visible.

// SIM STABILITY AT THE CLAMP — a deterministic fixed-step sim the load must not destabilize. A semi-implicit
// (symplectic) Euler harmonic oscillator: its advance is governed only by accumulated fixed time, so it
// stays correct + bounded however the load stretches the render frame-time (the dt clamp bounds steps/frame,
// each fixed step gets the constant FIXED_DT). ω·dt = 0.1 is a deep-stable regime; symplectic Euler's raw
// energy then ripples by Ω/√(4−Ω²) ≈ Ω/2 ≈ 0.05 about E₀, so a 0.1 band (2× the ripple) flags only gross
// instability (NaN / divergence / a variable dt leaking into the fixed step).
const FIXED_DT = 1 / 60;
const SIM_OMEGA = 6; // rad/s → SIM_OMEGA · FIXED_DT = 0.1
const SIM_E0 = 0.5 * SIM_OMEGA * SIM_OMEGA; // energy of (y, v) = (1, 0)
const ENERGY_BAND = 0.1;

// the live knob drives `setLoad` each frame (so the control panel mutates GPU load in place); the assert
// suspends it to own the base/hot schedule itself. Module scope, like every single-instance gym scene.
let assertActive = false;
let stressParams: Params | null = null;

// the fixed-step sim state (the harmonic oscillator above). `tick` counts fixed steps the sim saw, which
// must equal the scheduler's `fixedTick` (no step dropped from the sim's view).
const sim = { y: 1, v: 0, tick: 0 };
function resetSim(): void {
    sim.y = 1;
    sim.v = 0;
    sim.tick = 0;
}
const simEnergy = (): number => 0.5 * sim.v * sim.v + 0.5 * SIM_OMEGA * SIM_OMEGA * sim.y * sim.y;

const StressDriver: Plugin = {
    name: "StressDriver",
    systems: [
        {
            name: "stress-driver",
            group: "draw",
            annotations: { mode: "always" },
            update() {
                if (assertActive) return;
                setLoad(Number(stressParams?.compute ?? 0) | 0);
                setBandwidth(Number(stressParams?.bandwidth ?? 0) | 0);
                setSubmission(Number(stressParams?.submission ?? 0) | 0);
            },
        } satisfies System,
        {
            // the deterministic fixed-step sim — runs every fixed step the scheduler takes, never the
            // stretched render frame. The dt clamp bounds how many fire per frame (no spiral); each gets
            // the constant FIXED_DT (the sim stays correct + bounded). This is what "the dt clamp holds the
            // sim" protects — the sim-stability gate reads its final state.
            name: "stress-sim",
            group: "fixed",
            update() {
                sim.v += -SIM_OMEGA * SIM_OMEGA * sim.y * FIXED_DT;
                sim.y += sim.v * FIXED_DT;
                sim.tick++;
            },
        } satisfies System,
    ],
};

function box(state: State, pos: [number, number, number], color: [number, number, number]): void {
    const eid = state.create();
    state.add(eid, Part); // surface "default", mesh "cube" by trait default
    state.add(eid, Transform);
    state.add(eid, Color);
    Transform.pos.set(eid, pos[0], pos[1], pos[2], 0);
    Color.rgba.set(eid, color[0], color[1], color[2], 1);
}

const scenario: Scenario = {
    name: "stress",
    params: [
        // the COMPUTE axis: per-lane iterations of the `load` burn kernel. A LIVE knob (StressDriver
        // pushes it to setLoad each frame); 0 = clean scene. The assert auto-ramps from RAMP_BASE
        // regardless, so this is for live hand-exploration of the saturation.
        { key: "compute", type: "number", default: 0, min: 0, max: RAMP_MAX, step: 2048 },
        // the BANDWIDTH axis: per-frame sweeps of the DRAM streaming burn (../bandwidth). A LIVE knob, the
        // compute knob's sibling — the assert auto-ramps it regardless, so this is for live exploration.
        { key: "bandwidth", type: "number", default: 0, min: 0, max: BW_RAMP_MAX, step: 1 },
        // the SUBMISSION axis: per-frame query-and-write passes over the churn pool (../submission). A LIVE
        // knob, the CPU-bound sibling — the assert auto-ramps it regardless, so this is for live exploration.
        { key: "submission", type: "number", default: 0, min: 0, max: SUB_RAMP_MAX, step: 8 },
        // the GPU-MEMORY axis: an assert-phase gate (no live effect — a fail-loud ceiling has nothing to
        // ramp). When set, the assert induces an over-budget allocation past THIS device's real limits and
        // proves the chosen response is fail-loud — a named UnsupportedError, not a silent OOM.
        { key: "gpuMem", type: "bool", default: true, label: "gpu-mem fail-loud (assert gate)" },
    ],

    async build(_canvas, p: Params) {
        stressParams = p;
        assertActive = false;
        resetSim();
        const { state, dispose } = await run({
            defaults: false,
            // holds the small lit field + the ../submission churn pool (POOL entities); the GPU axes don't
            // depend on capacity (the churn entities carry no Part/Transform, so they never render)
            capacity: 8192,
            plugins: [
                ProfilePlugin,
                SlabPlugin,
                TransformsPlugin,
                InputPlugin,
                OrbitPlugin,
                RenderPlugin,
                PartPlugin,
                SearPlugin,
                GlazePlugin,
                LoadPlugin,
                BandwidthPlugin,
                SubmissionPlugin,
                MirrorPlugin,
                StressDriver,
            ],
        });

        state.add(state.create(), AmbientLight);
        state.add(state.create(), DirectionalLight);

        // a small lit field: a floor + a 7×7 grid of boxes + two point lights, so the renderer runs real
        // sear / glaze / part / light spans for the attribution to compare the induced compute against.
        box(state, [0, -1, 0], [0.28, 0.3, 0.34]);
        for (let ix = -3; ix <= 3; ix++)
            for (let iz = -3; iz <= 3; iz++) box(state, [ix * 2, 0, iz * 2], [0.55, 0.6, 0.7]);

        for (let i = 0; i < 2; i++) {
            const eid = state.create();
            state.add(eid, PointLight);
            state.add(eid, Transform);
            Transform.pos.set(eid, i === 0 ? -5 : 5, 3, 0, 0);
            PointLight.color.set(eid, i === 0 ? 0xffd9a0 : 0xa0c8ff);
            PointLight.intensity.set(eid, 2);
            PointLight.range.set(eid, 12);
        }

        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Camera);
        state.add(cam, Sear);
        state.add(cam, Orbit);
        Camera.mode.set(cam, CameraMode.Perspective);
        Orbit.yaw.set(cam, Math.PI / 6);
        Orbit.pitch.set(cam, Math.PI / 8);
        Orbit.distance.set(cam, 22);

        return {
            state,
            dispose() {
                setLoad(0);
                setBandwidth(0);
                disposeBandwidth();
                disposeSubmission();
                stressParams = null;
                assertActive = false;
                dispose();
            },
        };
    },

    async assert(state: State): Promise<Check[]> {
        const bench = window.__benchmark;
        if (!bench) return [{ name: "stress: profiler", pass: false, detail: "no __benchmark" }];
        assertActive = true; // own the load + bandwidth schedule; suspend the live driver
        const checks: Check[] = [];
        checks.push(...(await computeAxis(state, bench)));
        checks.push(...(await bandwidthAxis(bench)));
        checks.push(...(await submissionAxis(bench)));
        checks.push(...(await memoryAxis(state)));
        if (stressParams?.gpuMem) checks.push(...(await gpuMemChecks()));
        assertActive = false;
        return checks;
    },

    live(): string {
        return [
            "stress",
            `  compute     ${getLoad()} iters/lane`,
            `  bandwidth   ${getBandwidth()} sweeps`,
            `  submission  ${getSubmission()} passes × ${submissionPool()} ents`,
            "(assert ramps each axis to the wall)",
        ].join("\n");
    },
};

// ── the shared induce-to-wall ramp + even-pacing gate (sub-stage 0's compute gate, span-parameterized) ──
// One ramp to one operating point — the felt-lag wall — feeds every axis's gate. Run a clean baseline
// (knob 0), double the knob until the GPU's per-frame work enters the felt-lag regime (the wall), then a
// full hot measure there. BOTH even-pacing axes (compute, bandwidth) copy this — only the knob + the
// profiler span differ. The wall is GPU work/frame (busyPerFrameMs), axis-independent, so WALL_MS / FELT_MS
// are shared. This is "the full compute-axis gate the later even-pacing axes copy" (roadmap): one source.

const f3 = (x: number): string => x.toFixed(3);
// the wall metric per axis: GPU work/frame for the GPU-bound axes (compute, bandwidth), total per-system
// CPU/frame for the CPU-bound submission axis. WALL_MS / FELT_MS are the same felt-lag thresholds either
// way — the wall is "this resource's per-frame work entered the felt-lag regime", whichever resource.
const busyOf = (m: BenchmarkMeasurement): number => m.gpu?.busyPerFrameMs ?? 0;
const cpuTotalOf = (m: BenchmarkMeasurement): number => m.cpu?.total ?? 0;
const passPerFrame = (m: BenchmarkMeasurement, name: string): number =>
    m.gpu?.passes[name]?.perFrameMs ?? 0;
const spanShare = (m: BenchmarkMeasurement, span: string): number => {
    const busy = m.gpu?.busyPerFrameMs ?? 0;
    return busy > 0 ? passPerFrame(m, span) / busy : 0;
};
// the worst per-system CPU p99 — the precise spike signal (a GC pause / sync compile spikes one system).
const maxSystemP99 = (m: BenchmarkMeasurement): number =>
    Math.max(0, ...Object.values(m.cpu?.systemsP99 ?? {}));

// ramp `set` (the axis knob) from `base0`, doubling until the axis's per-frame work (`wall`) reaches the
// wall (hardware-adaptive: a 4090 needs far more than a Deck; a fast CPU more passes than a slow one),
// capped at `max`. Returns the clean baseline + the hot measure at the wall + the level reached. Short
// windows while ramping, full windows at base + hot. `wall` is the GPU-busy or CPU-total extractor per axis.
async function rampAxis(
    set: (n: number) => void,
    base0: number,
    max: number,
    bench: BenchmarkAPI,
    wall: (m: BenchmarkMeasurement) => number,
): Promise<{ base: BenchmarkMeasurement; hot: BenchmarkMeasurement; level: number }> {
    set(0);
    const base = await bench.measure(MEASURE_WARMUP, MEASURE_FRAMES);
    let level = base0;
    set(level);
    let probe = await bench.measure(RAMP_WARMUP, RAMP_FRAMES);
    while (wall(probe) < WALL_MS && level < max) {
        level = Math.min(level * 2, max);
        set(level);
        probe = await bench.measure(RAMP_WARMUP, RAMP_FRAMES);
    }
    const hot = await bench.measure(MEASURE_WARMUP, MEASURE_FRAMES);
    set(0);
    return { base, hot, level };
}

// the 5 even-pacing gates, keyed on the axis's own profiler `span`: saturation, attribution, the per-pass
// percentile metric resolving, the wall, and even pacing. `axis` / `unit` / `max` shape the messages.
// Returns the gates + a `data` telemetry object + the wall/fps the axis-specific reporter extends.
function paceChecks(
    axis: string,
    span: string,
    unit: string,
    max: number,
    base: BenchmarkMeasurement,
    hot: BenchmarkMeasurement,
    level: number,
): { checks: Check[]; data: Record<string, number>; wallMs: number; fps: number } {
    const inducedHot = passPerFrame(hot, span);
    const spanHot = hot.gpu?.passes[span];
    const busyBase = base.gpu?.busyPerFrameMs ?? 0;
    const busyHot = hot.gpu?.busyPerFrameMs ?? 0;
    const share = spanShare(hot, span);

    // total per-pass drift base→hot, excluding the induced span (it exists only in hot). The induced work
    // should account for nearly all the GPU-busy growth; everything else stays put.
    const names = new Set([
        ...Object.keys(base.gpu?.passes ?? {}),
        ...Object.keys(hot.gpu?.passes ?? {}),
    ]);
    let otherDrift = 0;
    for (const name of names) {
        if (name === span) continue;
        otherDrift += Math.abs(passPerFrame(hot, name) - passPerFrame(base, name));
    }
    const attribution = inducedHot > 0 ? otherDrift / inducedHot : Number.POSITIVE_INFINITY;

    // even-pacing inputs. The GPU-timeline signal (the induced pass occupancy) is the even-production proof;
    // the per-system CPU p99 is the spike guard.
    const occMs = spanHot?.occMs ?? 0;
    const occP99 = spanHot?.occP99 ?? 0;
    const occRatio = occMs > 0 ? occP99 / occMs : Number.POSITIVE_INFINITY;
    const wallMs = busyHot; // the wall = GPU work/frame = the present cadence (1000/busy = the framerate)
    const fps = busyHot > 0 ? 1000 / busyHot : 0;
    const rawMax = hot.frame?.rawMax ?? 0;
    const cpuSpikeBase = maxSystemP99(base);
    const cpuSpikeHot = maxSystemP99(hot);
    const cpuSpikeClimb = cpuSpikeHot - cpuSpikeBase;

    const data: Record<string, number> = {
        level,
        inducedPerFrameMs: inducedHot,
        busyBaseMs: busyBase,
        busyHotMs: busyHot,
        share,
        otherDriftMs: otherDrift,
        attribution,
        wallMs,
        fps,
        occMs,
        occP99,
        occP95: spanHot?.occP95 ?? 0,
        occRatio,
        frameRawMax: rawMax,
        cpuSpikeBase,
        cpuSpikeHot,
    };

    const saturated = share >= DOMINANCE && inducedHot > 0;
    const attributed = attribution < ATTRIB;
    const atWall = wallMs >= FELT_MS;
    // even slowdown = the GPU produces frames at a steady cadence (occP99 ≈ occMs) AND no CPU system spikes
    // beyond its clean baseline (no GC pause / sync compile injecting a frame).
    const evenPaced = occMs > 0 && occRatio <= OCC_K && cpuSpikeClimb <= CPU_SPIKE_MARGIN;

    const checks: Check[] = [
        {
            name: `stress: ${axis} — saturates its own span (the ${span} pass owns the GPU)`,
            pass: saturated,
            detail: saturated
                ? `level=${level} ${unit} → ${span} span ${f3(inducedHot)} ms/frame = ${(share * 100).toFixed(0)}% of GPU busy ${f3(busyHot)} ms (baseline ${f3(busyBase)} ms)`
                : `the knob couldn't dominate this GPU: ${span} span ${f3(inducedHot)} ms = ${(share * 100).toFixed(0)}% of busy ${f3(busyHot)} ms at level=${level} (max ${max}); need ≥${(DOMINANCE * 100).toFixed(0)}% — escalate (the knob or axis split needs rethinking)`,
            data,
        },
        {
            name: `stress: ${axis} — moves only its own span (attribution — no collateral on other passes)`,
            pass: saturated && attributed,
            detail: `other spans drifted Σ${f3(otherDrift)} ms = ${(attribution * 100).toFixed(1)}% of the ${f3(inducedHot)} ms induced ${axis} (bar < ${(ATTRIB * 100).toFixed(0)}%)`,
            data,
        },
        {
            // ties the gate to the per-pass percentile metric: the induced span resolved a spike tail.
            name: `stress: ${axis} — per-pass percentile metric resolves for the saturated span`,
            pass: (spanHot?.occP99 ?? 0) > 0 && (spanHot?.occMs ?? 0) > 0,
            detail: `${span} span occMs ${f3(spanHot?.occMs ?? 0)} · occP95 ${f3(spanHot?.occP95 ?? 0)} · occP99 ${f3(spanHot?.occP99 ?? 0)} ms`,
            data,
        },
        {
            // the WALL: the level that reproduces a low-end felt-lag on this faster hardware. Records the
            // hardware-specific level + frame-time (the regression reference Stage 5 reads).
            name: `stress: ${axis} — the wall: the GPU present cadence enters the felt-lag regime`,
            pass: atWall,
            detail: atWall
                ? `wall level=${level} ${unit} → ${f3(wallMs)} ms GPU work/frame = ${fps.toFixed(0)} fps present cadence ≥ ${FELT_MS} ms felt-lag floor`
                : `the ${axis} knob (max ${max}) couldn't slow this GPU to felt-lag: ${f3(wallMs)} ms GPU work/frame < ${FELT_MS} ms — escalate (the knob ceiling or axis split needs rethinking)`,
            data,
        },
        {
            // EVEN PACING: the chosen response is an even slowdown to a consistent framerate. Read the GPU
            // timeline (the steady-production proof), not the rAF-beat-bearing tick interval. A real spike
            // (GC pause, sync compile, big sync readback) adds wall time beyond the steady GPU work and
            // trips the CPU-spike guard; a bursty GPU load trips the occupancy ratio.
            name: `stress: ${axis} — even pacing: frames produced at a steady cadence, no added-work spike`,
            pass: evenPaced,
            detail: `GPU produces frames at ${fps.toFixed(0)} fps steady (occP99/occMs ${occRatio.toFixed(2)}× ≤ ${OCC_K}×) · no CPU spike (worst system p99 ${f3(cpuSpikeHot)} ms vs ${f3(cpuSpikeBase)} ms idle, +${f3(cpuSpikeClimb)} ≤ ${CPU_SPIKE_MARGIN} ms). Worst tick ${f3(rawMax)} ms is the chosen 2-in-flight beat the GPU queue absorbs — presents stay even.`,
            data,
        },
    ];
    return { checks, data, wallMs, fps };
}

// ── the COMPUTE axis (sub-stage 0, the template): ramp the GPU-ALU load to the wall, run the shared gate,
// then the compute-unique checks — the fixed-step sim stays stable at the dt clamp, plus the reporter.
async function computeAxis(state: State, bench: BenchmarkAPI): Promise<Check[]> {
    const { base, hot, level } = await rampAxis(setLoad, RAMP_BASE, RAMP_MAX, bench, busyOf);
    const { checks, data, wallMs, fps } = paceChecks(
        "compute",
        "load",
        "iters/lane",
        RAMP_MAX,
        base,
        hot,
        level,
    );

    // sim-stability: the fixed-step sim survived the load-stretched frame-time — finite, energy bounded, the
    // fixed clock exact w.r.t. accumulated time, it made progress, the dt clamp bounded steps/frame.
    const energyErr = SIM_E0 > 0 ? Math.abs(simEnergy() - SIM_E0) / SIM_E0 : 0;
    const fixedTick = state.time.fixedTick;
    const clockErr = Math.abs(state.time.elapsed - fixedTick * FIXED_DT);
    const simFinite = Number.isFinite(sim.y) && Number.isFinite(sim.v);
    const stepsPerFrame = hot.frame?.stepsPerFrame ?? 0;
    const clampedFrames = hot.frame?.clampedFrames ?? 0;
    const simStable =
        simFinite &&
        energyErr <= ENERGY_BAND &&
        clockErr < FIXED_DT &&
        fixedTick > 0 &&
        sim.tick === fixedTick && // the sim's fixed system fired on every fixed step — none dropped
        stepsPerFrame <= Time.MAX_FIXED_STEPS;
    Object.assign(data, {
        stepsPerFrame,
        clampedFrames,
        simEnergyErr: energyErr,
        simClockErr: clockErr,
        fixedTick,
    });

    checks.push({
        name: "stress: compute — the fixed-step sim stays stable at the dt clamp",
        pass: simStable,
        detail: `energy err ${(energyErr * 100).toFixed(2)}% (≤ ${(ENERGY_BAND * 100).toFixed(0)}%) · clock err ${f3(clockErr * 1000)} ms (< 1 step) · ${stepsPerFrame.toFixed(2)} steps/frame ≤ ${Time.MAX_FIXED_STEPS} (no spiral; clamp engaged ${clampedFrames}×) · ${sim.tick}/${fixedTick} sim/fixed ticks · finite ${simFinite}`,
        data,
    });
    checks.push({
        name: "measured (compute wall + cadence + timebase + clamp telemetry)",
        pass: true,
        detail: `wall ${f3(wallMs)} ms @ load ${level} · ${fps.toFixed(0)} fps steady · idle ${f3(base.frame?.median ?? 0)} ± ${f3(base.frame?.stddev ?? 0)} ms · ${stepsPerFrame.toFixed(2)} steps/frame · clamped ${clampedFrames}`,
        data,
    });
    return checks;
}

// ── the BANDWIDTH axis (sub-stage 1): ramp the DRAM streaming burn to the wall, run the SAME shared gate
// (the compute template, copied — the chosen response is the same even slowdown), then the readback-volume
// spike (the one bandwidth spike the roadmap names: a large Mirror readback must not stall the frame).
async function bandwidthAxis(bench: BenchmarkAPI): Promise<Check[]> {
    const { base, hot, level } = await rampAxis(
        setBandwidth,
        BW_RAMP_BASE,
        BW_RAMP_MAX,
        bench,
        busyOf,
    );
    const { checks, wallMs, fps } = paceChecks(
        "bandwidth",
        "bandwidth",
        "sweeps",
        BW_RAMP_MAX,
        base,
        hot,
        level,
    );
    const gbPerFrame = (level * BANDWIDTH_SWEEP_BYTES) / (1 << 30);
    // an UPPER bound on real DRAM bandwidth — counts the ~10% L2-resident hits (1 GB buffer / 98 MB L2) as
    // if they were DRAM, so the true DRAM rate is a bit below this. Above the 4090's ~1 TB/s spec ⇒ the
    // burn is L2-cache-contaminated; near or below it ⇒ genuinely DRAM-bound.
    const achievedTbs = wallMs > 0 ? gbPerFrame / (wallMs / 1000) / 1024 : 0;
    checks.push({
        name: "measured (bandwidth wall + achieved memory throughput)",
        pass: true,
        detail: `wall ${level} sweeps → ${gbPerFrame.toFixed(1)} GB/frame at ${f3(wallMs)} ms = ${achievedTbs.toFixed(2)} TB/s memory traffic (L2+DRAM upper bound) · ${fps.toFixed(0)} fps steady`,
    });
    checks.push(...(await readbackChecks(bench)));
    return checks;
}

// the per-frame CPU p99 of the Mirror flush system — the readback ENCODE cost (copy + map setup), which
// must be O(1) in readback size (the READ_RING async-staging property: no sync map-and-wait in the loop).
const mirrorP99 = (m: BenchmarkMeasurement): number => {
    const sys = m.cpu?.systemsP99 ?? {};
    return Math.max(
        0,
        ...Object.entries(sys)
            .filter(([k]) => k.startsWith("Mirror"))
            .map(([, v]) => v),
    );
};

// ── the readback-volume spike: Mirror's READ_RING async staging must keep a LARGE GPU→CPU readback off the
// per-frame critical path (the "big sync readback" spike). Per frame, MirrorSystem does one
// copyBufferToBuffer + one mapAsync — both async (the GPU copy pipelines; the map resolves a microtask
// later), so the per-frame ENCODE cost is O(1) in readback size, not a sync map-and-wait stall. Induce a
// 32 MB readback (a 4K tag buffer) every frame on an idle scene (a tight baseline, so a spike stands out)
// and prove: (1) the readback is LIVE — its snapshot resolves at the full size, so the ring really cycles;
// (2) the per-frame Mirror ENCODE stays bounded vs the no-readback baseline (the async-staging property).
async function readbackChecks(bench: BenchmarkAPI): Promise<Check[]> {
    const device = Compute.device;
    setBandwidth(0); // idle scene — a tight frame distribution, so any readback spike is visible
    const cold = await bench.measure(MEASURE_WARMUP, MEASURE_FRAMES);

    const buf = device.createBuffer({
        label: "readback-stress",
        size: READBACK_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const m = mirror(buf);
    const hot = await bench.measure(MEASURE_WARMUP, MEASURE_FRAMES);
    const snap = m.snapshot;
    m.dispose();
    buf.destroy();

    const live = snap !== null && snap.bytes.byteLength === READBACK_BYTES;
    const encodeBase = mirrorP99(cold);
    const encodeHot = mirrorP99(hot);
    const encodeClimb = encodeHot - encodeBase;
    const frameColdP99 = cold.frame?.p99 ?? 0;
    const frameHotP99 = hot.frame?.p99 ?? 0;
    const frameClimb = frameHotP99 - frameColdP99;
    const mb = READBACK_BYTES / (1 << 20);
    const data: Record<string, number> = {
        readbackMb: mb,
        encodeBaseMs: encodeBase,
        encodeHotMs: encodeHot,
        encodeClimbMs: encodeClimb,
        frameColdP99Ms: frameColdP99,
        frameHotP99Ms: frameHotP99,
        frameClimbMs: frameClimb,
    };

    return [
        {
            // the readback genuinely happens — the ring cycles a full-size snapshot to the CPU. (A skipped
            // ring would leave snapshot null; a partial copy a wrong byteLength.)
            name: "stress: readback — a large Mirror readback is live (the ring cycles a full snapshot)",
            pass: live,
            detail: live
                ? `${mb} MB readback resolved: snapshot ${(snap?.bytes.byteLength ?? 0) / (1 << 20)} MB @ frame ${snap?.frame}`
                : `the ${mb} MB readback never resolved a full-size snapshot (snapshot ${snap ? `${snap.bytes.byteLength} B` : "null"}) — the ring stalled`,
            data,
        },
        {
            // THE GATE — the async-staging property: the per-frame readback ENCODE (copyBufferToBuffer +
            // mapAsync setup) is O(1) in size, so a large readback injects NO synchronous map-and-wait stall
            // into the frame loop. This is the "big sync readback" spike the roadmap names, and Mirror's
            // READ_RING eliminates it (the transfer + the map-callback resolve async, off the critical path).
            name: "stress: readback — async-staging keeps the per-frame encode O(1) in size (no sync map-and-wait stall)",
            pass: live && encodeClimb <= READBACK_ENCODE_MARGIN,
            detail: `Mirror per-frame encode p99 ${f3(encodeHot)} ms vs ${f3(encodeBase)} ms idle (+${f3(encodeClimb)} ≤ ${READBACK_ENCODE_MARGIN} ms) for a ${mb} MB readback — O(1) command records, no sync stall`,
            data,
        },
        {
            // REPORT (the roadmap's "find the spike" — found + characterized + the residual named): the
            // per-frame encode is async-bounded (above), and Mirror copies the resolved range into a buffer
            // reused across readbacks (mirror/index.ts), so a large readback costs no per-frame allocation
            // (the CPU-memory axis's zero-allocation doctrine, applied to the readback path — a fresh
            // 32 MB ArrayBuffer per frame would be major-GC garbage). The residual is the size-dependent
            // main-thread memcpy itself — off the per-frame critical path (a microtask) yet still inflating
            // the frame p99 for a large RAW readback. Two ways to shrink it: (1) Mirror's own design
            // guidance — compact/crop to a small GPU buffer and Mirror THAT (its JSDoc; what physics'
            // pos+quat compaction does), so a raw 32 MB readback is misuse; (2) a zero-copy mapped-range
            // snapshot (defer unmap to slot reclaim, needs ring depth 3) to drop the copy entirely. A
            // scoped follow-on, not a regression — reported so the cost of ignoring (1) is visible.
            name: "measured (readback memcpy residual — no per-frame alloc; compact-first or the zero-copy follow-on)",
            pass: true,
            detail: `a RAW ${mb} MB readback's post-resolve memcpy climbs frame p99 ${f3(frameColdP99)} → ${f3(frameHotP99)} ms (+${f3(frameClimb)}); the per-frame encode stays ${f3(encodeHot)} ms (O(1), no sync stall) and the snapshot buffer is reused (no per-frame allocation). Shrink the copy: compact before Mirror (its design guidance), or a zero-copy snapshot (ring 3).`,
            data,
        },
    ];
}

// ── the SUBMISSION axis (sub-stage 1): the CPU-side per-frame residual — ECS query iteration + slab writes
// (../submission). The CPU-bound axis: GPU-driven rendering dissolves the draw-call overhead, so what's
// left is the JS the scheduler runs each frame — system dispatch, query loops, slab `.set` writes. It can't
// ride paceChecks (that reads the GPU timeline — busyPerFrameMs, per-pass occupancy); the induced span here
// is a SYSTEM (Submission/submission), the wall is CPU work/frame (cpu.total), and the even-pacing proof is
// the per-system CPU percentiles directly — the same CPU-spike guard paceChecks carries, here the primary
// signal. The chosen response is the same EVEN SLOWDOWN (roadmap guardrail, not adaptive quality): the cost
// scales smoothly with the knob (the induced system's p99 ≈ its mean), with no GC spike from the slab-write
// churn (no OTHER system's p99 climbs).

const sysMean = (m: BenchmarkMeasurement, name: string): number => m.cpu?.systems[name] ?? 0;
const sysP99 = (m: BenchmarkMeasurement, name: string): number => m.cpu?.systemsP99[name] ?? 0;

async function submissionAxis(bench: BenchmarkAPI): Promise<Check[]> {
    const { base, hot, level } = await rampAxis(
        setSubmission,
        SUB_RAMP_BASE,
        SUB_RAMP_MAX,
        bench,
        cpuTotalOf,
    );
    const span = SUBMISSION_SYSTEM;
    const inducedMean = sysMean(hot, span);
    const inducedP99 = sysP99(hot, span);
    const cpuHot = hot.cpu?.total ?? 0;
    const cpuBase = base.cpu?.total ?? 0;
    const share = cpuHot > 0 ? inducedMean / cpuHot : 0;

    // attribution: per-system MEAN drift base→hot, excluding the induced system. The induced submission work
    // lands on its own span; the other systems (the slab flush, render encode, membership, the profile
    // systems) stay put. The one expected non-zero drifter is the slab flush — it now packs the churn pool's
    // dirty slots — but that's bounded (a constant POOL slots/frame) and far under the bar.
    const names = new Set([
        ...Object.keys(base.cpu?.systems ?? {}),
        ...Object.keys(hot.cpu?.systems ?? {}),
    ]);
    let otherDrift = 0;
    for (const n of names) {
        if (n === span) continue;
        otherDrift += Math.abs(sysMean(hot, n) - sysMean(base, n));
    }
    const attribution = inducedMean > 0 ? otherDrift / inducedMean : Number.POSITIVE_INFINITY;

    // even pacing, CPU-side: (1) the induced cost is STEADY — the submission system's p99 ≈ its mean, so the
    // per-frame JS scales smoothly with the knob, no spike in the induced work itself (the occP99/occMs
    // analog, on the CPU). (2) no COLLATERAL spike — no OTHER system's p99 climbs beyond its clean baseline:
    // a GC pause from the slab-write churn (the CPU-memory axis it couples to) would surface on some system's
    // p99, immune to the 2-in-flight tick beat.
    const inducedRatio = inducedMean > 0 ? inducedP99 / inducedMean : Number.POSITIVE_INFINITY;
    let otherSpikeClimb = 0;
    let spikeSys = "";
    for (const n of names) {
        if (n === span) continue;
        const climb = sysP99(hot, n) - sysP99(base, n);
        if (climb > otherSpikeClimb) {
            otherSpikeClimb = climb;
            spikeSys = n;
        }
    }

    const saturated = share >= DOMINANCE && inducedMean > 0;
    const attributed = saturated && attribution < ATTRIB;
    const atWall = cpuHot >= FELT_MS;
    const evenPaced =
        inducedMean > 0 && inducedRatio <= OCC_K && otherSpikeClimb <= CPU_SPIKE_MARGIN;
    const fps = cpuHot > 0 ? 1000 / cpuHot : 0;
    const pool = submissionPool();

    const data: Record<string, number> = {
        level,
        poolEntities: pool,
        inducedMeanMs: inducedMean,
        inducedP99Ms: inducedP99,
        inducedRatio,
        cpuBaseMs: cpuBase,
        cpuHotMs: cpuHot,
        share,
        otherDriftMs: otherDrift,
        attribution,
        otherSpikeClimbMs: otherSpikeClimb,
        wallMs: cpuHot,
        fps,
    };

    return [
        {
            name: "stress: submission — saturates the CPU side (the submission system owns the frame's JS)",
            pass: saturated,
            detail: saturated
                ? `level=${level} passes × ${pool} ents → ${span} ${f3(inducedMean)} ms/frame = ${(share * 100).toFixed(0)}% of CPU total ${f3(cpuHot)} ms (baseline ${f3(cpuBase)} ms)`
                : `the knob couldn't dominate this CPU: ${span} ${f3(inducedMean)} ms = ${(share * 100).toFixed(0)}% of CPU total ${f3(cpuHot)} ms at level=${level} (max ${SUB_RAMP_MAX}); need ≥${(DOMINANCE * 100).toFixed(0)}% — escalate (the knob or axis split needs rethinking)`,
            data,
        },
        {
            name: "stress: submission — moves only its own system (attribution — no collateral on other systems)",
            pass: attributed,
            detail: `other systems drifted Σ${f3(otherDrift)} ms = ${(attribution * 100).toFixed(1)}% of the ${f3(inducedMean)} ms induced submission (bar < ${(ATTRIB * 100).toFixed(0)}%)`,
            data,
        },
        {
            // ties the gate to the per-system percentile metric the even-pacing read depends on
            name: "stress: submission — per-system CPU percentile metric resolves for the induced system",
            pass: inducedP99 > 0 && inducedMean > 0,
            detail: `${span} mean ${f3(inducedMean)} · p99 ${f3(inducedP99)} ms`,
            data,
        },
        {
            // the WALL: the pass count that reproduces a low-end CPU-bound felt-lag on this faster CPU.
            // Records the hardware-specific level + frame-time (the Stage 3 regression reference).
            name: "stress: submission — the wall: the CPU per-frame work enters the felt-lag regime",
            pass: atWall,
            detail: atWall
                ? `wall level=${level} passes → ${f3(cpuHot)} ms CPU work/frame = ${fps.toFixed(0)} fps cadence ≥ ${FELT_MS} ms felt-lag floor`
                : `the submission knob (max ${SUB_RAMP_MAX}) couldn't slow this CPU to felt-lag: ${f3(cpuHot)} ms CPU work/frame < ${FELT_MS} ms — escalate (the knob ceiling or axis split needs rethinking)`,
            data,
        },
        {
            // EVEN PACING: the chosen response is an even slowdown — the per-frame JS scales smoothly (the
            // induced system's p99 ≈ mean) with no GC spike (no other system's p99 climbs). Reads the
            // per-system CPU percentiles, not the rAF-beat-bearing tick interval.
            name: "stress: submission — even pacing: cost scales smoothly, no per-system outlier",
            pass: evenPaced,
            detail: `induced cost steady (p99/mean ${inducedRatio.toFixed(2)}× ≤ ${OCC_K}×: ${f3(inducedP99)} / ${f3(inducedMean)} ms) · no collateral spike (worst other system ${spikeSys || "—"} p99 +${f3(otherSpikeClimb)} ≤ ${CPU_SPIKE_MARGIN} ms vs idle) — the slab-write churn injects no GC pause`,
            data,
        },
        {
            name: "measured (submission wall + cadence)",
            pass: true,
            detail: `wall ${f3(cpuHot)} ms @ ${level} passes × ${pool} ents · ${fps.toFixed(0)} fps cadence · induced ${f3(inducedMean)} ms (${(share * 100).toFixed(0)}% of CPU) · idle CPU ${f3(cpuBase)} ms`,
            data,
        },
    ];
}

// ── the CPU-MEMORY axis (sub-stage 1): the GC-pause hunt. The chosen response is zero steady-state
// allocation — if the per-frame loop allocates nothing, there's no GC to pause on. Unlike the even-pacing
// axes there's no wall to ramp to; the proof is that allocPerFrame is FLAT as work scales (the slope ≈ 0,
// the intercept absorbing the work-independent floor — the command encoder, the dirty-slab mapped-range
// views, the per-camera quaternions). Two levers on the churn loop (../submission): the pool size (the
// firehose — per-entity slab writes) and the query-loop pass count (the for…of iterator, pooled in
// query.ts). allocPerFrame = the CDP heap delta (window.__probeAlloc, no forced GC) over the engine frames
// Compute.frame advances. Before-picture (measured by temp-reverting the pooled iterator): the old
// allocating iterator's pass-slope was ~6 B/pass — V8 escape-analyzes most of the simple-loop iterator, so
// it stayed under the bar — and the pooled iterator flattens it to ≈ 0, removing the dependence on that
// elision (provably zero for any loop shape, not just the ones V8 happens to optimize).

interface AllocBracket {
    perFrame: number;
    gcCount: number;
    gcPauseMs: number;
    frames: number;
    top: TopAlloc[];
}

// set the work scale, then measure heap growth over MEM_WINDOW_MS with no forced GC. The window runs
// entirely inside the probe's wall-clock wait (Node-side), so the only JS the page runs during it is the
// engine loop + the induced churn — the measurement itself injects no per-frame allocation. The
// denominator is the engine frames Compute.frame advanced (so a slower hot bracket still normalizes).
async function allocAt(state: State, passes: number, pool: number): Promise<AllocBracket> {
    setSubmission(passes);
    setSubmissionPool(state, pool);
    const f0 = Compute.frame;
    const r = await window.__probeAlloc!(MEM_WINDOW_MS);
    const frames = Math.max(1, Compute.frame - f0);
    return {
        perFrame: r.heapDelta / frames,
        gcCount: r.gcCount,
        gcPauseMs: r.gcPauseMs,
        frames,
        top: r.top,
    };
}

async function memoryAxis(state: State): Promise<Check[]> {
    if (!window.__probeAlloc) {
        return [
            {
                // no environment awareness (gym contract): a live tab / non-Chromium run has no CDP probe,
                // so the hunt is bench-only. Skip-pass with a clear note, not a hard fail.
                name: "stress: cpu-memory — allocation probe (skipped: needs the bench harness)",
                pass: true,
                detail: "window.__probeAlloc absent (live tab / non-Chromium) — the GC-pause hunt runs under bun bench",
            },
        ];
    }

    // ENTITY-slope (the firehose): hold passes at 1, ramp the churn pool. Per-entity slab writes are a
    // direct TypedArray index (no heap object), so allocPerFrame must be flat vs pool size.
    const eLo = await allocAt(state, 1, MEM_POOL_LO);
    const eHi = await allocAt(state, 1, MEM_POOL_HI);
    const entitySlope = (eHi.perFrame - eLo.perFrame) / (MEM_POOL_HI - MEM_POOL_LO);

    // PASS-slope (the iterator residual): hold a small pool, ramp the query-loop passes. Each for…of over
    // the [Churn] query must reuse its pooled iterator, so allocPerFrame must be flat vs pass count.
    const pLo = await allocAt(state, MEM_PASS_LO, MEM_POOL_LO);
    const pHi = await allocAt(state, MEM_PASS_HI, MEM_POOL_LO);
    const passSlope = (pHi.perFrame - pLo.perFrame) / (MEM_PASS_HI - MEM_PASS_LO);

    setSubmission(0);
    setSubmissionPool(state, MEM_POOL_HI); // restore the warm default pool

    const entityZero = entitySlope < OBJECT_FLOOR_BYTES;
    const passZero = passSlope < OBJECT_FLOOR_BYTES;
    // the no-GC confirm: PASS_HI passes drove the steady loop hardest. If it allocated per pass the young
    // gen would fill and scavenge; 0 GC over its window = no GC pause at the heaviest steady load.
    const noGc = pHi.gcCount === 0;
    const top = pHi.top[0] ?? eHi.top[0];
    const where = top ? `${top.name} @ ${top.location}` : "—";

    const data: Record<string, number> = {
        entitySlopeBytes: entitySlope,
        passSlopeBytes: passSlope,
        floorBytes: OBJECT_FLOOR_BYTES,
        entityLoPerFrame: eLo.perFrame,
        entityHiPerFrame: eHi.perFrame,
        passLoPerFrame: pLo.perFrame,
        passHiPerFrame: pHi.perFrame,
        wallGcCount: pHi.gcCount,
        wallGcPauseMs: pHi.gcPauseMs,
        wallFrames: pHi.frames,
    };

    return [
        {
            name: "stress: cpu-memory — per-entity zero allocation (the firehose: slab writes allocate nothing)",
            pass: entityZero,
            detail: entityZero
                ? `allocPerFrame flat vs pool ${MEM_POOL_LO}→${MEM_POOL_HI} ents: ${f3(eLo.perFrame)}→${f3(eHi.perFrame)} B/frame, slope ${f3(entitySlope)} B/entity < ${OBJECT_FLOOR_BYTES} B (no per-entity object)`
                : `allocPerFrame GREW with the pool: slope ${f3(entitySlope)} B/entity ≥ ${OBJECT_FLOOR_BYTES} B (${f3(eLo.perFrame)}→${f3(eHi.perFrame)} B/frame over ${MEM_POOL_LO}→${MEM_POOL_HI}) — a per-entity allocation on the slab-write path; top: ${where}`,
            data,
        },
        {
            name: "stress: cpu-memory — per-pass zero allocation (the query for…of reuses its iterator)",
            pass: passZero,
            detail: passZero
                ? `allocPerFrame flat vs passes ${MEM_PASS_LO}→${MEM_PASS_HI}: ${f3(pLo.perFrame)}→${f3(pHi.perFrame)} B/frame, slope ${f3(passSlope)} B/pass < ${OBJECT_FLOOR_BYTES} B (the pooled iterator allocates nothing)`
                : `allocPerFrame GREW with passes: slope ${f3(passSlope)} B/pass ≥ ${OBJECT_FLOOR_BYTES} B (${f3(pLo.perFrame)}→${f3(pHi.perFrame)} B/frame over ${MEM_PASS_LO}→${MEM_PASS_HI}) — the query loop allocates per pass; top: ${where}`,
            data,
        },
        {
            name: "stress: cpu-memory — no GC pause at the wall (the heaviest loop triggers no collection)",
            pass: noGc,
            detail: noGc
                ? `${MEM_PASS_HI} passes × ${MEM_POOL_LO} ents over ${pHi.frames} frames: 0 GC, 0 ms pause — the steady loop is zero-allocation`
                : `${pHi.gcCount} GC(s), ${f3(pHi.gcPauseMs)} ms pause over ${pHi.frames} frames at ${MEM_PASS_HI} passes — the loop allocates enough to scavenge; top: ${where}`,
            data,
        },
        {
            name: "measured (cpu-memory allocation slopes + the top allocator)",
            pass: true,
            detail: `entity-slope ${f3(entitySlope)} B/ent · pass-slope ${f3(passSlope)} B/pass · floor ${OBJECT_FLOOR_BYTES} B · wall ${pHi.gcCount} GC over ${pHi.frames} frames · top allocator: ${top ? `${top.name} @ ${top.location} (${top.bytes} B sampled)` : "none"}`,
            data,
        },
    ];
}

// ── the GPU-MEMORY axis (sub-stage 1): induce an over-budget allocation, prove the chosen response ──
// The chosen response on the memory axis is fail-loud (the roadmap guardrail — a hard ceiling, not an even
// slowdown): a named UnsupportedError (the buffer + needed-vs-available + remedy) before the bare allocation
// OOMs silently or surfaces an opaque WebGPU validation error. The contact store already had this guard
// (`checkContactStore`); this gap-fills every other large/fixed-cap allocation a heavy scene grows — the BVH
// node buffer (storage binding), the image array (texture-array layers), the VAT (texture dimension). Each
// check drives the real allocation entry point past THIS device's actual limit (so the recorded ceiling is
// hardware-specific, like the compute wall), proving the guard is wired, not merely present. Independent of
// the even-pacing axes; the pure guard logic is `bun test` red-first (engine/runtime/gpu.test.ts).

// pass = the over-limit ask threw a NAMED UnsupportedError. A non-named error (a generic GPU validation
// error) or no throw is the silent-OOM failure the guard exists to replace. Keyed on the error name, not
// instanceof, so any bundling boundary between the gym and the engine can't mask a real pass.
const unsupported = (err: unknown): boolean =>
    err instanceof Error && err.name === "UnsupportedError";
const memMb = (bytes: number): string => `${(bytes / (1 << 20)).toFixed(0)} MB`;
const memDetail = (err: unknown, induced: string): string =>
    unsupported(err)
        ? `${induced} → threw ${(err as Error).name}: ${(err as Error).message}`
        : err
          ? `${induced} → threw ${(err as Error).name} (NOT UnsupportedError — an opaque GPU error, the silent failure the guard must replace)`
          : `${induced} → did NOT throw (silent OOM — the allocation was unguarded)`;

async function gpuMemChecks(): Promise<Check[]> {
    const device = Compute.device;
    const maxBinding = device.limits.maxStorageBufferBindingSize;
    const maxLayers = device.limits.maxTextureArrayLayers;
    const maxDim = device.limits.maxTextureDimension2D;
    const checks: Check[] = [];

    // 1. BVH node buffer (storage binding). A maxPrims whose 2·maxPrims nodes (32 B each) exceed the
    // per-binding limit; the guard fires before createBvh allocates or compiles, so nothing heavy runs.
    // Overshoot to ~1.5× the binding so the node buffer trips (not the smaller prim buffer) AND the
    // needed-vs-available MB read distinctly (the minimal +1-byte trip rounds both to the same MB).
    const overPrims = Math.floor((maxBinding / 64) * 1.5); // node = 64·prims ≈ 1.5·maxBinding; prims ≈ 0.75·
    let bvhErr: unknown;
    try {
        (await createBvh(device, overPrims)).destroy(); // destroy only reached if unguarded (silent OOM)
    } catch (e) {
        bvhErr = e;
    }
    checks.push({
        name: "stress: gpu-mem — the BVH node buffer fails loud past maxStorageBufferBindingSize",
        pass: unsupported(bvhErr),
        detail: memDetail(
            bvhErr,
            `createBvh(${overPrims} prims) → node buffer > ${memMb(maxBinding)} binding`,
        ),
    });

    // 2. image array (texture-array layers). One 1×1 bitmap referenced (maxLayers+1)× — arrayFromBitmaps
    // guards on the layer count before any per-layer resize/upload, so this is cheap. The shared
    // gltf-baseColor / sprite-atlas path.
    const bmp = await createImageBitmap(new ImageData(1, 1));
    let arrErr: unknown;
    try {
        (await arrayFromBitmaps(device, new Array(maxLayers + 1).fill(bmp))).destroy();
    } catch (e) {
        arrErr = e;
    } finally {
        bmp.close();
    }
    checks.push({
        name: "stress: gpu-mem — an image array fails loud past maxTextureArrayLayers",
        pass: unsupported(arrErr),
        detail: memDetail(
            arrErr,
            `arrayFromBitmaps(${maxLayers + 1} layers) > ${maxLayers} maxTextureArrayLayers`,
        ),
    });

    // 3. VAT texture dimension. The skin path bakes a vertCount × frameCount texture; `assembleVat` is
    // gltf-internal (not a clean import), so drive the same `checkTextureLimits` guard it calls directly
    // against this device's real maxTextureDimension2D — the dimension branch the layers check (2) doesn't
    // reach. The guard's wiring into assembleVat is covered red-first in bun test.
    let vatErr: unknown;
    try {
        checkTextureLimits(
            "[gltf] a skinned mesh's VAT",
            { width: maxDim + 1, height: 1 },
            device.limits,
            "Reduce the unique vertex count or bake fewer frames.",
        );
    } catch (e) {
        vatErr = e;
    }
    checks.push({
        name: "stress: gpu-mem — a VAT texture fails loud past maxTextureDimension2D",
        pass: unsupported(vatErr),
        detail: memDetail(vatErr, `${maxDim + 1}×1 VAT > ${maxDim} maxTextureDimension2D`),
    });

    return checks;
}

register(scenario);
