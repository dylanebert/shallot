import type { State } from "../../engine";
import { Compute } from "../../engine";
import type { Profile } from "./index";

/** one GPU pass, measured per-occurrence — the robust unit for a mixed fixed/variable engine where a
 *  pass belongs to one clock (a sim pass fires per fixed step, a render pass per frame). */
export interface BenchmarkPass {
    /** per-occurrence (per-fire) GPU time — the per-step cost for a sim pass, per-frame for a render
     *  pass. Exact (cumulative time ÷ fire count), not the display map's greedy-held value. */
    occMs: number;
    /** p95 of the per-occurrence GPU time over the window — the spike tail the {@link occMs} window
     *  mean averages away. A sample is one drained frame's time (== per-occurrence for a per-frame
     *  render pass; the summed occurrences for a multi-step sim frame). 0 when no samples landed. */
    occP95: number;
    /** p99 of the per-occurrence GPU time — the worst-frame cost of this pass. A pass whose `occP99`
     *  far exceeds its `occMs` is spiking; pairing it with the frame-time spike attributes the stall to
     *  its span (the stress-suite saturation gate reads this). */
    occP99: number;
    /** amortized over every frame in the window (`occMs × firesPerFrame`) */
    perFrameMs: number;
    /** fire cadence: occurrences per frame — ≈1 for a per-frame render pass, ≈steps/frame for sim */
    firesPerFrame: number;
    /** the clock the pass fires on, inferred from its cadence vs the fixed-step rate */
    clock: "sim" | "render";
}

/** GPU timing for one measurement window — per-pass costs keyed by pass name, plus the per-frame busy
 *  total and the per-step simulation cost. */
export interface BenchmarkGpuStats {
    /** predicted Dawn indirect-draw validation floor in µs/frame, summed over every pass — the
     *  deterministic *untimed* cost (`#drawIndexedIndirect × INDIRECT_FLOOR_US`, gpu.md), invisible to
     *  the per-pass timers because it runs before each pass. Read it against the fence-wait delta under a
     *  draw-count ablation, not summed into a single frame's fence — it only surfaces in fence when the
     *  frame is GPU-bound. */
    indirectFloorUsPerFrame: number;
    /** per-pass indirect draws/frame + the floor µs each predicts (window-diffed from `indirectCount`
     *  like the pass timers from `gpuTime`). */
    indirect: Record<string, { drawsPerFrame: number; floorUs: number }>;
    /** total GPU busy time amortized per frame (Σ `perFrameMs`) — the frame-budget number */
    busyPerFrameMs: number;
    /** GPU busy per fixed step (Σ `occMs` over sim passes) — the simulation step cost, the number
     *  that climbs toward the fixed-step budget under load. Reported per-step so it never under-reads
     *  the cost of a heavy step by amortizing it across the idle frames between steps. */
    simPerStepMs: number;
    /** GPU busy per frame from render passes (Σ `perFrameMs` over render passes) */
    renderPerFrameMs: number;
    /** engine frames spanned by the window — the amortization denominator */
    frames: number;
    passes: Record<string, BenchmarkPass>;
}

/** per-system CPU timing for one measurement window — mean and p99 per system, plus the frame total. */
export interface BenchmarkCpuStats {
    /** per-system mean CPU time over the window, in ms (avg is the primary stat for CPU-mixed rows —
     *  `testing.md`, the 100µs `performance.now` quantization makes min biased) */
    systems: Record<string, number>;
    /** per-system p99 CPU time — the spike tail the {@link systems} mean hides, so the submission /
     *  GC-pause hunt can attribute a frame-time outlier to the system that churned (a slab-write burst,
     *  a query loop). Keyed identically to {@link systems}. */
    systemsP99: Record<string, number>;
    total: number;
}

/** frame-interval timing for one measurement window — the wall-clock distribution (percentiles, stddev)
 *  plus the cpu/fence/gap decomposition of the mean frame. */
export interface BenchmarkFrameStats {
    /** wall-clock frame-interval distribution (ms). In headless this is rAF-paced (~240Hz floor), so
     *  for a light scene it reads near the cadence regardless of work — read it with the decomposition
     *  below, not alone. */
    avg: number;
    median: number;
    p5: number;
    p95: number;
    /** p99 frame interval (ms) — the worst-1%-frame stutter, the even-pacing metric that matters more
     *  than the mean: a steady slowdown holds p99 close to the median, a spike pushes it far above. */
    p99: number;
    /** standard deviation of the frame interval (ms) — the variance signal an even-pacing response
     *  must keep bounded. Flat under a chosen even slowdown; climbs when the engine hitches. */
    stddev: number;
    min: number;
    max: number;
    /** p99 of the *un-clamped* `rawDeltaTime` (ms) — the frame interval before the spiral-of-death dt
     *  clamp (`scheduler.ts` MAX_FIXED_STEPS) caps it. {@link p99} reads the clamped value (≤ ~67 ms),
     *  so this exposes the spike the clamp hides — the worst-1% real frame interval. */
    rawP99: number;
    /** max un-clamped `rawDeltaTime` (ms) — the single worst spike magnitude. `clampedFrames` counts
     *  how many frames the dt clamp fired; this is how big the largest one actually was. */
    rawMax: number;
    samples: number;
    /** mean wall-clock decomposition of the frame interval (ms): the interval is CPU work, then the
     *  GPU fence-wait, then idle. `cpuMs + fenceMs + gapMs ≈ avg`. */
    cpuMs: number;
    /** mean GPU fence-wait — the canonical GPU-bound signal (the loop blocking on the prior frame's
     *  GPU before the next). Small + flat = GPU hidden under pipelining; climbing toward the frame
     *  budget = GPU becoming the bottleneck. */
    fenceMs: number;
    /** p95 GPU fence-wait — the worst-frame GPU-bound stall */
    fenceP95: number;
    /** mean idle gap (`frame − cpu − fence`) — rAF/vsync pacing, not work */
    gapMs: number;
    /** `device.queue.submit` calls per frame over the window — render + slab flush + any mirror
     *  readback + the profiler's own resolve. Each is an IPC round-trip + a GPU serialization point,
     *  untimed by the per-pass timers (it surfaces in {@link fenceMs}); window-diffed from
     *  `Profile.submitCount`. The before/after number for the submit-collapse lever (gpu.md "Single
     *  queue"). Reads ~1 higher under the profiler than in production (the resolve submit). */
    submitsPerFrame: number;
    /** mean fixed steps per frame — the bridge between per-step sim cost and per-frame budget */
    stepsPerFrame: number;
    /** frames whose raw delta was clamped by the spiral-of-death gate (MAX_FIXED_STEPS) */
    clampedFrames: number;
    /** peak GPU frames-in-flight observed (the loop caps at 2) */
    maxPending: number;
}

/** one-shot pipeline-compile timing from app startup — the total wall span plus per-pipeline durations
 *  keyed by pipeline label. */
export interface BenchmarkCompileStats {
    /** wall-clock span from the first pipeline build start to the last build end, in ms */
    totalMs: number;
    /** per-pipeline compile duration in ms, keyed by the pipeline's `label` */
    pipelines: Record<string, number>;
}

/** the result of one `measure()` window — the gpu, cpu, frame, and compile sections, each null when its
 *  data wasn't captured (no GPU passes ran, the profiler wasn't attached). */
export interface BenchmarkMeasurement {
    gpu: BenchmarkGpuStats | null;
    cpu: BenchmarkCpuStats | null;
    frame: BenchmarkFrameStats | null;
    compile: BenchmarkCompileStats | null;
    /** benchmark ticks in the window (warmup + measured), distinct from the engine frame count */
    frames: number;
}

/** the `window.__benchmark` contract `ProfilePlugin` installs — a readiness flag plus `measure`, which
 *  captures a window and resolves the aggregated stats. `bun bench` and any custom perf harness drive it. */
export interface BenchmarkAPI {
    /** true once the profiler has attached and a frame has drained — poll before `measure` */
    readonly ready: boolean;
    /** run `warmup` unmeasured frames, then aggregate `frames` measured ones into one measurement */
    measure(warmup: number, frames: number): Promise<BenchmarkMeasurement>;
}

declare global {
    interface Window {
        __benchmark?: BenchmarkAPI;
    }
}

const r2 = (v: number) => Math.round(v * 100) / 100;
const r3 = (v: number) => Math.round(v * 1000) / 1000;

function quantile(sorted: number[], q: number): number {
    if (sorted.length === 0) return 0;
    return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * q))];
}

/** Dawn's injected indirect-draw validation floor, µs per `drawIndexedIndirect` command. Chrome/D3D12
 *  runs this validation *before* the render pass, so it's untimed by `timestampWrites` and surfaces as
 *  fence wait, not a pass time (gpu.md "WebGPU-specific traps"). Calibrated 2026-06-14 via a gym
 *  indirect-draw calibration sweep (redundant instanceCount-0 indirect draws — pure validation, zero raster): measured
 *  ~1.0–1.3 µs/draw fence-mean slope on lovelace *in the GPU-bound regime*. Kept at 1 as the round,
 *  cross-device gauge — it's an order-of-magnitude predictor, not a budget-precise number: the cost only
 *  surfaces in fence once total GPU work exceeds the frame interval (below that it's absorbed by rAF/vsync
 *  idle — the calibration sweep crossed the threshold at ~4000 draws ≈ 4 ms on a 240 Hz rAF). Consistent with
 *  Toji + the 2026-06-14 point-shadow collapse (~1.17 µs/draw). */
export const INDIRECT_FLOOR_US = 1;

/** the indirect-draw validation floor for `count` issued commands, in µs (`count × INDIRECT_FLOOR_US`).
 *  Pure — unit-tested. */
export function indirectFloorUs(count: number): number {
    return count * INDIRECT_FLOOR_US;
}

/** fold one frame's per-pass indirect tally into the cumulative `count` / `fires` counters (one fire per
 *  pass per frame). The profiler calls it at frame begin so the benchmark window-diffs `count` / `fires`
 *  the way it does the GPU pass timers, deriving the per-frame draw count as `count / fires`. Pure —
 *  unit-tested. */
export function foldIndirect(
    frame: ReadonlyMap<string, number>,
    count: Map<string, number>,
    fires: Map<string, number>,
): void {
    for (const [name, n] of frame) {
        count.set(name, (count.get(name) ?? 0) + n);
        fires.set(name, (fires.get(name) ?? 0) + 1);
    }
}

/** distribution summary of a sample array (ms): central tendency + the p99 / stddev that read a spike
 *  the mean and p95 hide. Pure — unit-tested. */
export function distribution(samples: number[]): {
    avg: number;
    median: number;
    p5: number;
    p95: number;
    p99: number;
    stddev: number;
    min: number;
    max: number;
} {
    const sorted = [...samples].sort((a, b) => a - b);
    const n = sorted.length;
    const avg = sorted.reduce((s, v) => s + v, 0) / n;
    const variance = sorted.reduce((s, v) => s + (v - avg) ** 2, 0) / n;
    return {
        avg: r2(avg),
        median: r2(sorted[Math.floor(n / 2)]),
        p5: r2(quantile(sorted, 0.05)),
        p95: r2(quantile(sorted, 0.95)),
        p99: r2(quantile(sorted, 0.99)),
        stddev: r2(Math.sqrt(variance)),
        min: r2(sorted[0]),
        max: r2(sorted[n - 1]),
    };
}

const mean = (a: number[]): number => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0);

/** resolve one pass from its window-cumulative deltas. `dTime` ms over `dFires` occurrences across
 *  `dFrames` frames; `stepsPerFrame` sets the sim/render threshold. `samples` is the per-drained-frame
 *  time list for the percentile tail (empty → 0 percentiles). Pure — unit-tested. */
export function passStats(
    dTime: number,
    dFires: number,
    dFrames: number,
    stepsPerFrame: number,
    samples: number[] = [],
): BenchmarkPass {
    const occMs = dTime / dFires;
    const firesPerFrame = dFrames > 0 ? dFires / dFrames : 0;
    const perFrameMs = dFrames > 0 ? dTime / dFrames : 0;
    // a render pass fires ~once/frame; a sim pass ~steps/frame (< 1 whenever render outruns the fixed
    // rate). Split at the midpoint so the classification adapts to the measured step rate.
    const clock = firesPerFrame > (stepsPerFrame + 1) / 2 ? "render" : "sim";
    const sorted = [...samples].sort((a, b) => a - b);
    return {
        occMs: r3(occMs),
        occP95: r3(quantile(sorted, 0.95)),
        occP99: r3(quantile(sorted, 0.99)),
        perFrameMs: r3(perFrameMs),
        firesPerFrame: r3(firesPerFrame),
        clock,
    };
}

export function createMeasure(state: State, profile: Profile) {
    return (warmup: number, frames: number): Promise<BenchmarkMeasurement> => {
        return new Promise((resolve) => {
            const frameTimes: number[] = [];
            // the un-clamped rAF interval (rawDeltaTime), so the report exposes the spike magnitude the
            // spiral-of-death dt clamp hides — `frameTimes` is the clamped value, capped at ~67 ms.
            const rawFrameTimes: number[] = [];
            const fenceWaits: number[] = [];
            const cpuTotals: number[] = [];
            const fixedStepCounts: number[] = [];
            const cpuAccum = new Map<string, number[]>();
            let clampedFrames = 0;
            let maxPending = 0;
            let count = 0;

            // GPU is measured by diffing the profiler's cumulative per-pass counters over the window
            // (exact per-occurrence, immune to the display map's hold) rather than sampling each frame.
            let gpuTimeStart: Map<string, number> | null = null;
            let gpuFiresStart: Map<string, number> | null = null;
            // per-pass per-occurrence GPU sample lists (the cpuAccum twin) — for the percentile tail the
            // window mean can't reach. Built by frame-diffing the cumulative counters: a pass that drained
            // this frame contributes its per-drained-frame time (`dt/df`, normalized when >1 frame drained
            // between ticks). `prev*` hold the last frame's cumulative read; `ready` gates the first diff.
            const gpuAccum = new Map<string, number[]>();
            const prevGpuTime = new Map<string, number>();
            const prevGpuFires = new Map<string, number>();
            let gpuPrevReady = false;
            // the indirect-draw counters window-diff exactly like gpuTime/gpuFires (cumulative since
            // attach), so the floor reads on the same window the pass timers do
            let indirectCountStart: Map<string, number> | null = null;
            let frameStart = 0;
            let submitStart = 0;

            function tick() {
                count++;
                const measuring = count > warmup;

                if (measuring) {
                    if (gpuTimeStart === null) {
                        gpuTimeStart = new Map(profile.gpuTime);
                        gpuFiresStart = new Map(profile.gpuFires);
                        indirectCountStart = new Map(profile.indirectCount);
                        frameStart = Compute?.frame ?? 0;
                        submitStart = profile.submitCount;
                    }

                    frameTimes.push(state.time.deltaTime * 1000);
                    rawFrameTimes.push(state.time.rawDeltaTime * 1000);
                    fenceWaits.push(profile.fenceWaitMs);
                    if (state.time.throttled) clampedFrames++;
                    fixedStepCounts.push(state.time.fixedSteps);
                    const fif = Compute?.pending?.() ?? 0;
                    if (fif > maxPending) maxPending = fif;

                    let cpuTotal = 0;
                    for (const [name, ms] of profile.cpu) {
                        cpuTotal += ms;
                        if (!cpuAccum.has(name)) cpuAccum.set(name, []);
                        cpuAccum.get(name)!.push(ms);
                    }
                    cpuTotals.push(cpuTotal);

                    // per-pass GPU sampling: diff the cumulative counters since the last measured frame.
                    // `df` is drained-frames since last tick (usually 1); skip a pass that didn't drain.
                    for (const [name, t] of profile.gpuTime) {
                        const fires = profile.gpuFires.get(name) ?? 0;
                        if (gpuPrevReady) {
                            const dt = t - (prevGpuTime.get(name) ?? 0);
                            const df = fires - (prevGpuFires.get(name) ?? 0);
                            if (dt > 0 && df > 0) {
                                let arr = gpuAccum.get(name);
                                if (!arr) gpuAccum.set(name, (arr = []));
                                arr.push(dt / df);
                            }
                        }
                        prevGpuTime.set(name, t);
                        prevGpuFires.set(name, fires);
                    }
                    gpuPrevReady = true;
                }

                if (count < warmup + frames) {
                    requestAnimationFrame(tick);
                    return;
                }

                const stepsPerFrame = mean(fixedStepCounts);
                // engine frames elapsed in the window — the denominator for every engine-side cumulative
                // counter (gpuTime, indirectCount, submitCount), distinct from the benchmark's own tick count
                const dFrames = (Compute?.frame ?? 0) - frameStart;

                let frame: BenchmarkFrameStats | null = null;
                if (frameTimes.length > 0) {
                    const gaps = frameTimes.map((dt, i) =>
                        Math.max(0, dt - cpuTotals[i] - fenceWaits[i]),
                    );
                    const raw = distribution(rawFrameTimes);
                    frame = {
                        ...distribution(frameTimes),
                        rawP99: raw.p99,
                        rawMax: raw.max,
                        samples: frameTimes.length,
                        cpuMs: r3(mean(cpuTotals)),
                        fenceMs: r3(mean(fenceWaits)),
                        fenceP95: r3(
                            quantile(
                                [...fenceWaits].sort((a, b) => a - b),
                                0.95,
                            ),
                        ),
                        gapMs: r3(mean(gaps)),
                        submitsPerFrame: r2(
                            dFrames > 0 ? (profile.submitCount - submitStart) / dFrames : 0,
                        ),
                        stepsPerFrame: r2(stepsPerFrame),
                        clampedFrames,
                        maxPending,
                    };
                }

                let cpu: BenchmarkCpuStats | null = null;
                if (cpuAccum.size > 0) {
                    const systems: Record<string, number> = {};
                    const systemsP99: Record<string, number> = {};
                    for (const [name, samples] of cpuAccum) {
                        systems[name] = r3(mean(samples));
                        systemsP99[name] = r3(
                            quantile(
                                [...samples].sort((a, b) => a - b),
                                0.99,
                            ),
                        );
                    }
                    cpu = { systems, systemsP99, total: r3(mean(cpuTotals)) };
                }

                let gpu: BenchmarkGpuStats | null = null;
                if (gpuTimeStart !== null && gpuFiresStart !== null) {
                    const passes: Record<string, BenchmarkPass> = {};
                    let busyPerFrameMs = 0;
                    let simPerStepMs = 0;
                    let renderPerFrameMs = 0;
                    for (const [name, tEnd] of profile.gpuTime) {
                        const dTime = tEnd - (gpuTimeStart.get(name) ?? 0);
                        const dFires =
                            (profile.gpuFires.get(name) ?? 0) - (gpuFiresStart.get(name) ?? 0);
                        if (dFires <= 0 || dTime <= 0) continue;
                        const pass = passStats(
                            dTime,
                            dFires,
                            dFrames,
                            stepsPerFrame,
                            gpuAccum.get(name) ?? [],
                        );
                        passes[name] = pass;
                        busyPerFrameMs += pass.perFrameMs;
                        if (pass.clock === "sim") simPerStepMs += pass.occMs;
                        else renderPerFrameMs += pass.perFrameMs;
                    }
                    // the indirect-validation floor, window-diffed like the pass timers above: per pass,
                    // draws/frame = Δcount / Δframes, floor µs = draws × INDIRECT_FLOOR_US (gpu.md)
                    const indirect: Record<string, { drawsPerFrame: number; floorUs: number }> = {};
                    let indirectFloorUsPerFrame = 0;
                    if (indirectCountStart !== null && dFrames > 0) {
                        for (const [name, cEnd] of profile.indirectCount) {
                            const dCount = cEnd - (indirectCountStart.get(name) ?? 0);
                            if (dCount <= 0) continue;
                            const drawsPerFrame = dCount / dFrames;
                            const floorUs = indirectFloorUs(drawsPerFrame);
                            indirect[name] = {
                                drawsPerFrame: r2(drawsPerFrame),
                                floorUs: r2(floorUs),
                            };
                            indirectFloorUsPerFrame += floorUs;
                        }
                    }
                    gpu = {
                        indirectFloorUsPerFrame: r2(indirectFloorUsPerFrame),
                        indirect,
                        busyPerFrameMs: r3(busyPerFrameMs),
                        simPerStepMs: r3(simPerStepMs),
                        renderPerFrameMs: r3(renderPerFrameMs),
                        frames: dFrames,
                        passes,
                    };
                }

                let compile: BenchmarkCompileStats | null = null;
                if (profile.compile.size > 0) {
                    const pipelines: Record<string, number> = {};
                    for (const [label, ms] of profile.compile)
                        pipelines[label || "(unnamed)"] = r2(ms);
                    compile = { totalMs: r2(profile.compileMs), pipelines };
                }

                resolve({ gpu, cpu, frame, compile, frames: count });
            }

            requestAnimationFrame(tick);
        });
    };
}
