import { GpuProfile, GpuRegistryResource } from "@dylanebert/shallot/compute/core";
import { Compute, type State } from "@dylanebert/shallot";

export interface GpuStats {
    avg: number;
    median: number;
    p5: number;
    p95: number;
    min: number;
    max: number;
    samples: number;
    passes: Record<string, number>;
}

export interface CpuStats {
    systems: Record<string, number>;
    total: number;
}

export interface FrameStats {
    avg: number;
    median: number;
    p5: number;
    p95: number;
    min: number;
    max: number;
    samples: number;
    clampedFrames: number;
    avgFixedSteps: number;
    maxPending: number;
}

export interface CompileStats {
    totalMs: number;
    pipelines: Record<string, number>;
}

export interface Measurement {
    gpu: GpuStats | null;
    cpu: CpuStats | null;
    frame: FrameStats | null;
    compile: CompileStats | null;
    frames: number;
}

declare global {
    interface Window {
        __benchmark: {
            readonly ready: boolean;
            measure: (warmup: number, frames: number) => Promise<Measurement>;
            setPipeline: (pipeline: "raster" | "raytracing") => void;
            setEffects: (names: string[]) => void;
            setCount: (n: number) => void;
            setCamera: (mode: "static" | "pan") => void;
            setLayout: (mode: "lorenz" | "grid") => void;
            setRenderTestShape?: (name: "box" | "sphere" | "capsule" | "plane" | "cone") => void;
            setRenderTestVariant?: (
                name:
                    | "default"
                    | "transparent"
                    | "vertex"
                    | "fragment"
                    | "scaled"
                    | "reflective"
                    | "roughness"
                    | "refraction",
            ) => void;
            setRenderTestLighting?: (
                name: "directional" | "point" | "dir+pt" | "multipoint",
            ) => void;
            setRoom?: (room: "bathroom" | "living" | "cathedral" | "anechoic") => void;
            setPhysicsTestVariant?: (name: string) => void;
            readBodies?: () => Promise<any>;
        };
        __shallot?: Record<string, unknown>;
    }
}

const r2 = (v: number) => Math.round(v * 100) / 100;

export function createMeasure(state: State) {
    return (warmup: number, frames: number): Promise<Measurement> => {
        return new Promise((resolve) => {
            const gpuTotals: number[] = [];
            const passAccum = new Map<string, number[]>();
            const cpuAccum = new Map<string, number[]>();
            const cpuTotals: number[] = [];
            const frameTimes: number[] = [];
            const fixedStepCounts: number[] = [];
            let clampedFrames = 0;
            let maxPending = 0;
            let count = 0;

            function tick() {
                count++;
                const profiles = GpuProfile.from(state);
                const measuring = count > warmup;

                if (measuring && profiles && profiles.length > 0) {
                    let total = 0;
                    for (const p of profiles) for (const [, ms] of p) total += ms;
                    if (total > 0) {
                        gpuTotals.push(total);
                        for (const p of profiles) {
                            for (const [name, ms] of p) {
                                if (!passAccum.has(name)) passAccum.set(name, []);
                                passAccum.get(name)!.push(ms);
                            }
                        }
                    }
                }

                if (measuring) {
                    frameTimes.push(state.time.deltaTime * 1000);
                    if (state.time.throttled) clampedFrames++;
                    fixedStepCounts.push(state.time.fixedSteps);
                    const fif = Compute.from(state)?.pending ?? 0;
                    if (fif > maxPending) {
                        maxPending = fif;
                    }

                    const cpuTimings = state.scheduler.cpu;
                    if (cpuTimings.size > 0) {
                        let total = 0;
                        for (const [name, ms] of cpuTimings) {
                            total += ms;
                            if (!cpuAccum.has(name)) cpuAccum.set(name, []);
                            cpuAccum.get(name)!.push(ms);
                        }
                        cpuTotals.push(total);
                    }
                }

                if (count < warmup + frames) {
                    requestAnimationFrame(tick);
                } else {
                    let frameStats: FrameStats | null = null;
                    if (frameTimes.length > 0) {
                        const sorted = [...frameTimes].sort((a, b) => a - b);
                        const n = sorted.length;
                        frameStats = {
                            avg: r2(sorted.reduce((s, v) => s + v, 0) / n),
                            median: r2(sorted[Math.floor(n / 2)]),
                            p5: r2(sorted[Math.floor(n * 0.05)]),
                            p95: r2(sorted[Math.floor(n * 0.95)]),
                            min: r2(sorted[0]),
                            max: r2(sorted[n - 1]),
                            samples: n,
                            clampedFrames,
                            avgFixedSteps: r2(
                                fixedStepCounts.reduce((s, v) => s + v, 0) / fixedStepCounts.length,
                            ),
                            maxPending,
                        };
                    }

                    let cpuStats: CpuStats | null = null;
                    if (cpuTotals.length > 0) {
                        const systems: Record<string, number> = {};
                        for (const [name, samples] of cpuAccum) {
                            systems[name] = r2(samples.reduce((s, v) => s + v, 0) / samples.length);
                        }
                        cpuStats = {
                            systems,
                            total: r2(cpuTotals.reduce((s, v) => s + v, 0) / cpuTotals.length),
                        };
                    }

                    let compile: CompileStats | null = null;
                    const registry = GpuRegistryResource.from(state);
                    if (registry && registry.compileSpans.length > 0) registry.finalizeCompile();
                    if (registry && registry.compileTimings.length > 0) {
                        const pipelines: Record<string, number> = {};
                        for (const t of registry.compileTimings) {
                            pipelines[t.label || "(unnamed)"] = r2(t.ms);
                        }
                        compile = { totalMs: r2(registry.compileTotalMs), pipelines };
                    }

                    if (gpuTotals.length === 0) {
                        resolve({
                            gpu: null,
                            cpu: cpuStats,
                            frame: frameStats,
                            compile,
                            frames: count,
                        });
                        return;
                    }
                    const sorted = [...gpuTotals].sort((a, b) => a - b);
                    const avg = sorted.reduce((s, v) => s + v, 0) / sorted.length;
                    const passes: Record<string, number> = {};
                    for (const [name, samples] of passAccum) {
                        passes[name] = r2(samples.reduce((s, v) => s + v, 0) / samples.length);
                    }
                    const n = sorted.length;
                    resolve({
                        gpu: {
                            avg: r2(avg),
                            median: r2(sorted[Math.floor(n / 2)]),
                            p5: r2(sorted[Math.floor(n * 0.05)]),
                            p95: r2(sorted[Math.floor(n * 0.95)]),
                            min: r2(sorted[0]),
                            max: r2(sorted[n - 1]),
                            samples: n,
                            passes,
                        },
                        cpu: cpuStats,
                        frame: frameStats,
                        compile,
                        frames: count,
                    });
                }
            }
            requestAnimationFrame(tick);
        });
    };
}
