import { type Plugin, Profile, type System, showProfiler } from "@dylanebert/shallot";

// The same numbers behind the F3 overlay live on the `Profile` singleton, refreshed every frame. Read
// `Profile.gpu` (per-pass GPU time) or `Profile.cpu` (per-system) in a system to guard a frame budget or
// feed your own HUD.
const GPU_BUDGET_MS = 8;

export const budget = {
    name: "budget",
    group: "simulation",
    update() {
        let gpuMs = 0;
        for (const ms of Profile.gpu.values()) gpuMs += ms;
        if (gpuMs > GPU_BUDGET_MS) console.warn(`gpu over budget: ${gpuMs.toFixed(2)} ms`);
    },
} satisfies System;

// For an aggregated capture (percentiles, the cpu/fence/gap split) rather than a live frame,
// `window.__benchmark` exposes `measure(warmup, frames)`, the same window `bun bench` reads.
export const Budget = {
    name: "Budget",
    systems: [budget],
    // surface the profiler HUD on open — the same overlay F3 toggles, shown from code so the numbers are
    // visible without a keypress. Off by default engine-wide; a perf-focused project opts in like this.
    warm() {
        showProfiler();
    },
} satisfies Plugin;

export default Budget;
