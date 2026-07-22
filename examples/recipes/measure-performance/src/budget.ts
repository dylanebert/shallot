import { type Plugin, Profile, type System } from "@dylanebert/shallot";

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
} satisfies Plugin;

export default Budget;
