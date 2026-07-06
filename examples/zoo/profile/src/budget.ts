import { type Plugin, Profile, type System } from "@dylanebert/shallot";

// #doc:intro
// A live performance overlay: FPS, per-pass GPU and CPU timings, memory, and shader-compile stats,
// pinned in the corner of the view. Enable it, press F3, and see where every millisecond goes.

// #doc:code source:profile/public/scenes/profile.scene
// The overlay reads the running scene, so its numbers are your app's actual cost. Enable `ProfilePlugin`
// (like [any plugin](doc:guide/quick-start)) and press F3 to toggle it; the overlay stays hidden until you do.

// #doc:code
// ### Read the stats in code
//
// The same numbers behind the overlay live on the `Profile` singleton, refreshed every frame. Read
// `Profile.gpu` (per-pass GPU time) or `Profile.cpu` (per-system) in a system to guard a frame budget or
// feed your own HUD. This one warns when the frame's GPU work crosses a budget.
// #region budget
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
// #endregion

// #doc:code
// For an aggregated capture (percentiles, the cpu/fence/gap split) rather than a live frame, `window.__benchmark`
// exposes `measure(warmup, frames)`, the same window `bun bench` reads.

export const Budget = {
    name: "Budget",
    systems: [budget],
} satisfies Plugin;

export default Budget;
