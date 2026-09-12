import type { RealGpuLaunch } from "./browser";
import launch from "./browser.json" with { type: "json" };

export type { RealGpuLaunch };
/** the real-GPU Chromium launch recipe; the same data `./harness/browser` publishes as JSON. */
export const REAL_GPU_LAUNCH: RealGpuLaunch = launch as RealGpuLaunch;
export * from "./driver";
export * from "./runtime";
