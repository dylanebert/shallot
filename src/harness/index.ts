import type { RealGpuLaunch } from "./browser";
import launch from "./browser.json" with { type: "json" };

export type { RealGpuLaunch };
/** the real-GPU Chromium launch floor; the same data `./harness/browser` publishes as JSON. */
export const REAL_GPU_LAUNCH: RealGpuLaunch = launch as RealGpuLaunch;
export * from "./capture";
export * from "./driver";
export * from "./launch";
export * from "./runtime";
export * from "./seat";
export { compileWgsl } from "./wgsl";
