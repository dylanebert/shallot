// The real-GPU Chromium launch recipe. Pure data — no Playwright import, no node import — so any
// bun-side consumer (this package's own `bin/verify.ts`, or a project's own bench/harness driver
// script) can import it from the published package without pulling in a runtime it doesn't have.
//
// This is one of two exports whose only consumption context is Node (the other is `@dylanebert/shallot
// /vite`), so unlike every other export it ships a compiled `dist/harness-browser.js` alongside its
// `.ts` source (`exports.md` — `tsc` still type-checks against source, only the runtime load target
// differs). A `playwright.config.ts` resolves through Node's plain ESM loader, which applies no TS
// transform to a `node_modules` import, so the raw `.ts` export published through 0.9.0 threw
// `ERR_UNKNOWN_FILE_EXTENSION` there at any publish state (confirmed 2026-08-04 against the registry
// package, no staging involved). The compile landed in 0.9.1, which is the floor a config importing
// this module directly must depend on. This module never captures or launches anything itself, only
// names the floor.

/**
 * the `channel` + `args` a real-GPU Chromium launch needs, as a `chromium.launch(...)` opt or a
 * Playwright config's `use.launchOptions`. This is the floor, not a grab bag — a consumer appends its
 * own extra `args` (e.g. a timestamp-query quantization bypass) rather than this constant growing one.
 *
 * @example
 * ```
 * const browser = await chromium.launch({ headless: true, ...REAL_GPU_LAUNCH });
 * ```
 */
export interface RealGpuLaunch {
    channel: "chromium";
    args: string[];
}

/**
 * `channel: "chromium"` runs Playwright's full Chromium build. Bare `headless: true` with no channel
 * runs the stripped headless-shell build instead, whose GPU stack is software-only — SwiftShader misses
 * shallot's floor even on real hardware (probed 2026-07-14, M4 Metal: headless-shell = swiftshader, 4/5
 * floor features; `channel: "chromium"` = metal-3, full floor + subgroups). The same
 * `playwright install chromium` provides both builds, and this channel is real GPU in headed launches
 * too, not just headless. `args` requests WebGPU behind Chromium's dev flags.
 */
export const REAL_GPU_LAUNCH: RealGpuLaunch = {
    channel: "chromium",
    args: ["--enable-unsafe-webgpu", "--enable-features=WebGPUDeveloperFeatures"],
};
