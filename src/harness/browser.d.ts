// Types for `browser.json`, the real-GPU Chromium launch recipe. The recipe is pure data, so it ships as
// JSON under `./harness/browser` and loads in Node, Bun or a bundler with no compile step; import it
// with `with { type: "json" }`, which Node requires.

/**
 * the `channel` + `args` a real-GPU Chromium launch needs, as a `chromium.launch(...)` opt or a
 * Playwright config's `use.launchOptions`. This is the floor, not a grab bag — a consumer appends its
 * own extra `args` rather than this recipe growing one.
 *
 * @example
 * ```
 * import launch from "@dylanebert/shallot/harness/browser" with { type: "json" };
 * const browser = await chromium.launch({ headless: true, ...launch });
 * ```
 */
export interface RealGpuLaunch {
    channel: "chromium";
    args: string[];
}

/**
 * `channel: "chromium"` runs Playwright's full Chromium build; bare `headless: true` with no channel runs
 * the stripped headless-shell build, whose software-only GPU stack misses shallot's floor even on real
 * hardware. `args` requests WebGPU behind Chromium's dev flags and sets the window class (the Wayland
 * app id) for callers that select a visible window.
 */
declare const REAL_GPU_LAUNCH: RealGpuLaunch;
export default REAL_GPU_LAUNCH;
