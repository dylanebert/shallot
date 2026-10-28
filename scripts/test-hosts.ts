import { normalizePath } from "../src/project/paths";

/**
 * Test files and browser subjects hosted CI runs on only the listed platforms, keyed by repository-relative path;
 * every other one runs on every host. A claim runs on the cheapest host that can fail on it, and each host-only
 * class keeps files that red it on its own host. Outside GitHub Actions, as in scripts/lavapipe.sh, every path
 * runs on every platform.
 */
export const ONLY: Readonly<Record<string, readonly NodeJS.Platform[]>> = {
    // Converged fog images fail alike on every adapter, and Metal runs them in a fifth of llvmpipe's time and a
    // ninth of WARP's. The software adapters' float class stays held there by fog's other files, targets.node.ts
    // and examples/particles/src/motes.node.ts.
    "src/extras/fog/filter.node.ts": ["darwin"],
    "src/extras/fog/flicker.node.ts": ["darwin"],
    "src/extras/fog/fog-motion.node.ts": ["darwin"],
    "src/extras/fog/fog.node.ts": ["darwin"],
    "src/extras/fog/hall.node.ts": ["darwin"],
    "src/extras/fog/in-scatter.node.ts": ["darwin"],
    "src/extras/fog/lamp.node.ts": ["darwin"],
    "src/extras/fog/leak.node.ts": ["darwin"],
    "src/extras/fog/lighting.node.ts": ["darwin"],
    "src/extras/fog/silhouette.node.ts": ["darwin"],
    "examples/volumetric-fog/src/motes-fog.node.ts": ["darwin"],
    // These browser subjects' claims fail alike on every adapter, and hosted macOS's Chromium, on the Apple
    // Paravirtual device, runs each fastest. The subjects every host keeps hold the software adapters' browser
    // classes and Windows layout rounding.
    "examples/ambient-occlusion": ["darwin"],
    "examples/bloom": ["darwin"],
    "examples/day-night-cycle": ["darwin"],
    "examples/day-night-cycle/tests/frames": ["darwin"],
    "examples/interact": ["darwin"],
    "examples/tonemapping": ["darwin"],
    "examples/view-model": ["darwin"],
    "examples/volumetric-fog": ["darwin"],
    "src/project/browser": ["darwin"],
    // Its cost is DOM-bound, so SwiftShader's runners run it no slower than the other hosts.
    "examples/color-grading": ["linux"],
    // Its packaging classes are Windows paths and Linux case; macOS adds none, and there, on runners GitHub caps at
    // five, it is the costliest file (8 min). The packed CLI on macOS stays held by mac-native, packed-gpu and
    // icons-packed.
    "scripts/packed-install.node.ts": ["linux", "win32"],
    // Its desktop opener shim is POSIX.
    "scripts/publish-browser-session.node.ts": ["darwin", "linux"],
};

const HOSTED = process.env.GITHUB_ACTIONS === "true";

/** Whether the path, relative to the repository root, runs on `platform`; `hosted` applies {@link ONLY}. */
export function runsOn(
    path: string,
    platform: NodeJS.Platform = process.platform,
    hosted = HOSTED,
): boolean {
    return !hosted || (ONLY[normalizePath(path).replace(/^\.\//, "")]?.includes(platform) ?? true);
}

/** The `paths` that run on `platform`, in order. */
export function hostFiles(
    paths: readonly string[],
    platform: NodeJS.Platform = process.platform,
    hosted = HOSTED,
): string[] {
    return paths.filter((path) => runsOn(path, platform, hosted));
}

// Usage: bun scripts/test-hosts.ts <path>... prints, one per line, the paths this host runs.
if (import.meta.main) for (const path of hostFiles(process.argv.slice(2))) console.log(path);
