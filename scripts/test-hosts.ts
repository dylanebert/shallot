import { normalizePath } from "../src/project/paths";

/**
 * Test files CI runs on only the listed platforms, keyed by repository-relative path; every other file runs on
 * every host. A claim runs on the cheapest host that can fail on it, and each host-only class keeps files that
 * red it on its own host.
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
    // Its desktop opener shim is POSIX.
    "scripts/publish-browser-session.node.ts": ["darwin", "linux"],
};

/** Whether CI runs the test file at `path`, relative to the repository root, on `platform`. */
export function runsOn(path: string, platform: NodeJS.Platform = process.platform): boolean {
    return ONLY[normalizePath(path).replace(/^\.\//, "")]?.includes(platform) ?? true;
}

/** The `paths` CI runs on `platform`, in order. */
export function hostFiles(
    paths: readonly string[],
    platform: NodeJS.Platform = process.platform,
): string[] {
    return paths.filter((path) => runsOn(path, platform));
}

// Usage: bun scripts/test-hosts.ts <file>... prints, one per line, the files this host runs.
if (import.meta.main) for (const path of hostFiles(process.argv.slice(2))) console.log(path);
