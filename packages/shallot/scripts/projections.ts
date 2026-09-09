/** Public distribution paths written by the runtime projector. */
export const runtimeRoots = [
    "src/index.ts",
    "src/engine",
    "src/standard",
    "src/extras",
    "src/types",
    "src/harness/runtime.ts",
    "src/harness/pixels.ts",
    "src/harness/motion.ts",
    "src/harness/degraded-boot.ts",
    "rust/audio/pkg",
];
/** Public distribution paths copied from the tooling owner. */
export const toolingRoots = [
    "bin",
    "src/project",
    "src/harness/browser.ts",
    "rust/window",
    "assets",
];
/** Generated distribution roots; the producers and realization reader share these names. */
export const runtimeRecord = "runtime-inputs.json";
export const toolingDist = "dist";
export const recipeRoot = "examples";
