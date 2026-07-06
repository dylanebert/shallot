// Regenerates the contact-reduction gold vectors by building + running the verbatim-Jolt
// `reduce-gold` harness (reference/avbd-demo3d/reduce-gold.cpp). The output is the independent
// reference reduce.test.ts gates the oracle's `reduceManifold` against — the 4.8.1 reduction crux.
//
// Usage: bun run tests/avbd/gen-reduce-gold.ts
// Requires: g++ on PATH.

import { spawnSync } from "child_process";
import { existsSync, writeFileSync } from "fs";
import { resolve } from "path";

const repoRoot = resolve(import.meta.dir, "..", "..", "..", "..", "..");
const refDir = resolve(repoRoot, "reference", "avbd-demo3d");
const out = resolve(import.meta.dir, "reduce-gold.json");

if (!existsSync(refDir)) {
    console.error(`reference dir missing: ${refDir}`);
    process.exit(1);
}

const binary = resolve(refDir, "reduce-gold");
console.log("[gen-reduce-gold] building reduce-gold");
const build = spawnSync("g++", ["-std=c++17", "-O2", "reduce-gold.cpp", "-o", binary], {
    cwd: refDir,
    stdio: "inherit",
});
if (build.status !== 0) process.exit(build.status ?? 1);

const run = spawnSync(binary, [], { encoding: "utf8" });
if (run.status !== 0) {
    console.error(run.stderr);
    process.exit(run.status ?? 1);
}

writeFileSync(out, `${JSON.stringify(JSON.parse(run.stdout), null, 2)}\n`);
console.log(`[gen-reduce-gold] wrote ${out}`);
