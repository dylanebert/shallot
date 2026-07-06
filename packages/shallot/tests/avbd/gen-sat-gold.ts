// Regenerates the box-box SAT gold vectors by building and running the reference
// C++ `gold-sat` harness, which drives `Manifold::collide` over a fixed set of
// box-pair configs. Output is committed (small) and is the spec collide.ts must
// reproduce (sat.test.ts) — the SAT crux gate.
//
// Usage: bun run tests/avbd/gen-sat-gold.ts
// Requires: g++ on PATH.

import { spawnSync } from "child_process";
import { existsSync, writeFileSync } from "fs";
import { resolve } from "path";

const repoRoot = resolve(import.meta.dir, "..", "..", "..", "..", "..");
const refDir = resolve(repoRoot, "reference", "avbd-demo3d");
const out = resolve(import.meta.dir, "sat-gold-vectors.json");

if (!existsSync(refDir)) {
    console.error(`reference dir missing: ${refDir}`);
    process.exit(1);
}

const sources = [
    "source/solver.cpp",
    "source/rigid.cpp",
    "source/force.cpp",
    "source/manifold.cpp",
    "source/collide.cpp",
    "source/joint.cpp",
    "source/spring.cpp",
    "gold-sat.cpp",
];
const binary = resolve(refDir, "gold-sat");

console.log("[gen-sat-gold] building gold-sat");
const build = spawnSync("g++", ["-std=c++17", "-O2", ...sources, "-I", "source", "-o", binary], {
    cwd: refDir,
    stdio: "inherit",
});
if (build.status !== 0) process.exit(build.status ?? 1);

const run = spawnSync(binary, [], { encoding: "utf8" });
if (run.status !== 0) {
    console.error(run.stderr);
    process.exit(run.status ?? 1);
}

// pretty-print so the committed file diffs cleanly
writeFileSync(out, `${JSON.stringify(JSON.parse(run.stdout), null, 2)}\n`);
console.log(`[gen-sat-gold] wrote ${out}`);
