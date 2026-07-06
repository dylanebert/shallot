// Regenerates the hull-hull SAT gold vectors by building and running the independent
// reference/bullet3-sat-harness.cpp, which drives Bullet3's b3FindSeparatingAxis +
// b3ClipHullAgainstHull + b3ReduceContacts over a fixed set of hull-pair configs
// (cube / tetrahedron / cone). Output is committed (small) and is the geometric spec
// the oracle hull SAT (hull.ts collideHull) must reproduce — separating normal + the
// reduced contact world points + depths. The polytope-family gold for Phase 6.3 hull;
// non-box hulls have no AVBD reference (demo3d is boxes only), so Bullet is the gold.
//
// Usage: bun run tests/avbd/gen-hull-gold.ts
// Requires: g++ on PATH.

import { spawnSync } from "child_process";
import { existsSync, writeFileSync } from "fs";
import { resolve } from "path";

const repoRoot = resolve(import.meta.dir, "..", "..", "..", "..", "..");
const refDir = resolve(repoRoot, "reference");
const harness = resolve(refDir, "bullet3-sat-harness.cpp");
const out = resolve(import.meta.dir, "hull-gold-vectors.json");

if (!existsSync(harness)) {
    console.error(`harness missing: ${harness}`);
    process.exit(1);
}

// b3AlignedObjectArray pulls in the aligned allocator + logging symbols (b3AlignedAllocInternal,
// b3OutputErrorMessageVarArgsInternal), so the two Bullet3Common .cpp must link in. The binary lands in
// reference/ (the system /tmp is mounted noexec here).
const sources = [
    harness,
    resolve(refDir, "bullet3/src/Bullet3Common/b3AlignedAllocator.cpp"),
    resolve(refDir, "bullet3/src/Bullet3Common/b3Logging.cpp"),
];
const binary = resolve(refDir, "bullet3-sat-harness");

console.log("[gen-hull-gold] building bullet3-sat-harness");
const build = spawnSync(
    "g++",
    ["-std=c++17", "-O2", "-I", resolve(refDir, "bullet3/src"), ...sources, "-o", binary],
    { stdio: "inherit" },
);
if (build.status !== 0) process.exit(build.status ?? 1);

const run = spawnSync(binary, [], { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
if (run.status !== 0) {
    console.error(run.stderr);
    process.exit(run.status ?? 1);
}

// pretty-print so the committed file diffs cleanly
writeFileSync(out, `${JSON.stringify(JSON.parse(run.stdout), null, 2)}\n`);
console.log(`[gen-hull-gold] wrote ${out}`);
