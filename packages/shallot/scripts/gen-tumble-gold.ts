// Regenerates a committed bit-exact gold vector for the tumble engine by building and running the Box3D
// C reference (branch `harness`, scalar + force-overflow) `<name>_gold` target. Output lands in
// src/standard/tumble/engine/<name>.gold.json — COMMITTED (small), unlike the scene fixtures. These
// vectors pin the kernel's per-phase math (`cargo test`) and the engine's `*.test.ts` gold comparisons.
//
// One parameterized script covers every gold target — they differ only in the cmake target name and the
// output filename. The valid names are the committed engine/*.gold.json files.
//
// The reference lives at ../reference/box3d beside the shallot checkout (the kex workspace layout —
// sibling of kex/shallot). Absent it, this errors honestly. The committed gold is the frozen contract
// (pin 29bf523); only run this at a deliberate upstream sync.
//
// Usage: bun run scripts/gen-tumble-gold.ts <name>   (from packages/shallot)
//        e.g. bun run scripts/gen-tumble-gold.ts contact

import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

const GOLD_NAMES = [
    "math",
    "geometry",
    "distance",
    "tree",
    "manifold",
    "convex_manifold",
    "query",
    "mover",
    "integrate",
    "contact",
    "contact_wide",
    "finalize",
    "recycle",
    "joint",
];

const name = process.argv[2];
if (!name || !GOLD_NAMES.includes(name)) {
    console.error(
        `usage: bun run scripts/gen-tumble-gold.ts <name>\n  name one of: ${GOLD_NAMES.join(", ")}`,
    );
    process.exit(1);
}

const pkgRoot = resolve(import.meta.dir, "..");
const shallotRoot = resolve(pkgRoot, "..", "..");
const refDir = resolve(shallotRoot, "..", "reference", "box3d");
const buildDir = resolve(refDir, "build-fixtures");
const outPath = resolve(pkgRoot, "src", "standard", "tumble", "engine", `${name}.gold.json`);

if (!existsSync(refDir)) {
    console.error(`box3d reference missing: ${refDir}`);
    console.error(
        "expected the box3d reference at reference/box3d beside the shallot checkout (the kex workspace layout: kex/reference/box3d, sibling of kex/shallot) on branch `harness`.",
    );
    process.exit(1);
}

function run(cmd: string, args: string[]) {
    const r = spawnSync(cmd, args, { cwd: refDir, stdio: "inherit" });
    if (r.status !== 0) process.exit(r.status ?? 1);
}

console.log(`[gen-tumble-gold] configuring reference (scalar + force-overflow) for ${name}_gold`);
run("cmake", [
    "-S",
    refDir,
    "-B",
    buildDir,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBOX3D_DISABLE_SIMD=ON",
    "-DBOX3D_FORCE_OVERFLOW=ON",
    "-DBOX3D_FIXTURES=ON",
    "-DBOX3D_SAMPLES=OFF",
    "-DBOX3D_BENCHMARKS=OFF",
    "-DBOX3D_UNIT_TESTS=OFF",
    "-DBOX3D_DOCS=OFF",
]);

console.log(`[gen-tumble-gold] building ${name}_gold`);
run("cmake", ["--build", buildDir, "--target", `${name}_gold`, "-j"]);

console.log(`[gen-tumble-gold] generating gold vectors -> ${outPath}`);
run(resolve(buildDir, "bin", `${name}_gold`), [outPath]);
console.log("[gen-tumble-gold] done");
