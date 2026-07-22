// Regenerates the tumble engine's bit-exact scene fixtures by building and running the Box3D C reference
// at ../reference/box3d (branch `harness`, the kex workspace layout — sibling of the shallot checkout).
// Output lands in tests/tumble/fixtures/; the engine's step.fixture.ts replays each scene and asserts
// per-step hash equality against them.
//
// The reference is built with BOX3D_DISABLE_SIMD=ON (default overflow OFF) — the colored solver with
// graph coloring + the wide (4-lane) convex path + serial mesh/overflow spill, which the port mirrors
// per-lane. DISABLE_SIMD's scalar FloatW is bit-identical per lane to the SIMD build (proven 52/52), so
// these fixtures pin the wide-simd wasm path too. Requires cmake and a C toolchain.
//
// The committed fixtures are the frozen contract (pin 29bf523 — tests/tumble/fixtures/README.md); only
// run this at a deliberate upstream sync. Absent the reference (a plain shallot checkout with no kex
// workspace around it), it errors honestly.
//
// Usage: bun run scripts/gen-tumble-fixtures.ts   (from packages/shallot)

import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";

const pkgRoot = resolve(import.meta.dir, "..");
const shallotRoot = resolve(pkgRoot, "..", "..");
const refDir = resolve(shallotRoot, "..", "reference", "box3d");
const buildDir = resolve(refDir, "build-fixtures");
const outDir = resolve(pkgRoot, "tests", "tumble", "fixtures");

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

console.log("[gen-tumble-fixtures] configuring reference (scalar-lane, colored)");
run("cmake", [
    "-S",
    refDir,
    "-B",
    buildDir,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBOX3D_DISABLE_SIMD=ON",
    "-DBOX3D_FORCE_OVERFLOW=OFF",
    "-DBOX3D_FIXTURES=ON",
    "-DBOX3D_SAMPLES=OFF",
    "-DBOX3D_BENCHMARKS=OFF",
    "-DBOX3D_UNIT_TESTS=OFF",
    "-DBOX3D_DOCS=OFF",
]);

console.log("[gen-tumble-fixtures] building fixture_gen");
run("cmake", ["--build", buildDir, "--target", "fixture_gen", "-j"]);

mkdirSync(outDir, { recursive: true });
console.log(`[gen-tumble-fixtures] generating fixtures -> ${outDir}`);
run(resolve(buildDir, "bin", "fixture_gen"), [outDir]);
console.log("[gen-tumble-fixtures] done");
