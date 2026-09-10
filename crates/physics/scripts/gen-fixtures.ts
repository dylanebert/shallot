// Regenerates the physics engine's bit-exact scene fixtures by building and running the Box3D C reference
// pinned in crates/physics/reference.json, cloned on demand into the user cache (scripts/reference.ts).
// Output lands in tests/physics/fixtures/; the engine's step.fixture.ts replays each scene and asserts
// per-step hash equality against them.
//
// The reference is built with BOX3D_DISABLE_SIMD=ON (default overflow OFF) — the colored solver with
// graph coloring + the wide (4-lane) convex path + serial mesh/overflow spill, which the port mirrors
// per-lane. DISABLE_SIMD's scalar FloatW is bit-identical per lane to the SIMD build (proven across
// every scene), so
// these fixtures pin the wide-simd wasm path too. Requires cmake and a C toolchain.
//
// The committed fixtures are the frozen contract (pin 29bf523 — tests/physics/fixtures/README.md); only
// run this at a deliberate upstream sync. Offline with no cached checkout, it refuses with the remedy.
//
// Usage: bun run crates/physics/scripts/gen-fixtures.ts   (from the repo root)

import { spawnSync } from "node:child_process";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { ensureReference } from "./reference";

const pkgRoot = resolve(import.meta.dir, "../../..");
const refDir = ensureReference();
const buildDir = resolve(refDir, "build-fixtures");
const outDir = resolve(pkgRoot, "src/standard/physics/solver/fixtures");

function run(cmd: string, args: string[]) {
    const r = spawnSync(cmd, args, { cwd: refDir, stdio: "inherit" });
    if (r.status !== 0) process.exit(r.status ?? 1);
}

console.log("[physics/gen-fixtures] configuring reference (scalar-lane, colored)");
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

console.log("[physics/gen-fixtures] building fixture_gen");
run("cmake", ["--build", buildDir, "--target", "fixture_gen", "-j"]);

mkdirSync(outDir, { recursive: true });
console.log(`[physics/gen-fixtures] generating fixtures -> ${outDir}`);
run(resolve(buildDir, "bin", "fixture_gen"), [outDir]);
console.log("[physics/gen-fixtures] done");
