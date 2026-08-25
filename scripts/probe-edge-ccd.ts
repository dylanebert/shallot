// `bun run scripts/probe-edge-ccd.ts` — the distance.ts:1154 reachability reading.
//
// Stage S2 of audit-tumble-engine-bitexact-literals moved `kToleranceSquared` at
// `distance.ts:1154` from `f32(0.05 * 0.05)` (= 0.0024999999441206455) to
// `f32(f32(0.05) * f32(0.05))` (= 0.002500000176951289) to match C's `0.05f * 0.05f`.
// That constant sits inside `makeSeparationFunction`'s edge/edge sub-branch
// (`cache.count === 2 && uniqueCountA === 2 && uniqueCountB === 2`), reached during
// CCD conservative advancement. The spec's Validation criterion owes a differential
// reading: a continuous scene whose `SeparationType` branch is exercised, before and after.
//
// This script constructs that scene — two boxes at non-parallel orientations under CCD —
// steps it through the world, and prints the per-step FNV-1a world-state hash. What it
// establishes: the scene reaches the line-1154 edge/edge comparison (1 call to
// `makeSeparationFunction`, `cache.count === 2`, `uniqueCountA === 2 && uniqueCountB === 2`),
// where `lengthSquared` = 0.992161214351654 against `kToleranceSquared` =
// 0.002500000176951289 selects `SeparationType.Edges`; the operands differ by ~400× and the
// constants by 1 ULP, so the branch outcome is invariant to the 1-ULP move (the old
// `f32(0.05 * 0.05)` = 0.0024999999441206455 still selects `Edges`).
//
// What it does not establish: the printed hash series does not discriminate the
// `SeparationType` branch at this site — forcing the opposite (`Vertices`) branch leaves the
// series byte-identical (measured), so hash identity here is not evidence about that branch.
// Re-establishing the reachability reading requires temporary instrumentation of
// `makeSeparationFunction`; the committed probe reprints hashes only.
//
// Scene: "ccd-edge-edge" — a static box at the origin rotated 90° around Z, and a dynamic
// bullet box at [0, 2, 0] rotated 5° around Y falling at -20 m/s. The non-parallel edge
// orientations drive the GJK simplex to count=2 (edge/edge) during the CCD sweep.

import { hashWorldState } from "../packages/shallot/src/standard/tumble/engine/hash";
import { BodyType, makeBoxHull, World } from "../packages/shallot/src/standard/tumble/engine/index";
import { init, shutdown } from "../packages/shallot/src/standard/tumble/engine/kernel";
import { DEG_TO_RAD, quat } from "../packages/shallot/src/standard/tumble/engine/math";

function toHex(h: bigint): string {
    return `0x${h.toString(16).padStart(16, "0")}`;
}

async function main() {
    await init({ threads: 0 });

    const q90z = quat.fromAxisAngle({ x: 0, y: 0, z: 1 }, DEG_TO_RAD * 90);
    const q5y = quat.fromAxisAngle({ x: 0, y: 1, z: 0 }, DEG_TO_RAD * 5);

    const world = new World({
        gravity: { x: 0, y: -10, z: 0 },
        enableSleep: false,
        enableContinuous: true,
    });

    // Static box at origin, rotated 90° around Z — its long axis now points along X.
    const ground = world.createBody({
        type: BodyType.Static,
        position: { x: 0, y: 0, z: 0 },
        rotation: { v: { x: q90z.v.x, y: q90z.v.y, z: q90z.v.z }, s: q90z.s },
    });
    ground.createHull({}, makeBoxHull(0.5, 0.5, 0.5));

    // Dynamic bullet box at [0, 2, 0], rotated 5° around Y, falling at -20 m/s.
    const bullet = world.createBody({
        type: BodyType.Dynamic,
        position: { x: 0, y: 2, z: 0 },
        rotation: { v: { x: q5y.v.x, y: q5y.v.y, z: q5y.v.z }, s: q5y.s },
        linearVelocity: { x: 0, y: -20, z: 0 },
        isBullet: true,
    });
    bullet.createHull({}, makeBoxHull(0.5, 0.5, 0.5));

    const dt = 1 / 60;
    const stepCount = 60;
    const hashes: string[] = [];

    for (let step = 0; step < stepCount; ++step) {
        world.step(dt, 4);
        hashes.push(toHex(hashWorldState(world.state)));
    }

    // Report: the reachability proof (if instrumentation is present) and the hash series.
    const edgeEdgeCount = (globalThis as any).__edgeEdgeCount ?? "n/a (no instrumentation)";
    const lastLengthSq = (globalThis as any).__lastLengthSq ?? "n/a";
    const lastKTol = (globalThis as any).__lastKTol ?? "n/a";
    const lastSepType = (globalThis as any).__lastSepType ?? "n/a";
    const sepName = lastSepType === 1 ? "Vertices" : lastSepType === 2 ? "Edges" : "n/a";

    console.log("ccd-edge-edge probe");
    console.log(`  edgeEdgeCount: ${edgeEdgeCount}`);
    console.log(`  lastLengthSq:  ${lastLengthSq}`);
    console.log(`  lastKTol:      ${lastKTol}`);
    console.log(`  sepType:       ${sepName}`);
    console.log(`  stepCount:     ${stepCount}`);
    console.log(`  hashes:`);
    for (let i = 0; i < hashes.length; ++i) {
        console.log(`    [${i}] = ${hashes[i]}`);
    }

    world.destroy();
    await shutdown();
}

main();
