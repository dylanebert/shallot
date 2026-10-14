// One pool-slowdown run: a box pyramid pair in a sole world, stepped back to back. Prints one JSON line of
// per-step wall times and collide/solve phase times. Bundled per join variant by settle.oracle.ts. The
// `large` scene is 3,192 boxes; `small` is 72 boxes that never sleep, below the size where the pool's fixed
// cost outweighs its split.
import { BodyType } from "../../src/standard/physics/common/types";
import { init } from "../../src/standard/physics/kernel/kernel";
import { makeBoxHull } from "../../src/standard/physics/shapes/build-hull";
import { createHullShape } from "../../src/standard/physics/shapes/shape";
import { createTestBody, createTestWorld, stepTestWorld, testShapeDef } from "../../src/standard/physics/testing";
import { worldProfile } from "../../src/standard/physics/world/world";

const threads = Number(process.argv[2]);
const steps = Number(process.argv[3]);
const small = process.argv[4] === "small";
const base = small ? 8 : 56;

await init(undefined, { threads });
const hull = makeBoxHull(0.5, 0.5, 0.5);
const world = createTestWorld({ gravity: { x: 0, y: -10, z: 0 }, enableSleep: !small });
const shapeDef = testShapeDef();
const ground = createTestBody(world, { position: { x: 0, y: -1, z: 0 } });
createHullShape(world, ground, shapeDef, makeBoxHull(200, 1, 200));
for (let z = 0; z < 2; z++)
    for (let row = 0; row < base; row++)
        for (let i = 0; i < base - row; i++)
            createHullShape(
                world,
                createTestBody(world, {
                    type: BodyType.Dynamic,
                    position: { x: i * 1.05 + row * 0.525 - base * 0.5, y: 0.5 + row, z: z * 4 },
                }),
                shapeDef,
                hull,
            );

const step: number[] = [];
const collide: number[] = [];
const solve: number[] = [];
for (let i = 0; i < steps; i++) {
    const start = performance.now();
    stepTestWorld(world, 1 / 60, 4);
    step.push(performance.now() - start);
    const profile = worldProfile(world);
    collide.push(profile.collide);
    solve.push(profile.solve);
}
console.log(JSON.stringify({ step, collide, solve }));
process.exit(0);
