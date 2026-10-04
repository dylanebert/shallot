// One pool-slowdown run: a 3,192-box pyramid pair in a sole world, stepped back to back. Prints one JSON
// line of per-step wall times and collide/solve phase times. Bundled per join variant by settle.oracle.ts.
import { PhysicsWorld } from "../../src/standard/physics/api/world";
import { BodyType } from "../../src/standard/physics/common/types";
import { init } from "../../src/standard/physics/kernel/kernel";
import { makeBoxHull } from "../../src/standard/physics/shapes/hull";

const threads = Number(process.argv[2]);
const steps = Number(process.argv[3]);
const base = 56;

await init(undefined, { threads });
const hull = makeBoxHull(0.5, 0.5, 0.5);
const world = new PhysicsWorld({ gravity: { x: 0, y: -10, z: 0 } });
world.createBody({ position: { x: 0, y: -1, z: 0 } }).createHull({}, makeBoxHull(200, 1, 200));
for (let z = 0; z < 2; z++)
    for (let row = 0; row < base; row++)
        for (let i = 0; i < base - row; i++)
            world
                .createBody({
                    type: BodyType.Dynamic,
                    position: { x: i * 1.05 + row * 0.525 - base * 0.5, y: 0.5 + row, z: z * 4 },
                })
                .createHull({}, hull);

const step: number[] = [];
const collide: number[] = [];
const solve: number[] = [];
for (let i = 0; i < steps; i++) {
    const start = performance.now();
    world.step(1 / 60, 4);
    step.push(performance.now() - start);
    const profile = world.getProfile();
    collide.push(profile.collide);
    solve.push(profile.solve);
}
console.log(JSON.stringify({ step, collide, solve }));
process.exit(0);
