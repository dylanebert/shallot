// The Paddle pilot build — the tumble.js Paddle sample (`samples/src/samples/joints.ts` `Paddle`)
// reproduced near-verbatim through the escape-hatch `World` API. A motor joint spins a paddle about a
// fixed pivot at a constant angular velocity, batting a pile of loose boxes around. This is the stage-3
// pilot: its authoring is replayed against the committed `joints-paddle` gold and must reproduce it
// bit-exact (only authoring can differ — both sides are the same engine).
//
// Creation order is load-bearing for the hash (the colored solver keys contact ordering on it): ground,
// anchor, paddle, motor joint, then the eight boxes — the sample's exact order.

import { BodyType, makeBoxHull, type World } from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";

const IDENT = { v: { x: 0, y: 0, z: 0 }, s: 1 };

/**
 * Author the Paddle scene into `world`, reading the `speed` knob (rad/s the motor spins the paddle about
 * its local z). Reproduces the sample's `build()` — a static ground box, a static anchor and a dynamic
 * paddle coincident at the pivot joined by a z-axis motor, and eight loose density-0.5 boxes to bat.
 */
export function buildPaddle(world: World, params: SampleParams): void {
    const speed = params.speed as number;

    // static ground box: Sample.addGroundBox(1) → a 50×1×50 half-extent hull at the origin.
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const pivot = { x: 0, y: 2, z: 0 };
    const anchor = world.createBody({ type: BodyType.Static, position: pivot });
    const paddle = world.createBody({ type: BodyType.Dynamic, position: pivot });
    paddle.createHull({ density: 1 }, makeBoxHull(2.2, 0.2, 0.4));
    world.createMotorJoint(anchor, paddle, {
        localFrameA: { p: { x: 0, y: 0, z: 0 }, q: IDENT },
        localFrameB: { p: { x: 0, y: 0, z: 0 }, q: IDENT },
        angularVelocity: { x: 0, y: 0, z: speed },
        maxVelocityTorque: 800,
        maxVelocityForce: 4000,
    });

    for (let i = 0; i < 8; ++i) {
        const b = world.createBody({
            type: BodyType.Dynamic,
            position: { x: -3 + (i % 4) * 2, y: 5 + Math.floor(i / 4), z: -1 + (i % 2) * 2 },
        });
        b.createHull({ density: 0.5 }, makeBoxHull(0.3, 0.3, 0.3));
    }
}
