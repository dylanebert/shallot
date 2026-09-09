// The tumble.js `Elevator` sample (`samples/src/samples/joints.ts`) reproduced near-verbatim through the
// escape-hatch `World` API. A platform on a vertical prismatic joint, motor-driven up and down between
// limits, carrying crates. `update()` reverses the motor speed when the translation crosses a limit —
// world-mutating, load-bearing for the gold.
//
// Creation order is load-bearing for the hash: ground, anchor, platform, prismatic joint, then the three
// crates — the sample's exact order.

import {
    BodyType,
    makeBoxHull,
    type PrismaticJoint,
    type World,
} from "@dylanebert/shallot/tumble/core";
import type { SampleUpdate } from "./tumble-oracle";

const SPEED = 2.5;

let joint: PrismaticJoint | null = null;

/**
 * Author the Elevator scene into `world`: a static ground box, a static anchor and a dynamic platform
 * coincident at the origin joined by a vertical prismatic joint (motor-driven, limited 0..5), and three
 * loose density-1 crates it carries.
 */
export function buildElevator(world: World): void {
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    // Rotate the joint frame (not the bodies) so the slide axis — frame A's local x — points up.
    const up90 = { v: { x: 0, y: 0, z: Math.SQRT1_2 }, s: Math.SQRT1_2 };
    const anchor = world.createBody({ type: BodyType.Static, position: { x: 0, y: 1, z: 0 } });
    const platform = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 1, z: 0 } });
    platform.createHull({ density: 2 }, makeBoxHull(1.5, 0.2, 1.5));

    joint = world.createPrismaticJoint(anchor, platform, {
        localFrameA: { p: { x: 0, y: 0, z: 0 }, q: up90 },
        localFrameB: { p: { x: 0, y: 0, z: 0 }, q: up90 },
        enableMotor: true,
        motorSpeed: SPEED,
        maxMotorForce: 2000,
        enableLimit: true,
        lowerTranslation: 0,
        upperTranslation: 5,
    });

    for (let i = 0; i < 3; ++i) {
        const crate = world.createBody({
            type: BodyType.Dynamic,
            position: { x: -0.6 + 0.6 * i, y: 2 + i * 0.7, z: 0 },
        });
        crate.createHull({ density: 1 }, makeBoxHull(0.35, 0.35, 0.35));
    }
}

/**
 * Reverse the motor direction past each translation limit (the sample's `update()`): above 4.8 drive
 * down, below 0.2 drive up.
 */
export const updateElevator: SampleUpdate = () => {
    if (joint === null) return;
    const t = joint.getTranslation();
    if (t > 4.8) joint.setMotorSpeed(-SPEED);
    else if (t < 0.2) joint.setMotorSpeed(SPEED);
};
