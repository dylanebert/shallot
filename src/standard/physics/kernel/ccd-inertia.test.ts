import { expect, test } from "bun:test";
import { BodyType, makeBoxHull, PhysicsWorld } from "../api";
import { mat3 } from "../common/math";
import { BodyFlags, getBodySim } from "../world/body";
import {
    readSimInvInertiaLocal,
    readSimInvInertiaWorld,
    readSimTransform,
    simFlags,
} from "./bodycolumns";

test("fast non-bullet inertia follows its CCD-clipped rotation in the kernel", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: false });
    try {
        const ground = world.createBody({
            type: BodyType.Static,
            position: { x: 0, y: -0.5, z: 0 },
        });
        ground.createHull({}, makeBoxHull(5, 0.5, 5));
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 1, z: 0 },
            linearVelocity: { x: 0, y: -120, z: 0 },
            angularVelocity: { x: 0, y: 0, z: 20 },
        });
        body.createHull({ density: 1 }, makeBoxHull(0.2, 0.5, 0.1));
        world.step(1 / 60, 4);
        const sim = getBodySim(world.state, world.state.bodies[body.id.index1 - 1]);
        expect(simFlags(world.state, sim) & BodyFlags.isFast).not.toBe(0);
        expect(world.state.bodies[body.id.index1 - 1].flags & BodyFlags.hadTimeOfImpact).not.toBe(
            0,
        );
        const rotation = mat3.fromQuat(
            readSimTransform(world.state, sim, {
                p: { x: 0, y: 0, z: 0 },
                q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
            }).q,
        );
        expect(
            readSimInvInertiaWorld(world.state, sim, {
                cx: { x: 0, y: 0, z: 0 },
                cy: { x: 0, y: 0, z: 0 },
                cz: { x: 0, y: 0, z: 0 },
            }),
        ).toEqual(
            mat3.mul(
                mat3.mul(
                    rotation,
                    readSimInvInertiaLocal(world.state, sim, {
                        cx: { x: 0, y: 0, z: 0 },
                        cy: { x: 0, y: 0, z: 0 },
                        cz: { x: 0, y: 0, z: 0 },
                    }),
                ),
                mat3.transpose(rotation),
            ),
        );
    } finally {
        world.destroy();
    }
});
