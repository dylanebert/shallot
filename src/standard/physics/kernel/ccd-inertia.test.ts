import { expect, test } from "bun:test";
import { mat3 } from "../common/math";
import { BodyType } from "../common/types";
import { makeBoxHull } from "../shapes/build-hull";
import { createHullShape } from "../shapes/shape";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "../testing";
import { BodyFlags, getBodySim } from "../world/body";
import {
    readSimInvInertiaLocal,
    readSimInvInertiaWorld,
    readSimTransform,
    simFlags,
} from "./bodycolumns";

test("fast non-bullet inertia follows its CCD-clipped rotation in the kernel", () => {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: false });
    try {
        const ground = createTestBody(world, {
            type: BodyType.Static,
            position: { x: 0, y: -0.5, z: 0 },
        });
        createHullShape(world, ground, testShapeDef(), makeBoxHull(5, 0.5, 5));
        const body = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 0, y: 1, z: 0 },
            linearVelocity: { x: 0, y: -120, z: 0 },
            angularVelocity: { x: 0, y: 0, z: 20 },
        });
        createHullShape(world, body, testShapeDef({ density: 1 }), makeBoxHull(0.2, 0.5, 0.1));
        stepTestWorld(world, 1 / 60, 4);
        const sim = getBodySim(world, body);
        expect(simFlags(world, sim) & BodyFlags.isFast).not.toBe(0);
        expect(simFlags(world, sim) & BodyFlags.hadTimeOfImpact).not.toBe(0);
        const rotation = mat3.fromQuat(
            readSimTransform(world, sim, {
                p: { x: 0, y: 0, z: 0 },
                q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
            }).q,
        );
        expect(
            readSimInvInertiaWorld(world, sim, {
                cx: { x: 0, y: 0, z: 0 },
                cy: { x: 0, y: 0, z: 0 },
                cz: { x: 0, y: 0, z: 0 },
            }),
        ).toEqual(
            mat3.mul(
                mat3.mul(
                    rotation,
                    readSimInvInertiaLocal(world, sim, {
                        cx: { x: 0, y: 0, z: 0 },
                        cy: { x: 0, y: 0, z: 0 },
                        cz: { x: 0, y: 0, z: 0 },
                    }),
                ),
                mat3.transpose(rotation),
            ),
        );
    } finally {
        destroyTestWorld(world);
    }
});
