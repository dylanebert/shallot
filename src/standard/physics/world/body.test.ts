import { expect, test } from "bun:test";
import gold from "../../../../crates/physics/tests/reader_regressions.gold.json";
import { froundConfig } from "../common/config";
import { BodyType } from "../common/types";
import { readSimCenter, readStateAngularVelocity } from "../kernel/bodycolumns";
import { quaternion, vector } from "../kernel/math";
import { createSphereShape } from "../shapes/shape";
import { createTestBody, createTestWorld, destroyTestWorld, testShapeDef } from "../testing";
import {
    bodyApplyAngularImpulse,
    bodyApplyLinearImpulse,
    bodyApplyLinearImpulseToCenter,
    bodyGetLinearVelocity,
    bodyGetType,
    bodyIsAwake,
    bodySetLinearVelocity,
    bodySetTargetTransform,
    getBodySim,
    getBodyState,
} from "./body";

function nativeVector(words: number[]) {
    const v = new Float32Array(new Uint32Array(words).buffer);
    return { x: v[0], y: v[1], z: v[2] };
}

function angularVelocity(world: ReturnType<typeof createTestWorld>, body: number) {
    const state = getBodyState(world, body);
    return state === null
        ? vector.zero()
        : readStateAngularVelocity(world, state, { x: 0, y: 0, z: 0 });
}

function center(world: ReturnType<typeof createTestWorld>, body: number) {
    return readSimCenter(world, getBodySim(world, body), { x: 0, y: 0, z: 0 });
}

test("target-transform scratch preserves the f32 velocity expressions for both quaternion signs", () => {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const body = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 2.1, y: 3.2, z: 4.3 },
            rotation: quaternion.fromAxisAngle({ x: 0, y: 1, z: 0 }, 0.4),
        });
        createSphereShape(world, body, testShapeDef(), {
            center: { x: 0.2, y: 0.3, z: 0.4 },
            radius: 0.5,
        });
        const q = quaternion.fromAxisAngle({ x: 1, y: 0, z: 0 }, 0.7);
        const cases = [
            {
                p: { x: 5.6, y: 6.7, z: 7.8 },
                timeStep: 1 / 60,
                linear: gold.targetHostLinear,
                angular: gold.targetHostAngular,
            },
            {
                p: { x: 5.5, y: 6.5, z: 7.5 },
                timeStep: Math.fround(1 / 60),
                linear: gold.targetLinear,
                angular: gold.targetAngular,
            },
        ];
        for (const row of cases) {
            for (const rotation of [q, quaternion.negate(q)]) {
                bodySetTargetTransform(world, body, { p: row.p, q: rotation }, row.timeStep, false);
                expect(bodyGetLinearVelocity(world, body, vector.zero())).toEqual(
                    nativeVector(row.linear),
                );
                expect(angularVelocity(world, body)).toEqual(nativeVector(row.angular));
            }
        }
    } finally {
        destroyTestWorld(world);
    }
});

test("impulse scratch preserves local-inertia rotation and linear-speed clamping", () => {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 }, maximumLinearSpeed: 1 });
    try {
        const body = createTestBody(world, {
            type: BodyType.Dynamic,
            rotation: quaternion.fromAxisAngle({ x: 0, y: 1, z: 0 }, 0.4),
        });
        createSphereShape(world, body, testShapeDef(), {
            center: { x: 0, y: 0, z: 0 },
            radius: 0.5,
        });
        const impulse = froundConfig({ x: 1234.5, y: 2345.6, z: 3456.7 });
        const expectedAngular = nativeVector(gold.impulseAngular);
        bodyApplyAngularImpulse(world, body, impulse, false);
        expect(angularVelocity(world, body)).toEqual(expectedAngular);
        const expectedLinear = nativeVector(gold.impulseLinear);
        bodyApplyLinearImpulseToCenter(world, body, impulse, false);
        const velocity = { x: 0, y: 0, z: 0 };
        expect(bodyGetLinearVelocity(world, body, velocity)).toEqual(expectedLinear);
        bodySetLinearVelocity(world, body, vector.zero());
        bodyApplyLinearImpulse(world, body, impulse, center(world, body), false);
        expect(bodyGetLinearVelocity(world, body, velocity)).toEqual(expectedLinear);
        expect(angularVelocity(world, body)).toEqual(expectedAngular);
    } finally {
        destroyTestWorld(world);
    }
});

test("setting velocity leaves static and zero-length sleeping bodies asleep, and wakes a nonzero dynamic body", () => {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const fixed = createTestBody(world, { type: BodyType.Static });
        bodySetLinearVelocity(world, fixed, { x: 0.5, y: 0, z: 0 });
        expect(bodyIsAwake(world, fixed)).toBe(false);
        const fixedVelocity = { x: 0, y: 0, z: 0 };
        expect(bodyGetLinearVelocity(world, fixed, fixedVelocity)).toEqual({ x: 0, y: 0, z: 0 });
        const sleeping = createTestBody(world, { type: BodyType.Dynamic, isAwake: false });
        for (const x of [0, -0, 1e-30]) {
            bodySetLinearVelocity(world, sleeping, { x, y: 0, z: 0 });
            expect(bodyIsAwake(world, sleeping)).toBe(false);
        }
        bodySetLinearVelocity(world, sleeping, { x: 0.5, y: 0, z: 0 });
        expect(bodyIsAwake(world, sleeping)).toBe(true);
        expect(bodyGetType(world, sleeping)).toBe(BodyType.Dynamic);
        expect(bodyGetLinearVelocity(world, sleeping, { x: 0, y: 0, z: 0 })).toEqual({
            x: 0.5,
            y: 0,
            z: 0,
        });
    } finally {
        destroyTestWorld(world);
    }
});
