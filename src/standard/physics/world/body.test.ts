import { expect, test } from "bun:test";
import { f32, froundConfig, mat3, quat, transformWorldPoint, vec3 } from "../common/math";
import { BodyType } from "../common/types";
import {
    readSimCenter,
    readSimInvInertiaLocal,
    readStateAngularVelocity,
} from "../kernel/bodycolumns";
import { createSphereShape } from "../shapes/shape";
import { createTestBody, createTestWorld, destroyTestWorld, testShapeDef } from "../testing";
import {
    bodyApplyAngularImpulse,
    bodyApplyLinearImpulse,
    bodyApplyLinearImpulseToCenter,
    bodyGetLinearVelocity,
    bodyGetTransform,
    bodyGetType,
    bodyIsAwake,
    bodySetLinearVelocity,
    bodySetTargetTransform,
    getBodySim,
    getBodyState,
    getMassData,
} from "./body";

function angularVelocity(world: ReturnType<typeof createTestWorld>, body: number) {
    const state = getBodyState(world, body);
    return state === null
        ? vec3.zero()
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
            rotation: quat.fromAxisAngle({ x: 0, y: 1, z: 0 }, 0.4),
        });
        createSphereShape(world, body, testShapeDef(), {
            center: { x: 0.2, y: 0.3, z: 0.4 },
            radius: 0.5,
        });
        const q1 = bodyGetTransform(world, body, { p: vec3.zero(), q: quat.identity() }).q;
        const q = quat.fromAxisAngle({ x: 1, y: 0, z: 0 }, 0.7);
        for (const rotation of [q, quat.negate(q)]) {
            const target = { p: { x: 5.6, y: 6.7, z: 7.8 }, q: rotation };
            const invDt = f32(1 / (1 / 60));
            const linear = vec3.scale(
                invDt,
                vec3.sub(
                    transformWorldPoint(target, getMassData(world, body).center),
                    center(world, body),
                ),
            );
            const q2 = quat.dot(q1, target.q) < 0 ? quat.negate(target.q) : target.q;
            const dq = { v: vec3.sub(q2.v, q1.v), s: f32(q2.s - q1.s) };
            const angular = vec3.scale(f32(2 * invDt), quat.mul(dq, quat.conjugate(q1)).v);
            bodySetTargetTransform(world, body, target, 1 / 60, false);
            const velocity = { x: 0, y: 0, z: 0 };
            expect(bodyGetLinearVelocity(world, body, velocity)).toEqual(linear);
            expect(angularVelocity(world, body)).toEqual(angular);
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
            rotation: quat.fromAxisAngle({ x: 0, y: 1, z: 0 }, 0.4),
        });
        createSphereShape(world, body, testShapeDef(), {
            center: { x: 0, y: 0, z: 0 },
            radius: 0.5,
        });
        const impulse = froundConfig({ x: 1234.5, y: 2345.6, z: 3456.7 });
        const inertia = readSimInvInertiaLocal(world, getBodySim(world, body), mat3.zero());
        const q = bodyGetTransform(world, body, { p: vec3.zero(), q: quat.identity() }).q;
        const expectedAngular = quat.rotate(q, mat3.mulV(inertia, quat.invRotate(q, impulse)));
        bodyApplyAngularImpulse(world, body, impulse, false);
        expect(angularVelocity(world, body)).toEqual(expectedAngular);
        const expectedLinear = vec3.normalize(
            vec3.scale(f32(1 / getMassData(world, body).mass), impulse),
        );
        bodyApplyLinearImpulseToCenter(world, body, impulse, false);
        const velocity = { x: 0, y: 0, z: 0 };
        expect(bodyGetLinearVelocity(world, body, velocity)).toEqual(expectedLinear);
        bodySetLinearVelocity(world, body, vec3.zero());
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
