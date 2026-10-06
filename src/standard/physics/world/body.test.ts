import { expect, test } from "bun:test";
import { BodyType, PhysicsWorld } from "../api";
import { f32, froundConfig, mat3, quat, transformWorldPoint, vec3 } from "../common/math";
import { readSimInvInertiaLocal } from "../kernel/bodycolumns";
import { getBodySim } from "./body";

test("target-transform scratch preserves the f32 velocity expressions for both quaternion signs", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 2.1, y: 3.2, z: 4.3 },
            rotation: quat.fromAxisAngle({ x: 0, y: 1, z: 0 }, 0.4),
        });
        body.createSphere({}, { center: { x: 0.2, y: 0.3, z: 0.4 }, radius: 0.5 });
        const q1 = body.getRotation();
        const q = quat.fromAxisAngle({ x: 1, y: 0, z: 0 }, 0.7);
        for (const rotation of [q, quat.negate(q)]) {
            const target = { p: { x: 5.6, y: 6.7, z: 7.8 }, q: rotation };
            const invDt = f32(1 / (1 / 60));
            const linear = vec3.scale(
                invDt,
                vec3.sub(
                    transformWorldPoint(target, body.getMassData().center),
                    body.getWorldCenterOfMass(),
                ),
            );
            const q2 = quat.dot(q1, target.q) < 0 ? quat.negate(target.q) : target.q;
            const dq = { v: vec3.sub(q2.v, q1.v), s: f32(q2.s - q1.s) };
            const angular = vec3.scale(f32(2 * invDt), quat.mul(dq, quat.conjugate(q1)).v);
            body.setTargetTransform(target, 1 / 60);
            expect(body.getLinearVelocity()).toEqual(linear);
            expect(body.getAngularVelocity()).toEqual(angular);
        }
    } finally {
        world.destroy();
    }
});

test("impulse scratch preserves local-inertia rotation and linear-speed clamping", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 }, maximumLinearSpeed: 1 });
    try {
        const body = world.createBody({
            type: BodyType.Dynamic,
            rotation: quat.fromAxisAngle({ x: 0, y: 1, z: 0 }, 0.4),
        });
        body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        const impulse = froundConfig({ x: 1234.5, y: 2345.6, z: 3456.7 });
        const record = world.state.bodies[body.id.index1 - 1];
        const inertia = readSimInvInertiaLocal(
            world.state,
            getBodySim(world.state, record),
            mat3.zero(),
        );
        const q = body.getRotation();
        const expectedAngular = quat.rotate(q, mat3.mulV(inertia, quat.invRotate(q, impulse)));
        body.applyAngularImpulse(impulse, false);
        expect(body.getAngularVelocity()).toEqual(expectedAngular);
        const expectedLinear = vec3.normalize(vec3.scale(f32(1 / body.getMass()), impulse));
        body.applyLinearImpulseToCenter(impulse, false);
        expect(body.getLinearVelocity()).toEqual(expectedLinear);
        body.setLinearVelocity(vec3.zero());
        body.applyLinearImpulse(impulse, body.getWorldCenterOfMass(), false);
        expect(body.getLinearVelocity()).toEqual(expectedLinear);
        expect(body.getAngularVelocity()).toEqual(expectedAngular);
    } finally {
        world.destroy();
    }
});

test("setting velocity leaves static and zero-length sleeping bodies asleep, and wakes a nonzero dynamic body", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const fixed = world.createBody({ type: BodyType.Static });
        fixed.setLinearVelocity({ x: 0.5, y: 0, z: 0 });
        expect(fixed.isAwake()).toBe(false);
        expect(fixed.getLinearVelocity()).toEqual({ x: 0, y: 0, z: 0 });
        const sleeping = world.createBody({ type: BodyType.Dynamic, isAwake: false });
        for (const x of [0, -0, 1e-30]) {
            sleeping.setLinearVelocity({ x, y: 0, z: 0 });
            expect(sleeping.isAwake()).toBe(false);
        }
        sleeping.setLinearVelocity({ x: 0.5, y: 0, z: 0 });
        expect(sleeping.isAwake()).toBe(true);
        expect(sleeping.getLinearVelocity()).toEqual({ x: 0.5, y: 0, z: 0 });
        expect(sleeping.getType()).toBe(BodyType.Dynamic);
    } finally {
        world.destroy();
    }
});
