import { expect, test } from "bun:test";
import { BodyType, PhysicsWorld } from "../api";
import { J_EVENT } from "../kernel/columns";
import { collectJointEvents, writeJointFloat } from "../kernel/jointcolumns";

test("plain-id joint bindings preserve public frames, bodies, tuning and user-data snapshot identity", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const a = world.createBody({ type: BodyType.Dynamic });
        const b = world.createBody({ type: BodyType.Dynamic, position: { x: 2, y: 0, z: 0 } });
        const userData = { label: "joint" };
        const joint = world.createDistanceJoint(a, b, {
            userData,
            constraintHertz: 13,
            constraintDampingRatio: 2,
            forceThreshold: 23,
            torqueThreshold: 29,
            localFrameA: { p: { x: 0.25, y: 0.5, z: 0.75 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
            localFrameB: {
                p: { x: -0.25, y: -0.5, z: -0.75 },
                q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
            },
        });
        const output = () => ({
            bodies: joint.getBodies().map((body) => body.id),
            a: joint.getLocalFrameA(),
            b: joint.getLocalFrameB(),
            tuning: joint.getConstraintTuning(),
            thresholds: [joint.getForceThreshold(), joint.getTorqueThreshold()],
            collide: joint.getCollideConnected(),
            force: joint.getConstraintForce(),
            torque: joint.getConstraintTorque(),
        });
        const expected = {
            bodies: [a.id, b.id],
            a: { p: { x: 0.25, y: 0.5, z: 0.75 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
            b: { p: { x: -0.25, y: -0.5, z: -0.75 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
            tuning: { hertz: 13, dampingRatio: 2 },
            thresholds: [23, 29],
            collide: false,
            force: { x: 0, y: -0, z: -0 },
            torque: { x: 0, y: 0, z: 0 },
        };
        expect(output()).toEqual(expected);
        const saved = world.snapshot();
        joint.setUserData({ label: "replacement" });
        joint.setConstraintTuning(31, 3);
        world.restore(saved);
        expect(output()).toEqual(expected);
        expect(joint.getUserData()).toBe(userData);
    } finally {
        world.destroy();
    }
});

test("a stepped over-stressed joint publishes the same ordered public event ids and opaque user data as its creation handles", () => {
    const world = new PhysicsWorld();
    try {
        const anchor = world.createBody({ type: BodyType.Static });
        const data = [{ label: "left" }, { label: "right" }];
        const joints = data.map((userData, i) => {
            const body = world.createBody({
                type: BodyType.Dynamic,
                position: { x: i * 4, y: -2, z: 0 },
            });
            body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
            return world.createDistanceJoint(anchor, body, {
                length: 2,
                forceThreshold: 0,
                userData,
                localFrameA: { p: { x: i * 4, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
            });
        });
        world.step(1 / 60);
        const events = world.getJointEvents();
        expect(events.map((event) => event.joint.id)).toEqual(joints.map((joint) => joint.id));
        for (let i = 0; i < events.length; ++i) {
            expect(events[i].userData).toBe(data[i]);
            expect(events[i].joint.isValid()).toBe(true);
            expect(events[i].joint.getConstraintForce().y).toBeGreaterThan(0);
        }
    } finally {
        world.destroy();
    }
});

test("a foreign snapshot preserves captured joint-event id values while delivery handles bind to the target world", () => {
    const source = new PhysicsWorld();
    const target = new PhysicsWorld();
    try {
        const a = source.createBody({ type: BodyType.Dynamic });
        const b = source.createBody({ type: BodyType.Dynamic });
        const userData = { label: "captured" };
        const joint = source.createFilterJoint(a, b, { userData });
        writeJointFloat(source.state, joint.id.index1 - 1, J_EVENT, 1);
        collectJointEvents(source.state);
        const captured = source.getJointEvents()[0].joint.id;
        expect(source.state.worldId).not.toBe(target.state.worldId);
        target.restore(source.snapshot());
        const event = target.getJointEvents()[0];
        expect(event.joint.id).toEqual(captured);
        expect(event.userData).toBe(userData);
        expect(event.joint.getWorld().state).toBe(target.state);
    } finally {
        source.destroy();
        target.destroy();
    }
});

test("internal joint event collection preserves ordered public ids, generations and user-data identity across destroy, reuse and restore", () => {
    const world = new PhysicsWorld();
    try {
        const a = world.createBody({ type: BodyType.Dynamic });
        const b = world.createBody({ type: BodyType.Dynamic });
        const data = [{ label: "first" }, { label: "second" }];
        const joints = data.map((userData) => world.createFilterJoint(a, b, { userData }));
        for (let i = joints.length - 1; i >= 0; --i)
            writeJointFloat(world.state, joints[i].id.index1 - 1, J_EVENT, 1);
        collectJointEvents(world.state);
        const expectedIds = joints.map((joint) => joint.id);
        const saved = world.snapshot();
        joints[0].destroy(false);
        const replacement = world.createFilterJoint(a, b, { userData: { label: "replacement" } });
        expect(replacement.id.index1).toBe(expectedIds[0].index1);
        expect(replacement.id.generation).not.toBe(expectedIds[0].generation);
        const events = world.getJointEvents();
        expect(events.map((event) => event.joint.id)).toEqual(expectedIds);
        for (let i = 0; i < events.length; ++i) expect(events[i].userData).toBe(data[i]);
        expect(events[0].joint.isValid()).toBe(false);
        world.restore(saved);
        const restored = world.getJointEvents();
        expect(restored.map((event) => event.joint.id)).toEqual(expectedIds);
        for (let i = 0; i < restored.length; ++i) {
            expect(restored[i].userData).toBe(data[i]);
            expect(restored[i].joint.isValid()).toBe(true);
        }
        world.step(1 / 60);
        expect(world.getJointEvents()).toHaveLength(0);
    } finally {
        world.destroy();
    }
});
