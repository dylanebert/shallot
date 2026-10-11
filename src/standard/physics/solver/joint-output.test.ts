import { expect, test } from "bun:test";
import { makeJointId } from "../api/config";
import { restore, snapshot } from "../api/snapshot";
import { BodyType } from "../common/types";
import {
    J_CONSTRAINT_DAMPING,
    J_CONSTRAINT_HERTZ,
    J_FORCE_THRESHOLD,
    J_LOCAL_FRAME_A,
    J_LOCAL_FRAME_B,
    J_TORQUE_THRESHOLD,
} from "../kernel/columns";
import { EventKind, eventCount, eventId } from "../kernel/eventbuffers";
import {
    collectJointEvents,
    readJointEventUserData,
    readJointFloat,
    readJointTransform,
    writeJointFloat,
} from "../kernel/jointcolumns";
import { JointField, jointField } from "../kernel/jointrecords";
import { kernel } from "../kernel/kernel";
import { createSphereShape } from "../shapes/shape";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "../testing";
import { destroyBody } from "../world/body";
import { createDistanceJoint, defaultDistanceJointDef } from "./distanceJoint";
import {
    createFilterJoint,
    defaultJointDef,
    destroyJointInternal,
    getJointConstraintForce,
    getJointConstraintTorque,
    jointIsValid,
} from "./joint";

function distanceJoint(
    world: ReturnType<typeof createTestWorld>,
    a: number,
    b: number,
    config: Record<string, unknown> = {},
) {
    const base = { ...defaultJointDef(), bodyIdA: a, bodyIdB: b };
    for (const name of [
        "localFrameA",
        "localFrameB",
        "collideConnected",
        "constraintHertz",
        "constraintDampingRatio",
        "forceThreshold",
        "torqueThreshold",
        "drawScale",
        "userData",
    ] as const)
        if (config[name] !== undefined) Object.assign(base, { [name]: config[name] });
    return createDistanceJoint(world, { ...defaultDistanceJointDef(base), ...config }).joint;
}

function jointEvents(world: ReturnType<typeof createTestWorld>) {
    readJointEventUserData(world);
    return Array.from({ length: eventCount(world, EventKind.Joint) }, (_, index) => ({
        id: eventId(world, EventKind.Joint, index, 0),
        userData: world.jointEventUserData[index],
    }));
}

test("plain-id joint bindings preserve frames, bodies, tuning and user-data snapshot identity", () => {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const a = createTestBody(world, { type: BodyType.Dynamic });
        const b = createTestBody(world, { type: BodyType.Dynamic, position: { x: 2, y: 0, z: 0 } });
        const userData = { label: "joint" };
        const id = distanceJoint(world, a, b, {
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
            bodies: [
                jointField(world, id, JointField.bodyIdA),
                jointField(world, id, JointField.bodyIdB),
            ],
            a: readJointTransform(world, id, J_LOCAL_FRAME_A),
            b: readJointTransform(world, id, J_LOCAL_FRAME_B),
            tuning: {
                hertz: readJointFloat(world, id, J_CONSTRAINT_HERTZ),
                dampingRatio: readJointFloat(world, id, J_CONSTRAINT_DAMPING),
            },
            thresholds: [
                readJointFloat(world, id, J_FORCE_THRESHOLD),
                readJointFloat(world, id, J_TORQUE_THRESHOLD),
            ],
            collide: jointField(world, id, JointField.collideConnected) !== 0,
            force: getJointConstraintForce(world, id),
            torque: getJointConstraintTorque(world, id),
        });
        const expected = {
            bodies: [a, b],
            a: { p: { x: 0.25, y: 0.5, z: 0.75 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
            b: { p: { x: -0.25, y: -0.5, z: -0.75 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
            tuning: { hertz: 13, dampingRatio: 2 },
            thresholds: [23, 29],
            collide: false,
            force: { x: 0, y: -0, z: -0 },
            torque: { x: 0, y: 0, z: 0 },
        };
        expect(output()).toEqual(expected);
        const saved = snapshot(world);
        world.jointUserData[id] = { label: "replacement" };
        writeJointFloat(world, id, J_CONSTRAINT_HERTZ, 31);
        writeJointFloat(world, id, J_CONSTRAINT_DAMPING, 3);
        restore(world, saved);
        expect(output()).toEqual(expected);
        expect(world.jointUserData[id]).toBe(userData);
    } finally {
        destroyTestWorld(world);
    }
});

test("a stepped over-stressed joint publishes ordered event ids and opaque user data", () => {
    const world = createTestWorld();
    try {
        const anchor = createTestBody(world, { type: BodyType.Static });
        const data = [{ label: "left" }, { label: "right" }];
        const joints = data.map((userData, i) => {
            const body = createTestBody(world, {
                type: BodyType.Dynamic,
                position: { x: i * 4, y: -2, z: 0 },
            });
            createSphereShape(world, body, testShapeDef(), {
                center: { x: 0, y: 0, z: 0 },
                radius: 0.5,
            });
            return distanceJoint(world, anchor, body, {
                length: 2,
                forceThreshold: 0,
                userData,
                localFrameA: { p: { x: i * 4, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
            });
        });
        stepTestWorld(world, 1 / 60);
        const events = jointEvents(world);
        expect(events.map((event) => event.id)).toEqual(
            joints.map((joint) => makeJointId(world, joint)),
        );
        for (let i = 0; i < events.length; ++i) {
            expect(events[i]!.userData).toBe(data[i]);
            expect(jointIsValid(world, events[i]!.id)).toBe(true);
            expect(getJointConstraintForce(world, joints[i]!).y).toBeGreaterThan(0);
        }
    } finally {
        destroyTestWorld(world);
    }
});

test("body destruction preserves published joint-event user-data identity", () => {
    const world = createTestWorld();
    try {
        const anchor = createTestBody(world, { type: BodyType.Static });
        const body = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 0, y: -2, z: 0 },
        });
        createSphereShape(world, body, testShapeDef(), {
            center: { x: 0, y: 0, z: 0 },
            radius: 0.5,
        });
        const userData = { label: "captured" };
        distanceJoint(world, anchor, body, { length: 2, forceThreshold: 0, userData });
        stepTestWorld(world, 1 / 60);
        destroyBody(world, body);
        const events = jointEvents(world);
        expect(events).toHaveLength(1);
        expect(events[0]!.userData).toBe(userData);
    } finally {
        destroyTestWorld(world);
    }
});

test("a foreign snapshot preserves captured joint-event ids while records rebind to the target world", () => {
    const source = createTestWorld();
    const target = createTestWorld();
    try {
        const a = createTestBody(source, { type: BodyType.Dynamic });
        const b = createTestBody(source, { type: BodyType.Dynamic });
        const userData = { label: "captured" };
        const joint = createFilterJoint(source, {
            ...defaultJointDef(),
            bodyIdA: a,
            bodyIdB: b,
            userData,
        }).joint;
        const k = kernel(source.ecsState);
        k.jointResetEventBits(source.worldId);
        k.jointSetEventBit(source.worldId, joint);
        collectJointEvents(source);
        const captured = eventId(source, EventKind.Joint, 0, 0);
        expect(source.worldId).not.toBe(target.worldId);
        restore(target, snapshot(source));
        const event = jointEvents(target)[0]!;
        expect(event.id).toEqual(captured);
        expect(event.userData).toBe(userData);
        const rebound = makeJointId(target, event.id.index1 - 1);
        expect(rebound.world0).toBe(target.worldId);
        expect(rebound.generation).toBe(event.id.generation);
    } finally {
        destroyTestWorld(source);
        destroyTestWorld(target);
    }
});

test("joint event collection preserves ordered ids, generations and user-data across destroy, reuse and restore", () => {
    const world = createTestWorld();
    try {
        const a = createTestBody(world, { type: BodyType.Dynamic });
        const b = createTestBody(world, { type: BodyType.Dynamic });
        const data = [{ label: "first" }, { label: "second" }];
        const joints = data.map(
            (userData) =>
                createFilterJoint(world, { ...defaultJointDef(), bodyIdA: a, bodyIdB: b, userData })
                    .joint,
        );
        const k = kernel(world.ecsState);
        k.jointResetEventBits(world.worldId);
        for (let i = joints.length - 1; i >= 0; --i) k.jointSetEventBit(world.worldId, joints[i]!);
        collectJointEvents(world);
        const expectedIds = joints.map((joint) => makeJointId(world, joint));
        const saved = snapshot(world);
        destroyJointInternal(world, joints[0]!, false);
        const replacement = createFilterJoint(world, {
            ...defaultJointDef(),
            bodyIdA: a,
            bodyIdB: b,
            userData: { label: "replacement" },
        }).joint;
        expect(replacement).toBe(joints[0]);
        expect(makeJointId(world, replacement).generation).not.toBe(expectedIds[0]!.generation);
        const events = jointEvents(world);
        expect(events.map((event) => event.id)).toEqual(expectedIds);
        for (let i = 0; i < events.length; ++i) expect(events[i]!.userData).toBe(data[i]);
        restore(world, saved);
        expect(jointEvents(world).map((event) => event.id)).toEqual(expectedIds);
        for (let i = 0; i < data.length; ++i) expect(jointEvents(world)[i]!.userData).toBe(data[i]);
    } finally {
        destroyTestWorld(world);
    }
});
