// Direct-state joint behavior checks. Wrapper-specific accessor tests left with the removed handle API;
// kernel creation, ordering, filtering, solver response and output are exercised over state and IDs.
import { expect, test } from "bun:test";
import { froundConfig } from "../common/config";
import type { Vec3 } from "../common/records";
import { quaternion, vector } from "../kernel/math";

const f32 = Math.fround;

import { BodyType, type WorldDef } from "../common/types";
import { readSimCenter, readStateAngularVelocity } from "../kernel/bodycolumns";
import { JointField, jointField, setJointField } from "../kernel/jointrecords";
import { makeBoxHull, makeCubeHull } from "../shapes/build-hull";
import { createHullShape, createSphereShape } from "../shapes/shape";
import { createDistanceJoint, defaultDistanceJointDef } from "../solver/distanceJoint";
import {
    createFilterJoint,
    defaultJointDef,
    destroyJointInternal,
    getJointConstraintForce,
    getJointConstraintTorque,
    type JointDef,
    JointType,
    jointIsValid,
} from "../solver/joint";
import { createMotorJoint, defaultMotorJointDef } from "../solver/motorJoint";
import { createParallelJoint, defaultParallelJointDef } from "../solver/parallelJoint";
import {
    createPrismaticJoint,
    defaultPrismaticJointDef,
    prismaticJointSpeed,
} from "../solver/prismaticJoint";
import { createRevoluteJoint, defaultRevoluteJointDef } from "../solver/revoluteJoint";
import { createSphericalJoint, defaultSphericalJointDef } from "../solver/sphericalJoint";
import { createWeldJoint, defaultWeldJointDef } from "../solver/weldJoint";
import { createWheelJoint, defaultWheelJointDef } from "../solver/wheelJoint";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "../testing";
import {
    bodyGetLinearVelocity,
    bodyIsAwake,
    getBodySim,
    getBodyState,
    getMassData,
    readBodyTransform,
} from "../world/body";
import type { WorldState } from "../world/world";
import { worldCounters } from "../world/world";

const len = (v: Vec3): number => Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
const frame = (x = 0, y = 0, z = 0) => ({ p: { x, y, z }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } });
const identityPose = () => ({ p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } });
const baseDef = (a: number, b: number, fields: Partial<JointDef> = {}): JointDef => ({
    ...defaultJointDef(),
    bodyIdA: a,
    bodyIdB: b,
    ...fields,
});
function addHull(
    world: WorldState,
    body: number,
    hull: ReturnType<typeof makeBoxHull>,
    density = 1,
): void {
    createHullShape(world, body, testShapeDef({ density }), hull);
}
const baseJointKeys = [
    "localFrameA",
    "localFrameB",
    "collideConnected",
    "constraintHertz",
    "constraintDampingRatio",
    "forceThreshold",
    "torqueThreshold",
    "drawScale",
    "userData",
];
function splitJointFields(
    fields: Record<string, unknown>,
): [Partial<JointDef>, Record<string, unknown>] {
    const base: Record<string, unknown> = {};
    const specific = { ...fields };
    for (const key of baseJointKeys) {
        if (key in fields) {
            base[key] = fields[key];
            delete specific[key];
        }
    }
    return [base as Partial<JointDef>, specific];
}
function revolute(world: WorldState, a: number, b: number, fields: Record<string, unknown> = {}) {
    const [base, specific] = splitJointFields(fields);
    return createRevoluteJoint(world, {
        ...defaultRevoluteJointDef(baseDef(a, b, base)),
        ...specific,
    } as Parameters<typeof createRevoluteJoint>[1]).joint;
}
function prismatic(world: WorldState, a: number, b: number, fields: Record<string, unknown> = {}) {
    const [base, specific] = splitJointFields(fields);
    return createPrismaticJoint(world, {
        ...defaultPrismaticJointDef(baseDef(a, b, base)),
        ...specific,
    } as Parameters<typeof createPrismaticJoint>[1]).joint;
}
function spherical(world: WorldState, a: number, b: number, fields: Record<string, unknown> = {}) {
    const [base, specific] = splitJointFields(fields);
    return createSphericalJoint(world, {
        ...defaultSphericalJointDef(baseDef(a, b, base)),
        ...specific,
    } as Parameters<typeof createSphericalJoint>[1]).joint;
}
function weld(world: WorldState, a: number, b: number, fields: Record<string, unknown> = {}) {
    const [base, specific] = splitJointFields(fields);
    return createWeldJoint(world, {
        ...defaultWeldJointDef(baseDef(a, b, base)),
        ...specific,
    } as Parameters<typeof createWeldJoint>[1]).joint;
}
function parallel(world: WorldState, a: number, b: number, fields: Record<string, unknown> = {}) {
    const [base, specific] = splitJointFields(fields);
    return createParallelJoint(world, {
        ...defaultParallelJointDef(baseDef(a, b, base)),
        ...specific,
    } as Parameters<typeof createParallelJoint>[1]).joint;
}
function motor(world: WorldState, a: number, b: number, fields: Record<string, unknown> = {}) {
    const [base, specific] = splitJointFields(fields);
    return createMotorJoint(world, {
        ...defaultMotorJointDef(baseDef(a, b, base)),
        ...specific,
    } as Parameters<typeof createMotorJoint>[1]).joint;
}
function wheel(world: WorldState, a: number, b: number, fields: Record<string, unknown> = {}) {
    const [base, specific] = splitJointFields(fields);
    return createWheelJoint(world, {
        ...defaultWheelJointDef(baseDef(a, b, base)),
        ...specific,
    } as Parameters<typeof createWheelJoint>[1]).joint;
}
function distance(world: WorldState, a: number, b: number, fields: Record<string, unknown> = {}) {
    const [base, specific] = splitJointFields(fields);
    return createDistanceJoint(world, {
        ...defaultDistanceJointDef(baseDef(a, b, base)),
        ...specific,
    } as Parameters<typeof createDistanceJoint>[1]).joint;
}
function position(world: WorldState, body: number) {
    return readBodyTransform(world, body, identityPose()).p;
}
function angularVelocity(world: WorldState, body: number): Vec3 {
    const state = getBodyState(world, body);
    return state === null ? vector.zero() : readStateAngularVelocity(world, state, vector.zero());
}
function rig(def: Partial<WorldDef> = {}) {
    const world = createTestWorld({ gravity: vector.zero(), ...def });
    const anchor = createTestBody(world);
    const body = createTestBody(world, { type: BodyType.Dynamic, position: { x: 0, y: 4, z: 0 } });
    addHull(world, body, makeCubeHull(0.5));
    return { world, anchor, body };
}

// A recycled numeric slot is not a live joint identity unless its generation also matches.
test("joint generations wrap at sixteen bits in the kernel slot", () => {
    const { world, anchor, body } = rig();
    try {
        const first = createFilterJoint(world, baseDef(anchor, body)).joint;
        setJointField(world, first, JointField.generation, 0xfffe);
        const stale = { index1: first + 1, world0: world.worldId, generation: 0xfffe };
        destroyJointInternal(world, first, true);
        const last = createFilterJoint(world, baseDef(anchor, body)).joint;
        expect(last).toBe(first);
        expect(jointField(world, last, JointField.generation)).toBe(0xffff);
        expect(jointIsValid(world, stale)).toBe(false);
        destroyJointInternal(world, last, true);
        const wrapped = createFilterJoint(world, baseDef(anchor, body)).joint;
        expect(wrapped).toBe(first);
        expect(jointField(world, wrapped, JointField.generation)).toBe(0);
        destroyJointInternal(world, wrapped, true);
        expect(
            jointField(
                world,
                createFilterJoint(world, baseDef(anchor, body)).joint,
                JointField.generation,
            ),
        ).toBe(1);
    } finally {
        destroyTestWorld(world);
    }
});

test("prismatic speed uses f32 anchors, center offsets and rotating-axis terms", () => {
    const world = createTestWorld({ gravity: vector.zero() });
    try {
        const a = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 1.2, y: 2.3, z: 3.4 },
            rotation: quaternion.fromAxisAngle({ x: 0, y: 1, z: 0 }, 0.4),
            linearVelocity: { x: 0.1, y: 0.2, z: 0.3 },
            angularVelocity: { x: 0.4, y: 0.5, z: 0.6 },
        });
        const b = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 4.5, y: 5.6, z: 6.7 },
            rotation: quaternion.fromAxisAngle({ x: 1, y: 0, z: 0 }, 0.7),
            linearVelocity: { x: 0.7, y: 0.8, z: 0.9 },
            angularVelocity: { x: 1.1, y: 1.2, z: 1.3 },
        });
        createSphereShape(world, a, testShapeDef(), {
            center: { x: 0.2, y: 0.3, z: 0.4 },
            radius: 0.5,
        });
        createSphereShape(world, b, testShapeDef(), {
            center: { x: 0.3, y: 0.4, z: 0.5 },
            radius: 0.5,
        });
        const localFrameA = froundConfig({
            p: { x: 0.6, y: 0.7, z: 0.8 },
            q: quaternion.fromAxisAngle({ x: 0, y: 0, z: 1 }, 0.3),
        });
        const localFrameB = froundConfig(frame(0.9, 1.1, 1.2));
        const joint = prismatic(world, a, b, { localFrameA, localFrameB });
        const poseA = readBodyTransform(world, a, identityPose());
        const poseB = readBodyTransform(world, b, identityPose());
        const centerA = getMassData(world, a).center;
        const centerB = getMassData(world, b).center;
        const axis = quaternion.rotate(poseA.q, quaternion.rotate(localFrameA.q, vector.axisX()));
        const rA = quaternion.rotate(poseA.q, vector.sub(localFrameA.p, centerA));
        const rB = quaternion.rotate(poseB.q, vector.sub(localFrameB.p, centerB));
        const centerAWorld = readSimCenter(world, getBodySim(world, a), vector.zero());
        const centerBWorld = readSimCenter(world, getBodySim(world, b), vector.zero());
        const d = vector.add(vector.sub(centerBWorld, centerAWorld), vector.sub(rB, rA));
        const wA = angularVelocity(world, a);
        const wB = angularVelocity(world, b);
        const vA = bodyGetLinearVelocity(world, a, vector.zero());
        const vB = bodyGetLinearVelocity(world, b, vector.zero());
        const vRel = vector.sub(
            vector.add(vB, vector.cross(wB, rB)),
            vector.add(vA, vector.cross(wA, rA)),
        );
        const expected = f32(vector.dot(d, vector.cross(wA, axis)) + vector.dot(axis, vRel));
        expect(prismaticJointSpeed(world, joint)).toBe(expected);
    } finally {
        destroyTestWorld(world);
    }
});

test("joint filtering suppresses a touching pair only when collideConnected is false", () => {
    function contacts(collide: boolean): number {
        const world = createTestWorld({ gravity: vector.zero(), enableContinuous: false });
        try {
            const a = createTestBody(world, {
                type: BodyType.Dynamic,
                position: { x: 0, y: 5, z: 0 },
            });
            const b = createTestBody(world, {
                type: BodyType.Dynamic,
                position: { x: 1, y: 5, z: 0 },
            });
            addHull(world, a, makeBoxHull(0.5, 0.5, 0.5));
            addHull(world, b, makeBoxHull(0.5, 0.5, 0.5));
            revolute(world, a, b, {
                collideConnected: collide,
                localFrameA: frame(0.5),
                localFrameB: frame(-0.5),
            });
            stepTestWorld(world, 1 / 60, 4);
            return worldCounters(world).contactCount;
        } finally {
            destroyTestWorld(world);
        }
    }
    expect(contacts(false)).toBe(0);
    expect(contacts(true)).toBe(1);
});

test("revolute joint holds a pendulum pivot and carries load", () => {
    const world = createTestWorld({ gravity: { x: 0, y: -10, z: 0 }, enableContinuous: false });
    try {
        const anchor = createTestBody(world, { position: { x: 0, y: 5, z: 0 } });
        const arm = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 1, y: 5, z: 0 },
        });
        addHull(world, arm, makeBoxHull(1, 0.2, 0.2));
        const joint = revolute(world, anchor, arm, {
            localFrameA: frame(),
            localFrameB: frame(-1),
        });
        for (let i = 0; i < 60; ++i) stepTestWorld(world, 1 / 60, 4);
        const p = position(world, arm);
        const radius = len({ x: p.x, y: p.y - 5, z: p.z });
        expect(radius).toBeGreaterThan(0.9);
        expect(radius).toBeLessThan(1.1);
        expect(len(getJointConstraintForce(world, joint))).toBeGreaterThan(0);
    } finally {
        destroyTestWorld(world);
    }
});

test("weld joint keeps its body at the authored pose and carries load", () => {
    const world = createTestWorld({ gravity: { x: 0, y: -10, z: 0 }, enableContinuous: false });
    try {
        const anchor = createTestBody(world, { position: { x: 0, y: 5, z: 0 } });
        const box = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 0, y: 4, z: 0 },
        });
        addHull(world, box, makeBoxHull(0.25, 0.25, 0.25));
        const joint = weld(world, anchor, box, { localFrameA: frame(0, -1), localFrameB: frame() });
        for (let i = 0; i < 90; ++i) stepTestWorld(world, 1 / 60, 4);
        expect(
            len({
                x: position(world, box).x,
                y: position(world, box).y - 4,
                z: position(world, box).z,
            }),
        ).toBeLessThan(0.05);
        expect(len(getJointConstraintForce(world, joint))).toBeGreaterThan(0);
    } finally {
        destroyTestWorld(world);
    }
});

test("parallel joint resists off-axis spin with constraint torque", () => {
    const world = createTestWorld({ gravity: { x: 0, y: -10, z: 0 }, enableContinuous: false });
    try {
        const a = createTestBody(world, { type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
        const b = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 1, y: 5, z: 0 },
            angularVelocity: { x: 3, y: 2, z: 0 },
        });
        addHull(world, a, makeBoxHull(0.2, 0.2, 0.2));
        addHull(world, b, makeBoxHull(0.2, 0.2, 0.2));
        const joint = parallel(world, a, b, { maxTorque: 5 });
        for (let i = 0; i < 30; ++i) stepTestWorld(world, 1 / 60, 4);
        expect(jointField(world, joint, JointField.type)).toBe(JointType.Parallel);
        expect(len(getJointConstraintTorque(world, joint))).toBeGreaterThan(0);
    } finally {
        destroyTestWorld(world);
    }
});

test("motor joint target speed controls which way its body spins", () => {
    function spin(targetZ: number): number {
        const world = createTestWorld({ gravity: { x: 0, y: -10, z: 0 }, enableContinuous: false });
        try {
            const anchor = createTestBody(world, { position: { x: 0, y: 5, z: 0 } });
            const arm = createTestBody(world, {
                type: BodyType.Dynamic,
                position: { x: 1, y: 5, z: 0 },
            });
            addHull(world, arm, makeBoxHull(1, 0.2, 0.2));
            motor(world, anchor, arm, {
                localFrameA: frame(),
                localFrameB: frame(-1),
                maxVelocityForce: 1000,
                angularVelocity: { x: 0, y: 0, z: targetZ },
                maxVelocityTorque: 500,
            });
            for (let i = 0; i < 60; ++i) stepTestWorld(world, 1 / 60, 4);
            return angularVelocity(world, arm).z;
        } finally {
            destroyTestWorld(world);
        }
    }
    expect(spin(5)).toBeGreaterThan(spin(-5));
});

test("prismatic joint motor reaches its translation limit without falling off the axis", () => {
    const world = createTestWorld({ gravity: { x: 0, y: -10, z: 0 }, enableContinuous: false });
    try {
        const anchor = createTestBody(world, { position: { x: 0, y: 5, z: 0 } });
        const slider = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 0, y: 5, z: 0 },
        });
        addHull(world, slider, makeBoxHull(0.25, 0.25, 0.25));
        const joint = prismatic(world, anchor, slider, {
            enableMotor: true,
            motorSpeed: 3,
            maxMotorForce: 1000,
            enableLimit: true,
            lowerTranslation: -2,
            upperTranslation: 2,
        });
        for (let i = 0; i < 180; ++i) stepTestWorld(world, 1 / 60, 4);
        const p = position(world, slider);
        expect(p.x).toBeGreaterThan(1.8);
        expect(p.x).toBeLessThan(2.1);
        expect(Math.abs(p.y - 5)).toBeLessThan(0.1);
        expect(len(getJointConstraintForce(world, joint))).toBeGreaterThan(0);
    } finally {
        destroyTestWorld(world);
    }
});

test("spherical joint holds its pivot while its arm swings", () => {
    const world = createTestWorld({ gravity: { x: 0, y: -10, z: 0 }, enableContinuous: false });
    try {
        const anchor = createTestBody(world, { position: { x: 0, y: 5, z: 0 } });
        const arm = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 1, y: 5, z: 0 },
        });
        addHull(world, arm, makeBoxHull(1, 0.2, 0.2));
        const joint = spherical(world, anchor, arm, { localFrameB: frame(-1) });
        for (let i = 0; i < 90; ++i) stepTestWorld(world, 1 / 60, 4);
        const p = position(world, arm);
        const radius = len({ x: p.x, y: p.y - 5, z: p.z });
        expect(radius).toBeGreaterThan(0.9);
        expect(radius).toBeLessThan(1.1);
        expect(len(getJointConstraintForce(world, joint))).toBeGreaterThan(0);
    } finally {
        destroyTestWorld(world);
    }
});

test("wheel joint spin motor follows the sign of its target speed", () => {
    function spin(speed: number): number {
        const world = createTestWorld({ gravity: { x: 0, y: -10, z: 0 }, enableContinuous: false });
        try {
            const chassis = createTestBody(world, { position: { x: 0, y: 5, z: 0 } });
            const body = createTestBody(world, {
                type: BodyType.Dynamic,
                position: { x: 0, y: 5, z: 0 },
            });
            addHull(world, body, makeBoxHull(0.3, 0.3, 0.3));
            wheel(world, chassis, body, {
                enableSpinMotor: true,
                spinSpeed: speed,
                maxSpinTorque: 50,
            });
            for (let i = 0; i < 60; ++i) stepTestWorld(world, 1 / 60, 4);
            return angularVelocity(world, body).z;
        } finally {
            destroyTestWorld(world);
        }
    }
    expect(spin(10)).toBeGreaterThan(spin(-10));
});

test("filter joint suppresses contact without adding a solver constraint", () => {
    function contacts(filtered: boolean): number {
        const world = createTestWorld({ gravity: vector.zero(), enableContinuous: false });
        try {
            const a = createTestBody(world, {
                type: BodyType.Dynamic,
                position: { x: 0, y: 5, z: 0 },
            });
            const b = createTestBody(world, {
                type: BodyType.Dynamic,
                position: { x: 1, y: 5, z: 0 },
            });
            addHull(world, a, makeBoxHull(0.5, 0.5, 0.5));
            addHull(world, b, makeBoxHull(0.5, 0.5, 0.5));
            if (filtered) {
                const joint = createFilterJoint(world, baseDef(a, b)).joint;
                expect(jointField(world, joint, JointField.type)).toBe(JointType.Filter);
            }
            stepTestWorld(world, 1 / 60, 4);
            return worldCounters(world).contactCount;
        } finally {
            destroyTestWorld(world);
        }
    }
    expect(contacts(true)).toBe(0);
    expect(contacts(false)).toBe(1);
});

test("mixed-joint island sleeps as one unit", () => {
    const world = createTestWorld({ gravity: { x: 0, y: -10, z: 0 }, enableContinuous: false });
    try {
        const ground = createTestBody(world, { position: { x: 0, y: -1, z: 0 } });
        addHull(world, ground, makeBoxHull(20, 1, 20));
        const torso = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 0, y: 3, z: 0 },
            angularDamping: 0.5,
            linearDamping: 0.5,
        });
        addHull(world, torso, makeBoxHull(0.25, 0.5, 0.25));
        const limb = (x: number, y: number, box: [number, number, number]) => {
            const body = createTestBody(world, {
                type: BodyType.Dynamic,
                position: { x, y, z: 0 },
                angularDamping: 0.5,
                linearDamping: 0.5,
            });
            addHull(world, body, makeBoxHull(...box));
            return body;
        };
        const armL = limb(-0.75, 3.5, [0.5, 0.125, 0.125]);
        const armR = limb(0.75, 3.5, [0.5, 0.125, 0.125]);
        const legL = limb(-0.25, 2, [0.125, 0.5, 0.125]);
        const legR = limb(0.25, 2, [0.125, 0.5, 0.125]);
        spherical(world, torso, armL, { localFrameA: frame(-0.25, 0.5), localFrameB: frame(0.5) });
        spherical(world, torso, armR, { localFrameA: frame(0.25, 0.5), localFrameB: frame(-0.5) });
        revolute(world, torso, legL, {
            localFrameA: frame(-0.25, -0.5),
            localFrameB: frame(0, 0.5),
        });
        revolute(world, torso, legR, {
            localFrameA: frame(0.25, -0.5),
            localFrameB: frame(0, 0.5),
        });
        for (let i = 0; i < 300; ++i) stepTestWorld(world, 1 / 60, 4);
        expect(bodyIsAwake(world, torso)).toBe(false);
        expect(bodyIsAwake(world, armL)).toBe(false);
        expect(bodyIsAwake(world, legR)).toBe(false);
    } finally {
        destroyTestWorld(world);
    }
});

test("distance joint holds its authored length under gravity", () => {
    const world = createTestWorld({ gravity: { x: 0, y: -10, z: 0 }, enableContinuous: false });
    try {
        const anchor = createTestBody(world, { position: { x: 0, y: 5, z: 0 } });
        const ball = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 2, y: 5, z: 0 },
        });
        addHull(world, ball, makeBoxHull(0.2, 0.2, 0.2));
        const joint = distance(world, anchor, ball, { length: 2 });
        for (let i = 0; i < 120; ++i) stepTestWorld(world, 1 / 60, 4);
        const p = position(world, ball);
        const d = len({ x: p.x, y: p.y - 5, z: p.z });
        expect(d).toBeGreaterThan(1.9);
        expect(d).toBeLessThan(2.1);
        expect(len(getJointConstraintForce(world, joint))).toBeGreaterThan(0);
    } finally {
        destroyTestWorld(world);
    }
});

test("all kernel joint constructors retain their declared kind", () => {
    const { world, anchor, body } = rig();
    try {
        const joints = [
            parallel(world, anchor, body),
            distance(world, anchor, body),
            createFilterJoint(world, baseDef(anchor, body)).joint,
            motor(world, anchor, body),
            prismatic(world, anchor, body),
            revolute(world, anchor, body),
            spherical(world, anchor, body),
            weld(world, anchor, body),
            wheel(world, anchor, body),
        ];
        expect(joints.map((joint) => jointField(world, joint, JointField.type))).toEqual([
            JointType.Parallel,
            JointType.Distance,
            JointType.Filter,
            JointType.Motor,
            JointType.Prismatic,
            JointType.Revolute,
            JointType.Spherical,
            JointType.Weld,
            JointType.Wheel,
        ]);
    } finally {
        destroyTestWorld(world);
    }
});
