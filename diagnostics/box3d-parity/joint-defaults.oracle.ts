import { expect, setDefaultTimeout, test } from "bun:test";
import * as C from "../../src/standard/physics/kernel/columns";
import { J_CONSTRAINT_SOFTNESS } from "../../src/standard/physics/kernel/joint-layout";
import {
    readJointEventUserData,
    readJointFlag,
    readJointFloat,
    readJointQuat,
    readJointTransform,
    readJointVec3,
    writeJointFloat,
    writeJointQuat,
    writeJointTransform,
    writeJointVec3,
} from "../../src/standard/physics/kernel/jointcolumns";
import { JointField, jointDrawScale, jointField } from "../../src/standard/physics/kernel/jointrecords";
import { kernel, init } from "../../src/standard/physics/kernel/kernel";
import { BodyType } from "../../src/standard/physics/common/types";
import { createTestBody, createTestWorld, destroyTestWorld } from "../../src/standard/physics/testing";
import { createFilterJoint, defaultJointDef, destroyJointInternal } from "../../src/standard/physics/solver/joint";
import { createDistanceJoint, defaultDistanceJointDef } from "../../src/standard/physics/solver/distanceJoint";
import { createMotorJoint, defaultMotorJointDef } from "../../src/standard/physics/solver/motorJoint";
import { createParallelJoint, defaultParallelJointDef } from "../../src/standard/physics/solver/parallelJoint";
import { createPrismaticJoint, defaultPrismaticJointDef } from "../../src/standard/physics/solver/prismaticJoint";
import { createRevoluteJoint, defaultRevoluteJointDef } from "../../src/standard/physics/solver/revoluteJoint";
import { createSphericalJoint, defaultSphericalJointDef } from "../../src/standard/physics/solver/sphericalJoint";
import { createWeldJoint, defaultWeldJointDef } from "../../src/standard/physics/solver/weldJoint";
import { createWheelJoint, defaultWheelJointDef } from "../../src/standard/physics/solver/wheelJoint";
import { wakeJointBodies } from "../../src/standard/physics/solver/joint";
import { nativeSseOutput } from "./native-evidence";
import { assertPublicOracleKernel } from "./oracle-kernel";
import type { WorldState } from "../../src/standard/physics/world/world";

setDefaultTimeout(180_000);
await init(undefined, { threads: 0 });
await assertPublicOracleKernel();

const kinds = ["parallel", "distance", "filter", "motor", "prismatic", "revolute", "spherical", "weld", "wheel"] as const;
type Kind = (typeof kinds)[number];
type Values = { s: number[]; p: number[]; flags: number };

function generated(kind: number, sample: number): Values {
    let state = (0x47d7f7cc ^ Math.imul(kind, 0x9e3779b9) ^ Math.imul(sample, 0x85ebca6b)) >>> 0;
    const next = () => {
        state ^= state << 13;
        state ^= state >>> 17;
        state ^= state << 5;
        return state >>> 0;
    };
    const s = Array.from({ length: 16 }, () => Math.fround(((next() & 2047) - 1024) / 64));
    const p = Array.from({ length: 16 }, () => Math.fround(((next() & 1023) + 1) / 64));
    return { s, p, flags: next() };
}

const ordered = (a: number, b: number) => (a < b ? [a, b] : [b, a]) as [number, number];
const vec = (a: number[]) => ({ x: a[0]!, y: a[1]!, z: a[2]! });
const quat = (axis: "x" | "y" | "z" | "identity") => ({
    v: { x: axis === "x" ? 1 : 0, y: axis === "y" ? 1 : 0, z: axis === "z" ? 1 : 0 },
    s: axis === "identity" ? 1 : 0,
});

function jointBase(bodyIdA: number, bodyIdB: number) {
    return { ...defaultJointDef(), bodyIdA, bodyIdB };
}

function create(kind: Kind, world: WorldState, a: number, b: number): number {
    const base = jointBase(a, b);
    switch (kind) {
        case "parallel":
            return createParallelJoint(world, defaultParallelJointDef(base)).joint;
        case "distance":
            return createDistanceJoint(world, defaultDistanceJointDef(base)).joint;
        case "filter":
            return createFilterJoint(world, base).joint;
        case "motor":
            return createMotorJoint(world, defaultMotorJointDef(base)).joint;
        case "prismatic":
            return createPrismaticJoint(world, defaultPrismaticJointDef(base)).joint;
        case "revolute":
            return createRevoluteJoint(world, defaultRevoluteJointDef(base)).joint;
        case "spherical":
            return createSphericalJoint(world, defaultSphericalJointDef(base)).joint;
        case "weld":
            return createWeldJoint(world, defaultWeldJointDef(base)).joint;
        case "wheel":
            return createWheelJoint(world, defaultWheelJointDef(base)).joint;
    }
}

function setValues(kind: Kind, world: WorldState, joint: number, v: Values): void {
    const k = kernel(world.ecsState);
    const set = (field: number, value: number) => writeJointFloat(world, joint, field, value);
    const enable = (field: number, value: boolean) => k.jointEnable(world.worldId, joint, field, value);
    writeJointTransform(world, joint, C.J_LOCAL_FRAME_A, {
        p: vec(v.s.slice(0, 3)),
        q: quat(v.flags & 2 ? "z" : "identity"),
    });
    writeJointTransform(world, joint, C.J_LOCAL_FRAME_B, {
        p: vec(v.s.slice(3, 6)),
        q: quat(v.flags & 4 ? "x" : "identity"),
    });
    k.jointSetCollideConnected(world.worldId, joint, !!(v.flags & 1));
    set(C.J_CONSTRAINT_HERTZ, v.p[0]!);
    set(C.J_CONSTRAINT_DAMPING, v.p[1]!);
    set(C.J_FORCE_THRESHOLD, v.p[2]!);
    set(C.J_TORQUE_THRESHOLD, v.p[3]!);
    readJointEventUserData(world);
    world.jointUserData[joint] = 1;

    switch (kind) {
        case "parallel":
            set(C.PLJ_HERTZ, v.p[4]!);
            set(C.PLJ_DAMPING_RATIO, v.p[5]!);
            set(C.PLJ_MAX_TORQUE, v.p[6]!);
            break;
        case "distance":
            k.distanceJointSetLength(world.worldId, joint, v.p[4]!);
            writeJointFloat(world, joint, C.DJ_LOWER_SPRING_FORCE, Math.min(v.s[6]!, v.s[7]!));
            writeJointFloat(world, joint, C.DJ_UPPER_SPRING_FORCE, Math.max(v.s[6]!, v.s[7]!));
            enable(C.DJ_ENABLE_SPRING, !!(v.flags & 8));
            set(C.DJ_HERTZ, v.p[5]!);
            set(C.DJ_DAMPING_RATIO, v.p[6]!);
            enable(C.DJ_ENABLE_LIMIT, !!(v.flags & 16));
            k.jointSetLimits(world.worldId, joint, ...ordered(v.p[7]!, v.p[8]!));
            enable(C.DJ_ENABLE_MOTOR, !!(v.flags & 32));
            writeJointFloat(world, joint, C.DJ_MOTOR_SPEED, v.s[8]!);
            wakeJointBodies(world, joint);
            set(C.DJ_MAX_MOTOR_FORCE, v.p[9]!);
            break;
        case "filter":
            break;
        case "motor":
            writeJointVec3(world, joint, C.MJ_LINEAR_VELOCITY, vec(v.s.slice(6, 9)));
            wakeJointBodies(world, joint);
            writeJointVec3(world, joint, C.MJ_ANGULAR_VELOCITY, vec(v.s.slice(9, 12)));
            wakeJointBodies(world, joint);
            set(C.MJ_MAX_VELOCITY_FORCE, v.p[4]!);
            set(C.MJ_MAX_VELOCITY_TORQUE, v.p[5]!);
            set(C.MJ_LINEAR_HERTZ, v.p[6]!);
            set(C.MJ_LINEAR_DAMPING_RATIO, v.p[7]!);
            set(C.MJ_MAX_SPRING_FORCE, v.p[8]!);
            set(C.MJ_ANGULAR_HERTZ, v.p[9]!);
            set(C.MJ_ANGULAR_DAMPING_RATIO, v.p[10]!);
            set(C.MJ_MAX_SPRING_TORQUE, v.p[11]!);
            break;
        case "prismatic":
            enable(C.PJ_ENABLE_SPRING, !!(v.flags & 8));
            set(C.PJ_HERTZ, v.p[4]!);
            set(C.PJ_DAMPING_RATIO, v.p[5]!);
            set(C.PJ_TARGET_TRANSLATION, v.s[6]!);
            enable(C.PJ_ENABLE_LIMIT, !!(v.flags & 16));
            k.jointSetLimits(world.worldId, joint, ...ordered(v.s[7]!, v.s[8]!));
            enable(C.PJ_ENABLE_MOTOR, !!(v.flags & 32));
            set(C.PJ_MOTOR_SPEED, v.s[9]!);
            wakeJointBodies(world, joint);
            set(C.PJ_MAX_MOTOR_FORCE, v.p[6]!);
            break;
        case "revolute":
            enable(C.RJ_ENABLE_SPRING, !!(v.flags & 8));
            set(C.RJ_HERTZ, v.p[4]!);
            set(C.RJ_DAMPING_RATIO, v.p[5]!);
            set(C.RJ_TARGET_ANGLE, v.s[6]! / 8);
            enable(C.RJ_ENABLE_LIMIT, !!(v.flags & 16));
            k.jointSetLimits(world.worldId, joint, ...ordered(v.s[7]! / 8, v.s[8]! / 8));
            enable(C.RJ_ENABLE_MOTOR, !!(v.flags & 32));
            set(C.RJ_MOTOR_SPEED, v.s[9]!);
            wakeJointBodies(world, joint);
            set(C.RJ_MAX_MOTOR_TORQUE, v.p[6]!);
            break;
        case "spherical":
            enable(C.SJ_ENABLE_CONE_LIMIT, !!(v.flags & 8));
            set(C.SJ_CONE_ANGLE, v.p[4]! / 8);
            enable(C.SJ_ENABLE_TWIST_LIMIT, !!(v.flags & 16));
            k.jointSetLimits(world.worldId, joint, ...ordered(v.s[6]! / 8, v.s[7]! / 8));
            enable(C.SJ_ENABLE_SPRING, !!(v.flags & 32));
            writeJointQuat(world, joint, C.SJ_TARGET_ROTATION, quat(v.flags & 64 ? "y" : "identity"));
            set(C.SJ_HERTZ, v.p[5]!);
            set(C.SJ_DAMPING_RATIO, v.p[6]!);
            enable(C.SJ_ENABLE_MOTOR, !!(v.flags & 128));
            writeJointVec3(world, joint, C.SJ_MOTOR_VELOCITY, vec(v.s.slice(8, 11)));
            wakeJointBodies(world, joint);
            set(C.SJ_MAX_MOTOR_TORQUE, v.p[7]!);
            break;
        case "weld":
            set(C.WJ_LINEAR_HERTZ, v.p[4]!);
            set(C.WJ_LINEAR_DAMPING_RATIO, v.p[5]!);
            set(C.WJ_ANGULAR_HERTZ, v.p[6]!);
            set(C.WJ_ANGULAR_DAMPING_RATIO, v.p[7]!);
            break;
        case "wheel":
            enable(C.WHJ_ENABLE_SUSPENSION_SPRING, !!(v.flags & 8));
            set(C.WHJ_SUSPENSION_HERTZ, v.p[4]!);
            set(C.WHJ_SUSPENSION_DAMPING_RATIO, v.p[5]!);
            enable(C.WHJ_ENABLE_SUSPENSION_LIMIT, !!(v.flags & 16));
            k.jointSetLimits(world.worldId, joint, ...ordered(v.s[6]!, v.s[7]!));
            enable(C.WHJ_ENABLE_SPIN_MOTOR, !!(v.flags & 32));
            writeJointFloat(world, joint, C.WHJ_SPIN_SPEED, v.s[8]!);
            wakeJointBodies(world, joint);
            set(C.WHJ_MAX_SPIN_TORQUE, v.p[6]!);
            enable(C.WHJ_ENABLE_STEERING, !!(v.flags & 64));
            set(C.WHJ_STEERING_HERTZ, v.p[7]!);
            set(C.WHJ_STEERING_DAMPING_RATIO, v.p[8]!);
            set(C.WHJ_MAX_STEERING_TORQUE, v.p[9]!);
            enable(C.WHJ_ENABLE_STEERING_LIMIT, !!(v.flags & 128));
            writeJointFloat(world, joint, C.WHJ_LOWER_STEERING_LIMIT, Math.min(v.s[9]! / 8, v.s[10]! / 8));
            writeJointFloat(world, joint, C.WHJ_UPPER_STEERING_LIMIT, Math.max(v.s[9]! / 8, v.s[10]! / 8));
            writeJointFloat(world, joint, C.WHJ_TARGET_STEERING_ANGLE, v.s[11]! / 8);
            wakeJointBodies(world, joint);
            break;
    }
}

function flatten(value: unknown, out: number[] = []): number[] {
    if (typeof value === "number") out.push(Math.fround(value));
    else if (typeof value === "boolean") out.push(value ? 1 : 0);
    else if (Array.isArray(value)) for (const item of value) flatten(item, out);
    else if (value && typeof value === "object")
        for (const item of Object.values(value)) flatten(item, out);
    return out;
}

function dump(kind: Kind, world: WorldState, joint: number): number[] {
    const value: unknown[] = [
        readJointTransform(world, joint, C.J_LOCAL_FRAME_A),
        readJointTransform(world, joint, C.J_LOCAL_FRAME_B),
        !!jointField(world, joint, JointField.collideConnected),
        {
            hertz: readJointFloat(world, joint, C.J_CONSTRAINT_HERTZ),
            dampingRatio: readJointFloat(world, joint, C.J_CONSTRAINT_DAMPING),
        },
        readJointFloat(world, joint, C.J_FORCE_THRESHOLD),
        readJointFloat(world, joint, C.J_TORQUE_THRESHOLD),
        jointField(world, joint, JointField.type),
        world.jointUserData[joint] == null ? 0 : 1,
        jointDrawScale(world, joint),
    ];
    const flag = (layout: number, bit: number) => readJointFlag(world, joint, layout, bit);
    const read = (field: number) => readJointFloat(world, joint, field);
    switch (kind) {
        case "parallel":
            value.push(read(C.PLJ_HERTZ), read(C.PLJ_DAMPING_RATIO), read(C.PLJ_MAX_TORQUE));
            break;
        case "distance":
            value.push(
                read(C.DJ_LENGTH),
                flag(C.DJ_ENABLE, C.DJ_ENABLE_SPRING),
                read(C.DJ_LOWER_SPRING_FORCE),
                read(C.DJ_UPPER_SPRING_FORCE),
                read(C.DJ_HERTZ),
                read(C.DJ_DAMPING_RATIO),
                flag(C.DJ_ENABLE, C.DJ_ENABLE_LIMIT),
                read(C.DJ_MIN_LENGTH),
                read(C.DJ_MAX_LENGTH),
                flag(C.DJ_ENABLE, C.DJ_ENABLE_MOTOR),
                read(C.DJ_MAX_MOTOR_FORCE),
                read(C.DJ_MOTOR_SPEED),
            );
            break;
        case "filter":
            break;
        case "motor":
            value.push(
                readJointVec3(world, joint, C.MJ_LINEAR_VELOCITY),
                read(C.MJ_MAX_VELOCITY_FORCE),
                readJointVec3(world, joint, C.MJ_ANGULAR_VELOCITY),
                read(C.MJ_MAX_VELOCITY_TORQUE),
                read(C.MJ_LINEAR_HERTZ),
                read(C.MJ_LINEAR_DAMPING_RATIO),
                read(C.MJ_MAX_SPRING_FORCE),
                read(C.MJ_ANGULAR_HERTZ),
                read(C.MJ_ANGULAR_DAMPING_RATIO),
                read(C.MJ_MAX_SPRING_TORQUE),
            );
            break;
        case "prismatic":
            value.push(
                flag(C.PJ_ENABLE, C.PJ_ENABLE_SPRING),
                read(C.PJ_HERTZ),
                read(C.PJ_DAMPING_RATIO),
                read(C.PJ_TARGET_TRANSLATION),
                flag(C.PJ_ENABLE, C.PJ_ENABLE_LIMIT),
                read(C.PJ_LOWER_TRANSLATION),
                read(C.PJ_UPPER_TRANSLATION),
                flag(C.PJ_ENABLE, C.PJ_ENABLE_MOTOR),
                read(C.PJ_MAX_MOTOR_FORCE),
                read(C.PJ_MOTOR_SPEED),
            );
            break;
        case "revolute":
            value.push(
                flag(C.RJ_ENABLE, C.RJ_ENABLE_SPRING),
                read(C.RJ_HERTZ),
                read(C.RJ_DAMPING_RATIO),
                read(C.RJ_TARGET_ANGLE),
                flag(C.RJ_ENABLE, C.RJ_ENABLE_LIMIT),
                read(C.RJ_LOWER_ANGLE),
                read(C.RJ_UPPER_ANGLE),
                flag(C.RJ_ENABLE, C.RJ_ENABLE_MOTOR),
                read(C.RJ_MAX_MOTOR_TORQUE),
                read(C.RJ_MOTOR_SPEED),
            );
            break;
        case "spherical":
            value.push(
                flag(C.SJ_ENABLE, C.SJ_ENABLE_CONE_LIMIT),
                read(C.SJ_CONE_ANGLE),
                flag(C.SJ_ENABLE, C.SJ_ENABLE_TWIST_LIMIT),
                read(C.SJ_LOWER_TWIST_ANGLE),
                read(C.SJ_UPPER_TWIST_ANGLE),
                flag(C.SJ_ENABLE, C.SJ_ENABLE_SPRING),
                readJointQuat(world, joint, C.SJ_TARGET_ROTATION),
                read(C.SJ_HERTZ),
                read(C.SJ_DAMPING_RATIO),
                flag(C.SJ_ENABLE, C.SJ_ENABLE_MOTOR),
                readJointVec3(world, joint, C.SJ_MOTOR_VELOCITY),
                read(C.SJ_MAX_MOTOR_TORQUE),
            );
            break;
        case "weld":
            value.push(
                read(C.WJ_LINEAR_HERTZ),
                read(C.WJ_LINEAR_DAMPING_RATIO),
                read(C.WJ_ANGULAR_HERTZ),
                read(C.WJ_ANGULAR_DAMPING_RATIO),
            );
            break;
        case "wheel":
            value.push(
                flag(C.WHJ_ENABLE, C.WHJ_ENABLE_SUSPENSION_SPRING),
                read(C.WHJ_SUSPENSION_HERTZ),
                read(C.WHJ_SUSPENSION_DAMPING_RATIO),
                flag(C.WHJ_ENABLE, C.WHJ_ENABLE_SUSPENSION_LIMIT),
                read(C.WHJ_LOWER_SUSPENSION_LIMIT),
                read(C.WHJ_UPPER_SUSPENSION_LIMIT),
                flag(C.WHJ_ENABLE, C.WHJ_ENABLE_SPIN_MOTOR),
                read(C.WHJ_SPIN_SPEED),
                read(C.WHJ_MAX_SPIN_TORQUE),
                flag(C.WHJ_ENABLE, C.WHJ_ENABLE_STEERING),
                read(C.WHJ_STEERING_HERTZ),
                read(C.WHJ_STEERING_DAMPING_RATIO),
                read(C.WHJ_MAX_STEERING_TORQUE),
                flag(C.WHJ_ENABLE, C.WHJ_ENABLE_STEERING_LIMIT),
                read(C.WHJ_LOWER_STEERING_LIMIT),
                read(C.WHJ_UPPER_STEERING_LIMIT),
                read(C.WHJ_TARGET_STEERING_ANGLE),
            );
            break;
    }
    value.push([
        readJointFloat(world, joint, J_CONSTRAINT_SOFTNESS),
        readJointFloat(world, joint, J_CONSTRAINT_SOFTNESS + 1),
        readJointFloat(world, joint, J_CONSTRAINT_SOFTNESS + 2),
    ]);
    return flatten(value);
}

const native = new Map<string, string[]>();
for (const line of nativeSseOutput("joint-defaults.c", "").trim().split("\n")) {
    const [kind, sample, ...words] = line.split(" ");
    native.set(`${kind}:${sample}`, words);
}

const bits = new Uint32Array(1);
const floats = new Float32Array(bits.buffer);
const hex = (value: number) => {
    floats[0] = value;
    return bits[0]!.toString(16).padStart(8, "0");
};
const rows: { kind: Kind; sample: number; actual: string[]; native: string[] }[] = [];
const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 }, workerCount: 1 });
try {
    const a = createTestBody(world, { type: BodyType.Dynamic });
    const b = createTestBody(world, { type: BodyType.Dynamic, position: { x: 2, y: 0, z: 0 } });
    for (const [kindIndex, kind] of kinds.entries()) {
        for (let sample = 0; sample <= 8; ++sample) {
            const joint = create(kind, world, a, b);
            if (sample !== 0) setValues(kind, world, joint, generated(kindIndex, sample));
            rows.push({
                kind,
                sample,
                actual: dump(kind, world, joint).map(hex),
                native: native.get(`${kindIndex}:${sample}`) ?? [],
            });
            destroyJointInternal(world, joint, false);
        }
    }
} finally {
    destroyTestWorld(world);
}

for (const kind of kinds) {
    test(`${kind} creation defaults equal b3Default*JointDef through direct state reads`, () => {
        const row = rows.find((entry) => entry.kind === kind && entry.sample === 0)!;
        expect(row.actual.slice(0, -3)).toEqual(row.native.slice(0, -3));
    });
    test(`${kind} seeded setter writes equal Box3D`, () => {
        const selected = rows.filter((entry) => entry.kind === kind && entry.sample !== 0);
        expect(selected.map((row) => row.actual.slice(0, -3))).toEqual(
            selected.map((row) => row.native.slice(0, -3)),
        );
    });
}

test("joint_lifecycle.rs:127-135 vs Box3D src/joint.c:309-313: creation and spring enable/disable preserve rigid default constraint softness [0, 1, 0]", () => {
    expect(rows.map((row) => row.actual.slice(-3))).toEqual(
        rows.map((row) => row.native.slice(-3)),
    );
});
