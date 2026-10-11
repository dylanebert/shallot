import type { ScalarField, Vector4Field, World } from "../../engine";
import type { Quat, Transform, Vec3 } from "./api";
import { makeJointId } from "./api/config";
import type { JointBinding, JointDef } from "./authoring";
import { JOINT_KIND_COUNT, jointDef } from "./authoring";
import type { EntityId } from "./common/ids";
import { froundConfig } from "./common/math";
import {
    clearFieldCandidates,
    createFieldCandidates,
    type FieldCandidates,
    markFieldCandidate,
} from "./field-candidates";
import * as C from "./kernel/columns";
import {
    DJ_LOWER_SPRING_FORCE,
    DJ_UPPER_SPRING_FORCE,
    J_CONSTRAINT_DAMPING,
    J_CONSTRAINT_HERTZ,
    J_LOCAL_FRAME_A,
    J_LOCAL_FRAME_B,
    MJ_ANGULAR_VELOCITY,
    MJ_LINEAR_VELOCITY,
    SJ_MOTOR_VELOCITY,
    SJ_TARGET_ROTATION,
} from "./kernel/columns";
import {
    readJointFlag,
    readJointFloat,
    readJointQuat,
    readJointVec3,
    writeJointFloat,
    writeJointQuat,
    writeJointTransform,
    writeJointVec3,
} from "./kernel/jointcolumns";
import {
    JointField,
    jointDrawScale,
    jointField,
    jointViews,
    setJointDrawScale,
} from "./kernel/jointrecords";
import { kernel } from "./kernel/kernel";
import { createDistanceJoint, defaultDistanceJointDef } from "./solver/distanceJoint";
import {
    createFilterJoint,
    defaultJointDef,
    destroyJointInternal,
    jointIsValid,
    setJointCollideConnected,
    wakeJointBodies,
} from "./solver/joint";
import { createMotorJoint, defaultMotorJointDef } from "./solver/motorJoint";
import { createParallelJoint, defaultParallelJointDef } from "./solver/parallelJoint";
import { createPrismaticJoint, defaultPrismaticJointDef } from "./solver/prismaticJoint";
import { createRevoluteJoint, defaultRevoluteJointDef } from "./solver/revoluteJoint";
import { createSphericalJoint, defaultSphericalJointDef } from "./solver/sphericalJoint";
import { createWeldJoint, defaultWeldJointDef } from "./solver/weldJoint";
import { createWheelJoint, defaultWheelJointDef } from "./solver/wheelJoint";
import type { WorldState } from "./world/world";

export interface ConstraintCache {
    liveJoints: Map<number, Map<number, EntityId>>;
    candidates: FieldCandidates[];
    warnedJoints: Set<number>;
}
export function createConstraintCache(): ConstraintCache {
    return {
        liveJoints: new Map(),
        candidates: Array.from({ length: JOINT_KIND_COUNT }, createFieldCandidates),
        warnedJoints: new Set(),
    };
}

function identity(index: number, eid: number): number {
    return eid * JOINT_KIND_COUNT + index;
}
function fieldChanged(binding: JointBinding, mask: number, name: string): boolean {
    const index = binding.fieldIndices.get(name);
    return index !== undefined && (mask & (1 << index)) !== 0;
}
export function markAllJointCandidates(
    world: World,
    cache: ConstraintCache,
    bindings: readonly JointBinding[],
): void {
    for (let i = 0; i < bindings.length; i++) {
        const binding = bindings[i];
        const candidates = cache.candidates[binding.index];
        for (const eid of world.query([binding.component])) markFieldCandidate(candidates, eid);
        const live = cache.liveJoints.get(binding.index);
        if (live) for (const eid of live.keys()) markFieldCandidate(candidates, eid);
    }
}
function warnOnce(warned: Set<number>, key: number, message: string): void {
    if (warned.has(key)) return;
    console.warn(message);
    warned.add(key);
}
function clearJointWarnings(warned: Set<number>, key: number): void {
    for (const warning of warned)
        if (warning === key || (warning < 0 && Math.floor((-warning - 1) / 32) === key))
            warned.delete(warning);
}
const WARNING_FIELD: Record<string, string> = {
    "constraint tuning": "constraintHertz",
    localFrameA: "localAnchorA",
    localFrameB: "localAnchorB",
    "lowerSpringForce/upperSpringForce": "lowerSpringForce",
    "lowerTranslation/upperTranslation": "lowerTranslation",
    "lowerAngle/upperAngle": "lowerAngle",
    "lowerTwistAngle/upperTwistAngle": "lowerTwistAngle",
    "lowerSuspensionLimit/upperSuspensionLimit": "lowerSuspensionLimit",
    "lowerSteeringLimit/upperSteeringLimit": "lowerSteeringLimit",
};
function fieldWarningKey(binding: JointBinding, key: number, field: string): number {
    const name = WARNING_FIELD[field] ?? field;
    return -(key * 32 + (binding.fieldIndices.get(name) ?? 0) + 1);
}
function clearFieldWarning(
    warned: Set<number>,
    binding: JointBinding,
    key: number,
    field: string,
): void {
    warned.delete(fieldWarningKey(binding, key, field));
}

function endpoints(
    bodies: ReadonlyMap<number, EntityId>,
    def: JointDef,
    isDeferred: (eid: number) => boolean,
    warned: Set<number>,
    key: number,
): [EntityId, EntityId] | null {
    const a = bodies.get(def.a);
    const b = bodies.get(def.b);
    if (a && b) return [a, b];
    const cause = (label: string, eid: number) =>
        isDeferred(eid)
            ? `${label}: ${eid} is a deferred body (marshal pending, will retry)`
            : `${label}: ${eid} is not a Body (skipped)`;
    const parts: string[] = [];
    if (!a) parts.push(cause("a", def.a));
    if (!b) parts.push(cause("b", def.b));
    warnOnce(
        warned,
        key,
        `[physics] ${def.kind}Joint ${def.eid} endpoint unavailable — ${parts.join("; ")}`,
    );
    return null;
}
const validQuat = (q: Quat): boolean => {
    if (![q.v.x, q.v.y, q.v.z, q.s].every(Number.isFinite)) return false;
    const f = Math.fround;
    const lengthSquared = f(
        f(f(f(q.v.x * q.v.x) + f(q.v.y * q.v.y)) + f(q.v.z * q.v.z)) + f(q.s * q.s),
    );
    const tolerance = 20 * 2 ** -23;
    return 1 - tolerance < lengthSquared && lengthSquared < 1 + tolerance;
};
const validFrame = (frame: Transform): boolean =>
    [frame.p.x, frame.p.y, frame.p.z].every(Number.isFinite) && validQuat(frame.q);

/** Only the assertions in Box3D's create functions refuse authored values; solver clamping stays in the solver. */
export function invalidJointField(def: JointDef): string | null {
    const c = def.config;
    if (!validFrame(c.localFrameA)) return "localFrameA";
    if (!validFrame(c.localFrameB)) return "localFrameB";
    switch (def.kind) {
        case "Distance":
            if (!Number.isFinite(c.length) || !(c.length! > 0)) return "length";
            if (!(c.lowerSpringForce! <= c.upperSpringForce!))
                return "lowerSpringForce/upperSpringForce";
            break;
        case "Parallel":
            for (const name of ["hertz", "dampingRatio", "maxTorque"] as const)
                if (!Number.isFinite(c[name]) || !(c[name]! >= 0)) return name;
            break;
        case "Prismatic":
            if (!(c.lowerTranslation! <= c.upperTranslation!))
                return "lowerTranslation/upperTranslation";
            break;
        case "Spherical":
            if (
                !(
                    c.coneAngle! >= 0 &&
                    c.coneAngle! <= Math.fround(Math.fround(0.99) * Math.fround(Math.PI))
                )
            )
                return "coneAngle";
            if (!validQuat(c.targetRotation!)) return "targetRotation";
            break;
        case "Weld":
            for (const name of [
                "angularHertz",
                "angularDampingRatio",
                "linearHertz",
                "linearDampingRatio",
            ] as const)
                if (!(c[name]! >= 0)) return name;
            break;
        case "Wheel":
            if (!(c.lowerSuspensionLimit! <= c.upperSuspensionLimit!))
                return "lowerSuspensionLimit/upperSuspensionLimit";
            break;
    }
    return null;
}
function createJoint(
    state: WorldState,
    bodies: ReadonlyMap<number, EntityId>,
    def: JointDef,
    isDeferred: (eid: number) => boolean,
    warned: Set<number>,
    key: number,
): EntityId | null {
    const pair = endpoints(bodies, def, isDeferred, warned, key);
    if (!pair) return null;
    const invalid = invalidJointField(def);
    if (invalid) {
        warnOnce(
            warned,
            key,
            `[physics] ${def.kind}Joint ${def.eid} has invalid ${invalid} — skipped`,
        );
        return null;
    }
    const [a, b] = pair;
    const config = froundConfig(def.config);
    const base = {
        ...defaultJointDef(),
        bodyIdA: a.index1 - 1,
        bodyIdB: b.index1 - 1,
        ...config,
    };
    let joint: number;
    switch (def.kind) {
        case "Distance":
            joint = createDistanceJoint(state, {
                ...defaultDistanceJointDef(base),
                ...config,
                base,
            } as never).joint;
            break;
        case "Filter":
            joint = createFilterJoint(state, base).joint;
            break;
        case "Motor":
            joint = createMotorJoint(state, {
                ...defaultMotorJointDef(base),
                ...config,
                base,
            } as never).joint;
            break;
        case "Parallel":
            joint = createParallelJoint(state, {
                ...defaultParallelJointDef(base),
                ...config,
                base,
            } as never).joint;
            break;
        case "Prismatic":
            joint = createPrismaticJoint(state, {
                ...defaultPrismaticJointDef(base),
                ...config,
                base,
            } as never).joint;
            break;
        case "Revolute":
            joint = createRevoluteJoint(state, {
                ...defaultRevoluteJointDef(base),
                ...config,
                base,
            } as never).joint;
            break;
        case "Spherical":
            joint = createSphericalJoint(state, {
                ...defaultSphericalJointDef(base),
                ...config,
                base,
            } as never).joint;
            break;
        case "Weld":
            joint = createWeldJoint(state, {
                ...defaultWeldJointDef(base),
                ...config,
                base,
            } as never).joint;
            break;
        case "Wheel":
            joint = createWheelJoint(state, {
                ...defaultWheelJointDef(base),
                ...config,
                base,
            } as never).joint;
            break;
    }
    return makeJointId(state, joint);
}

const frame: Transform = {
    p: { x: 0, y: 0, z: 0 },
    q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
};
const vector: Vec3 = { x: 0, y: 0, z: 0 };
const rotation: Quat = { v: { x: 0, y: 0, z: 0 }, s: 1 };
const same = (a: number, b: number): boolean => Object.is(a, b);
const flag = (value: number): boolean => value !== 0;
const differs = (value: number, current: number): boolean => !same(value, current);
const vectorDiffers = (x: number, y: number, z: number, current: Vec3): boolean =>
    differs(x, current.x) || differs(y, current.y) || differs(z, current.z);
const rotationDiffers = (x: number, y: number, z: number, s: number, current: Quat): boolean =>
    differs(x, current.v.x) ||
    differs(y, current.v.y) ||
    differs(z, current.v.z) ||
    differs(s, current.s);
function jointValue(binding: JointBinding, eid: number, name: string): number {
    return (binding.storage[name] as ScalarField).column[eid] ?? 0;
}
type ScalarEdit = readonly [field: string, getter: string, setter: string];
const REVOLUTE_MOTOR_SPEED: ScalarEdit = ["motorSpeed", "getMotorSpeed", "setMotorSpeed"];
const BASE_SCALAR_EDITS: readonly ScalarEdit[] = [
    ["collideConnected", "getCollideConnected", "setCollideConnected"],
    ["forceThreshold", "getForceThreshold", "setForceThreshold"],
    ["torqueThreshold", "getTorqueThreshold", "setTorqueThreshold"],
];
const JOINT_SCALAR_EDITS: Record<JointBinding["kind"], readonly ScalarEdit[]> = {
    Distance: [
        ["length", "getLength", "setLength"],
        ["enableSpring", "isSpringEnabled", "enableSpring"],
        ["hertz", "getSpringHertz", "setSpringHertz"],
        ["dampingRatio", "getSpringDampingRatio", "setSpringDampingRatio"],
        ["enableLimit", "isLimitEnabled", "enableLimit"],
        ["enableMotor", "isMotorEnabled", "enableMotor"],
        ["maxMotorForce", "getMaxMotorForce", "setMaxMotorForce"],
        ["motorSpeed", "getMotorSpeed", "setMotorSpeed"],
    ],
    Filter: [],
    Motor: [
        ["maxVelocityForce", "getMaxVelocityForce", "setMaxVelocityForce"],
        ["maxVelocityTorque", "getMaxVelocityTorque", "setMaxVelocityTorque"],
        ["linearHertz", "getLinearHertz", "setLinearHertz"],
        ["linearDampingRatio", "getLinearDampingRatio", "setLinearDampingRatio"],
        ["maxSpringForce", "getMaxSpringForce", "setMaxSpringForce"],
        ["angularHertz", "getAngularHertz", "setAngularHertz"],
        ["angularDampingRatio", "getAngularDampingRatio", "setAngularDampingRatio"],
        ["maxSpringTorque", "getMaxSpringTorque", "setMaxSpringTorque"],
    ],
    Parallel: [
        ["hertz", "getSpringHertz", "setSpringHertz"],
        ["dampingRatio", "getSpringDampingRatio", "setSpringDampingRatio"],
        ["maxTorque", "getMaxTorque", "setMaxTorque"],
    ],
    Prismatic: [
        ["enableSpring", "isSpringEnabled", "enableSpring"],
        ["hertz", "getSpringHertz", "setSpringHertz"],
        ["dampingRatio", "getSpringDampingRatio", "setSpringDampingRatio"],
        ["targetTranslation", "getTargetTranslation", "setTargetTranslation"],
        ["enableLimit", "isLimitEnabled", "enableLimit"],
        ["enableMotor", "isMotorEnabled", "enableMotor"],
        ["maxMotorForce", "getMaxMotorForce", "setMaxMotorForce"],
        ["motorSpeed", "getMotorSpeed", "setMotorSpeed"],
    ],
    Revolute: [
        ["targetAngle", "getTargetAngle", "setTargetAngle"],
        ["enableSpring", "isSpringEnabled", "enableSpring"],
        ["hertz", "getSpringHertz", "setSpringHertz"],
        ["dampingRatio", "getSpringDampingRatio", "setSpringDampingRatio"],
        ["enableLimit", "isLimitEnabled", "enableLimit"],
        ["enableMotor", "isMotorEnabled", "enableMotor"],
        ["maxMotorTorque", "getMaxMotorTorque", "setMaxMotorTorque"],
        REVOLUTE_MOTOR_SPEED,
    ],
    Spherical: [
        ["enableSpring", "isSpringEnabled", "enableSpring"],
        ["hertz", "getSpringHertz", "setSpringHertz"],
        ["dampingRatio", "getSpringDampingRatio", "setSpringDampingRatio"],
        ["enableConeLimit", "isConeLimitEnabled", "enableConeLimit"],
        ["coneAngle", "getConeLimit", "setConeLimit"],
        ["enableTwistLimit", "isTwistLimitEnabled", "enableTwistLimit"],
        ["enableMotor", "isMotorEnabled", "enableMotor"],
        ["maxMotorTorque", "getMaxMotorTorque", "setMaxMotorTorque"],
    ],
    Weld: [
        ["linearHertz", "getLinearHertz", "setLinearHertz"],
        ["angularHertz", "getAngularHertz", "setAngularHertz"],
        ["linearDampingRatio", "getLinearDampingRatio", "setLinearDampingRatio"],
        ["angularDampingRatio", "getAngularDampingRatio", "setAngularDampingRatio"],
    ],
    Wheel: [
        ["enableSuspensionSpring", "isSuspensionEnabled", "enableSuspension"],
        ["suspensionHertz", "getSuspensionHertz", "setSuspensionHertz"],
        ["suspensionDampingRatio", "getSuspensionDampingRatio", "setSuspensionDampingRatio"],
        ["enableSuspensionLimit", "isSuspensionLimitEnabled", "enableSuspensionLimit"],
        ["enableSpinMotor", "isSpinMotorEnabled", "enableSpinMotor"],
        ["maxSpinTorque", "getMaxSpinTorque", "setMaxSpinTorque"],
        ["spinSpeed", "getSpinMotorSpeed", "setSpinMotorSpeed"],
        ["enableSteering", "isSteeringEnabled", "enableSteering"],
        ["steeringHertz", "getSteeringHertz", "setSteeringHertz"],
        ["steeringDampingRatio", "getSteeringDampingRatio", "setSteeringDampingRatio"],
        ["targetSteeringAngle", "getTargetSteeringAngle", "setTargetSteeringAngle"],
        ["maxSteeringTorque", "getMaxSteeringTorque", "setMaxSteeringTorque"],
        ["enableSteeringLimit", "isSteeringLimitEnabled", "enableSteeringLimit"],
    ],
};
const JOINT_SCALAR_COLUMNS: Record<JointBinding["kind"], Record<string, number>> = {
    Distance: {
        length: C.DJ_LENGTH,
        hertz: C.DJ_HERTZ,
        dampingRatio: C.DJ_DAMPING_RATIO,
        maxMotorForce: C.DJ_MAX_MOTOR_FORCE,
        motorSpeed: C.DJ_MOTOR_SPEED,
    },
    Filter: {},
    Motor: {
        maxVelocityForce: C.MJ_MAX_VELOCITY_FORCE,
        maxVelocityTorque: C.MJ_MAX_VELOCITY_TORQUE,
        linearHertz: C.MJ_LINEAR_HERTZ,
        linearDampingRatio: C.MJ_LINEAR_DAMPING_RATIO,
        maxSpringForce: C.MJ_MAX_SPRING_FORCE,
        angularHertz: C.MJ_ANGULAR_HERTZ,
        angularDampingRatio: C.MJ_ANGULAR_DAMPING_RATIO,
        maxSpringTorque: C.MJ_MAX_SPRING_TORQUE,
    },
    Parallel: {
        hertz: C.PLJ_HERTZ,
        dampingRatio: C.PLJ_DAMPING_RATIO,
        maxTorque: C.PLJ_MAX_TORQUE,
    },
    Prismatic: {
        hertz: C.PJ_HERTZ,
        dampingRatio: C.PJ_DAMPING_RATIO,
        targetTranslation: C.PJ_TARGET_TRANSLATION,
        maxMotorForce: C.PJ_MAX_MOTOR_FORCE,
        motorSpeed: C.PJ_MOTOR_SPEED,
    },
    Revolute: {
        targetAngle: C.RJ_TARGET_ANGLE,
        hertz: C.RJ_HERTZ,
        dampingRatio: C.RJ_DAMPING_RATIO,
        maxMotorTorque: C.RJ_MAX_MOTOR_TORQUE,
        motorSpeed: C.RJ_MOTOR_SPEED,
    },
    Spherical: {
        hertz: C.SJ_HERTZ,
        dampingRatio: C.SJ_DAMPING_RATIO,
        coneAngle: C.SJ_CONE_ANGLE,
        maxMotorTorque: C.SJ_MAX_MOTOR_TORQUE,
    },
    Weld: {
        linearHertz: C.WJ_LINEAR_HERTZ,
        linearDampingRatio: C.WJ_LINEAR_DAMPING_RATIO,
        angularHertz: C.WJ_ANGULAR_HERTZ,
        angularDampingRatio: C.WJ_ANGULAR_DAMPING_RATIO,
    },
    Wheel: {
        suspensionHertz: C.WHJ_SUSPENSION_HERTZ,
        suspensionDampingRatio: C.WHJ_SUSPENSION_DAMPING_RATIO,
        maxSpinTorque: C.WHJ_MAX_SPIN_TORQUE,
        spinSpeed: C.WHJ_SPIN_SPEED,
        steeringHertz: C.WHJ_STEERING_HERTZ,
        steeringDampingRatio: C.WHJ_STEERING_DAMPING_RATIO,
        targetSteeringAngle: C.WHJ_TARGET_STEERING_ANGLE,
        maxSteeringTorque: C.WHJ_MAX_STEERING_TORQUE,
    },
};
type JointFlag = readonly [field: number, mask: number];
const JOINT_SCALAR_FLAGS: Partial<Record<JointBinding["kind"], Record<string, JointFlag>>> = {
    Distance: {
        enableSpring: [C.DJ_ENABLE, C.DJ_ENABLE_SPRING],
        enableLimit: [C.DJ_ENABLE, C.DJ_ENABLE_LIMIT],
        enableMotor: [C.DJ_ENABLE, C.DJ_ENABLE_MOTOR],
    },
    Prismatic: {
        enableSpring: [C.PJ_ENABLE, C.PJ_ENABLE_SPRING],
        enableLimit: [C.PJ_ENABLE, C.PJ_ENABLE_LIMIT],
        enableMotor: [C.PJ_ENABLE, C.PJ_ENABLE_MOTOR],
    },
    Revolute: {
        enableSpring: [C.RJ_ENABLE, C.RJ_ENABLE_SPRING],
        enableLimit: [C.RJ_ENABLE, C.RJ_ENABLE_LIMIT],
        enableMotor: [C.RJ_ENABLE, C.RJ_ENABLE_MOTOR],
    },
    Spherical: {
        enableSpring: [C.SJ_ENABLE, C.SJ_ENABLE_SPRING],
        enableConeLimit: [C.SJ_ENABLE, C.SJ_ENABLE_CONE_LIMIT],
        enableTwistLimit: [C.SJ_ENABLE, C.SJ_ENABLE_TWIST_LIMIT],
        enableMotor: [C.SJ_ENABLE, C.SJ_ENABLE_MOTOR],
    },
    Wheel: {
        enableSuspensionSpring: [C.WHJ_ENABLE, C.WHJ_ENABLE_SUSPENSION_SPRING],
        enableSuspensionLimit: [C.WHJ_ENABLE, C.WHJ_ENABLE_SUSPENSION_LIMIT],
        enableSpinMotor: [C.WHJ_ENABLE, C.WHJ_ENABLE_SPIN_MOTOR],
        enableSteering: [C.WHJ_ENABLE, C.WHJ_ENABLE_STEERING],
        enableSteeringLimit: [C.WHJ_ENABLE, C.WHJ_ENABLE_STEERING_LIMIT],
    },
};
function readJointScalar(
    state: WorldState,
    id: number,
    kind: JointBinding["kind"],
    field: string,
): number {
    if (field === "collideConnected") return jointField(state, id, JointField.collideConnected);
    const flagAccess = JOINT_SCALAR_FLAGS[kind]?.[field];
    if (flagAccess) return Number(readJointFlag(state, id, flagAccess[0], flagAccess[1]));
    const column =
        field === "forceThreshold"
            ? C.J_FORCE_THRESHOLD
            : field === "torqueThreshold"
              ? C.J_TORQUE_THRESHOLD
              : JOINT_SCALAR_COLUMNS[kind][field];
    if (column === undefined) throw new Error(`physics: unknown ${kind} joint field ${field}`);
    return readJointFloat(state, id, column);
}
function writeJointScalar(
    state: WorldState,
    id: number,
    kind: JointBinding["kind"],
    field: string,
    value: number,
): void {
    if (field === "collideConnected") {
        setJointCollideConnected(state, id, flag(value));
        return;
    }
    const flagAccess = JOINT_SCALAR_FLAGS[kind]?.[field];
    if (flagAccess) {
        kernel(state.ecsState).jointEnable(state.worldId, id, flagAccess[1], Boolean(value));
        return;
    }
    if (field === "length") {
        kernel(state.ecsState).distanceJointSetLength(state.worldId, id, value);
        return;
    }
    if (field === "maxSpringForce" || field === "maxSpringTorque") {
        kernel(state.ecsState).motorJointSetMaxSpring(
            state.worldId,
            id,
            field === "maxSpringTorque",
            value,
        );
        return;
    }
    const column =
        field === "forceThreshold"
            ? C.J_FORCE_THRESHOLD
            : field === "torqueThreshold"
              ? C.J_TORQUE_THRESHOLD
              : JOINT_SCALAR_COLUMNS[kind][field];
    if (column === undefined) throw new Error(`physics: unknown ${kind} joint field ${field}`);
    writeJointFloat(state, id, column, value);
    if (field === "motorSpeed" || field === "spinSpeed" || field === "targetSteeringAngle")
        wakeJointBodies(state, id);
}

function setterAccepts(kind: JointBinding["kind"], field: string, value: number): boolean {
    const finite = Number.isFinite(value);
    const nonnegative = finite && value >= 0;
    if (field === "forceThreshold" || field === "torqueThreshold") return nonnegative;
    switch (kind) {
        case "Parallel":
            if (field === "hertz" || field === "dampingRatio" || field === "maxTorque")
                return nonnegative;
            break;
        case "Prismatic":
            if (field === "hertz" || field === "dampingRatio" || field === "maxMotorForce")
                return nonnegative;
            if (field === "targetTranslation" || field === "motorSpeed") return finite;
            break;
        case "Revolute":
            if (field === "hertz" || field === "dampingRatio" || field === "maxMotorTorque")
                return nonnegative;
            if (field === "motorSpeed") return finite;
            if (field === "targetAngle") {
                const pi = Math.fround(Math.PI);
                return finite && -pi <= value && value <= pi;
            }
            break;
        case "Spherical":
            if (field === "hertz" || field === "dampingRatio" || field === "maxMotorTorque")
                return nonnegative;
            if (field === "coneAngle") {
                const max = Math.fround(Math.fround(0.5) * Math.fround(Math.PI));
                return finite && 0 <= value && value <= max;
            }
            break;
        case "Weld":
            if (
                field === "linearHertz" ||
                field === "angularHertz" ||
                field === "linearDampingRatio" ||
                field === "angularDampingRatio"
            )
                return nonnegative;
            break;
    }
    return true;
}

function rejectFieldEdit(
    warned: Set<number>,
    key: number,
    binding: JointBinding,
    eid: number,
    field: string,
): void {
    warnOnce(
        warned,
        fieldWarningKey(binding, key, field),
        `[physics] ${binding.kind}Joint ${eid} ignored invalid ${field}; the solver value is unchanged`,
    );
}

function applyScalarEdits(
    binding: JointBinding,
    eid: number,
    state: WorldState,
    id: number,
    mask: number,
    edits: readonly ScalarEdit[],
    warned: Set<number>,
    key: number,
): void {
    for (const [field] of edits) {
        if (!fieldChanged(binding, mask, field)) continue;
        const value = jointValue(binding, eid, field);
        const finiteMotorSpeed =
            field === "motorSpeed" && (binding.kind === "Prismatic" || binding.kind === "Revolute");
        const accepted = finiteMotorSpeed
            ? Number.isFinite(value)
            : setterAccepts(binding.kind, field, value);
        if (!accepted) {
            rejectFieldEdit(warned, key, binding, eid, field);
            continue;
        }
        clearFieldWarning(warned, binding, key, field);
        if (differs(value, readJointScalar(state, id, binding.kind, field))) {
            if (field === "motorSpeed") {
                const k = kernel(state.ecsState);
                k.bodySetActiveWorld(state.worldId);
                const column = JOINT_SCALAR_COLUMNS[binding.kind][field];
                jointViews(k).floats[(k.jointSimPtr(id) >>> 2) + column] = Math.fround(value);
                // The kernel wake is idempotent for awake/static bodies; avoid probing them in JS.
                state.locked = true;
                k.jointWakeBodies(state.worldId, id);
                state.locked = false;
            } else {
                writeJointScalar(state, id, binding.kind, field, value);
            }
        }
    }
}

function applyBaseFields(
    binding: JointBinding,
    eid: number,
    state: WorldState,
    id: number,
    mask: number,
    warned: Set<number>,
    key: number,
): void {
    applyScalarEdits(binding, eid, state, id, mask, BASE_SCALAR_EDITS, warned, key);
    if (
        fieldChanged(binding, mask, "constraintHertz") ||
        fieldChanged(binding, mask, "constraintDampingRatio")
    ) {
        const hertz = jointValue(binding, eid, "constraintHertz");
        const damping = jointValue(binding, eid, "constraintDampingRatio");
        if (!Number.isFinite(hertz) || hertz < 0 || !Number.isFinite(damping) || damping < 0) {
            rejectFieldEdit(warned, key, binding, eid, "constraint tuning");
        } else {
            clearFieldWarning(warned, binding, key, "constraint tuning");
            if (
                differs(hertz, readJointFloat(state, id, J_CONSTRAINT_HERTZ)) ||
                differs(damping, readJointFloat(state, id, J_CONSTRAINT_DAMPING))
            ) {
                writeJointFloat(state, id, J_CONSTRAINT_HERTZ, hertz);
                writeJointFloat(state, id, J_CONSTRAINT_DAMPING, damping);
            }
        }
    }
    if (fieldChanged(binding, mask, "drawScale")) {
        const drawScale = jointValue(binding, eid, "drawScale");
        if (differs(drawScale, jointDrawScale(state, id))) setJointDrawScale(state, id, drawScale);
    }
    if (
        fieldChanged(binding, mask, "localAnchorA") ||
        fieldChanged(binding, mask, "localRotationA")
    )
        applyLocalFrame(binding, eid, state, id, "A", J_LOCAL_FRAME_A, warned, key);
    if (
        fieldChanged(binding, mask, "localAnchorB") ||
        fieldChanged(binding, mask, "localRotationB")
    )
        applyLocalFrame(binding, eid, state, id, "B", J_LOCAL_FRAME_B, warned, key);
}

function applyLocalFrame(
    binding: JointBinding,
    eid: number,
    state: WorldState,
    id: number,
    suffix: "A" | "B",
    offset: number,
    warned: Set<number>,
    key: number,
): void {
    const localFrameName = suffix === "A" ? "localFrameA" : "localFrameB";
    const anchor = (
        suffix === "A" ? binding.storage.localAnchorA : binding.storage.localAnchorB
    ) as Vector4Field;
    const localRotation = (
        suffix === "A" ? binding.storage.localRotationA : binding.storage.localRotationB
    ) as Vector4Field;
    const x = anchor.x.get(eid),
        y = anchor.y.get(eid),
        z = anchor.z.get(eid);
    const qx = localRotation.x.get(eid),
        qy = localRotation.y.get(eid);
    const qz = localRotation.z.get(eid),
        qw = localRotation.w.get(eid);
    if (
        !differs(x, readJointFloat(state, id, offset)) &&
        !differs(y, readJointFloat(state, id, offset + 1)) &&
        !differs(z, readJointFloat(state, id, offset + 2)) &&
        !differs(qx, readJointFloat(state, id, offset + 3)) &&
        !differs(qy, readJointFloat(state, id, offset + 4)) &&
        !differs(qz, readJointFloat(state, id, offset + 5)) &&
        !differs(qw, readJointFloat(state, id, offset + 6))
    )
        return;
    frame.p.x = x;
    frame.p.y = y;
    frame.p.z = z;
    frame.q.v.x = qx;
    frame.q.v.y = qy;
    frame.q.v.z = qz;
    frame.q.s = qw;
    if (!validFrame(frame)) {
        rejectFieldEdit(warned, key, binding, eid, localFrameName);
        return;
    }
    clearFieldWarning(warned, binding, key, localFrameName);
    writeJointTransform(state, id, offset, frame);
}

function applyJointFields(
    binding: JointBinding,
    eid: number,
    state: WorldState,
    id: number,
    mask: number,
    warned: Set<number>,
    key: number,
): void {
    applyBaseFields(binding, eid, state, id, mask, warned, key);
    applyScalarEdits(binding, eid, state, id, mask, JOINT_SCALAR_EDITS[binding.kind], warned, key);
    const storage = binding.storage;
    switch (binding.kind) {
        case "Distance": {
            const lowerForce = jointValue(binding, eid, "lowerSpringForce");
            const upperForce = jointValue(binding, eid, "upperSpringForce");
            if (
                (fieldChanged(binding, mask, "lowerSpringForce") ||
                    fieldChanged(binding, mask, "upperSpringForce")) &&
                (differs(lowerForce, readJointFloat(state, id, DJ_LOWER_SPRING_FORCE)) ||
                    differs(upperForce, readJointFloat(state, id, DJ_UPPER_SPRING_FORCE)))
            ) {
                if (lowerForce <= upperForce) {
                    clearFieldWarning(warned, binding, key, "lowerSpringForce/upperSpringForce");
                    writeJointFloat(state, id, DJ_LOWER_SPRING_FORCE, lowerForce);
                    writeJointFloat(state, id, DJ_UPPER_SPRING_FORCE, upperForce);
                } else {
                    rejectFieldEdit(warned, key, binding, eid, "lowerSpringForce/upperSpringForce");
                }
            }
            const minLength = jointValue(binding, eid, "minLength");
            const maxLength = jointValue(binding, eid, "maxLength");
            if (
                (fieldChanged(binding, mask, "minLength") ||
                    fieldChanged(binding, mask, "maxLength")) &&
                (differs(minLength, readJointFloat(state, id, C.DJ_MIN_LENGTH)) ||
                    differs(maxLength, readJointFloat(state, id, C.DJ_MAX_LENGTH)))
            )
                kernel(state.ecsState).jointSetLimits(state.worldId, id, minLength, maxLength);
            break;
        }
        case "Filter":
        case "Parallel":
        case "Weld":
            break;
        case "Motor": {
            if (fieldChanged(binding, mask, "linearVelocity")) {
                const linear = storage.linearVelocity as Vector4Field;
                vector.x = linear.x.get(eid);
                vector.y = linear.y.get(eid);
                vector.z = linear.z.get(eid);
                readJointVec3(state, id, MJ_LINEAR_VELOCITY, kinVector);
                if (vectorDiffers(vector.x, vector.y, vector.z, kinVector))
                    writeJointVec3(state, id, MJ_LINEAR_VELOCITY, vector);
            }
            if (fieldChanged(binding, mask, "angularVelocity")) {
                const angular = storage.angularVelocity as Vector4Field;
                vector.x = angular.x.get(eid);
                vector.y = angular.y.get(eid);
                vector.z = angular.z.get(eid);
                readJointVec3(state, id, MJ_ANGULAR_VELOCITY, kinVector);
                if (vectorDiffers(vector.x, vector.y, vector.z, kinVector))
                    writeJointVec3(state, id, MJ_ANGULAR_VELOCITY, vector);
            }
            break;
        }
        case "Prismatic": {
            if (
                fieldChanged(binding, mask, "lowerTranslation") ||
                fieldChanged(binding, mask, "upperTranslation")
            ) {
                const lower = jointValue(binding, eid, "lowerTranslation");
                const upper = jointValue(binding, eid, "upperTranslation");
                if (!Number.isFinite(lower) || !Number.isFinite(upper)) {
                    rejectFieldEdit(warned, key, binding, eid, "lowerTranslation/upperTranslation");
                } else {
                    clearFieldWarning(warned, binding, key, "lowerTranslation/upperTranslation");
                    if (
                        differs(lower, readJointFloat(state, id, C.PJ_LOWER_TRANSLATION)) ||
                        differs(upper, readJointFloat(state, id, C.PJ_UPPER_TRANSLATION))
                    )
                        kernel(state.ecsState).jointSetLimits(state.worldId, id, lower, upper);
                }
            }
            break;
        }
        case "Revolute": {
            if (
                fieldChanged(binding, mask, "lowerAngle") ||
                fieldChanged(binding, mask, "upperAngle")
            ) {
                const lower = jointValue(binding, eid, "lowerAngle");
                const upper = jointValue(binding, eid, "upperAngle");
                if (!Number.isFinite(lower) || !Number.isFinite(upper)) {
                    rejectFieldEdit(warned, key, binding, eid, "lowerAngle/upperAngle");
                } else {
                    clearFieldWarning(warned, binding, key, "lowerAngle/upperAngle");
                    if (
                        differs(lower, readJointFloat(state, id, C.RJ_LOWER_ANGLE)) ||
                        differs(upper, readJointFloat(state, id, C.RJ_UPPER_ANGLE))
                    )
                        kernel(state.ecsState).jointSetLimits(state.worldId, id, lower, upper);
                }
            }
            break;
        }
        case "Spherical": {
            if (fieldChanged(binding, mask, "targetRotation")) {
                const target = storage.targetRotation as Vector4Field;
                rotation.v.x = target.x.get(eid);
                rotation.v.y = target.y.get(eid);
                rotation.v.z = target.z.get(eid);
                rotation.s = target.w.get(eid);
                if (!validQuat(rotation)) {
                    rejectFieldEdit(warned, key, binding, eid, "targetRotation");
                } else {
                    clearFieldWarning(warned, binding, key, "targetRotation");
                    readJointQuat(state, id, SJ_TARGET_ROTATION, kinRotation);
                    if (
                        rotationDiffers(
                            rotation.v.x,
                            rotation.v.y,
                            rotation.v.z,
                            rotation.s,
                            kinRotation,
                        )
                    )
                        writeJointQuat(state, id, SJ_TARGET_ROTATION, rotation);
                }
            }
            if (
                fieldChanged(binding, mask, "lowerTwistAngle") ||
                fieldChanged(binding, mask, "upperTwistAngle")
            ) {
                const lower = jointValue(binding, eid, "lowerTwistAngle");
                const upper = jointValue(binding, eid, "upperTwistAngle");
                if (!Number.isFinite(lower) || !Number.isFinite(upper)) {
                    rejectFieldEdit(warned, key, binding, eid, "lowerTwistAngle/upperTwistAngle");
                } else {
                    clearFieldWarning(warned, binding, key, "lowerTwistAngle/upperTwistAngle");
                    if (
                        differs(lower, readJointFloat(state, id, C.SJ_LOWER_TWIST_ANGLE)) ||
                        differs(upper, readJointFloat(state, id, C.SJ_UPPER_TWIST_ANGLE))
                    )
                        kernel(state.ecsState).jointSetLimits(state.worldId, id, lower, upper);
                }
            }
            if (fieldChanged(binding, mask, "motorVelocity")) {
                const velocity = storage.motorVelocity as Vector4Field;
                vector.x = velocity.x.get(eid);
                vector.y = velocity.y.get(eid);
                vector.z = velocity.z.get(eid);
                if (!finiteVec3(vector)) {
                    rejectFieldEdit(warned, key, binding, eid, "motorVelocity");
                } else {
                    clearFieldWarning(warned, binding, key, "motorVelocity");
                    readJointVec3(state, id, SJ_MOTOR_VELOCITY, kinVector);
                    if (vectorDiffers(vector.x, vector.y, vector.z, kinVector))
                        writeJointVec3(state, id, SJ_MOTOR_VELOCITY, vector);
                }
            }
            break;
        }
        case "Wheel": {
            if (
                fieldChanged(binding, mask, "lowerSuspensionLimit") ||
                fieldChanged(binding, mask, "upperSuspensionLimit")
            ) {
                const lower = jointValue(binding, eid, "lowerSuspensionLimit");
                const upper = jointValue(binding, eid, "upperSuspensionLimit");
                if (lower <= upper) {
                    clearFieldWarning(
                        warned,
                        binding,
                        key,
                        "lowerSuspensionLimit/upperSuspensionLimit",
                    );
                    if (
                        differs(lower, readJointFloat(state, id, C.WHJ_LOWER_SUSPENSION_LIMIT)) ||
                        differs(upper, readJointFloat(state, id, C.WHJ_UPPER_SUSPENSION_LIMIT))
                    ) {
                        writeJointFloat(state, id, C.WHJ_LOWER_SUSPENSION_LIMIT, lower);
                        writeJointFloat(state, id, C.WHJ_UPPER_SUSPENSION_LIMIT, upper);
                    }
                } else {
                    rejectFieldEdit(
                        warned,
                        key,
                        binding,
                        eid,
                        "lowerSuspensionLimit/upperSuspensionLimit",
                    );
                }
            }
            if (
                fieldChanged(binding, mask, "lowerSteeringLimit") ||
                fieldChanged(binding, mask, "upperSteeringLimit")
            ) {
                const lower = jointValue(binding, eid, "lowerSteeringLimit");
                const upper = jointValue(binding, eid, "upperSteeringLimit");
                if (lower <= upper) {
                    clearFieldWarning(
                        warned,
                        binding,
                        key,
                        "lowerSteeringLimit/upperSteeringLimit",
                    );
                    if (
                        differs(lower, readJointFloat(state, id, C.WHJ_LOWER_STEERING_LIMIT)) ||
                        differs(upper, readJointFloat(state, id, C.WHJ_UPPER_STEERING_LIMIT))
                    ) {
                        writeJointFloat(state, id, C.WHJ_LOWER_STEERING_LIMIT, lower);
                        writeJointFloat(state, id, C.WHJ_UPPER_STEERING_LIMIT, upper);
                    }
                } else {
                    rejectFieldEdit(
                        warned,
                        key,
                        binding,
                        eid,
                        "lowerSteeringLimit/upperSteeringLimit",
                    );
                }
            }
            break;
        }
    }
}

function finiteVec3(value: Vec3): boolean {
    return Number.isFinite(value.x) && Number.isFinite(value.y) && Number.isFinite(value.z);
}
const kinVector: Vec3 = { x: 0, y: 0, z: 0 };
const kinRotation: Quat = { v: { x: 0, y: 0, z: 0 }, s: 1 };

function bodyEndpointsMatch(
    state: WorldState,
    joint: EntityId,
    bodyA: EntityId | undefined,
    bodyB: EntityId | undefined,
): boolean {
    if (!bodyA || !bodyB || !jointIsValid(state, joint)) return false;
    const id = joint.index1 - 1;
    return (
        jointField(state, id, JointField.bodyIdA) === bodyA.index1 - 1 &&
        jointField(state, id, JointField.bodyIdB) === bodyB.index1 - 1
    );
}

export function syncJoints(
    cache: ConstraintCache,
    state: WorldState,
    bodies: ReadonlyMap<number, EntityId>,
    ecs: World,
    bindings: readonly JointBinding[],
    isDeferred: (eid: number) => boolean,
): void {
    for (let i = 0; i < bindings.length; i++) {
        const binding = bindings[i];
        const candidates = cache.candidates[binding.index];
        if (candidates.count === 0) continue;
        let live = cache.liveJoints.get(binding.index);
        for (let candidate = 0; candidate < candidates.count; candidate++) {
            const eid = candidates.eids[candidate];
            const fieldMask = candidates.fieldMarks[eid];
            const key = identity(binding.index, eid);
            const joint = live?.get(eid);
            if (!ecs.has(eid, binding.component)) {
                if (joint && jointIsValid(state, joint))
                    destroyJointInternal(state, joint.index1 - 1, true);
                live?.delete(eid);
                clearJointWarnings(cache.warnedJoints, key);
                continue;
            }
            const a = (binding.storage.a as import("../../engine").ScalarField).get(eid);
            const b = (binding.storage.b as import("../../engine").ScalarField).get(eid);
            const bodyA = bodies.get(a);
            const bodyB = bodies.get(b);
            if (joint && bodyEndpointsMatch(state, joint, bodyA, bodyB)) {
                applyJointFields(
                    binding,
                    eid,
                    state,
                    joint.index1 - 1,
                    fieldMask,
                    cache.warnedJoints,
                    key,
                );
                continue;
            }
            if (joint && jointIsValid(state, joint))
                destroyJointInternal(state, joint.index1 - 1, true);
            live?.delete(eid);
            const def = jointDef(ecs, binding, eid);
            const created = createJoint(state, bodies, def, isDeferred, cache.warnedJoints, key);
            if (created) {
                live ??= new Map();
                live.set(eid, created);
                cache.liveJoints.set(binding.index, live);
                clearJointWarnings(cache.warnedJoints, key);
            }
        }
        clearFieldCandidates(candidates);
    }
}

export interface ConstraintIds {
    joints: [number, number, number, number][];
    candidates: [number, number, number][];
}
export function captureConstraints(cache: ConstraintCache, state: WorldState): ConstraintIds {
    const joints: ConstraintIds["joints"] = [];
    for (const [index, live] of cache.liveJoints)
        for (const [eid, joint] of live)
            if (jointIsValid(state, joint))
                joints.push([index, eid, joint.index1, joint.generation]);
    const candidates: ConstraintIds["candidates"] = [];
    for (let index = 0; index < cache.candidates.length; index++) {
        const pending = cache.candidates[index];
        for (let i = 0; i < pending.count; i++) {
            const eid = pending.eids[i];
            candidates.push([index, eid, pending.fieldMarks[eid]]);
        }
    }
    return { joints, candidates };
}
export function restoreConstraints(
    cache: ConstraintCache,
    ids: ConstraintIds,
    state: WorldState,
): void {
    const liveJoints = new Map<number, Map<number, EntityId>>();
    for (const [index, eid, index1, generation] of ids.joints) {
        let live = liveJoints.get(index);
        if (!live) liveJoints.set(index, (live = new Map()));
        live.set(eid, { index1, world0: state.worldId, generation });
    }
    cache.liveJoints = liveJoints;
    for (const candidates of cache.candidates) clearFieldCandidates(candidates);
    for (const [index, eid, fieldMask] of ids.candidates) {
        markFieldCandidate(cache.candidates[index], eid);
        cache.candidates[index].fieldMarks[eid] = fieldMask;
    }
    cache.warnedJoints.clear();
}
export function resetConstraints(cache: ConstraintCache): void {
    cache.liveJoints.clear();
    for (const candidates of cache.candidates) clearFieldCandidates(candidates);
    cache.warnedJoints.clear();
}
