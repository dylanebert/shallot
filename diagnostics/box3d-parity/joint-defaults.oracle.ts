import { expect, setDefaultTimeout, test } from "bun:test";
import { BodyType, init, type Joint, PhysicsWorld } from "../../src/standard/physics/api";
import { J_CONSTRAINT_SOFTNESS } from "../../src/standard/physics/kernel/joint-layout";
import { readJointFloat } from "../../src/standard/physics/kernel/jointcolumns";
import { jointDrawScale } from "../../src/standard/physics/kernel/jointrecords";
import { nativeSseOutput } from "./native-evidence";
import { assertPublicOracleKernel } from "./oracle-kernel";

setDefaultTimeout(180_000);
await init(undefined, { threads: 0 });
await assertPublicOracleKernel();

const kinds = [
    "parallel",
    "distance",
    "filter",
    "motor",
    "prismatic",
    "revolute",
    "spherical",
    "weld",
    "wheel",
] as const;
type Kind = (typeof kinds)[number];
type Values = { s: number[]; p: number[]; flags: number };
type PublicJoint = Joint & Record<string, (...args: unknown[]) => unknown>;
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

function setValues(kind: Kind, joint: PublicJoint, v: Values): void {
    joint.setLocalFrameA({ p: vec(v.s.slice(0, 3)), q: quat(v.flags & 2 ? "z" : "identity") });
    joint.setLocalFrameB({ p: vec(v.s.slice(3, 6)), q: quat(v.flags & 4 ? "x" : "identity") });
    joint.setCollideConnected(!!(v.flags & 1));
    joint.setConstraintTuning(v.p[0]!, v.p[1]!);
    joint.setForceThreshold(v.p[2]!);
    joint.setTorqueThreshold(v.p[3]!);
    joint.setUserData(1);

    switch (kind) {
        case "parallel":
            joint.setSpringHertz(v.p[4]!);
            joint.setSpringDampingRatio(v.p[5]!);
            joint.setMaxTorque(v.p[6]!);
            break;
        case "distance": {
            joint.setLength(v.p[4]!);
            joint.setSpringForceRange(...ordered(v.s[6]!, v.s[7]!));
            joint.enableSpring(!!(v.flags & 8));
            joint.setSpringHertz(v.p[5]!);
            joint.setSpringDampingRatio(v.p[6]!);
            joint.enableLimit(!!(v.flags & 16));
            joint.setLengthRange(...ordered(v.p[7]!, v.p[8]!));
            joint.enableMotor(!!(v.flags & 32));
            joint.setMotorSpeed(v.s[8]!);
            joint.setMaxMotorForce(v.p[9]!);
            break;
        }
        case "filter":
            break;
        case "motor":
            joint.setLinearVelocity(vec(v.s.slice(6, 9)));
            joint.setAngularVelocity(vec(v.s.slice(9, 12)));
            joint.setMaxVelocityForce(v.p[4]!);
            joint.setMaxVelocityTorque(v.p[5]!);
            joint.setLinearHertz(v.p[6]!);
            joint.setLinearDampingRatio(v.p[7]!);
            joint.setMaxSpringForce(v.p[8]!);
            joint.setAngularHertz(v.p[9]!);
            joint.setAngularDampingRatio(v.p[10]!);
            joint.setMaxSpringTorque(v.p[11]!);
            break;
        case "prismatic":
            joint.enableSpring(!!(v.flags & 8));
            joint.setSpringHertz(v.p[4]!);
            joint.setSpringDampingRatio(v.p[5]!);
            joint.setTargetTranslation(v.s[6]!);
            joint.enableLimit(!!(v.flags & 16));
            joint.setLimits(...ordered(v.s[7]!, v.s[8]!));
            joint.enableMotor(!!(v.flags & 32));
            joint.setMotorSpeed(v.s[9]!);
            joint.setMaxMotorForce(v.p[6]!);
            break;
        case "revolute":
            joint.enableSpring(!!(v.flags & 8));
            joint.setSpringHertz(v.p[4]!);
            joint.setSpringDampingRatio(v.p[5]!);
            joint.setTargetAngle(v.s[6]! / 8);
            joint.enableLimit(!!(v.flags & 16));
            joint.setLimits(...ordered(v.s[7]! / 8, v.s[8]! / 8));
            joint.enableMotor(!!(v.flags & 32));
            joint.setMotorSpeed(v.s[9]!);
            joint.setMaxMotorTorque(v.p[6]!);
            break;
        case "spherical":
            joint.enableConeLimit(!!(v.flags & 8));
            joint.setConeLimit(v.p[4]! / 8);
            joint.enableTwistLimit(!!(v.flags & 16));
            joint.setTwistLimits(...ordered(v.s[6]! / 8, v.s[7]! / 8));
            joint.enableSpring(!!(v.flags & 32));
            joint.setTargetRotation(quat(v.flags & 64 ? "y" : "identity"));
            joint.setSpringHertz(v.p[5]!);
            joint.setSpringDampingRatio(v.p[6]!);
            joint.enableMotor(!!(v.flags & 128));
            joint.setMotorVelocity(vec(v.s.slice(8, 11)));
            joint.setMaxMotorTorque(v.p[7]!);
            break;
        case "weld":
            joint.setLinearHertz(v.p[4]!);
            joint.setLinearDampingRatio(v.p[5]!);
            joint.setAngularHertz(v.p[6]!);
            joint.setAngularDampingRatio(v.p[7]!);
            break;
        case "wheel":
            joint.enableSuspension(!!(v.flags & 8));
            joint.setSuspensionHertz(v.p[4]!);
            joint.setSuspensionDampingRatio(v.p[5]!);
            joint.enableSuspensionLimit(!!(v.flags & 16));
            joint.setSuspensionLimits(...ordered(v.s[6]!, v.s[7]!));
            joint.enableSpinMotor(!!(v.flags & 32));
            joint.setSpinMotorSpeed(v.s[8]!);
            joint.setMaxSpinTorque(v.p[6]!);
            joint.enableSteering(!!(v.flags & 64));
            joint.setSteeringHertz(v.p[7]!);
            joint.setSteeringDampingRatio(v.p[8]!);
            joint.setMaxSteeringTorque(v.p[9]!);
            joint.enableSteeringLimit(!!(v.flags & 128));
            joint.setSteeringLimits(...ordered(v.s[9]! / 8, v.s[10]! / 8));
            joint.setTargetSteeringAngle(v.s[11]! / 8);
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

function dump(kind: Kind, joint: PublicJoint): number[] {
    const values: unknown[] = [
        joint.getLocalFrameA(),
        joint.getLocalFrameB(),
        joint.getCollideConnected(),
        joint.getConstraintTuning(),
        joint.getForceThreshold(),
        joint.getTorqueThreshold(),
        joint.getType(),
        joint.getUserData() == null ? 0 : 1,
        jointDrawScale(joint.world, joint.id.index1 - 1),
    ];
    switch (kind) {
        case "parallel":
            values.push(
                joint.getSpringHertz(),
                joint.getSpringDampingRatio(),
                joint.getMaxTorque(),
            );
            break;
        case "distance": {
            const range = joint.getSpringForceRange() as { lowerForce: number; upperForce: number };
            values.push(
                joint.getLength(),
                joint.isSpringEnabled(),
                range.lowerForce,
                range.upperForce,
                joint.getSpringHertz(),
                joint.getSpringDampingRatio(),
                joint.isLimitEnabled(),
                joint.getMinLength(),
                joint.getMaxLength(),
                joint.isMotorEnabled(),
                joint.getMaxMotorForce(),
                joint.getMotorSpeed(),
            );
            break;
        }
        case "filter":
            break;
        case "motor":
            values.push(
                joint.getLinearVelocity(),
                joint.getMaxVelocityForce(),
                joint.getAngularVelocity(),
                joint.getMaxVelocityTorque(),
                joint.getLinearHertz(),
                joint.getLinearDampingRatio(),
                joint.getMaxSpringForce(),
                joint.getAngularHertz(),
                joint.getAngularDampingRatio(),
                joint.getMaxSpringTorque(),
            );
            break;
        case "prismatic":
            values.push(
                joint.isSpringEnabled(),
                joint.getSpringHertz(),
                joint.getSpringDampingRatio(),
                joint.getTargetTranslation(),
                joint.isLimitEnabled(),
                joint.getLowerLimit(),
                joint.getUpperLimit(),
                joint.isMotorEnabled(),
                joint.getMaxMotorForce(),
                joint.getMotorSpeed(),
            );
            break;
        case "revolute":
            values.push(
                joint.isSpringEnabled(),
                joint.getSpringHertz(),
                joint.getSpringDampingRatio(),
                joint.getTargetAngle(),
                joint.isLimitEnabled(),
                joint.getLowerLimit(),
                joint.getUpperLimit(),
                joint.isMotorEnabled(),
                joint.getMaxMotorTorque(),
                joint.getMotorSpeed(),
            );
            break;
        case "spherical":
            values.push(
                joint.isConeLimitEnabled(),
                joint.getConeLimit(),
                joint.isTwistLimitEnabled(),
                joint.getLowerTwistLimit(),
                joint.getUpperTwistLimit(),
                joint.isSpringEnabled(),
                joint.getTargetRotation(),
                joint.getSpringHertz(),
                joint.getSpringDampingRatio(),
                joint.isMotorEnabled(),
                joint.getMotorVelocity(),
                joint.getMaxMotorTorque(),
            );
            break;
        case "weld":
            values.push(
                joint.getLinearHertz(),
                joint.getLinearDampingRatio(),
                joint.getAngularHertz(),
                joint.getAngularDampingRatio(),
            );
            break;
        case "wheel":
            values.push(
                joint.isSuspensionEnabled(),
                joint.getSuspensionHertz(),
                joint.getSuspensionDampingRatio(),
                joint.isSuspensionLimitEnabled(),
                joint.getLowerSuspensionLimit(),
                joint.getUpperSuspensionLimit(),
                joint.isSpinMotorEnabled(),
                joint.getSpinMotorSpeed(),
                joint.getMaxSpinTorque(),
                joint.isSteeringEnabled(),
                joint.getSteeringHertz(),
                joint.getSteeringDampingRatio(),
                joint.getMaxSteeringTorque(),
                joint.isSteeringLimitEnabled(),
                joint.getLowerSteeringLimit(),
                joint.getUpperSteeringLimit(),
                joint.getTargetSteeringAngle(),
            );
            break;
    }
    const record = joint.id.index1 - 1;
    values.push([
        readJointFloat(joint.world, record, J_CONSTRAINT_SOFTNESS),
        readJointFloat(joint.world, record, J_CONSTRAINT_SOFTNESS + 1),
        readJointFloat(joint.world, record, J_CONSTRAINT_SOFTNESS + 2),
    ]);
    return flatten(values);
}

function create(
    kind: Kind,
    world: PhysicsWorld,
    a: Parameters<PhysicsWorld["createFilterJoint"]>[0],
    b: Parameters<PhysicsWorld["createFilterJoint"]>[1],
): PublicJoint {
    switch (kind) {
        case "parallel":
            return world.createParallelJoint(a, b) as unknown as PublicJoint;
        case "distance":
            return world.createDistanceJoint(a, b) as unknown as PublicJoint;
        case "filter":
            return world.createFilterJoint(a, b) as unknown as PublicJoint;
        case "motor":
            return world.createMotorJoint(a, b) as unknown as PublicJoint;
        case "prismatic":
            return world.createPrismaticJoint(a, b) as unknown as PublicJoint;
        case "revolute":
            return world.createRevoluteJoint(a, b) as unknown as PublicJoint;
        case "spherical":
            return world.createSphericalJoint(a, b) as unknown as PublicJoint;
        case "weld":
            return world.createWeldJoint(a, b) as unknown as PublicJoint;
        case "wheel":
            return world.createWheelJoint(a, b) as unknown as PublicJoint;
    }
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
const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
try {
    const a = world.createBody({ type: BodyType.Dynamic });
    const b = world.createBody({ type: BodyType.Dynamic, position: { x: 2, y: 0, z: 0 } });
    for (const [kindIndex, kind] of kinds.entries()) {
        for (let sample = 0; sample <= 8; ++sample) {
            const joint = create(kind, world, a, b);
            if (sample !== 0) setValues(kind, joint, generated(kindIndex, sample));
            rows.push({
                kind,
                sample,
                actual: dump(kind, joint).map(hex),
                native: native.get(`${kindIndex}:${sample}`) ?? [],
            });
            joint.destroy(false);
        }
    }
} finally {
    world.destroy();
}

for (const kind of kinds) {
    test(`${kind} creation defaults equal b3Default*JointDef through public getters`, () => {
        const row = rows.find((entry) => entry.kind === kind && entry.sample === 0)!;
        expect(row.actual.slice(0, -3)).toEqual(row.native.slice(0, -3));
    });
    test(`${kind} seeded public setters equal Box3D setters`, () => {
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
