import { afterAll, beforeAll, expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { type Component, type FieldType, Time, type World } from "@dylanebert/shallot";
import * as physics from "@dylanebert/shallot/physics";
import { StandardPhysicsPlugin } from "@dylanebert/shallot/standard/physics";
import { createTestGpuApp as createApp } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { JointField, jointField } from "./kernel/jointrecords";
import { kernel } from "./kernel/kernel";
import { physicsBodyIndex, physicsState } from "./runtime";
import { defaultDistanceJointDef } from "./solver/distanceJoint";
import { defaultJointDef, JointType } from "./solver/joint";
import { jointIds } from "./solver/joint.fixture";
import { defaultMotorJointDef } from "./solver/motorJoint";
import { defaultParallelJointDef } from "./solver/parallelJoint";
import { defaultPrismaticJointDef } from "./solver/prismaticJoint";
import { defaultRevoluteJointDef } from "./solver/revoluteJoint";
import { defaultSphericalJointDef } from "./solver/sphericalJoint";
import { defaultWeldJointDef } from "./solver/weldJoint";
import { defaultWheelJointDef } from "./solver/wheelJoint";

setDefaultTimeout(CEILING.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();
let app: Awaited<ReturnType<typeof createApp>>;
beforeAll(async () => {
    app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
});
afterAll(() => app?.dispose());

const kinds = [
    ["Distance", defaultDistanceJointDef],
    ["Filter", (base: ReturnType<typeof defaultJointDef>) => ({ base })],
    ["Motor", defaultMotorJointDef],
    ["Parallel", defaultParallelJointDef],
    ["Prismatic", defaultPrismaticJointDef],
    ["Revolute", defaultRevoluteJointDef],
    ["Spherical", defaultSphericalJointDef],
    ["Weld", defaultWeldJointDef],
    ["Wheel", defaultWheelJointDef],
] as const;

type Values = Record<string, number | readonly [number, number, number, number]>;
function authored(config: Record<string, unknown>): Values {
    const out: Values = {};
    for (const [name, value] of Object.entries(config)) {
        if (name === "userData" || name === "bodyIdA" || name === "bodyIdB") continue;
        if (name === "localFrameA" || name === "localFrameB") {
            const frame = value as {
                p: { x: number; y: number; z: number };
                q: { v: { x: number; y: number; z: number }; s: number };
            };
            const suffix = name.at(-1);
            out[`localAnchor${suffix}`] = [frame.p.x, frame.p.y, frame.p.z, 0];
            out[`localRotation${suffix}`] = [frame.q.v.x, frame.q.v.y, frame.q.v.z, frame.q.s];
        } else if (typeof value === "boolean") out[name] = Number(value);
        else if (typeof value === "number") out[name] = value;
        else {
            const v = value as {
                x: number;
                y: number;
                z: number;
                v?: { x: number; y: number; z: number };
                s?: number;
            };
            out[name] = v.v ? [v.v.x, v.v.y, v.v.z, v.s!] : [v.x, v.y, v.z, 0];
        }
    }
    return out;
}
function kernelJointArguments(
    kind: string,
    config: Record<string, unknown>,
    bodyIds?: readonly [number, number],
): unknown[] {
    const number = (name: string) => config[name] as number;
    const boolean = (name: string) => config[name] as boolean;
    const vector = (name: string) => {
        const value = config[name] as { x: number; y: number; z: number };
        return [value.x, value.y, value.z];
    };
    const quaternion = (name: string) => {
        const value = config[name] as { v: { x: number; y: number; z: number }; s: number };
        return [value.v.x, value.v.y, value.v.z, value.s];
    };
    switch (kind) {
        case "Distance":
            return [
                number("length"),
                number("hertz"),
                number("dampingRatio"),
                number("lowerSpringForce"),
                number("upperSpringForce"),
                number("minLength"),
                number("maxLength"),
                number("maxMotorForce"),
                number("motorSpeed"),
                boolean("enableSpring"),
                boolean("enableLimit"),
                boolean("enableMotor"),
            ];
        case "Filter": {
            const frame = (name: string) => {
                const value = config[name] as {
                    p: { x: number; y: number; z: number };
                    q: { v: { x: number; y: number; z: number }; s: number };
                };
                return [
                    value.p.x,
                    value.p.y,
                    value.p.z,
                    value.q.v.x,
                    value.q.v.y,
                    value.q.v.z,
                    value.q.s,
                ];
            };
            return [
                bodyIds![0],
                bodyIds![1],
                JointType.Filter,
                number("drawScale"),
                Number(boolean("collideConnected")),
                ...frame("localFrameA"),
                ...frame("localFrameB"),
                number("forceThreshold"),
                number("torqueThreshold"),
                number("constraintHertz"),
                number("constraintDampingRatio"),
            ];
        }
        case "Motor":
            return [
                ...vector("linearVelocity"),
                number("maxVelocityForce"),
                ...vector("angularVelocity"),
                number("maxVelocityTorque"),
                number("linearHertz"),
                number("linearDampingRatio"),
                number("maxSpringForce"),
                number("angularHertz"),
                number("angularDampingRatio"),
                number("maxSpringTorque"),
            ];
        case "Parallel":
            return [number("hertz"), number("dampingRatio"), number("maxTorque")];
        case "Prismatic":
            return [
                number("hertz"),
                number("dampingRatio"),
                number("targetTranslation"),
                number("lowerTranslation"),
                number("upperTranslation"),
                number("maxMotorForce"),
                number("motorSpeed"),
                boolean("enableSpring"),
                boolean("enableLimit"),
                boolean("enableMotor"),
            ];
        case "Revolute":
            return [
                number("hertz"),
                number("dampingRatio"),
                number("targetAngle"),
                number("lowerAngle"),
                number("upperAngle"),
                number("maxMotorTorque"),
                number("motorSpeed"),
                boolean("enableSpring"),
                boolean("enableLimit"),
                boolean("enableMotor"),
            ];
        case "Spherical":
            return [
                number("hertz"),
                number("dampingRatio"),
                ...quaternion("targetRotation"),
                number("coneAngle"),
                number("lowerTwistAngle"),
                number("upperTwistAngle"),
                number("maxMotorTorque"),
                ...vector("motorVelocity"),
                boolean("enableSpring"),
                boolean("enableConeLimit"),
                boolean("enableTwistLimit"),
                boolean("enableMotor"),
            ];
        case "Weld":
            return [
                number("linearHertz"),
                number("linearDampingRatio"),
                number("angularHertz"),
                number("angularDampingRatio"),
            ];
        case "Wheel":
            return [
                boolean("enableSuspensionSpring"),
                number("suspensionHertz"),
                number("suspensionDampingRatio"),
                boolean("enableSuspensionLimit"),
                number("lowerSuspensionLimit"),
                number("upperSuspensionLimit"),
                boolean("enableSpinMotor"),
                number("maxSpinTorque"),
                number("spinSpeed"),
                boolean("enableSteering"),
                number("steeringHertz"),
                number("steeringDampingRatio"),
                number("targetSteeringAngle"),
                number("maxSteeringTorque"),
                boolean("enableSteeringLimit"),
                number("lowerSteeringLimit"),
                number("upperSteeringLimit"),
            ];
    }
    throw new Error(`unknown joint kind ${kind}`);
}

function read(world: World, component: Component, eid: number): Values {
    const out: Values = {};
    const storage = world.storage(component) as unknown as Record<
        string,
        { get(eid: number): number; read(eid: number, out: Float32Array): Float32Array }
    >;
    for (const [name, type] of Object.entries(component)) {
        out[name] =
            (type as FieldType).lanes === 1
                ? storage[name].get(eid)
                : (Array.from(storage[name].read(eid, new Float32Array(4))) as [
                      number,
                      number,
                      number,
                      number,
                  ]);
    }
    return out;
}
for (const [kind, defaults] of kinds) {
    test(`${kind}Joint defaults equal Box3D and every off-default field reaches its matching solver create`, () => {
        const component = (physics as unknown as Record<string, Component>)[`${kind}Joint`];
        expect(component, `${kind}Joint is authored by core`).toBeDefined();
        const world = app.world;
        const state = physicsState(world)!;
        const { base, ...specific } = defaults(defaultJointDef());
        const expected = { ...base, ...specific } as Record<string, unknown>;
        const eid = world.create();
        const a = world.create();
        const b = world.create();
        try {
            world.add(eid, component);
            const values = read(world, component, eid);
            expect(Object.keys(values).sort()).toEqual(
                ["a", "b", ...Object.keys(authored(expected))].sort(),
            );
            for (const [field, value] of Object.entries(authored(expected)))
                expect(values[field], `${kind}.${field} default`).toEqual(value);
            expect(values.a).toBe(0);
            expect(values.b).toBe(0);
            for (const [field, value] of Object.entries(expected)) {
                if (field === "userData" || field.startsWith("bodyId")) continue;
                if (typeof value === "boolean") expected[field] = !value;
                else if (typeof value === "number")
                    expected[field] = field.startsWith("lower") ? -0.5 : value === 0 ? 0.5 : 2;
                else if (field === "localFrameA")
                    expected[field] = {
                        p: { x: 1, y: 2, z: 3 },
                        q: { v: { x: 0.5, y: 0.5, z: 0.5 }, s: 0.5 },
                    };
                else if (field === "localFrameB")
                    expected[field] = {
                        p: { x: -1, y: -2, z: -3 },
                        q: { v: { x: 0.5, y: -0.5, z: 0.5 }, s: -0.5 },
                    };
                else if (field === "targetRotation")
                    expected[field] = { v: { x: -0.5, y: 0.5, z: 0.5 }, s: 0.5 };
                else
                    expected[field] =
                        field === "angularVelocity"
                            ? { x: -1, y: -2, z: -3 }
                            : { x: 1, y: 2, z: 3 };
            }
            world.add(a, physics.Body);
            world.add(b, physics.Body);
            const authoredValues = { ...authored(expected), a, b };
            world.remove(eid, component);
            world.add(eid, component, authoredValues);
            const method = kind === "Filter" ? "jointCreate" : `jointCreate${kind}`;
            const k = kernel(state.ecsState);
            const methods = k as unknown as Record<string, (...args: unknown[]) => number>;
            const original = methods[method].bind(k);
            let calls = 0;
            const spy = spyOn(methods, method).mockImplementation((...args: unknown[]) => {
                calls++;
                if (kind === "Filter") {
                    expect(args, `${kind} solver arguments`).toEqual(
                        kernelJointArguments(kind, expected, [
                            physicsBodyIndex(world, a),
                            physicsBodyIndex(world, b),
                        ]),
                    );
                } else {
                    expect(args.slice(2), `${kind} solver arguments`).toEqual(
                        kernelJointArguments(kind, expected),
                    );
                }
                return original(...args);
            });
            try {
                world.step(Time.FIXED_DT);
            } finally {
                spy.mockRestore();
            }
            expect(calls).toBe(1);
            const joint = jointIds(state).find((id) => state.jointUserData[id] === world.ref(eid))!;
            expect(joint, `${kind} between two static bodies is created`).toBeDefined();
            expect(jointField(state, joint, JointField.type)).toBe(JointType[kind]);
        } finally {
            world.destroy(eid);
            world.destroy(a);
            world.destroy(b);
            world.step(Time.FIXED_DT);
        }
    });
}
