import { afterAll, beforeAll, expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { type Component, createApp, type FieldType, Time, type World } from "@dylanebert/shallot";
import * as physics from "@dylanebert/shallot/physics";
import { physicsWorld, StandardPhysicsPlugin } from "@dylanebert/shallot/standard/physics";
import { CEILING } from "../../../scripts/test-tiers";
import { defaultDistanceJointDef } from "./solver/distanceJoint";
import { defaultJointDef, JointType } from "./solver/joint";
import { defaultMotorJointDef } from "./solver/motorJoint";
import { defaultParallelJointDef } from "./solver/parallelJoint";
import { defaultPrismaticJointDef } from "./solver/prismaticJoint";
import { defaultRevoluteJointDef } from "./solver/revoluteJoint";
import { defaultSphericalJointDef } from "./solver/sphericalJoint";
import { defaultWeldJointDef } from "./solver/weldJoint";
import { defaultWheelJointDef } from "./solver/wheelJoint";

setDefaultTimeout(CEILING.node);
const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
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
        const solver = physicsWorld(world)!;
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
            const method = `create${kind}Joint`;
            const methods = solver as unknown as Record<string, (...args: unknown[]) => unknown>;
            const original = methods[method].bind(solver);
            let calls = 0;
            const spy = spyOn(methods, method).mockImplementation((bodyA, bodyB, cfg) => {
                calls++;
                expect(bodyA).toBe(solver.getBody(a));
                expect(bodyB).toBe(solver.getBody(b));
                const actual = cfg as Record<string, unknown>;
                for (const [field, value] of Object.entries(expected)) {
                    if (field === "userData" || field.startsWith("bodyId")) continue;
                    expect(actual[field], `${kind}.${field} marshal`).toEqual(value);
                }
                expect(actual.userData).toBe(eid);
                return original(bodyA, bodyB, cfg);
            });
            try {
                world.step(Time.FIXED_DT);
            } finally {
                spy.mockRestore();
            }
            expect(calls).toBe(1);
            const joint = solver.state.joints.find((j) => j.userData === eid)!;
            expect(joint, `${kind} between two static bodies is created`).toBeDefined();
            expect(joint.type).toBe(JointType[kind]);
        } finally {
            world.destroy(eid);
            world.destroy(a);
            world.destroy(b);
            world.step(Time.FIXED_DT);
        }
    });
}
