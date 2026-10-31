import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Body,
    BodyType,
    MoverJoint,
    PogoJoint,
    Shape,
    ShapeKind,
    Transform,
} from "@dylanebert/shallot";
import {
    getBodyTransform,
    getPogoJointCurrentLength,
    getPogoJointCurrentSpeed,
    getPogoJointImpulse,
    PhysicsWorldDefinition,
    StandardPhysicsPlugin,
} from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import gold from "../../../crates/physics/tests/mover_pogo.gold.json";
import { createTestGpuApp as createApp, disposeGpuApps } from "../../../scripts/gpu.fixture";
import { GlobalTransform } from "../../core/transform";
import {
    MVJ_LINEAR_VELOCITY,
    MVJ_MAX_FORCE,
    PGJ_DAMPING_RATIO,
    PGJ_HERTZ,
    PGJ_IMPULSE,
    PGJ_MAX_COMPRESSION_FORCE,
    PGJ_MAX_TENSION_FORCE,
    PGJ_NORMAL,
    PGJ_REST_LENGTH,
    PGJ_VELOCITY,
} from "./kernel/columns";
import { readJointFloat, readJointVec3 } from "./kernel/jointcolumns";
import { physicsState } from "./runtime";
import { jointIds } from "./solver/joint.fixture";

setDefaultTimeout(TIMEOUT.node);
await setupGlobals();
const bits = (value: number) => new Uint32Array(new Float32Array([value]).buffer)[0]!;
for (const c of gold) {
    test(`authored ${c.kind === 9 ? "MoverJoint" : "PogoJoint"}, mode ${c.mode}, follows native through limits and recreation`, async () => {
        const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
        try {
            const world = app.world;
            const definition = world.resource(PhysicsWorldDefinition);
            definition.gravity = { x: 0, y: 0, z: 0 };
            definition.enableSleep = false;
            definition.enableAutomaticStep = false;
            const a = world.create();
            world.add(a, Body);
            const b = world.create();
            world.add(b, Transform, { translation: [0, c.mode === 1 ? 2 : 0.5, 0] });
            world.add(b, Body, {
                type: BodyType.Dynamic,
                linearVelocity: [4, c.mode === 1 ? 4 : -4, -3],
            });
            world.add(world.create(), Shape, { body: b, kind: ShapeKind.Sphere, radius: 0.5 });
            const joint = world.create();
            const mover = { a, b, linearVelocity: [-6, 0, 8] as const, maxForce: 0.75 };
            const pogo = {
                a,
                b,
                restLength: 1,
                hertz: 4,
                dampingRatio: 0.7,
                maxTensionForce: 0.25,
                maxCompressionForce: 0.75,
                impulse: c.mode === 2 ? 0.03125 : 0,
                velocity: c.mode === 2 ? -0.875 : 0,
            };
            if (c.kind === 9) world.add(joint, MoverJoint, mover);
            else world.add(joint, PogoJoint, pogo);
            expect(getPogoJointImpulse(world, joint)).toBeNull();
            world.tick();
            if (c.kind === 10) {
                expect(getPogoJointImpulse(world, joint)).toBe(pogo.impulse);
                expect(getPogoJointCurrentSpeed(world, joint)).toBe(pogo.velocity);
                expect(getPogoJointCurrentLength(world, joint)).toBe(c.mode === 1 ? 2 : 0.5);
            }
            definition.enableAutomaticStep = true;
            for (let i = 0; i < c.samples.length; ++i) {
                if (i && c.mode === 2) {
                    if (c.kind === 9) {
                        world.remove(joint, MoverJoint);
                        world.add(joint, MoverJoint, mover);
                    } else {
                        const impulse = getPogoJointImpulse(world, joint)!,
                            velocity = getPogoJointCurrentSpeed(world, joint)!;
                        world.remove(joint, PogoJoint);
                        expect(getPogoJointImpulse(world, joint)).toBeNull();
                        world.add(joint, PogoJoint, { ...pogo, impulse, velocity });
                    }
                }
                world.tick();
                const pose = getBodyTransform(world, b)!,
                    v = world.storage(GlobalTransform).linearVelocity;
                expect(
                    [pose.p.x, pose.p.y, pose.p.z, v.x.get(b), v.y.get(b), v.z.get(b)].map(bits),
                    `step ${i} motion`,
                ).toEqual(c.samples[i]!.slice(0, 6));
                if (c.kind === 10)
                    expect(
                        [
                            bits(getPogoJointImpulse(world, joint)!),
                            bits(getPogoJointCurrentSpeed(world, joint)!),
                        ],
                        `step ${i} carried state`,
                    ).toEqual(c.samples[i]!.slice(9));
            }
        } finally {
            await disposeGpuApps([app]);
        }
    });
}

test("mover and pogo live edits use their setters while carried impulse and velocity remain spawn-only", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        world.resource(PhysicsWorldDefinition).enableAutomaticStep = false;
        const a = world.create(),
            b = world.create();
        world.add(a, Body);
        world.add(b, Body, { type: BodyType.Dynamic });
        const mover = world.create(),
            pogo = world.create();
        world.add(mover, MoverJoint, { a, b });
        world.add(pogo, PogoJoint, { a, b, impulse: 0.125, velocity: -0.25 });
        world.tick();
        const state = physicsState(world)!;
        const id = (eid: number) =>
            jointIds(state).find((j) => state.jointUserData[j] === world.ref(eid))!;
        const m = id(mover),
            p = id(pogo);
        world.storage(MoverJoint).linearVelocity.set(mover, 2, -3, 4);
        world.storage(MoverJoint).maxForce.set(mover, 7);
        const ps = world.storage(PogoJoint);
        ps.normal.set(pogo, 1, 0, 0);
        ps.restLength.set(pogo, 2);
        ps.hertz.set(pogo, 5);
        ps.dampingRatio.set(pogo, 0.5);
        ps.maxTensionForce.set(pogo, 6);
        ps.maxCompressionForce.set(pogo, 8);
        ps.impulse.set(pogo, 99);
        ps.velocity.set(pogo, 99);
        expect(readJointFloat(state, m, MVJ_MAX_FORCE)).toBe(0);
        world.tick();
        expect(readJointFloat(state, m, MVJ_MAX_FORCE)).toBe(7);
        for (const [column, value] of [
            [PGJ_REST_LENGTH, 2],
            [PGJ_HERTZ, 5],
            [PGJ_DAMPING_RATIO, 0.5],
            [PGJ_MAX_TENSION_FORCE, 6],
            [PGJ_MAX_COMPRESSION_FORCE, 8],
            [PGJ_IMPULSE, 0.125],
            [PGJ_VELOCITY, -0.25],
        ])
            expect(readJointFloat(state, p, column!)).toBe(value!);
        expect(readJointVec3(state, m, MVJ_LINEAR_VELOCITY)).toEqual({ x: 2, y: -3, z: 4 });
        expect(readJointVec3(state, p, PGJ_NORMAL)).toEqual({ x: 1, y: 0, z: 0 });
        const image = world.snapshot();
        ps.restLength.set(pogo, 3);
        world.tick();
        world.restore(image);
        expect(getPogoJointImpulse(world, pogo)).toBe(0.125);
        expect(getPogoJointCurrentSpeed(world, pogo)).toBe(-0.25);
        world.remove(pogo, PogoJoint);
        expect(getPogoJointCurrentLength(world, pogo)).toBeNull();
    } finally {
        await disposeGpuApps([app]);
    }
});
