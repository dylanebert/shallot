import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { type Component, createApp, Time } from "@dylanebert/shallot";
import * as core from "@dylanebert/shallot/physics";
import { physicsWorld, StandardPhysicsPlugin } from "@dylanebert/shallot/standard/physics";
import { CEILING } from "../../../scripts/test-tiers";
import {
    DJ_LOWER_SPRING_FORCE,
    DJ_UPPER_SPRING_FORCE,
    J_CONSTRAINT_HERTZ,
    J_FORCE_THRESHOLD,
    J_LOCAL_FRAME_A,
    J_LOCAL_FRAME_B,
    PJ_HERTZ,
    PJ_LOWER_TRANSLATION,
    PJ_TARGET_TRANSLATION,
    PJ_UPPER_TRANSLATION,
    PLJ_HERTZ,
    RJ_LOWER_ANGLE,
    RJ_TARGET_ANGLE,
    RJ_UPPER_ANGLE,
    SJ_CONE_ANGLE,
    SJ_MOTOR_VELOCITY,
    SJ_TARGET_ROTATION,
    WHJ_LOWER_STEERING_LIMIT,
    WHJ_LOWER_SUSPENSION_LIMIT,
    WHJ_UPPER_STEERING_LIMIT,
    WHJ_UPPER_SUSPENSION_LIMIT,
    WJ_ANGULAR_HERTZ,
} from "./kernel/joint-layout";
import { readJointFloat, readJointQuat, readJointVec3 } from "./kernel/jointcolumns";
import { jointIds } from "./solver/joint.fixture";

setDefaultTimeout(CEILING.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

test("values asserted against by Box3D create warn once and skip, while positive infinite weld hertz is accepted", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const warning = spyOn(console, "warn").mockImplementation(() => {});
    const world = app.world;
    try {
        const a = world.create();
        const b = world.create();
        world.add(a, core.Body);
        world.add(b, core.Body);
        world.step(Time.FIXED_DT);
        const cases = [
            ["Filter", { localAnchorA: [Infinity, 0, 0, 0] }, "localFrameA"],
            ["Motor", { localRotationB: [0, 0, 0, 0] }, "localFrameB"],
            ["Distance", { length: 0 }, "length"],
            ["Distance", { length: Infinity }, "length"],
            [
                "Distance",
                { lowerSpringForce: 2, upperSpringForce: 1 },
                "lowerSpringForce/upperSpringForce",
            ],
            ["Parallel", { hertz: -1 }, "hertz"],
            ["Parallel", { dampingRatio: Infinity }, "dampingRatio"],
            ["Parallel", { maxTorque: NaN }, "maxTorque"],
            [
                "Prismatic",
                { lowerTranslation: 1, upperTranslation: 0 },
                "lowerTranslation/upperTranslation",
            ],
            ["Spherical", { coneAngle: Math.PI }, "coneAngle"],
            ["Spherical", { targetRotation: [0, 0, 0, 1.00001] }, "targetRotation"],
            ["Weld", { angularHertz: NaN }, "angularHertz"],
            ["Weld", { angularDampingRatio: -1 }, "angularDampingRatio"],
            ["Weld", { linearHertz: -1 }, "linearHertz"],
            ["Weld", { linearDampingRatio: -1 }, "linearDampingRatio"],
            [
                "Wheel",
                { lowerSuspensionLimit: 1, upperSuspensionLimit: 0 },
                "lowerSuspensionLimit/upperSuspensionLimit",
            ],
        ] as const;
        for (const [kind, values, field] of cases) {
            const component = (core as unknown as Record<string, Component>)[`${kind}Joint`];
            const eid = world.create();
            world.add(eid, component, { a, b, ...values });
            warning.mockClear();
            world.step(Time.FIXED_DT);
            world.step(Time.FIXED_DT);
            expect(warning).toHaveBeenCalledTimes(1);
            expect(String(warning.mock.calls[0]![0])).toContain(`${kind}Joint ${eid}`);
            expect(String(warning.mock.calls[0]![0])).toContain(field);
            expect(jointIds(physicsWorld(world)!.state)).toHaveLength(0);
            world.destroy(eid);
            world.step(Time.FIXED_DT);
        }
        const infinite = world.create();
        world.add(infinite, core.WeldJoint, { a, b, angularHertz: Infinity });
        warning.mockClear();
        world.step(Time.FIXED_DT);
        expect(warning).not.toHaveBeenCalled();
        expect(jointIds(physicsWorld(world)!.state)).toHaveLength(1);
        world.destroy(infinite);
        world.step(Time.FIXED_DT);

        const eid = world.create();
        world.add(eid, core.WeldJoint, { a, b, angularHertz: 2 });
        world.step(Time.FIXED_DT);
        expect(jointIds(physicsWorld(world)!.state)).toHaveLength(1);
        world.storage(core.WeldJoint).angularHertz.set(eid, NaN);
        world.step(Time.FIXED_DT);
        expect(warning).toHaveBeenCalledTimes(1);
        world.storage(core.WeldJoint).angularHertz.set(eid, NaN);
        world.step(Time.FIXED_DT);
        expect(warning).toHaveBeenCalledTimes(1);
        const ids = jointIds(physicsWorld(world)!.state);
        expect(ids).toHaveLength(1);
        expect(readJointFloat(physicsWorld(world)!.state, ids[0]!, WJ_ANGULAR_HERTZ)).toBe(2);
    } finally {
        warning.mockRestore();
        app.dispose();
    }
});

test("live joint edits reject Box3D setter precondition failures", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const warning = spyOn(console, "warn").mockImplementation(() => {});
    const world = app.world;
    try {
        const a = world.create();
        const b = world.create();
        world.add(a, core.Body);
        world.add(b, core.Body, { type: core.BodyType.Dynamic });
        const distance = world.create();
        world.add(distance, core.DistanceJoint, {
            a,
            b,
            lowerSpringForce: -1,
            upperSpringForce: 1,
        });
        const filter = world.create();
        world.add(filter, core.FilterJoint, { a, b });
        const parallel = world.create();
        world.add(parallel, core.ParallelJoint, { a, b });
        const spherical = world.create();
        world.add(spherical, core.SphericalJoint, { a, b });
        const prismatic = world.create();
        world.add(prismatic, core.PrismaticJoint, { a, b });
        const revolute = world.create();
        world.add(revolute, core.RevoluteJoint, { a, b });
        const wheel = world.create();
        world.add(wheel, core.WheelJoint, { a, b });
        world.step(Time.FIXED_DT);

        const state = physicsWorld(world)!.state;
        const id = (eid: number) => {
            const found = jointIds(state).find((joint) => state.jointUserData[joint] === eid);
            if (found === undefined) throw new Error(`joint ${eid} did not spawn`);
            return found;
        };
        const ids = {
            distance: id(distance),
            filter: id(filter),
            parallel: id(parallel),
            spherical: id(spherical),
            prismatic: id(prismatic),
            revolute: id(revolute),
            wheel: id(wheel),
        };
        const before = {
            distanceForce: [
                readJointFloat(state, ids.distance, DJ_LOWER_SPRING_FORCE),
                readJointFloat(state, ids.distance, DJ_UPPER_SPRING_FORCE),
            ],
            forceThreshold: readJointFloat(state, ids.distance, J_FORCE_THRESHOLD),
            constraintHertz: readJointFloat(state, ids.distance, J_CONSTRAINT_HERTZ),
            frameAX: readJointFloat(state, ids.filter, J_LOCAL_FRAME_A),
            frameBW: readJointFloat(state, ids.filter, J_LOCAL_FRAME_B + 6),
            parallelHertz: readJointFloat(state, ids.parallel, PLJ_HERTZ),
            sphericalTarget: readJointQuat(state, ids.spherical, SJ_TARGET_ROTATION),
            sphericalVelocity: readJointVec3(state, ids.spherical, SJ_MOTOR_VELOCITY),
            sphericalCone: readJointFloat(state, ids.spherical, SJ_CONE_ANGLE),
            prismaticHertz: readJointFloat(state, ids.prismatic, PJ_HERTZ),
            prismaticTarget: readJointFloat(state, ids.prismatic, PJ_TARGET_TRANSLATION),
            prismaticLimits: [
                readJointFloat(state, ids.prismatic, PJ_LOWER_TRANSLATION),
                readJointFloat(state, ids.prismatic, PJ_UPPER_TRANSLATION),
            ],
            revoluteTarget: readJointFloat(state, ids.revolute, RJ_TARGET_ANGLE),
            revoluteLimits: [
                readJointFloat(state, ids.revolute, RJ_LOWER_ANGLE),
                readJointFloat(state, ids.revolute, RJ_UPPER_ANGLE),
            ],
            wheelSuspension: [
                readJointFloat(state, ids.wheel, WHJ_LOWER_SUSPENSION_LIMIT),
                readJointFloat(state, ids.wheel, WHJ_UPPER_SUSPENSION_LIMIT),
            ],
            wheelSteering: [
                readJointFloat(state, ids.wheel, WHJ_LOWER_STEERING_LIMIT),
                readJointFloat(state, ids.wheel, WHJ_UPPER_STEERING_LIMIT),
            ],
        };

        warning.mockClear();
        const distanceFields = world.storage(core.DistanceJoint);
        distanceFields.forceThreshold.set(distance, -1);
        distanceFields.constraintHertz.set(distance, NaN);
        distanceFields.lowerSpringForce.set(distance, 2);
        distanceFields.upperSpringForce.set(distance, 1);
        const filterFields = world.storage(core.FilterJoint);
        filterFields.localAnchorA.set(filter, Infinity, 0, 0, 0);
        filterFields.localRotationB.set(filter, 0, 0, 0, 2);
        world.storage(core.ParallelJoint).hertz.set(parallel, -1);
        const sphericalFields = world.storage(core.SphericalJoint);
        sphericalFields.targetRotation.set(spherical, 0, 0, 0, 2);
        sphericalFields.motorVelocity.set(spherical, NaN, 0, 0, 0);
        sphericalFields.coneAngle.set(spherical, Math.PI);
        const prismaticFields = world.storage(core.PrismaticJoint);
        prismaticFields.hertz.set(prismatic, -1);
        prismaticFields.targetTranslation.set(prismatic, NaN);
        prismaticFields.lowerTranslation.set(prismatic, NaN);
        const revoluteFields = world.storage(core.RevoluteJoint);
        revoluteFields.targetAngle.set(revolute, Math.PI * 1.1);
        revoluteFields.upperAngle.set(revolute, Infinity);
        const wheelFields = world.storage(core.WheelJoint);
        wheelFields.lowerSuspensionLimit.set(wheel, 2);
        wheelFields.upperSuspensionLimit.set(wheel, 1);
        wheelFields.lowerSteeringLimit.set(wheel, 2);
        wheelFields.upperSteeringLimit.set(wheel, 1);
        world.step(Time.FIXED_DT);
        world.step(Time.FIXED_DT);

        expect(warning).toHaveBeenCalledTimes(16);
        expect([
            readJointFloat(state, ids.distance, DJ_LOWER_SPRING_FORCE),
            readJointFloat(state, ids.distance, DJ_UPPER_SPRING_FORCE),
        ]).toEqual(before.distanceForce);
        expect(readJointFloat(state, ids.distance, J_FORCE_THRESHOLD)).toBe(before.forceThreshold);
        expect(readJointFloat(state, ids.distance, J_CONSTRAINT_HERTZ)).toBe(
            before.constraintHertz,
        );
        expect(readJointFloat(state, ids.filter, J_LOCAL_FRAME_A)).toBe(before.frameAX);
        expect(readJointFloat(state, ids.filter, J_LOCAL_FRAME_B + 6)).toBe(before.frameBW);
        expect(readJointFloat(state, ids.parallel, PLJ_HERTZ)).toBe(before.parallelHertz);
        expect(readJointQuat(state, ids.spherical, SJ_TARGET_ROTATION)).toEqual(
            before.sphericalTarget,
        );
        expect(readJointVec3(state, ids.spherical, SJ_MOTOR_VELOCITY)).toEqual(
            before.sphericalVelocity,
        );
        expect(readJointFloat(state, ids.spherical, SJ_CONE_ANGLE)).toBe(before.sphericalCone);
        expect(readJointFloat(state, ids.prismatic, PJ_HERTZ)).toBe(before.prismaticHertz);
        expect(readJointFloat(state, ids.prismatic, PJ_TARGET_TRANSLATION)).toBe(
            before.prismaticTarget,
        );
        expect([
            readJointFloat(state, ids.prismatic, PJ_LOWER_TRANSLATION),
            readJointFloat(state, ids.prismatic, PJ_UPPER_TRANSLATION),
        ]).toEqual(before.prismaticLimits);
        expect(readJointFloat(state, ids.revolute, RJ_TARGET_ANGLE)).toBe(before.revoluteTarget);
        expect([
            readJointFloat(state, ids.revolute, RJ_LOWER_ANGLE),
            readJointFloat(state, ids.revolute, RJ_UPPER_ANGLE),
        ]).toEqual(before.revoluteLimits);
        expect([
            readJointFloat(state, ids.wheel, WHJ_LOWER_SUSPENSION_LIMIT),
            readJointFloat(state, ids.wheel, WHJ_UPPER_SUSPENSION_LIMIT),
        ]).toEqual(before.wheelSuspension);
        expect([
            readJointFloat(state, ids.wheel, WHJ_LOWER_STEERING_LIMIT),
            readJointFloat(state, ids.wheel, WHJ_UPPER_STEERING_LIMIT),
        ]).toEqual(before.wheelSteering);
    } finally {
        warning.mockRestore();
        app.dispose();
    }
});
