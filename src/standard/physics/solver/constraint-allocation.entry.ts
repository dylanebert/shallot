import {
    Body,
    createApp,
    DistanceJoint,
    Shape,
    SphericalJoint,
    StandardPhysicsPlugin,
} from "@dylanebert/shallot";
import { BodyType } from "../common/types";
import { init, shutdown } from "../kernel/kernel";
import { quaternion } from "../kernel/math";
import { physicsBodyIndex, physicsState } from "../runtime";
import { makeBoxHull } from "../shapes/build-hull";
import { createHullShape } from "../shapes/shape";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "../testing";
import {
    bodyApplyAngularImpulse,
    bodyApplyForce,
    bodyApplyForceToCenter,
    bodyApplyLinearImpulse,
    bodyApplyLinearImpulseToCenter,
    bodyApplyTorque,
    bodyIsAwake,
    bodySetTargetTransform,
} from "../world/body";
import { defaultJointDef } from "./joint";
import {
    createPrismaticJoint,
    defaultPrismaticJointDef,
    prismaticJointSpeed,
} from "./prismaticJoint";
import { createSphericalJoint, defaultSphericalJointDef } from "./sphericalJoint";

export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};

const COUNT = 64;
const SPACING = 8;

export default async function create(input: string) {
    if (input === "authored") return authored();
    if (input === "spherical") return spherical();
    if (input === "force") return forced();
    if (
        [
            "target",
            "target-return",
            "impulse-linear",
            "impulse-center",
            "impulse-angular",
            "prismatic-read",
        ].includes(input)
    )
        return bodyReads(input);
    throw new Error(`constraint allocation has no scene ${input}`);
}

async function solverWorld() {
    await init(undefined, { threads: 0 });
    return createTestWorld({ enableSleep: false, enableContinuous: false });
}

async function bodyReads(input: string) {
    const solver = await solverWorld();
    const fixed = createTestBody(solver);
    const body = createTestBody(solver, {
        type: BodyType.Dynamic,
        rotation:
            input === "target"
                ? quaternion.fromAxisAngle({ x: 0, y: 1, z: 0 }, 0.4)
                : quaternion.identity(),
        linearVelocity: { x: 1, y: 2, z: 3 },
        angularVelocity: { x: 4, y: 5, z: 6 },
    });
    createHullShape(solver, body, testShapeDef(), makeBoxHull(0.25, 0.25, 0.25));
    const joint = createPrismaticJoint(
        solver,
        defaultPrismaticJointDef({ ...defaultJointDef(), bodyIdA: fixed, bodyIdB: body }),
    ).joint;
    const sleeping = createTestBody(solver, { type: BodyType.Dynamic, isAwake: false });
    const stationary = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
    const target = {
        p: { x: 0.1, y: 0.2, z: 0.3 },
        q: quaternion.fromAxisAngle({ x: 1, y: 0, z: 0 }, 0.7),
    };
    const negativeTarget = { p: target.p, q: quaternion.negate(target.q) };
    const impulse = { x: 0.1, y: 0.2, z: 0.3 };
    const point = { x: 1, y: 0, z: 0 };
    const app = await createApp({
        defaults: false,
        plugins: [
            {
                name: "body-read-allocation",
                systems: [
                    {
                        name: "body-read",
                        group: "fixed",
                        update: () => {
                            if (input === "target") {
                                bodySetTargetTransform(solver, body, target, 1 / 60, false);
                                bodySetTargetTransform(solver, body, negativeTarget, 1 / 60, false);
                            } else if (input === "target-return") {
                                bodySetTargetTransform(solver, fixed, target, 1 / 60, false);
                                bodySetTargetTransform(solver, body, target, 0, false);
                                bodySetTargetTransform(solver, sleeping, stationary, 1 / 60, false);
                                bodySetTargetTransform(solver, sleeping, stationary, 1 / 60, true);
                            } else if (input === "impulse-linear") {
                                bodyApplyLinearImpulse(solver, body, impulse, point, false);
                                bodyApplyLinearImpulse(solver, fixed, impulse, point, false);
                            } else if (input === "impulse-center") {
                                bodyApplyLinearImpulseToCenter(solver, body, impulse, false);
                                bodyApplyLinearImpulseToCenter(solver, fixed, impulse, false);
                            } else if (input === "impulse-angular") {
                                bodyApplyAngularImpulse(solver, body, impulse, false);
                                bodyApplyAngularImpulse(solver, fixed, impulse, false);
                            } else if (input === "prismatic-read") {
                                if (prismaticJointSpeed(solver, joint) !== 1)
                                    throw new Error("prismatic speed changed");
                            }
                        },
                    },
                ],
            },
        ],
    });
    return {
        step: () => app.world.step(1 / 60),
        dispose: async () => {
            const sleepingStayedAsleep = !bodyIsAwake(solver, sleeping);
            app.dispose();
            destroyTestWorld(solver);
            await shutdown(undefined);
            if (!sleepingStayedAsleep)
                throw new Error("target-transform early return woke the sleeping body");
        },
    };
}

async function forced() {
    const solver = await solverWorld();
    const body = createTestBody(solver, { type: BodyType.Dynamic });
    createHullShape(solver, body, testShapeDef(), makeBoxHull(0.25, 0.25, 0.25));
    const force = { x: 1, y: 2, z: 3 };
    const point = { x: 1, y: 0, z: 0 };
    const app = await createApp({
        defaults: false,
        plugins: [
            {
                name: "force-allocation",
                systems: [
                    {
                        name: "force-step",
                        group: "fixed",
                        update: () => {
                            bodyApplyForce(solver, body, force, point, false);
                            bodyApplyForceToCenter(solver, body, force, false);
                            bodyApplyTorque(solver, body, force, false);
                            stepTestWorld(solver, 1 / 60, 4);
                        },
                    },
                ],
            },
        ],
    });
    return {
        step: () => app.world.step(1 / 60),
        dispose: async () => {
            app.dispose();
            destroyTestWorld(solver);
            await shutdown(undefined);
        },
    };
}

async function authored() {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const world = app.world;
    const bobs: number[] = [];
    for (let i = 0; i < COUNT; i++) {
        const anchor = world.create();
        world.add(anchor, Body, { position: [i * SPACING, 10, 0, 0] });
        world.add(anchor, Shape);
        const bob = world.create();
        world.add(bob, Body, { type: BodyType.Dynamic, position: [i * SPACING + 2.5, 10, 0, 0] });
        world.add(bob, Shape);
        bobs.push(bob);
        if (i % 2 === 0)
            world.add(world.create(), DistanceJoint, {
                a: anchor,
                b: bob,
                enableSpring: 1,
                length: 2.5,
                hertz: 2,
                dampingRatio: 1,
            });
        else
            world.add(world.create(), SphericalJoint, {
                a: anchor,
                b: bob,
                localAnchorB: [-2.5, 0, 0, 0],
            });
    }
    return {
        step: () => world.step(1 / 60),
        dispose: () => {
            const state = physicsState(world)!;
            const asleep = bobs.filter((eid) => {
                const body = physicsBodyIndex(world, eid);
                return body >= 0 && !bodyIsAwake(state, body);
            }).length;
            app.dispose();
            if (asleep > 0) throw new Error(`${asleep} authored pendulums fell asleep`);
        },
    };
}

async function spherical() {
    const solver = await solverWorld();
    const hull = makeBoxHull(0.25, 0.25, 0.25);
    for (let i = 0; i < COUNT; i++) {
        const anchor = createTestBody(solver, { position: { x: i * SPACING, y: 10, z: 0 } });
        const bob = createTestBody(solver, {
            type: BodyType.Dynamic,
            position: { x: i * SPACING + 2, y: 10, z: 0 },
        });
        createHullShape(solver, bob, testShapeDef(), hull);
        createSphericalJoint(
            solver,
            defaultSphericalJointDef({
                ...defaultJointDef(),
                bodyIdA: anchor,
                bodyIdB: bob,
                localFrameB: {
                    p: { x: -2, y: 0, z: 0 },
                    q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
                },
            }),
        );
    }
    const app = await createApp({
        defaults: false,
        plugins: [
            {
                name: "constraint-allocation",
                systems: [
                    {
                        name: "constraint-step",
                        group: "fixed",
                        update: () => stepTestWorld(solver, 1 / 60, 4),
                    },
                ],
            },
        ],
    });
    return {
        step: () => app.world.step(1 / 60),
        dispose: async () => {
            app.dispose();
            destroyTestWorld(solver);
            await shutdown(undefined);
        },
    };
}
