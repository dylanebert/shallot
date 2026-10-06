import {
    Body,
    createApp,
    DistanceJoint,
    SphericalJoint,
    StandardPhysicsPlugin,
} from "@dylanebert/shallot";
import { physicsWorld } from "@dylanebert/shallot/standard/physics";
import { PhysicsWorld } from "../api/world";
import { BodyType } from "../common/types";
import { makeBoxHull } from "../shapes/hull";

export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};

const COUNT = 64;
// Pendulums 8 m apart reach at most 3 m from their anchors, so no pair makes a contact.
const SPACING = 8;

// `authored`: DistanceJoint and SphericalJoint entities alternate, each swinging a bob from a static anchor through the
// plugin's sync. `spherical`: solver-API spherical joints with a `localFrameB`, each on a static anchor.
export default async function create(input: string) {
    if (input === "authored") return authored();
    if (input === "spherical") return spherical();
    if (input === "force") return forced();
    throw new Error(`constraint allocation has no scene ${input}`);
}

async function forced() {
    const solver = new PhysicsWorld({ enableSleep: false, enableContinuous: false });
    const body = solver.createBody({ type: BodyType.Dynamic });
    body.createHull({}, makeBoxHull(0.25, 0.25, 0.25));
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
                            body.applyForce(force, point, false);
                            body.applyForceToCenter(force, false);
                            body.applyTorque(force, false);
                            solver.step(1 / 60, 4);
                        },
                    },
                ],
            },
        ],
    });
    return {
        step: () => app.world.step(1 / 60),
        wait: () => app.world.gpu.device.queue.onSubmittedWorkDone(),
        dispose: () => {
            app.dispose();
            solver.destroy();
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
        const bob = world.create();
        world.add(bob, Body, { type: BodyType.Dynamic, position: [i * SPACING + 2.5, 10, 0, 0] });
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
        wait: () => world.gpu.device.queue.onSubmittedWorkDone(),
        dispose: () => {
            // A bob that slept would have left the joint column, so its windows would prove nothing.
            const physics = physicsWorld(world)!;
            const asleep = bobs.filter((bob) => !physics.getBody(bob)!.isAwake()).length;
            app.dispose();
            if (asleep > 0) throw new Error(`${asleep} authored pendulums fell asleep`);
        },
    };
}

async function spherical() {
    const solver = new PhysicsWorld({ enableSleep: false, enableContinuous: false });
    const hull = makeBoxHull(0.25, 0.25, 0.25);
    for (let i = 0; i < COUNT; i++) {
        const anchor = solver.createBody({ position: { x: i * SPACING, y: 10, z: 0 } });
        const bob = solver.createBody({
            type: BodyType.Dynamic,
            position: { x: i * SPACING + 2, y: 10, z: 0 },
        });
        bob.createHull({}, hull);
        solver.createSphericalJoint(anchor, bob, {
            localFrameB: { p: { x: -2, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
        });
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
                        update: () => solver.step(1 / 60, 4),
                    },
                ],
            },
        ],
    });
    return {
        step: () => app.world.step(1 / 60),
        wait: () => app.world.gpu.device.queue.onSubmittedWorkDone(),
        dispose: () => {
            app.dispose();
            solver.destroy();
        },
    };
}
