import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Body,
    createApp,
    DistanceJoint,
    FilterJoint,
    GlobalTransform,
    hashPhysics,
    MotorJoint,
    ParallelJoint,
    PrismaticJoint,
    RevoluteJoint,
    Shape,
    SphericalJoint,
    StandardPhysicsPlugin,
    Transform,
    WeldJoint,
    WheelJoint,
    type World,
} from "@dylanebert/shallot";
import { PhysicsEvents } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { EventSample, EventSamplePlugin } from "../src/events";
import { Grab, GrabPlugin, pick } from "../src/grab";
import { sceneSwitcher } from "../src/sample";
import { changeControl, controlValues, createScene, type Scene, scenes } from "../src/scenes";
import { sampleGold } from "./gold";

setDefaultTimeout(TIMEOUT.node);

function observations(world: World, controls: Record<string, number>) {
    return {
        sample: world.resource(EventSample),
        grab: world.resource(Grab),
        controls,
        events: Object.fromEntries(
            Object.entries(world.resource(PhysicsEvents)).map(([name, channel]) => [
                name,
                channel.records.slice(0, channel.count),
            ]),
        ),
        authored: [
            Body,
            Shape,
            Transform,
            DistanceJoint,
            FilterJoint,
            MotorJoint,
            ParallelJoint,
            PrismaticJoint,
            RevoluteJoint,
            SphericalJoint,
            WeldJoint,
            WheelJoint,
        ].map((component) =>
            [...world.query([component])]
                .sort((a, b) => a - b)
                .map((eid) => [eid, world.get(eid, component)]),
        ),
    };
}

test("every switched scene equals a fresh scene through its recorded run after a changed, grabbed Bridge", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [
            StandardPhysicsPlugin,
            EventSamplePlugin,
            { name: "GrabRecovery", recovery: GrabPlugin.recovery },
        ],
    });
    try {
        const world = app.world;
        const select = sceneSwitcher(world);
        for (const scene of Object.keys(scenes) as Scene[]) {
            const bodies = select("bridge");
            const values = controlValues("bridge");
            changeControl(world, "bridge", bodies, values, "Gravity scale", -0.5);
            world.tick();
            const grab = world.resource(Grab);
            const p = world.storage(GlobalTransform).translation;
            grab.origin = {
                x: p.x.get(bodies[0]!),
                y: p.y.get(bodies[0]!),
                z: p.z.get(bodies[0]!) + 10,
            };
            grab.translation = { x: 0, y: 0, z: -20 };
            pick(world);
            expect(grab.body).not.toBe(0);
            grab.held = true;
            const sample = world.resource(EventSample);
            sample.hits.push({} as never);
            sample.keys.W = 1;
            sample.time = 123;
            select(scene);
            const switchedControls = controlValues(scene);
            const freshControls = controlValues(scene);
            const fresh = await createApp({
                defaults: false,
                plugins: [StandardPhysicsPlugin, EventSamplePlugin],
                setup(world) {
                    createScene(world, scene);
                },
            });
            try {
                expect(observations(world, switchedControls), `${scene} construction`).toEqual(
                    observations(fresh.world, freshControls),
                );
                const gold = sampleGold(
                    await Bun.file(new URL(`./${scene}.gold.txt`, import.meta.url)).text(),
                );
                for (const [step, expected] of gold.entries()) {
                    world.tick();
                    fresh.world.tick();
                    expect(hashPhysics(world), `${scene} step ${step} native`).toBe(expected.hash);
                    expect(hashPhysics(world), `${scene} step ${step} fresh`).toBe(
                        hashPhysics(fresh.world),
                    );
                    expect(
                        observations(world, switchedControls),
                        `${scene} step ${step} gameplay`,
                    ).toEqual(observations(fresh.world, freshControls));
                }
            } finally {
                fresh.dispose();
            }
        }
    } finally {
        app.dispose();
    }
});
