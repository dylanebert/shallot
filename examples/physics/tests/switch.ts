import { expect } from "bun:test";
import {
    Body,
    createApp,
    DistanceJoint,
    FilterJoint,
    GlobalTransform,
    hashPhysics,
    MotorJoint,
    MoverJoint,
    ParallelJoint,
    type Plugin,
    PogoJoint,
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
import { PhysicsEvents, PhysicsWorldDefinition } from "@dylanebert/shallot/standard/physics";
import { DeterminismPlugin } from "../src/determinism";
import { DynamicMoverPlugin, DynamicMoverState } from "../src/dynamic-mover-scene";
import { EventSample, EventSamplePlugin } from "../src/events";
import { Grab, GrabPlugin, pick } from "../src/grab";
import { sceneSwitcher } from "../src/sample";
import { changeControl, controlValues, createScene, type Scene } from "../src/scenes";
import { TreeBenchmarkPlugin, TreeBenchmarkState, treeRecords } from "../src/tree";
import { advanceIteration } from "./iteration";

function observations(world: World, controls: Record<string, number>) {
    return {
        sample: world.resource(EventSample),
        grab: world.resource(Grab),
        controls,
        dynamic: world.resource(DynamicMoverState).active
            ? world.resource(DynamicMoverState)
            : null,
        tree: world.resource(TreeBenchmarkState).active ? treeRecords(world) : null,
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
            MoverJoint,
            PogoJoint,
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

/** Switches from a changed, grabbed Bridge to each destination and holds it equal to a fresh
 * scene at construction and after each sample iteration in the requested range (end exclusive).
 * Earlier iterations run without observation; the default holds all 600 iterations. */
export async function expectSwitchesEqualFresh(
    destinations: readonly Scene[],
    plugins: Plugin[] = [],
    iterations = { start: 0, end: 600 },
): Promise<void> {
    const app = await createApp({
        defaults: false,
        plugins: [
            StandardPhysicsPlugin,
            EventSamplePlugin,
            DeterminismPlugin,
            TreeBenchmarkPlugin,
            DynamicMoverPlugin,

            ...plugins,
            { name: "GrabRecovery", recovery: GrabPlugin.recovery },
        ],
        // createScene's single-thread request is read only at build, which this app passes empty.
        setup(world) {
            world.resource(PhysicsWorldDefinition).workerCount = 1;
            world.resource(PhysicsWorldDefinition).maximumMemory = 4 * 1024 ** 3;
        },
    });
    try {
        const world = app.world;
        const select = sceneSwitcher(world);
        for (const scene of destinations) {
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
                plugins: [
                    StandardPhysicsPlugin,
                    EventSamplePlugin,
                    DeterminismPlugin,
                    TreeBenchmarkPlugin,
                    DynamicMoverPlugin,

                    ...plugins,
                ],
                setup(world) {
                    createScene(world, scene);
                },
            });
            try {
                expect(observations(world, switchedControls), `${scene} construction`).toEqual(
                    observations(fresh.world, freshControls),
                );
                for (let iteration = 0; iteration < iterations.end; iteration++) {
                    advanceIteration(world, scene);
                    advanceIteration(fresh.world, scene);
                    if (iteration < iterations.start) continue;
                    const hash = hashPhysics(world);
                    expect(hash, `${scene} iteration ${iteration} fresh`).toBe(
                        hashPhysics(fresh.world),
                    );
                    expect(
                        observations(world, switchedControls),
                        `${scene} iteration ${iteration} gameplay`,
                    ).toEqual(observations(fresh.world, freshControls));
                }
            } finally {
                fresh.dispose();
            }
        }
    } finally {
        app.dispose();
    }
}
