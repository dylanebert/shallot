import {
    Body,
    BodyType,
    Hulls,
    PhysicsMeshes,
    type Plugin,
    recoverResource,
    Shape,
    ShapeKind,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    getMassData,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    SyncPhysicsConstraintsSystem,
    setLinearVelocity,
} from "@dylanebert/shallot/standard/physics";
import { SceneRestart } from "./scene-restart";
import type { Controls } from "./scenes";

const f = Math.fround;
const TreeSample = { create: () => ({ bodies: [] as number[], ready: false }) };
export const TreeSamplePlugin: Plugin = {
    name: "TreeSample",
    dependencies: [StandardPhysicsPlugin],
    systems: [
        {
            name: "tree-velocities",
            group: "fixed",
            after: [SyncPhysicsConstraintsSystem],
            before: [StepPhysicsSystem],
            update(world) {
                const sample = world.resource(TreeSample);
                if (sample.ready) return;
                for (let i = 0; i < sample.bodies.length; i++) {
                    const eid = sample.bodies[i]!;
                    const center = getMassData(world, eid)!.center;
                    const omega = f(f(0.5 + f(f(0.5 * i) / 50)) * (i % 2 ? 0.5 : -0.5));
                    setLinearVelocity(world, eid, {
                        x: f(-f(omega * center.y)),
                        y: f(omega * center.x),
                        z: 0,
                    });
                }
                sample.ready = true;
            },
        },
    ],
    recovery: recoverResource(TreeSample),
};

export function fallingTrees(world: World, gridSize = 100): number[] {
    const scale = 100 / gridSize;
    const geometry = world.resource(PhysicsMeshes).add({
        name: "terrain",
        wave: {
            xCount: scale * 150,
            zCount: scale * 200,
            cellWidth: 1 / scale,
            amplitude: f(0.4),
            rowFrequency: f(0.05),
            columnFrequency: f(0.1),
        },
    });
    const ground = world.create();
    world.add(ground, Body);
    world.add(ground, Transform);
    world.add(ground, Shape, { kind: ShapeKind.Mesh, geometry });
    const hulls: number[] = [];
    let y = 1,
        r = 0.75;
    for (let i = 0; i < 22; i++) {
        hulls.push(
            world.resource(Hulls).add({
                name: "trunk",
                cylinder: { height: f(1.5 + f(2 * r)), radius: r, yOffset: f(y - r), sides: 6 },
            }),
        );
        y = f(y + f(1.5 + f(2 * r)));
        r = f(f(0.95) * r);
    }
    const bodies: number[] = [];
    for (let i = 0; i < 50; i++) {
        const eid = world.create();
        const omega = f(f(0.5 + f(f(0.5 * i) / 50)) * (i % 2 ? 0.5 : -0.5));
        world.add(eid, Transform, { translation: [0, 1, -70 + 3 * i] });
        world.add(eid, Body, {
            type: BodyType.Dynamic,
            sleepThreshold: f(0.2),
            angularVelocity: [0, 0, omega],
        });
        for (const geometry of hulls) {
            const shape = world.create();
            world.add(shape, Shape, {
                body: eid,
                geometry,
                density: 1,
                friction: f(0.9),
                rollingResistance: f(0.05),
            });
        }
        bodies.push(eid);
    }
    Object.assign(world.resource(TreeSample), { bodies, ready: false });
    return bodies;
}
export const treeControls = {
    "Grid size": {
        label: "Grid size",
        value: 100,
        options: [
            { label: "100cm", value: 100 },
            { label: "50cm", value: 50 },
            { label: "25cm", value: 25 },
        ],
        set(world, _bodies, value) {
            if (![100, 50, 25].includes(value))
                throw new Error("Grid size must be 100, 50 or 25 cm");
            world.resource(SceneRestart).restart((world) => fallingTrees(world, value));
        },
    },
} satisfies Controls;
