import {
    Body,
    BodyType,
    GlobalTransform,
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
    isAwake,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
} from "@dylanebert/shallot/standard/physics";
import { human } from "./human";
import { SceneRestart } from "./scene-restart";
import type { Controls } from "./scenes";

const f = Math.fround;
type V3 = [number, number, number];
export const WorldSampleState = {
    create: () => ({
        offset: 0,
        top: -1,
        bodies: [] as number[],
        meshDrop: false,
        steps: 0,
        failed: false,
        moving: false,
    }),
};
export const WorldSamplePlugin: Plugin = {
    name: "WorldSample",
    dependencies: [StandardPhysicsPlugin],
    recovery: recoverResource(WorldSampleState),
    systems: [
        {
            name: "WorldMotion",
            group: "fixed",
            before: [StepPhysicsSystem],
            update(world) {
                const s = world.resource(WorldSampleState);
                if (s.meshDrop) s.moving = s.bodies.some((e) => isAwake(world, e));
            },
        },
        {
            name: "WorldFailure",
            group: "fixed",
            after: [StepPhysicsSystem],
            update(world) {
                const s = world.resource(WorldSampleState);
                if (s.meshDrop && ++s.steps >= 300 && s.moving) s.failed = true;
            },
        },
    ],
};
function box(world: World, halfExtents: V3) {
    return world.resource(Hulls).add({ name: "box", halfExtents });
}
function body(world: World, translation: V3, dynamic = false) {
    const eid = world.create();
    world.add(eid, Transform, { translation });
    world.add(eid, Body, { type: dynamic ? BodyType.Dynamic : BodyType.Static });
    return eid;
}
function ground(world: World, x: number, extent: number) {
    const eid = body(world, [x, -1, 0]);
    world.add(eid, Shape, { geometry: box(world, [extent, 1, extent]) });
}
function farStack(world: World, offset = 0): number[] {
    const s = world.resource(WorldSampleState);
    s.offset = offset;
    const x = f(1000 * offset);
    ground(world, x, 12);
    const geometry = box(world, [0.5, 0.5, 0.5]);
    const bodies: number[] = [];
    for (let i = 0; i < 6; i++) {
        const eid = body(world, [f(x + f(0.02 * (i & 1 ? 1 : -1))), 0.5 + i, 0], true);
        world.add(eid, Shape, { geometry });
        bodies.push(eid);
        s.top = eid;
    }
    return bodies;
}
function farPyramid(world: World): number[] {
    world.resource(WorldSampleState).offset = 10000;
    ground(world, 10000000, 400);
    const geometry = box(world, [0.5, 0.5, 0.5]);
    const bodies: number[] = [];
    for (let i = 0; i < 40; i++)
        for (let j = i; j < 40; j++) {
            const x = (i + 1) * 0.5 + (j - i) - 20;
            const eid = body(world, [f(10000000 + x), (2 * i + 1) * 0.5, 0], true);
            world.add(eid, Shape, { geometry, density: 100 });
            bodies.push(eid);
        }
    return bodies;
}
function farRagdolls(world: World): number[] {
    world.resource(WorldSampleState).offset = 1000;
    const eid = body(world, [1000000, -1, 0]);
    world.add(eid, Shape, {
        kind: ShapeKind.Mesh,
        geometry: world.resource(PhysicsMeshes).add({
            name: "grid",
            grid: {
                xCount: 20,
                zCount: 20,
                cellWidth: 1,
                materialCount: 1,
                identifyEdges: true,
            },
        }),
    });
    const bodies: number[] = [];
    for (let i = 0; i < 20; i++)
        bodies.push(
            ...human(
                world,
                [f(1000000 + f(f(0.15) * (i - 10))), 2 + 0.25 * i, f(f(0.15) * (10 - i))],
                i,
                10,
                0.5,
                0.7,
            ).bodies,
        );
    return bodies;
}
function farMeshDrop(world: World): number[] {
    const s = world.resource(WorldSampleState);
    s.offset = 1000;
    s.meshDrop = true;
    const ground = body(world, [1000000, 0, 1000000]);
    world.add(ground, Shape, {
        kind: ShapeKind.Mesh,
        filterCategoryLow: 1,
        filterCategoryHigh: 0,
        geometry: world.resource(PhysicsMeshes).add({
            name: "wave",
            wave: {
                xCount: 40,
                zCount: 40,
                cellWidth: 1,
                amplitude: 0.5,
                rowFrequency: 0.1,
                columnFrequency: 0.2,
            },
        }),
    });
    let seed = 3963634789;
    const random = (lo: number, hi: number) => {
        seed ^= seed << 13;
        seed ^= seed >>> 17;
        seed ^= seed << 5;
        seed >>>= 0;
        return f(f(f(hi - lo) * f((seed % 32768) / 32767)) + lo);
    };
    const geometry = box(world, [0.02, 0.2, 0.04]);
    for (let i = 0; i < 20; i++)
        for (let j = 0; j < 20; j++) {
            const eid = body(world, [1000000 + 0.5 * (i - 10), 5, 1000000 + 0.5 * (j - 10)], true);
            world
                .storage(Body)
                .linearVelocity.set(eid, random(-1, 1), random(-1, 1), random(-1, 1));
            world
                .storage(Body)
                .angularVelocity.set(eid, random(-5, 5), random(-5, 5), random(-5, 5));
            world.add(eid, Shape, {
                geometry,
                rollingResistance: 0.1,
                filterCategoryLow: 2,
                filterCategoryHigh: 0,
                filterMaskLow: 1,
                filterMaskHigh: 0,
            });
            s.bodies.push(eid);
        }
    return s.bodies;
}
const offset = {
    label: "Offset, km",
    value: 0,
    decimals: 1,
    read(world: World) {
        return world.resource(WorldSampleState).offset;
    },
    set() {},
};
const stackControls: Controls = Object.fromEntries(
    [
        ["origin", 0],
        ["10km", 10],
        ["100km", 100],
        ["1000km", 1000],
        ["10000km", 10000],
    ].map(([label, value]) => [
        label,
        {
            label: String(label),
            value: 0,
            button: true,
            set(world: World) {
                world.resource(SceneRestart).restart((w) => farStack(w, Number(value)));
            },
        },
    ]),
);
export const worldScenes = {
    farStack: {
        construct: farStack,
        gridExtent: 12,
        gridSpacing: 1,
        camera: { yaw: 25, pitch: 20, distance: 12, phoneDistance: 18, target: [0, 2.5, 0] },
        controls: {
            ...stackControls,
            Offset: offset,
            Height: {
                label: "Top height, m",
                value: 0,
                decimals: 4,
                read(world: World) {
                    return world
                        .storage(GlobalTransform)
                        .translation.y.get(world.resource(WorldSampleState).top);
                },
                set() {},
            },
        },
    },
    farPyramid: {
        construct: farPyramid,
        gridExtent: 400,
        gridSpacing: 5,
        gridCenter: [10000000, 0],
        camera: { yaw: 15, pitch: 20, distance: 70, phoneDistance: 135, target: [10000000, 18, 0] },
        controls: { Offset: { ...offset, decimals: 0 } },
    },
    farRagdolls: {
        construct: farRagdolls,
        gridExtent: 10,
        gridSpacing: 1,
        gridCenter: [1000000, 0],
        gridHeight: -1,
        camera: { yaw: 35, pitch: 25, distance: 18, phoneDistance: 24, target: [1000000, 2.5, 0] },
        controls: { Offset: { ...offset, decimals: 0 } },
    },
    farMeshDrop: {
        construct: farMeshDrop,
        gridExtent: 20,
        gridSpacing: 1,
        gridCenter: [1000000, 1000000],
        camera: {
            yaw: 25,
            pitch: 35,
            distance: 24,
            phoneDistance: 38,
            target: [1000000, 1, 1000000],
        },
        controls: {
            Offset: { ...offset, decimals: 0 },
            Failure: {
                label: "Result",
                value: 0,
                read(world: World) {
                    return world.resource(WorldSampleState).failed ? "failed!" : null;
                },
                set() {},
            },
        },
    },
} as const;
