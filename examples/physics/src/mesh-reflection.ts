import {
    Body,
    BodyType,
    Hulls,
    PhysicsMeshes,
    type Plugin,
    recoverResource,
    Shape,
    ShapeKind,
    ShapeMaterials,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { StandardPhysicsPlugin } from "@dylanebert/shallot/standard/physics";
import buildingData from "./building.json" with { type: "json" };
import { human } from "./human";
import { decodeMesh } from "./mesh-data";
import type { Controls } from "./scenes";

const building = decodeMesh(buildingData);
const f = Math.fround;
export const ReflectionState = { create: () => ({ shape: 0 }) };
export const ReflectionPlugin: Plugin = {
    name: "Reflection",
    dependencies: [StandardPhysicsPlugin],
    recovery: recoverResource(ReflectionState),
};
function construct(world: World): number[] {
    const ground = world.create();
    world.add(ground, Body);
    world.add(ground, Shape, {
        kind: ShapeKind.Mesh,
        geometry: world.resource(PhysicsMeshes).add({
            name: "grid",
            grid: {
                xCount: 20,
                zCount: 20,
                cellWidth: 2,
                materialCount: 2,
                identifyEdges: true,
            },
        }),
    });
    const geometry = world.resource(PhysicsMeshes).add({
        name: "building",
        ...building,
        useMedianSplit: false,
        identifyEdges: true,
        weldVertices: true,
        weldTolerance: 0.002,
    });
    const materialSet = world.resource(ShapeMaterials).add({
        name: "building",
        materials: [
            { friction: 0.6, restitution: 0, userMaterialId: 0n },
            { friction: 0, restitution: 0.95, userMaterialId: 1n },
            { friction: 0.2, restitution: 0.2, userMaterialId: 2n },
        ].map((m) => ({
            ...m,
            rollingResistance: 0,
            tangentVelocity: [0, 0, 0] as [number, number, number],
            customColor: 0,
        })),
    });
    for (const x of [-10, 10]) {
        const eid = world.create();
        world.add(eid, Transform, { translation: [x, 0, 0] });
        world.add(eid, Body);
        world.add(eid, Shape, {
            kind: ShapeKind.Mesh,
            geometry,
            materialSet,
            scale: [x === 10 ? -1 : 1, 1, 1],
        });
        if (x === 10) world.resource(ReflectionState).shape = eid;
    }
    const bodies: number[] = [];
    for (let i = 0; i < 3; i++) {
        const eid = world.create();
        bodies.push(eid);
        world.add(eid, Transform, { translation: [6 + 3 * i, 15, 0] });
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.add(eid, Shape, {
            rollingResistance: 0.2,
            userMaterialIdLow: [42, 11, 555][i]!,
            ...(i === 0
                ? { kind: ShapeKind.Sphere, radius: 0.5 }
                : i === 1
                  ? {
                        kind: ShapeKind.Capsule,
                        capsuleCenter1: [-0.5, 0.5, 0] as [number, number, number],
                        capsuleCenter2: [0.5, 0, 0] as [number, number, number],
                        radius: 0.25,
                    }
                  : {
                        geometry: world
                            .resource(Hulls)
                            .add({ name: "box", halfExtents: [0.25, 0.5, 0.75] }),
                    }),
        });
    }
    for (let i = 0; i < 20; i++)
        bodies.push(...human(world, [f(-14 + f(1.5 * i)), 8, 0], i).bodies);
    return bodies;
}
const controls: Controls = Object.fromEntries(
    ["X", "Y", "Z"].map((axis, index) => [
        `Scale ${axis}`,
        {
            label: `Scale ${axis}`,
            value: index === 0 ? -1 : 1,
            select: true,
            options: [
                { label: "Neg", value: -1 },
                { label: "Pos", value: 1 },
            ],
            set(world: World, _bodies: readonly number[], value: number) {
                const eid = world.resource(ReflectionState).shape;
                const scale = world.get(eid, Shape)!.scale;
                scale[index] = value;
                world.storage(Shape).scale.set(eid, ...scale);
            },
        },
    ]),
);
export const reflectionScene = {
    construct,
    controls,
    gridExtent: 20,
    gridSpacing: 2,
    camera: { yaw: 35, pitch: 25, distance: 48, phoneDistance: 78, target: [0, 5, 0] },
};
