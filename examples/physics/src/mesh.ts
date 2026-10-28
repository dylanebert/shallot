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
import { rotationFromAxisAngle, StandardPhysicsPlugin } from "@dylanebert/shallot/standard/physics";
import voxelData from "./collision_mesh_01.json" with { type: "json" };
import { decodeMesh } from "./mesh-data";
import { HeightFieldSamplePlugin, heightFieldScene } from "./mesh-height-field";
import { ReflectionPlugin, reflectionScene } from "./mesh-reflection";
import type { Controls } from "./scenes";

const f = Math.fround;
type V3 = [number, number, number];
type ShapeScene = "gridMesh" | "bigBoxMesh" | "boxMesh";
// Box3D's collision_mesh_01.obj, LoadTempMesh at scale 0.01 with Z up.
const voxelMesh = decodeMesh(voxelData);
export const MeshSampleState = {
    create: () => ({ scene: "gridMesh" as ShapeScene, body: 0, groundShape: 0, cylinder: 0 }),
};
export const MeshSamplePlugin: Plugin = {
    name: "MeshSample",
    dependencies: [StandardPhysicsPlugin, ReflectionPlugin, HeightFieldSamplePlugin],
    recovery: recoverResource(MeshSampleState),
};
function boxMesh(world: World, center: V3, extent: V3, hollow = false): number {
    const [x, y, z] = extent;
    const vertices = [
        [x, y, z],
        [-x, y, z],
        [-x, -y, z],
        [x, -y, z],
        [x, y, -z],
        [-x, y, -z],
        [-x, -y, -z],
        [x, -y, -z],
    ];
    const indices = hollow
        ? [
              3, 1, 0, 3, 2, 1, 1, 4, 0, 5, 4, 1, 7, 3, 0, 7, 0, 4, 5, 7, 4, 7, 5, 6, 2, 5, 1, 5, 2,
              6, 7, 2, 3, 2, 7, 6,
          ]
        : [
              0, 1, 3, 1, 2, 3, 0, 4, 1, 1, 4, 5, 0, 3, 7, 4, 0, 7, 4, 7, 5, 6, 5, 7, 1, 5, 2, 6, 2,
              5, 3, 2, 7, 6, 7, 2,
          ];
    return world.resource(PhysicsMeshes).add({
        name: hollow ? "hollow box" : "box mesh",
        vertices: new Float32Array(vertices.flatMap((p) => p.map((v, i) => f(v + center[i]!)))),
        indices: new Uint32Array(indices),
        identifyEdges: true,
    });
}
function ground(world: World, extent: number): void {
    const eid = world.create();
    world.add(eid, Transform, { translation: [0, -1, 0] });
    world.add(eid, Body);
    world.add(eid, Shape, {
        geometry: world.resource(Hulls).add({ name: "ground", halfExtents: [extent, 1, extent] }),
    });
}
function spawn(world: World, type: number): number {
    const state = world.resource(MeshSampleState);
    if (state.body) world.destroy(state.body);
    const eid = world.create();
    state.body = eid;
    const box = state.scene === "boxMesh";
    const big = state.scene === "bigBoxMesh";
    world.add(eid, Transform, {
        translation: box ? [0, type === 3 ? 1 : 1.5, 0] : big ? [0.5, 0, 0] : [0.1, 1, -0.1],
    });
    world.add(eid, Body, {
        type: BodyType.Dynamic,
        angularDamping: !box && !big && type === 3 ? 0.1 : 0,
    });
    const rollingResistance =
        big || (!box && (type === 0 || type === 1)) ? 0.05 : !box && type === 3 ? 0.02 : 0;
    if (type === 0)
        world.add(eid, Shape, { kind: ShapeKind.Sphere, radius: 0.5, rollingResistance });
    else if (type === 1) {
        const materials = big
            ? world.resource(ShapeMaterials).add({
                  name: "capsule",
                  materials: [
                      {
                          friction: 0.6,
                          restitution: 0,
                          rollingResistance: 0.1,
                          tangentVelocity: [0, 0, 0],
                          userMaterialId: 0n,
                          customColor: 0,
                      },
                  ],
              })
            : -1;
        world.add(eid, Shape, {
            kind: ShapeKind.Capsule,
            capsuleCenter1: box ? [-0.5, 0, 0] : [0, 0, 1.276],
            capsuleCenter2: box ? [0.5, 0, 0] : [0, 0, 0.476],
            radius: box ? 0.1 : 0.15,
            rollingResistance,
            materialSet: materials,
        });
    } else
        world.add(eid, Shape, {
            geometry:
                type === 3
                    ? state.cylinder
                    : world.resource(Hulls).add({ name: "box", halfExtents: [0.5, 0.5, 0.5] }),
            rollingResistance,
        });
    return eid;
}
function construct(world: World, scene: ShapeScene): number[] {
    const state = world.resource(MeshSampleState);
    Object.assign(state, MeshSampleState.create(), { scene });
    if (scene === "boxMesh") ground(world, 20);
    const eid = world.create();
    if (scene === "boxMesh")
        world.add(eid, Transform, {
            translation: [0, -1, 0],
            rotation: rotationFromAxisAngle(world, { x: 0, y: 1, z: 0 }, f(0.25 * f(Math.PI))),
        });
    world.add(eid, Body);
    const geometry =
        scene === "gridMesh"
            ? world.resource(PhysicsMeshes).add({
                  name: "grid",
                  grid: {
                      xCount: 20,
                      zCount: 20,
                      cellWidth: 1,
                      materialCount: 0,
                      identifyEdges: true,
                  },
              })
            : boxMesh(
                  world,
                  scene === "boxMesh" ? [0, 1, 0] : [0, -1, 0],
                  scene === "boxMesh" ? [1, 1, 1] : [50, 1, 50],
              );
    world.add(eid, Shape, {
        kind: ShapeKind.Mesh,
        geometry,
        scale: scene === "gridMesh" ? [2, 2, 2] : [1, 1, 1],
        friction: scene === "bigBoxMesh" ? 0.5 : 0.6,
    });
    state.groundShape = eid;
    state.cylinder = world.resource(Hulls).add({
        name: "cylinder",
        cylinder:
            scene === "gridMesh"
                ? { height: 1, radius: 0.25, yOffset: 0, sides: 15 }
                : scene === "bigBoxMesh"
                  ? { height: 0.3, radius: 0.15, yOffset: 0, sides: 32 }
                  : { height: 1, radius: 0.75, yOffset: 0, sides: 8 },
    });
    return [spawn(world, scene === "boxMesh" ? 2 : 3)];
}
function controls(type: number, scale = 1): Controls {
    return {
        Shape: {
            label: "Shape",
            value: type,
            select: true,
            options: ["Sphere", "Capsule", "Box", "Cylinder"].map((label, value) => ({
                label,
                value,
            })),
            set(world, _bodies, value) {
                spawn(world, value);
            },
        },
        ...Object.fromEntries(
            (
                [
                    ["Scale X", 0],
                    ["Scale Z", 2],
                ] as const
            ).map(([label, axis]) => [
                label,
                {
                    label,
                    value: scale,
                    min: -2,
                    max: 2,
                    step: 0.1,
                    decimals: 1,
                    set(world: World, _bodies: readonly number[], value: number) {
                        const shape = world.storage(Shape);
                        const eid = world.resource(MeshSampleState).groundShape;
                        const scale: V3 = [
                            shape.scale.x.get(eid),
                            shape.scale.y.get(eid),
                            shape.scale.z.get(eid),
                        ];
                        scale[axis] = value;
                        shape.scale.set(eid, ...scale);
                    },
                },
            ]),
        ),
    };
}
function hollowBox(world: World): number[] {
    const eid = world.create();
    world.add(eid, Body);
    world.add(eid, Shape, {
        kind: ShapeKind.Mesh,
        geometry: boxMesh(world, [0, 0, 0], [10, 10, 10], true),
    });
    const cylinder = world
        .resource(Hulls)
        .add({ name: "cylinder", cylinder: { height: 1, radius: 0.25, yOffset: 0, sides: 8 } });
    const bodies: number[] = [];
    for (const [capsule, positions] of [
        [
            false,
            [
                [0, -10.2, 0],
                [0, 9.2, 0],
                [-9.8, 0, 0],
                [9.8, 0, 0],
                [0, 0, -9.8],
                [0, 0, 9.8],
            ],
        ],
        [
            true,
            [
                [0, -10.2, 2],
                [0, 9.2, 2],
                [0, -9.9, 4],
                [0, 8.9, 4],
                [-9.8, 2, 0],
                [9.8, 2, 0],
                [0, 2, -9.8],
                [0, 2, 9.8],
            ],
        ],
    ] as [boolean, V3[]][])
        for (const translation of positions) {
            const body = world.create();
            bodies.push(body);
            world.add(body, Transform, { translation });
            world.add(body, Body, { type: BodyType.Dynamic, gravityScale: 0, enableSleep: 0 });
            world.add(
                body,
                Shape,
                capsule
                    ? {
                          kind: ShapeKind.Capsule,
                          capsuleCenter1: [0, 0, 0],
                          capsuleCenter2: [0, 1, 0],
                          radius: 0.25,
                      }
                    : { geometry: cylinder },
            );
        }
    return bodies;
}
function voxel(world: World): number[] {
    const ground = world.create();
    world.add(ground, Transform, { translation: [5000, 3500, -7000] });
    world.add(ground, Body);
    world.add(ground, Shape, {
        kind: ShapeKind.Mesh,
        geometry: world.resource(PhysicsMeshes).add({
            name: "voxel",
            ...voxelMesh,
            useMedianSplit: true,
            identifyEdges: true,
            weldVertices: true,
            weldTolerance: 0.002,
        }),
    });
    const points = [
        [-3.13548756, 3.81141949, 237.289047],
        [-16.2333279, -23.4977913, 235.486603],
        [-13.8834839, 6.20244455, 23.7760544],
        [14.0794125, 4.63170528, 24.9530792],
        [3.98322797, -16.4192238, 236.704071],
        [-23.3520412, -3.2671442, 236.071594],
        [13.451786, -6.94963741, 24.4085312],
        [-5.24953651, 13.9316301, 24.505806],
        [-4.65071201, -24.1484108, 235.974121],
        [-14.5111103, -5.37889385, 23.2315063],
        [6.33307076, 13.2810068, 24.993515],
        [4.81784487, -14.6788225, 23.6787796],
        [-14.7180958, 4.46204281, 236.801331],
        [-23.9796677, -14.8484812, 235.527039],
        [4.61085415, -4.83788204, 237.248611],
        [-6.76476669, -14.0281992, 23.1910706],
    ];
    const geometry = world.resource(Hulls).add({
        name: "cylinder",
        points: new Float32Array(
            points.flatMap(([x, y, z]) => [y!, z!, x!].map((v) => f(f(0.01) * f(v)))),
        ),
    });
    const body = world.create();
    world.add(body, Transform, {
        translation: [5020.27734, 3506.22559, -6986.48584],
        rotation: [0.664546967, 0.669287264, 0.135021493, 0.303646326],
    });
    world.add(body, Body, { type: BodyType.Dynamic });
    world.add(body, Shape, { geometry, rollingResistance: 0.1 });
    return [body];
}
export const meshScenes = {
    meshReflection: reflectionScene,
    meshHeightField: heightFieldScene,
    voxel: {
        construct: voxel,
        gridExtent: 40,
        gridSpacing: 4,
        gridHeight: 3500,
        gridCenter: [5000, -7000],
        camera: {
            yaw: 35,
            pitch: 30,
            distance: 18,
            phoneDistance: 28,
            target: [5020, 3505, -6986],
        },
        controls: undefined,
    },
    hollowBox: {
        construct: hollowBox,
        gridExtent: 10,
        gridSpacing: 1,
        gridHeight: -10,
        camera: { yaw: 35, pitch: 25, distance: 40, phoneDistance: 65, target: [0, 0, 0] },
        controls: undefined,
    },
    gridMesh: {
        construct: (world: World) => construct(world, "gridMesh"),
        gridExtent: 20,
        gridSpacing: 2,
        camera: { yaw: 35, pitch: 25, distance: 7, phoneDistance: 11, target: [0, 0.5, 0] },
        controls: controls(3, 2),
    },
    bigBoxMesh: {
        construct: (world: World) => construct(world, "bigBoxMesh"),
        gridExtent: 50,
        gridSpacing: 1,
        camera: { yaw: 35, pitch: 25, distance: 5, phoneDistance: 8, target: [0.5, 0.4, 0] },
        controls: controls(3),
    },
    boxMesh: {
        construct: (world: World) => construct(world, "boxMesh"),
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 35, pitch: 25, distance: 8, phoneDistance: 12, target: [0, 0.5, 0] },
        controls: controls(2),
    },
};
