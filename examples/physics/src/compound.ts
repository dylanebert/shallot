import {
    Body,
    BodyType,
    type Compound,
    Compounds,
    Hulls,
    PhysicsMeshes,
    Shape,
    ShapeKind,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    computeCosSin,
    PhysicsWorldDefinition,
    rotationFromAxisAngle,
} from "@dylanebert/shallot/standard/physics";
import boxMesh from "./compound-box.json" with { type: "json" };
import { decodeMesh } from "./mesh-data";

const f = Math.fround;
type V3 = [number, number, number];
const identity = (): [number, number, number, number] => [0, 0, 0, 1];
function random() {
    let seed = 12345;
    return (lo: number, hi: number) => {
        seed ^= seed << 13;
        seed ^= seed >>> 17;
        seed ^= seed << 5;
        seed >>>= 0;
        return f(f(f(f(hi) - f(lo)) * f((seed % 32768) / 32767)) + f(lo));
    };
}
function ground(
    world: World,
    definition: Compound,
    translation: V3 = [0, 0, 0],
    rotation = identity(),
): number {
    const geometry = world.resource(Compounds).add(definition);
    const eid = world.create();
    world.add(eid, Transform, { translation, rotation });
    world.add(eid, Body);
    world.add(eid, Shape, { kind: ShapeKind.Compound, geometry });
    return eid;
}
function sphere(world: World, translation: V3): number {
    const eid = world.create();
    world.add(eid, Transform, { translation });
    world.add(eid, Body, { type: BodyType.Dynamic });
    world.add(eid, Shape, { kind: ShapeKind.Sphere, radius: 0.25 });
    return eid;
}
function simple(world: World): number[] {
    const hull = world.resource(Hulls).add({ name: "box", halfExtents: [4, 0.5, 4] });
    ground(
        world,
        {
            name: "Simple",
            hulls: [{ hull, transform: { position: [1, -0.5, 0], rotation: identity() } }],
        },
        [2, -1, 0],
        rotationFromAxisAngle(world, { x: 0, y: 1, z: 0 }, f(0.25 * f(Math.PI))),
    );
    world.resource(PhysicsWorldDefinition).contactRecycleDistance = 0;
    return [sphere(world, [0, 2, 0])];
}
function spheres(world: World): number[] {
    const rand = random();
    const spheres: NonNullable<Compound["spheres"]> = [];
    for (let i = 0; i < 20; i++) {
        const center: V3 = [rand(-10, 10), rand(-10, 10), rand(-10, 10)];
        spheres.push({ center, radius: rand(f(f(0.01) * 10), f(f(0.05) * 10)) });
    }
    ground(world, { name: "Spheres", spheres });
    return [];
}
function hulls(world: World): number[] {
    const rand = random();
    const hulls: NonNullable<Compound["hulls"]> = [];
    for (let i = 0; i < 20; i++) {
        const lo = f(f(0.01) * 10),
            hi = f(f(0.05) * 10);
        const halfExtents: V3 = [rand(lo, hi), rand(lo, hi), rand(lo, hi)];
        const position: V3 = [rand(-10, 10), rand(-10, 10), rand(-10, 10)];
        const u1 = rand(0, 1),
            u2 = rand(0, f(2 * f(Math.PI))),
            u3 = rand(0, f(2 * f(Math.PI)));
        const a = f(Math.sqrt(f(1 - u1))),
            b = f(Math.sqrt(u1));
        const cs2 = computeCosSin(world, u2),
            cs3 = computeCosSin(world, u3);
        const rotation: [number, number, number, number] = [
            f(a * cs2[1]),
            f(a * cs2[0]),
            f(b * cs3[1]),
            f(b * cs3[0]),
        ];
        const hull = world.resource(Hulls).add({ name: "box", halfExtents });
        hulls.push({ hull, transform: { position, rotation } });
    }
    ground(world, { name: "Hulls", hulls });
    return [];
}
function tileFloor(world: World): number[] {
    const rand = random();
    const hull = world.resource(Hulls).add({ name: "tile", halfExtents: [4, 2, 4] });
    const hulls: NonNullable<Compound["hulls"]> = [];
    for (let i = 0; i < 50; i++)
        for (let j = 0; j < 50; j++) {
            hulls.push({
                hull,
                transform: {
                    position: [(2 * i - 50) * 4, f(rand(-0.5, 0.25) * 4), (2 * j - 50) * 4],
                    rotation: identity(),
                },
            });
        }
    ground(
        world,
        { name: "Tile Floor", hulls },
        [-2, 1, -3],
        rotationFromAxisAngle(world, { x: f(2 / 3), y: f(-2 / 3), z: f(1 / 3) }, 0),
    );
    return [sphere(world, [3, 12, 0])];
}
function meshTile(world: World): number[] {
    const rand = random();
    // Box3D b3CreateBoxMesh({0,0,0}, {4,2,4}, true).
    const mesh = world
        .resource(PhysicsMeshes)
        .add({ name: "tile", ...decodeMesh(boxMesh), identifyEdges: true });
    const meshes: NonNullable<Compound["meshes"]> = [];
    for (let i = 0; i < 2; i++)
        for (let j = 0; j < 2; j++) {
            meshes.push({
                mesh,
                transform: {
                    position: [(2 * i - 2) * 4, f(rand(-0.5, 0.25) * 4), (2 * j - 2) * 4],
                    rotation: identity(),
                },
                scale: [1, 1, 1],
            });
        }
    ground(world, { name: "Mesh Tile", meshes });
    return [];
}
export const compoundScenes = {
    simpleCompound: {
        construct: simple,
        gridExtent: 8,
        gridSpacing: 1,
        gridHeight: -1,
        camera: { yaw: 40, pitch: 25, distance: 15, phoneDistance: 23, target: [2.7, -0.5, -0.7] },
        controls: undefined,
    },
    compoundSpheres: {
        construct: spheres,
        gridExtent: 12,
        gridSpacing: 2,
        camera: { yaw: 40, pitch: 20, distance: 26, phoneDistance: 38, target: [0, 0, 0] },
        controls: undefined,
    },
    compoundHulls: {
        construct: hulls,
        gridExtent: 12,
        gridSpacing: 2,
        camera: { yaw: 40, pitch: 20, distance: 26, phoneDistance: 38, target: [0, 0, 0] },
        controls: undefined,
    },
    tileFloor: {
        construct: tileFloor,
        gridExtent: 208,
        gridSpacing: 8,
        camera: { yaw: 40, pitch: 35, distance: 45, target: [3, 3, 0] },
        controls: undefined,
    },
    meshTile: {
        construct: meshTile,
        gridExtent: 12,
        gridSpacing: 4,
        camera: { yaw: 40, pitch: 35, distance: 30, phoneDistance: 50, target: [-4, 0, -4] },
        controls: undefined,
    },
} as const;
