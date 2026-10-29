import {
    Body,
    type Compound,
    Compounds,
    Hulls,
    InputPlugin,
    PhysicsMeshes,
    type Plugin,
    recoverResource,
    Shape,
    ShapeKind,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    castRayClosest,
    castShapeClosest,
    emptyQueryHit,
    overlapShape,
    rotationFromAxisAngle,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    SyncPhysicsConstraintsSystem,
} from "@dylanebert/shallot/standard/physics";
import building from "./building.json" with { type: "json" };
import { decodeMesh } from "./mesh-data";
import { add, move, moverState, mul, sub, v } from "./mover";
import { readPlayerControls } from "./player-controls";

const f = Math.fround;
const identity = (): [number, number, number, number] => [0, 0, 0, 1];
export const VillageState = {
    create: () => ({
        active: false,
        entity: 0,
        mover: { ...moverState(), position: v(0, 10, 0) },
        keys: { W: 0, A: 0, S: 0, D: 0, Space: 0, Shift: 0 },
        forward: v(0, 0, -1),
        right: v(1, 0, 0),
        policies: new Map(),
        rayOrigin: v(-720, 20, -720),
        drawing: [] as number[][],
        overlap: false,
    }),
};
function construct(world: World): number[] {
    const s = world.resource(VillageState);
    Object.assign(s, VillageState.create());
    s.active = true;
    let seed = 12345;
    const rand = (lo: number, hi: number) => {
        seed ^= seed << 13;
        seed ^= seed >>> 17;
        seed ^= seed << 5;
        seed >>>= 0;
        return f(f(f(f(hi) - f(lo)) * f((seed % 32768) / 32767)) + f(lo));
    };
    const randomVec = (lo: [number, number, number], hi: [number, number, number]) =>
        v(rand(lo[0], hi[0]), rand(lo[1], hi[1]), rand(lo[2], hi[2]));
    const tuple = (p: ReturnType<typeof v>): [number, number, number] => [p.x, p.y, p.z];
    const hull = world.resource(Hulls).add({ name: "tile", halfExtents: [4, 2, 4] });
    const capsules: NonNullable<Compound["capsules"]> = [],
        hulls: NonNullable<Compound["hulls"]> = [],
        spheres: NonNullable<Compound["spheres"]> = [],
        meshes: NonNullable<Compound["meshes"]> = [];
    for (let i = 0; i < 200; i++) {
        for (let j = 0; j < 200; j++) {
            const p = v((2 * i - 200) * 4, f(rand(-0.25, 0.125) * 4), (2 * j - 200) * 4);
            if (i & 1 && j & 1) {
                const p1 = add(p, randomVec([-4, 4, -4], [4, 8, 4]));
                const p2 = add(p, randomVec([-4, 4, -4], [4, 8, 4]));
                const radius = rand(0.1, 0.5);
                if (capsules.length < spheres.length)
                    capsules.push({ center1: tuple(p1), center2: tuple(p2), radius });
                else spheres.push({ center: tuple(p1), radius });
            }
            hulls.push({ hull, transform: { position: tuple(p), rotation: identity() } });
        }
    }
    const data = decodeMesh(building);
    const mesh = world.resource(PhysicsMeshes).add({
        name: "building",
        ...data,
        identifyEdges: true,
        weldVertices: true,
        weldTolerance: 0.002,
    });
    const materials = Array.from({ length: Math.max(...data.materialIndices) + 1 }, (_, i) => ({
        friction: i === 0 ? 0 : 0.6,
        restitution: i === 1 ? 0.5 : 0,
        rollingResistance: 0,
        tangentVelocity: [0, 0, 0] as [number, number, number],
        customColor: 0,
        userMaterialId: BigInt(i + 42),
    }));
    for (let i = 0; i < 50; i++) {
        for (let j = 0; j < 50; j++) {
            const rotation = rotationFromAxisAngle(
                world,
                v(0, 1, 0),
                rand(-f(Math.PI), f(Math.PI)),
            );
            const scale = tuple(randomVec([0.5, 0.5, 0.5], [2, 2, 2]));
            const index = meshes.length;
            if (index & 1) scale[0] = -scale[0];
            if (index & 3) scale[2] = -scale[2];
            meshes.push({
                mesh,
                transform: {
                    position: [(2 * i - 50) * 16 + 8, 2, (2 * j - 50) * 16 + 8],
                    rotation,
                },
                scale,
                materials,
            });
        }
    }
    const geometry = world
        .resource(Compounds)
        .add({ name: "Village", capsules, hulls, meshes, spheres });
    const ground = world.create();
    world.add(ground, Transform, {
        translation: [-1, -0.5, 2],
        rotation: rotationFromAxisAngle(world, v(0, 1, 0), f(f(-1.15) * f(Math.PI))),
    });
    world.add(ground, Body);
    world.add(ground, Shape, { kind: ShapeKind.Compound, geometry });
    s.entity = world.create();
    world.add(s.entity, Transform, { translation: [0, 10, 0] });
    return [];
}

export const VillagePlugin: Plugin = {
    name: "VillageSample",
    dependencies: [StandardPhysicsPlugin, InputPlugin],
    recovery: recoverResource(VillageState),
    systems: [
        {
            name: "village-input",
            group: "setup",
            update(world) {
                const s = world.resource(VillageState);
                if (!s.active) return;
                readPlayerControls(world, s.keys, s.forward, s.right);
            },
        },
        {
            name: "village-before-step",
            group: "fixed",
            after: [SyncPhysicsConstraintsSystem],
            before: [StepPhysicsSystem],
            update(world) {
                const s = world.resource(VillageState);
                if (!s.active) return;
                move(world, s.mover, s.forward, s.right, s.keys, true, 0, s.policies);
                world
                    .storage(Transform)
                    .translation.set(
                        s.entity,
                        s.mover.position.x,
                        s.mover.position.y,
                        s.mover.position.z,
                    );
                const translation = v(10, -40, -5);
                s.drawing = [];
                const line = (a: ReturnType<typeof v>, b: ReturnType<typeof v>) =>
                    s.drawing.push([1, a.x, a.y, a.z, b.x, b.y, b.z]);
                for (const shape of [false, true]) {
                    const origin = shape ? sub(s.rayOrigin, v(1, 0, 1)) : s.rayOrigin;
                    const hit = emptyQueryHit();
                    const found = shape
                        ? castShapeClosest(
                              world,
                              origin,
                              { points: [v()], count: 1, radius: 0.25 },
                              translation,
                              hit,
                          )
                        : castRayClosest(world, origin, translation, hit);
                    line(origin, add(origin, translation));
                    if (found) {
                        const point = v(f(hit.point.x), f(hit.point.y), f(hit.point.z));
                        line(point, add(point, mul(0.5, hit.normal)));
                        s.drawing.push([0, point.x, point.y, point.z, 8]);
                        if (shape) {
                            const p = add(origin, mul(hit.fraction, translation));
                            s.drawing.push([4, p.x, p.y, p.z, 0, 0, 0, 1, 0, 0, 0, 0.25]);
                        }
                    }
                }
                s.overlap = false;
                overlapShape(
                    world,
                    v(f(s.rayOrigin.x - 1), 2, f(s.rayOrigin.z - 1)),
                    { points: [v()], count: 1, radius: 0.3 },
                    () => {
                        s.overlap = true;
                        return false;
                    },
                );
                if (s.rayOrigin.x > 720) {
                    s.rayOrigin.x = -720;
                    s.rayOrigin.z += 8;
                }
                if (s.rayOrigin.z > 720) s.rayOrigin.z = -720;
                s.rayOrigin.x = f(s.rayOrigin.x + f(2 * f(1 / 60)));
            },
        },
    ],
};
export const villageScene = {
    construct,
    gridExtent: 800,
    gridSpacing: 8,
    camera: { yaw: 45, pitch: 20, distance: 9, phoneDistance: 12, target: [0, 10, 0] },
    controls: undefined,
};
