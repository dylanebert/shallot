import {
    Body,
    BodyType,
    type Compound,
    Compounds,
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
    castShapeClosest,
    computeCosSin,
    emptyQueryHit,
    PhysicsWorldDefinition,
    rotationFromAxisAngle,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    SyncPhysicsConstraintsSystem,
} from "@dylanebert/shallot/standard/physics";
import { add, cross, mul, v } from "./mover";
import type { Controls } from "./scenes";

const f = Math.fround;
export const BenchmarkVillageState = {
    create: () => ({
        active: false,
        step: 0,
        drawCasts: true,
        drawSpheres: true,
        rotation: [0, 0, 0, 1] as [number, number, number, number],
        casts: [] as {
            origin: ReturnType<typeof v>;
            hit: boolean;
            result: ReturnType<typeof emptyQueryHit>;
        }[],
        translation: v(),
    }),
};
function construct(world: World): number[] {
    const s = world.resource(BenchmarkVillageState);
    Object.assign(s, BenchmarkVillageState.create(), { active: true });
    world.resource(PhysicsWorldDefinition).capacity = {
        staticShapeCount: 1,
        staticBodyCount: 1,
        dynamicShapeCount: 4900,
        dynamicBodyCount: 4900,
        contactCount: 19600,
    };
    const hull = world.resource(Hulls).add({ name: "tile", halfExtents: [2, 0.5, 2] });
    const wave = world.resource(PhysicsMeshes).add({
        name: "wave patch",
        wave: {
            xCount: 8,
            zCount: 8,
            cellWidth: 0.5,
            amplitude: 0.25,
            rowFrequency: 0.25,
            columnFrequency: 0.25,
        },
    });
    const compound: Compound = {
        name: "village",
        hulls: [],
        meshes: [],
        capsules: [],
        spheres: [],
    };
    for (let i = 0; i < 100; i++)
        for (let j = 0; j < 100; j++) {
            const x = -200 + (i + 0.5) * 4,
                z = -200 + (j + 0.5) * 4;
            if (i % 16 === 8 && j % 16 === 8) {
                compound.meshes!.push({
                    mesh: wave,
                    transform: { position: [x, 0, z], rotation: [0, 0, 0, 1] },
                    scale: [1, 1, 1],
                });
                continue;
            }
            compound.hulls!.push({
                hull,
                transform: { position: [x, -0.5, z], rotation: [0, 0, 0, 1] },
            });
            const index = i * 100 + j;
            if (index % 8) continue;
            if ((index / 8) & 1)
                compound.capsules!.push({
                    center1: [x - 1, 0.25, z],
                    center2: [x + 1, 0.25, z],
                    radius: 0.25,
                });
            else compound.spheres!.push({ center: [x, 0.5, z], radius: 0.5 });
        }
    s.rotation = rotationFromAxisAngle(world, v(0, 1, 0), f(f(-1.15) * f(Math.PI)));
    const base = world.create();
    world.add(base, Transform, { rotation: s.rotation });
    world.add(base, Body);
    world.add(base, Shape, {
        kind: ShapeKind.Compound,
        geometry: world.resource(Compounds).add(compound),
    });
    let seed = 2718281828;
    const rand = (lo: number, hi: number) => {
        seed ^= seed << 13;
        seed ^= seed >>> 17;
        seed ^= seed << 5;
        seed >>>= 0;
        return f(f(f(f(hi) - f(lo)) * f((seed % 32768) / 32767)) + f(lo));
    };
    const box = world.resource(Hulls).add({ name: "box", halfExtents: [0.3, 0.3, 0.3] });
    const bodies: number[] = [];
    const spacing = f(1.6),
        offset = f(f(0.5 * spacing) * -69);
    for (let i = 0; i < 70; i++)
        for (let j = 0; j < 70; j++) {
            const index = i * 70 + j;
            let y = rand(2, 6);
            const angularVelocity: [number, number, number] = [
                rand(-2, 2),
                rand(-2, 2),
                rand(-2, 2),
            ];
            if (index % 20 === 0) y = 20;
            const e = world.create();
            world.add(e, Transform, {
                translation: [f(offset + f(spacing * i)), y, f(offset + f(spacing * j))],
            });
            world.add(e, Body, {
                type: BodyType.Dynamic,
                angularVelocity,
                linearVelocity: index % 20 === 0 ? [0, -80, 0] : [0, 0, 0],
            });
            world.add(
                e,
                Shape,
                index % 3 === 0
                    ? { kind: ShapeKind.Sphere, radius: 0.3 }
                    : index % 3 === 1
                      ? {
                            kind: ShapeKind.Capsule,
                            capsuleCenter1: [-0.3, 0, 0],
                            capsuleCenter2: [0.3, 0, 0],
                            radius: 0.2,
                        }
                      : { geometry: box },
            );
            bodies.push(e);
        }
    s.casts = Array.from({ length: 401 }, () => ({
        origin: v(),
        hit: false,
        result: emptyQueryHit(),
    }));
    return bodies;
}
export const BenchmarkVillagePlugin: Plugin = {
    name: "BenchmarkVillage",
    dependencies: [StandardPhysicsPlugin],
    recovery: recoverResource(BenchmarkVillageState),
    systems: [
        {
            name: "village-curtain",
            group: "fixed",
            after: [SyncPhysicsConstraintsSystem],
            before: [StepPhysicsSystem],
            update(world) {
                const s = world.resource(BenchmarkVillageState);
                if (!s.active) return;
                const t = f((s.step++ % 600) / 599),
                    x = f(-200 + f(400 * t));
                const [c, sine] = computeCosSin(world, f(f(20 / 180) * f(Math.PI)));
                const q = {
                    v: v(...(s.rotation.slice(0, 3) as [number, number, number])),
                    s: s.rotation[3],
                };
                // b3RotateVector's nested cross order is observable in the query records.
                const rotate = (a: ReturnType<typeof v>) =>
                    add(a, mul(2, cross(q.v, add(cross(q.v, a), mul(q.s, a)))));
                s.translation = rotate(v(f(12 * sine), f(-12 * c), 0));
                for (const [i, cast] of s.casts.entries()) {
                    cast.origin = rotate(v(x, 10, -200 + i));
                    cast.hit = castShapeClosest(
                        world,
                        cast.origin,
                        { points: [v()], count: 1, radius: 0.05 },
                        s.translation,
                        cast.result,
                    );
                }
            },
        },
    ],
};
export function benchmarkVillageDrawing(world: World): number[][] {
    const s = world.resource(BenchmarkVillageState),
        rows: number[][] = [];
    if (!s.drawCasts) return rows;
    for (const cast of s.casts) {
        const end = add(cast.origin, mul(cast.hit ? cast.result.fraction : 1, s.translation));
        rows.push([1, cast.origin.x, cast.origin.y, cast.origin.z, end.x, end.y, end.z]);
        if (!cast.hit) continue;
        const p = cast.result.point,
            n = add(p, mul(0.5, cast.result.normal));
        rows.push([0, p.x, p.y, p.z, 4], [1, p.x, p.y, p.z, n.x, n.y, n.z]);
        if (s.drawSpheres) rows.push([4, end.x, end.y, end.z, 0, 0, 0, 1, 0, 0, 0, 0.05]);
    }
    return rows;
}
export const benchmarkVillageScene = {
    construct,
    gridExtent: 200,
    gridSpacing: 8,
    camera: { yaw: 45, pitch: 35, distance: 500, phoneDistance: 800, target: [0, 0, 0] },
    controls: {
        "Draw Casts": {
            label: "Draw casts",
            value: 1,
            set(world, _b, value) {
                world.resource(BenchmarkVillageState).drawCasts = !!value;
            },
        },
        "Draw Spheres": {
            label: "Draw spheres",
            value: 1,
            set(world, _b, value) {
                world.resource(BenchmarkVillageState).drawSpheres = !!value;
            },
        },
    } satisfies Controls,
};
