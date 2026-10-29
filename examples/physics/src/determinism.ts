import {
    Body,
    BodyType,
    HeightFields,
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
    castRayClosest,
    castShape,
    computeCosSin,
    emptyQueryHit,
    getBodyTransform,
    isAwake,
    overlapAABB,
    PhysicsWorldDefinition,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
} from "@dylanebert/shallot/standard/physics";
import { human } from "./human";
import type { Controls } from "./scenes";

const f = Math.fround;
type V3 = [number, number, number];
const vec = (v: V3) => ({ x: v[0], y: v[1], z: v[2] });
export const DeterminismState = {
    create: () => ({
        scene: "",
        bodies: [] as number[],
        step: 0,
        frame: 0,
        sleepStep: 0,
        hash: 0,
        seed: 0,
        spawnCount: 0,
        hits: 0,
        queryHash: 0,
        asleep: false,
        origin: [0, 0, 0] as V3,
        translation: [0, 0, 0] as V3,
        rayPoint: [0, 0, 0] as V3,
        normal: [0, 0, 0] as V3,
        rayHit: false,
        lower: [0, 0, 0] as V3,
        upper: [0, 0, 0] as V3,
        fraction: 1,
    }),
};
function random(world: World, lo: number, hi: number): number {
    const s = world.resource(DeterminismState);
    let x = s.seed;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    s.seed = x >>> 0;
    return f(f(f(f(hi) - f(lo)) * f((s.seed % 32768) / 32767)) + f(lo));
}
function randomVector(w: World, lo: number, hi: number): V3 {
    return [random(w, lo, hi), random(w, lo, hi), random(w, lo, hi)];
}
function randomRotation(w: World): [number, number, number, number] {
    const u = random(w, 0, 1);
    const a = random(w, 0, f(2 * f(Math.PI)));
    const b = random(w, 0, f(2 * f(Math.PI)));
    const r = f(Math.sqrt(f(1 - u))),
        t = f(Math.sqrt(u));
    const ca = computeCosSin(w, a),
        cb = computeCosSin(w, b);
    return [f(r * ca[1]), f(r * ca[0]), f(t * cb[1]), f(t * cb[0])];
}
function body(w: World, translation: V3, dynamic = false) {
    const e = w.create();
    w.add(e, Transform, { translation });
    w.add(e, Body, { type: dynamic ? BodyType.Dynamic : BodyType.Static });
    return e;
}
function box(w: World, halfExtents: V3) {
    return w.resource(Hulls).add({ name: "box", halfExtents });
}
function fallingRagdolls(w: World): number[] {
    const s = w.resource(DeterminismState);
    s.scene = "fallingRagdolls";
    const meshes = w.resource(PhysicsMeshes);
    const grid = meshes.add({
        name: "grid",
        grid: { xCount: 8, zCount: 8, cellWidth: 1.875, materialCount: 0, identifyEdges: true },
    });
    const torus = meshes.add({
        name: "torus",
        torus: { radialResolution: 16, tubularResolution: 16, radius: 3.75, thickness: 1 },
    });
    for (let i = 0; i < 2; i++)
        for (let j = 0; j < 2; j++) {
            const ground = body(w, [-7.5 + 15 * i, 0, -7.5 + 15 * j]);
            w.add(ground, Shape, { kind: ShapeKind.Mesh, geometry: grid });
            w.add(w.create(), Shape, { body: ground, kind: ShapeKind.Mesh, geometry: torus });
            for (let k = 0; k < 2; k++)
                s.bodies.push(
                    ...human(w, [-7.5 + 15 * j + 0.75 * k, 15, -7.5 + 15 * i], 2 * i + j).bodies,
                );
        }
    return s.bodies;
}
function wavePile(w: World): number[] {
    const s = w.resource(DeterminismState);
    s.scene = "wavePile";
    s.seed = 52977;
    const ground = body(w, [-10, 0, -10]);
    w.add(ground, Shape, {
        kind: ShapeKind.HeightField,
        geometry: w.resource(HeightFields).add({
            name: "wave",
            wave: {
                rowCount: 21,
                columnCount: 21,
                scale: [1, 0.6, 1],
                rowFrequency: 0.08,
                columnFrequency: 0.06,
                makeHoles: false,
            },
        }),
    });
    const phi = f(f(1 + f(Math.sqrt(5))) / 2);
    const [dc, ds] = computeCosSin(w, f(f(2 * f(Math.PI)) / phi));
    let c = 1,
        sn = 0;
    const points: number[] = [];
    for (let i = 0; i < 10; i++) {
        const z = f(1 - f((2 * i + 1) / 10));
        const r = f(Math.sqrt(f(1 - f(z * z))));
        points.push(f(f(f(0.55) * r) * c), f(f(f(0.55) * r) * sn), f(f(0.55) * z));
        const previous = c;
        c = f(f(dc * c) - f(ds * sn));
        sn = f(f(ds * previous) + f(dc * sn));
    }
    const rock = w
        .resource(Hulls)
        .add({ name: "rock", points: new Float32Array(points), maxVertexCount: 10 });
    const geometry = box(w, [0.45, 0.3, 0.55]);
    for (let layer = 0; layer < 4; layer++)
        for (let i = 0; i < 5; i++)
            for (let j = 0; j < 5; j++) {
                const jitter = randomVector(w, -0.3, 0.3);
                const e = body(
                    w,
                    [
                        f(f(f(1.7) * (i - 2)) + jitter[0]),
                        f(f(2.5 + f(f(1.6) * layer)) + f(f(0.3) * jitter[1])),
                        f(f(f(1.7) * (j - 2)) + jitter[2]),
                    ],
                    true,
                );
                w.storage(Transform).rotation.set(e, ...randomRotation(w));
                const index = s.bodies.length;
                w.add(e, Shape, {
                    rollingResistance: 0.3,
                    ...(index % 4 === 0
                        ? { kind: ShapeKind.Sphere, radius: 0.5 }
                        : index % 4 === 1
                          ? {
                                kind: ShapeKind.Capsule,
                                capsuleCenter1: [0, -0.3, 0] as V3,
                                capsuleCenter2: [0, 0.3, 0] as V3,
                                radius: 0.35,
                            }
                          : { geometry: index % 4 === 2 ? geometry : rock }),
                });
                s.bodies.push(e);
            }
    return s.bodies;
}
function meshDrop(w: World): number[] {
    const s = w.resource(DeterminismState);
    s.scene = "meshDropDeterminism";
    s.seed = 3963634789;
    const ground = body(w, [0, 0, 0]);
    w.add(ground, Shape, {
        kind: ShapeKind.Mesh,
        geometry: w.resource(PhysicsMeshes).add({
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
        filterCategoryLow: 1,
        filterCategoryHigh: 0,
    });
    const geometry = box(w, [0.02, 0.2, 0.04]);
    for (let i = 0; i < 20; i++)
        for (let j = 0; j < 20; j++) {
            const linear = randomVector(w, -1, 1),
                angular = randomVector(w, -5, 5);
            const e = body(w, [0.5 * (i - 10), 5, 0.5 * (j - 10)], true);
            w.storage(Body).linearVelocity.set(e, ...linear);
            w.storage(Body).angularVelocity.set(e, ...angular);
            w.add(e, Shape, {
                geometry,
                rollingResistance: 0.1,
                filterCategoryLow: 2,
                filterCategoryHigh: 0,
                filterMaskLow: 1,
                filterMaskHigh: 0,
            });
            s.bodies.push(e);
        }
    return s.bodies;
}
function hashBytes(hash: number, bytes: Uint8Array): number {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    let i = 0;
    for (; i + 8 <= bytes.length; i += 8) {
        hash = (Math.imul(hash, 33) + view.getUint32(i, true)) >>> 0;
        hash = (Math.imul(hash, 33) + view.getUint32(i + 4, true)) >>> 0;
    }
    for (; i < bytes.length; i++) hash = (Math.imul(hash, 33) + bytes[i]!) >>> 0;
    return hash;
}
function hashFloats(hash: number, values: number[]): number {
    const bytes = new Uint8Array(values.length * 4),
        view = new DataView(bytes.buffer);
    values.forEach((v, i) => {
        view.setFloat32(i * 4, v, true);
    });
    return hashBytes(hash, bytes);
}
function sleepHash(w: World): number {
    let hash = 5381;
    const bytes = new Uint8Array(28),
        view = new DataView(bytes.buffer);
    for (const e of w.resource(DeterminismState).bodies) {
        const { p, q } = getBodyTransform(w, e)!;
        [p.x, p.y, p.z, q.v.x, q.v.y, q.v.z, q.s].forEach((v, i) => {
            view.setFloat32(i * 4, v, true);
        });
        hash = hashBytes(hash, bytes);
    }
    return hash;
}
function spawn(w: World) {
    const s = w.resource(DeterminismState);
    s.origin = randomVector(w, -12, 12);
    const rotation = randomRotation(w);
    s.translation = rotation.slice(0, 3).map((v) => f(30 * v)) as V3;
    const hit = emptyQueryHit();
    s.rayHit = castRayClosest(w, vec(s.origin), vec(s.translation), hit);
    let position: V3;
    if (s.rayHit) {
        s.hits++;
        s.queryHash = hashFloats(s.queryHash, [hit.fraction]);
        s.queryHash = hashFloats(s.queryHash, [hit.normal.x, hit.normal.y, hit.normal.z]);
        position = [
            f(f(hit.point.x) + f(f(1.2) * hit.normal.x)),
            f(f(hit.point.y) + f(f(1.2) * hit.normal.y)),
            f(f(hit.point.z) + f(f(1.2) * hit.normal.z)),
        ];
        s.rayPoint = [hit.point.x, hit.point.y, hit.point.z];
        s.normal = [hit.normal.x, hit.normal.y, hit.normal.z];
    } else {
        position = randomVector(w, -6, 6);
        s.rayPoint = s.origin.map((v, i) => v + s.translation[i]!) as V3;
        s.normal = [0, 0, 0];
    }
    const center = randomVector(w, -10, 10),
        extent = random(w, 1, 4);
    s.lower = center.map((v) => f(v - extent)) as V3;
    s.upper = center.map((v) => f(v + extent)) as V3;
    let count = 0;
    overlapAABB(w, { lowerBound: vec(s.lower), upperBound: vec(s.upper) }, (shape) => {
        count++;
        s.hits++;
        const index = s.bodies.indexOf(shape) + 1;
        const bytes = new Uint8Array(4);
        new DataView(bytes.buffer).setInt32(0, index, true);
        s.queryHash = hashBytes(s.queryHash, bytes);
        return true;
    });
    s.fraction = 1;
    castShape(
        w,
        vec(s.origin),
        { points: [{ x: 0, y: 0, z: 0 }], count: 1, radius: 0.5 },
        vec(s.translation),
        (h) => {
            s.fraction = h.fraction;
            return h.fraction;
        },
    );
    if (s.fraction < 1) {
        s.hits++;
        s.queryHash = hashFloats(s.queryHash, [s.fraction]);
    }
    const size = f(f(0.3) + f(f(0.2) * s.fraction));
    const e = body(w, position, true);
    w.storage(Transform).rotation.set(e, ...randomRotation(w));
    w.storage(Body).linearVelocity.set(e, ...randomVector(w, -0.2, 0.2));
    w.storage(Body).angularVelocity.set(e, ...randomVector(w, -0.5, 0.5));
    w.storage(Body).linearDamping.set(e, 1);
    w.storage(Body).angularDamping.set(e, 1);
    const kind = (s.spawnCount + count) % 3;
    w.add(e, Shape, {
        rollingResistance: 0.2,
        ...(kind === 0
            ? { kind: ShapeKind.Sphere, radius: size }
            : kind === 1
              ? {
                    kind: ShapeKind.Capsule,
                    capsuleCenter1: [0, -size, 0] as V3,
                    capsuleCenter2: [0, size, 0] as V3,
                    radius: f(f(0.7) * size),
                }
              : { geometry: box(w, [size, f(f(0.7) * size), f(0.5 * size)]) }),
    });
    s.bodies.push(e);
    s.spawnCount++;
}
export const DeterminismPlugin: Plugin = {
    name: "Determinism",
    dependencies: [StandardPhysicsPlugin],
    recovery: recoverResource(DeterminismState),
    systems: [
        {
            name: "DeterminismUpdate",
            group: "fixed",
            after: [StepPhysicsSystem],
            update(w) {
                const s = w.resource(DeterminismState);
                if (!s.scene || s.hash) return;
                if (s.scene === "querySpawn" && ++s.frame % 10 !== 1) return;
                if (s.scene === "querySpawn" && s.spawnCount < 50) spawn(w);
                else if (s.bodies.every((e) => !isAwake(w, e))) {
                    if (s.scene !== "fallingRagdolls" || s.asleep) {
                        s.hash = sleepHash(w);
                        s.sleepStep = s.step;
                    }
                    s.asleep = true;
                } else s.asleep = false;
                s.step++;
            },
        },
    ],
};
const hex = (n: number) => `0x${n.toString(16).toUpperCase().padStart(8, "0")}`;
const readouts: Controls = {
    Sleep: {
        label: "Sleep step",
        value: 0,
        read: (w) => w.resource(DeterminismState).sleepStep,
        set() {},
    },
    Hash: {
        label: "Hash",
        value: 0,
        read: (w) => hex(w.resource(DeterminismState).hash),
        set() {},
    },
};
export const determinismScenes = {
    fallingRagdolls: {
        construct: fallingRagdolls,
        gridExtent: 15,
        gridSpacing: 1,
        camera: { yaw: 35, pitch: 25, distance: 48, phoneDistance: 65, target: [0, 7, 0] },
        controls: readouts,
    },
    wavePile: {
        construct: wavePile,
        gridExtent: 10,
        gridSpacing: 1,
        camera: { yaw: 35, pitch: 30, distance: 24, phoneDistance: 34, target: [0, 3, 0] },
        controls: readouts,
    },
    querySpawn: {
        construct(w: World) {
            const s = w.resource(DeterminismState);
            s.scene = "querySpawn";
            s.seed = 71689;
            w.resource(PhysicsWorldDefinition).gravity = { x: 0, y: 0, z: 0 };
            return s.bodies;
        },
        gridExtent: 12,
        gridSpacing: 1,
        camera: { yaw: 35, pitch: 25, distance: 35, phoneDistance: 50, target: [0, 0, 0] },
        controls: {
            ...readouts,
            Spawned: {
                label: "Spawned",
                value: 0,
                read: (w: World) => w.resource(DeterminismState).spawnCount,
                set() {},
            },
            Hits: {
                label: "Query hits",
                value: 0,
                read: (w: World) => w.resource(DeterminismState).hits,
                set() {},
            },
            QueryHash: {
                label: "Query hash",
                value: 0,
                read: (w: World) => hex(w.resource(DeterminismState).queryHash),
                set() {},
            },
        },
    },
    meshDropDeterminism: {
        construct: meshDrop,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 25, pitch: 35, distance: 24, phoneDistance: 38, target: [0, 1, 0] },
        controls: readouts,
    },
} as const;
