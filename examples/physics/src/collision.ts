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
    ShapeMaterials,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    castBodyRay,
    castRay,
    castRayClosest,
    castShape,
    computeCosSin,
    emptyQueryHit,
    overlapShape,
    type PhysicsQueryHit,
    PhysicsWorldDefinition,
    rotationFromAxisAngle,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
} from "@dylanebert/shallot/standard/physics";
import type { Controls } from "./scenes";

const f = Math.fround;
type Vec = { x: number; y: number; z: number };
type Point = [number, number, number];
const vec = (x = 0, y = 0, z = 0): Vec => ({ x, y, z });
const point = (v: Vec): Point => [v.x, v.y, v.z];
const add = (a: Vec, b: Vec, t = 1): Vec =>
    vec(f(a.x + f(t * b.x)), f(a.y + f(t * b.y)), f(a.z + f(t * b.z)));
export const collisionNames = [
    "rayCurtain",
    "castWorld",
    "meshScale",
    "shapeCast",
    "overlapWorld",
    "longRayCast",
    "initialOverlap",
    "capsuleCastRay",
] as const;
type Scene = (typeof collisionNames)[number];
export const CollisionState = {
    create: () => ({
        scene: "" as Scene | "",
        bodies: [] as number[],
        ignored: [] as number[],
        bodyIndex: 0,
        seed: 12345,
        mesh: 0,
        box: 0,
        field: 0,
        initialOverlap: false,
        mode: 1,
        castType: 0,
        radius: 0.5,
        offset: vec(),
        curtainOffset: 2,
        curtainSpeed: f(-0.015),
        start: vec(-2, 0, 0),
        sphereCast: true,
        phase: 0,
        rayLength: 1,
        coneAngle: 5,
        errors: Array<number | null>(5).fill(null),
        failures: Array<number>(5).fill(0),
        trails: Array.from({ length: 5 }, () => [] as { point: Point; color: number }[]),
        queries: [] as QueryRecord[],
        sweeps: [] as Sweep[],
        tracking: false,
        dragOrigin: vec(),
        dragBase: vec(),
    }),
};
export interface Sweep {
    origin: Vec;
    translation: Vec;
    points: Vec[];
    radius: number;
    hits: PhysicsQueryHit[];
    overlap?: boolean;
    color?: number;
    direction?: Vec;
}
export interface QueryRecord {
    kind: "ray" | "shape" | "closest" | "body" | "overlap";
    hits: { hit: PhysicsQueryHit; result?: number }[];
    hit: boolean;
    overlaps: boolean[];
}
const copyHit = (h: PhysicsQueryHit): PhysicsQueryHit => ({
    ...h,
    point: { ...h.point },
    normal: { ...h.normal },
});
function query(
    world: World,
    sweep: Sweep,
    kind: "ray" | "shape" | "closest" | "body" | "overlap",
    mode = 1,
): void {
    const s = world.resource(CollisionState);
    const record: QueryRecord = { kind, hits: [], hit: false, overlaps: [] };
    s.queries.push(record);
    const { origin, translation, points, radius } = sweep;
    const proxy = { points, count: points.length, radius };
    if (kind === "overlap") {
        overlapShape(world, origin, proxy, () => {
            sweep.overlap = true;
            record.overlaps.push(false);
            return false;
        });
    } else if (kind === "closest" || kind === "body") {
        const h = emptyQueryHit();
        const hit =
            kind === "closest"
                ? castRayClosest(world, origin, translation, h)
                : castBodyRay(
                      world,
                      s.bodies[0]!,
                      origin,
                      translation,
                      { p: vec(), q: { v: vec(), s: 1 } },
                      h,
                  );
        record.hit = hit;
        if (hit) {
            record.hits.push({ hit: copyHit(h) });
            sweep.hits.push(copyHit(h));
        }
    } else {
        const callback = (h: PhysicsQueryHit) => {
            let result = -1;
            if ((s.initialOverlap || h.fraction !== 0) && !s.ignored.includes(h.shape)) {
                if (mode === 0 || mode === 1) {
                    sweep.hits[0] = copyHit(h);
                    result = mode === 0 ? 0 : h.fraction;
                } else if (mode === 2) {
                    sweep.hits.push(copyHit(h));
                    result = sweep.hits.length === 3 ? 0 : 1;
                } else {
                    sweep.hits.push(copyHit(h));
                    sweep.hits.sort((a, b) => a.fraction - b.fraction);
                    sweep.hits.length = Math.min(3, sweep.hits.length);
                    result = sweep.hits.length === 3 ? sweep.hits[2]!.fraction : 1;
                }
            }
            record.hits.push({ hit: copyHit(h), result });
            return result;
        };
        if (kind === "ray") castRay(world, origin, translation, callback);
        else castShape(world, origin, proxy, translation, callback);
    }
    s.sweeps.push(sweep);
}
function sweep(origin: Vec, translation: Vec, points: Vec[] = [], radius = 0): Sweep {
    return { origin, translation, points, radius, hits: [] };
}
function boxPoints(extent: Point, offset = vec()): Vec[] {
    const [x, y, z] = extent;
    return [
        [x, y, z],
        [-x, y, z],
        [-x, -y, z],
        [x, -y, z],
        [x, y, -z],
        [-x, y, -z],
        [-x, -y, -z],
        [x, -y, -z],
    ].map((p) => add(offset, vec(...(p as Point))));
}
function runQueries(world: World): void {
    const s = world.resource(CollisionState);
    s.queries.length = 0;
    s.sweeps.length = 0;
    if (!s.scene) return;
    if (s.scene === "rayCurtain") {
        for (let x = -8; x <= 8; x = f(x + f(0.1)))
            query(world, sweep(vec(x, 8, s.curtainOffset), vec(0, -8, 0)), "closest");
        if (s.curtainOffset > 2) s.curtainSpeed = f(-0.015);
        else if (s.curtainOffset < -2) s.curtainSpeed = f(0.015);
        s.curtainOffset = f(s.curtainOffset + s.curtainSpeed);
    } else if (s.scene === "capsuleCastRay")
        query(world, sweep(vec(-1, 0.5, 0), vec(2, 0, 0)), "body");
    else if (s.scene === "castWorld") {
        const points =
            s.castType === 0
                ? []
                : s.castType === 1
                  ? [vec()]
                  : s.castType === 2
                    ? [vec(), vec(0, 1, 0)]
                    : boxPoints([s.radius, f(0.5 * s.radius), f(0.25 * s.radius)]);
        query(
            world,
            sweep(vec(-20, 10, 0), vec(20, 10, 0), points, s.castType === 3 ? 0 : s.radius),
            s.castType === 0 ? "ray" : "shape",
            s.mode,
        );
    } else if (s.scene === "meshScale")
        query(
            world,
            sweep({ ...s.start }, vec(4, 0, 0), [vec()], 0.25),
            s.sphereCast ? "shape" : "closest",
        );
    else if (s.scene === "initialOverlap")
        query(
            world,
            sweep(
                vec(),
                vec(),
                [vec(-2.1, f(-0.8), f(0.95)), vec(-2.1, f(f(-0.8) + 1), f(0.95))],
                0.25,
            ),
            "shape",
        );
    else if (s.scene === "shapeCast" || s.scene === "overlapWorld") {
        const overlap = s.scene === "overlapWorld";
        for (let row = 0; row < 3; row++)
            for (let i = 0; i < (overlap ? 5 : 4); i++) {
                const offset = add(vec(-6 + (overlap ? 3 : 4) * i, 3 + 2 * row, -5), s.offset);
                let points =
                    row === 0
                        ? [offset]
                        : row === 1
                          ? [
                                add(offset, vec(f(-0.2), f(-0.2), f(-0.2))),
                                add(offset, vec(f(0.2), f(0.2), f(0.2))),
                            ]
                          : boxPoints([f(0.3), f(0.3), f(0.3)], offset);
                if (row === 2 && !overlap) {
                    const angle = f(f(0.25) * f(Math.PI));
                    const qx = rotationFromAxisAngle(world, vec(1, 0, 0), angle),
                        qy = rotationFromAxisAngle(world, vec(0, 1, 0), angle),
                        qz = rotationFromAxisAngle(world, vec(0, 0, 1), angle);
                    const q = mulQuat(qx, mulQuat(qy, qz));
                    points = boxPoints([f(0.3), f(0.3), f(0.3)]).map((p) =>
                        add(offset, rotate(q, p)),
                    );
                }
                query(
                    world,
                    sweep(
                        vec(),
                        overlap ? vec() : vec(0, 0, 10),
                        points,
                        row === 0 ? f(0.3) : row === 1 ? f(0.2) : 0,
                    ),
                    overlap ? "overlap" : "shape",
                );
            }
    } else if (s.scene === "longRayCast") {
        const tau = f(2 * f(Math.PI));
        s.phase = f(s.phase + f(tau / 180));
        if (s.phase > tau) s.phase = f(s.phase - tau);
        const tilted = rotate(
            rotationFromAxisAngle(world, vec(1, 0, 0), f(s.coneAngle * f(f(Math.PI) / 180))),
            vec(0, 1, 0),
        );
        const direction = rotate(rotationFromAxisAngle(world, vec(0, 1, 0), s.phase), tilted);
        for (let i = 0; i < 5; i++) {
            const aim = vec((i - 2) * 5, 2.5, 0);
            const cast = (distance: number) => {
                const t = f(-(distance + 5));
                const q = sweep(
                    add(aim, direction, distance),
                    vec(f(t * direction.x), f(t * direction.y), f(t * direction.z)),
                );
                query(world, q, "ray");
                return q;
            };
            const truth = cast(50),
                actual = cast(f(1000 * s.rayLength));
            actual.direction = direction;
            const a = actual.hits[0],
                b = truth.hits[0];
            let fail = 0;
            if (a) {
                const delta = b
                    ? vec(
                          f(a.point.x - b.point.x),
                          f(a.point.y - b.point.y),
                          f(a.point.z - b.point.z),
                      )
                    : vec();
                const error = f(
                    Math.sqrt(
                        f(f(f(delta.x * delta.x) + f(delta.y * delta.y)) + f(delta.z * delta.z)),
                    ),
                );
                s.errors[i] = error;
                actual.color = error < f(0.05) ? 0xeee9df : 0xff8135;
                s.trails[i]!.push({ point: point(a.point), color: actual.color });
                if (s.trails[i]!.length > 180) s.trails[i]!.shift();
            } else {
                s.errors[i] = null;
                fail = Number(!!b);
                actual.color = b ? 0xff8135 : 0x797369;
            }
            s.failures[i] = f(f(f(0.95) * s.failures[i]!) + f(f(0.05) * fail));
        }
    }
}
function mulQuat(a: number[], b: number[]): number[] {
    const [ax, ay, az, aw] = a as [number, number, number, number],
        [bx, by, bz, bw] = b as [number, number, number, number];
    return [
        f(f(f(f(ay * bz) - f(az * by)) + f(aw * bx)) + f(bw * ax)),
        f(f(f(f(az * bx) - f(ax * bz)) + f(aw * by)) + f(bw * ay)),
        f(f(f(f(ax * by) - f(ay * bx)) + f(aw * bz)) + f(bw * az)),
        f(f(aw * bw) - f(f(f(ax * bx) + f(ay * by)) + f(az * bz))),
    ];
}
function rotate(q: number[], v: Vec): Vec {
    const [x, y, z, w] = q as [number, number, number, number];
    const tx = f(f(f(y * v.z) - f(z * v.y)) + f(w * v.x)),
        ty = f(f(f(z * v.x) - f(x * v.z)) + f(w * v.y)),
        tz = f(f(f(x * v.y) - f(y * v.x)) + f(w * v.z));
    return vec(
        f(v.x + f(2 * f(f(y * tz) - f(z * ty)))),
        f(v.y + f(2 * f(f(z * tx) - f(x * tz)))),
        f(v.z + f(2 * f(f(x * ty) - f(y * tx)))),
    );
}
export const CollisionPlugin: Plugin = {
    name: "CollisionSamples",
    dependencies: [StandardPhysicsPlugin],
    recovery: recoverResource(CollisionState),
    systems: [
        {
            name: "collision-queries",
            group: "fixed",
            after: [StepPhysicsSystem],
            update: runQueries,
        },
    ],
};
function torus(world: World): number {
    return world.resource(PhysicsMeshes).add({
        name: "torus",
        torus: {
            radialResolution: 10,
            tubularResolution: 12,
            radius: f(0.65),
            thickness: f(0.35),
        },
    });
}
function wave(
    world: World,
    scale: number,
    rowFrequency: number,
    columnFrequency: number,
    count = 10,
): number {
    return world.resource(HeightFields).add({
        name: "wave",
        wave: {
            rowCount: count,
            columnCount: count,
            scale: [scale, scale, scale],
            rowFrequency: f(rowFrequency),
            columnFrequency: f(columnFrequency),
            makeHoles: false,
        },
    });
}
function spawn(
    world: World,
    p: Point,
    type: number,
    shape: {
        kind?: ShapeKind;
        geometry?: number;
        radius?: number;
        capsuleCenter1?: Point;
        capsuleCenter2?: Point;
        scale?: Point;
    },
    rotation: Point | number[] = [0, 0, 0, 1],
): number {
    const eid = world.create();
    world.add(eid, Transform, {
        translation: p,
        rotation: rotation as [number, number, number, number],
    });
    world.add(eid, Body, { type });
    world.add(eid, Shape, shape);
    world.resource(CollisionState).bodies.push(eid);
    return eid;
}
function construct(world: World, scene: Scene): number[] {
    const s = world.resource(CollisionState);
    Object.assign(s, CollisionState.create());
    s.scene = scene;
    if (scene === "meshScale") world.resource(PhysicsWorldDefinition).enableAutomaticStep = false;
    if (scene === "overlapWorld" || scene === "longRayCast")
        world.resource(PhysicsWorldDefinition).gravity = vec();
    if (scene === "capsuleCastRay")
        spawn(world, [0, 0, 0], BodyType.Kinematic, {
            kind: ShapeKind.Capsule,
            capsuleCenter1: [0, 0, 0],
            capsuleCenter2: [0, 1, 0],
            radius: 0.5,
        });
    else if (scene === "initialOverlap") {
        const mesh = world.resource(PhysicsMeshes).add({
            name: "plane",
            vertices: new Float32Array([
                -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5,
            ]),
            indices: new Uint32Array([0, 1, 2, 2, 3, 0]),
        });
        spawn(
            world,
            [0, 0, 0],
            BodyType.Static,
            { kind: ShapeKind.Mesh, geometry: mesh, scale: [4, 4, 4] },
            rotationFromAxisAngle(world, vec(0, 0, 1), f(10 * f(f(Math.PI) / 180))),
        );
        s.initialOverlap = true;
    } else if (scene === "meshScale") {
        const x = 0.5,
            y = 0.5,
            z = 0.5;
        s.mesh = world.resource(PhysicsMeshes).add({
            name: "box mesh",
            vertices: new Float32Array([
                x,
                y,
                z,
                -x,
                y,
                z,
                -x,
                -y,
                z,
                x,
                -y,
                z,
                x,
                y,
                -z,
                -x,
                y,
                -z,
                -x,
                -y,
                -z,
                x,
                -y,
                -z,
            ]),
            indices: new Uint32Array([
                0, 1, 3, 1, 2, 3, 0, 4, 1, 1, 4, 5, 0, 3, 7, 4, 0, 7, 4, 7, 5, 6, 5, 7, 1, 5, 2, 6,
                2, 5, 3, 2, 7, 6, 7, 2,
            ]),
            identifyEdges: true,
        });
        spawn(world, [0, 0, 0], BodyType.Static, { kind: ShapeKind.Mesh, geometry: s.mesh });
    } else if (scene === "longRayCast") {
        const phi = f(f(1 + f(Math.sqrt(5))) / 2),
            theta = f(f(2 * f(Math.PI)) / phi);
        const [dc, ds] = computeCosSin(world, theta);
        let c = 1,
            sn = 0;
        const points: number[] = [];
        for (let i = 0; i < 10; i++) {
            const z = f(1 - f(f(2 * i + 1) / 10)),
                r = f(Math.sqrt(f(1 - f(z * z))));
            points.push(f(r * c), f(r * sn), z);
            const old = c;
            c = f(f(dc * c) - f(ds * sn));
            sn = f(f(ds * old) + f(dc * sn));
        }
        const rock = world
            .resource(Hulls)
            .add({ name: "rock", points: new Float32Array(points), maxVertexCount: 10 });
        const mesh = world.resource(PhysicsMeshes).add({
            name: "wave",
            wave: {
                xCount: 8,
                zCount: 8,
                cellWidth: 0.5,
                amplitude: 0.25,
                rowFrequency: f(0.2),
                columnFrequency: f(0.2),
            },
        });
        const field = wave(world, 0.5, 0.08, 0.16, 9);
        spawn(world, [-10, 0, 0], BodyType.Static, { kind: ShapeKind.Sphere, radius: 1 });
        spawn(world, [-5, 0, 0], BodyType.Static, {
            kind: ShapeKind.Capsule,
            capsuleCenter1: [-1, 0, 0],
            capsuleCenter2: [1, 0, 0],
            radius: f(0.7),
        });
        spawn(world, [0, 0, 0], BodyType.Static, { geometry: rock });
        spawn(world, [5, 0, 0], BodyType.Static, { kind: ShapeKind.Mesh, geometry: mesh });
        spawn(world, [8, 0, -2], BodyType.Static, { kind: ShapeKind.HeightField, geometry: field });
    } else {
        s.mesh = torus(world);
        s.box = world.resource(Hulls).add({ name: "box", halfExtents: [f(0.6), f(0.6), f(0.6)] });
        if (scene === "castWorld") s.field = wave(world, 0.5, 0.03, 0.09);
        else {
            const overlap = scene === "overlapWorld";
            if (overlap) s.field = wave(world, f(0.2), 0.03, 0.09);
            for (let row = 0; row < (scene === "rayCurtain" ? 1 : 3); row++)
                for (let col = 0; col < (overlap ? 5 : 4); col++) {
                    const p: Point = [-6 + (overlap ? 3 : 4) * col, 3 + 2 * row, 0];
                    const type =
                        scene === "rayCurtain"
                            ? BodyType.Kinematic
                            : overlap
                              ? row
                              : BodyType.Static;
                    const rotation =
                        scene === "rayCurtain"
                            ? [0, 0, 0, 1]
                            : rotationFromAxisAngle(
                                  world,
                                  col === 1 || col === 2 ? vec(0, 0, 1) : vec(1, 0, 0),
                                  f(
                                      (col === 1 || col === 2 ? 0.25 : col === 4 ? -0.5 : 0.5) *
                                          f(Math.PI),
                                  ),
                              );
                    const shape =
                        col === 0
                            ? { kind: ShapeKind.Sphere, radius: f(overlap ? 0.8 : 0.9) }
                            : col === 1
                              ? {
                                    kind: ShapeKind.Capsule,
                                    capsuleCenter1: [-0.5, 0, 0] as Point,
                                    capsuleCenter2: [0.5, 0, 0] as Point,
                                    radius: f(scene === "rayCurtain" ? 0.8 : overlap ? 0.5 : 0.7),
                                }
                              : col === 2
                                ? { geometry: s.box }
                                : col === 3
                                  ? {
                                        kind: ShapeKind.Mesh,
                                        geometry: s.mesh,
                                        scale: (overlap ? [-0.5, 1.5, -1] : [1, 1, 1]) as Point,
                                    }
                                  : { kind: ShapeKind.HeightField, geometry: s.field };
                    if (col === 4) {
                        p[0] = 5;
                        p[1] = 2 + 2 * row;
                    }
                    const eid = spawn(
                        world,
                        p,
                        col === 4 ? BodyType.Static : type,
                        shape,
                        rotation,
                    );
                    if (scene === "rayCurtain")
                        world.storage(Body).angularVelocity.set(eid, f(0.8), f(0.4), f(0.8));
                }
        }
    }
    return s.bodies;
}
const cameras = {
    rayCurtain: { yaw: 10, pitch: 18, distance: 26, phoneDistance: 45, target: [0, 4, 0] },
    castWorld: { yaw: 20, pitch: 20, distance: 70, phoneDistance: 115, target: [0, 0, 0] },
    meshScale: { yaw: 25, pitch: 20, distance: 9, phoneDistance: 14, target: [0, 0, 0] },
    shapeCast: { yaw: 20, pitch: 20, distance: 30, phoneDistance: 48, target: [0, 5, 0] },
    overlapWorld: { yaw: 20, pitch: 20, distance: 30, phoneDistance: 48, target: [0, 5, 0] },
    longRayCast: { yaw: 15, pitch: 20, distance: 35, phoneDistance: 60, target: [0, 1, 0] },
    initialOverlap: { yaw: -70, pitch: 15, distance: 10, phoneDistance: 16, target: [-1, 0, 0] },
    capsuleCastRay: { yaw: 0, pitch: 15, distance: 5, phoneDistance: 7, target: [0, 0.5, 0] },
};
export const collisionScenes = Object.fromEntries(
    collisionNames.map((scene) => [
        scene,
        {
            construct: (world: World) => construct(world, scene),
            gridExtent: scene === "castWorld" ? 30 : 10,
            gridSpacing: scene === "castWorld" ? 2 : 1,
            camera: cameras[scene],
            controls: undefined as Controls | undefined,
        },
    ]),
) as Record<
    Scene,
    {
        construct: (world: World) => number[];
        gridExtent: number;
        gridSpacing: number;
        camera: (typeof cameras)[Scene];
        controls: Controls | undefined;
    }
>;

function row(
    label: string,
    value: number,
    set: (world: World, value: number) => void,
    min?: number,
    max?: number,
    step?: number,
): Controls[string] {
    return {
        label,
        value,
        min,
        max,
        step,
        decimals: step === 0.1 ? 1 : 0,
        set(world, _bodies, value) {
            set(world, f(value));
        },
    };
}
function choice(
    label: string,
    value: number,
    options: string[],
    set: (world: World, value: number) => void,
): Controls[string] {
    return {
        ...row(label, value, set),
        select: true,
        options: options.map((label, value) => ({ label, value })),
    };
}
function random(world: World, lo: number, hi: number): number {
    const s = world.resource(CollisionState);
    let x = s.seed;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    s.seed = x >>> 0;
    return f(f(f(hi - lo) * f((s.seed % 32768) / 32767)) + lo);
}
function createShapes(world: World, kind: ShapeKind, count: number): void {
    const s = world.resource(CollisionState);
    for (let i = 0; i < count; i++) {
        const index = s.bodyIndex,
            prior = s.bodies[index];
        if (prior) {
            world.destroy(prior);
            s.ignored = s.ignored.filter((eid) => eid !== prior);
        }
        const type =
            kind === ShapeKind.HeightField
                ? BodyType.Static
                : index % 3 === 0
                  ? BodyType.Kinematic
                  : index % 2 === 0
                    ? BodyType.Dynamic
                    : BodyType.Static;
        const p: Point = [random(world, -20, 20), random(world, -20, 20), random(world, -20, 20)];
        const axis = vec(random(world, -1, 1), random(world, -1, 1), random(world, -1, 1));
        const inv = f(
            1 / f(Math.sqrt(f(f(f(axis.x * axis.x) + f(axis.y * axis.y)) + f(axis.z * axis.z)))),
        );
        axis.x = f(axis.x * inv);
        axis.y = f(axis.y * inv);
        axis.z = f(axis.z * inv);
        const angle = random(world, -f(Math.PI), f(Math.PI));
        const shape =
            kind === ShapeKind.Sphere
                ? { kind, radius: f(0.9) }
                : kind === ShapeKind.Capsule
                  ? {
                        kind,
                        capsuleCenter1: [-0.5, 0, 0] as Point,
                        capsuleCenter2: [0.5, 0, 0] as Point,
                        radius: f(0.8),
                    }
                  : kind === ShapeKind.Hull
                    ? { kind, geometry: s.box }
                    : kind === ShapeKind.Mesh
                      ? { kind, geometry: s.mesh, scale: [4, 3, -2] as Point }
                      : { kind, geometry: s.field };
        const eid = spawn(world, p, type, shape, rotationFromAxisAngle(world, axis, angle));
        s.bodies.pop();
        s.bodies[index] = eid;
        world.storage(Body).gravityScale.set(eid, 0);
        world
            .storage(Shape)
            .userMaterialIdLow.set(
                eid,
                kind === ShapeKind.Sphere
                    ? 11
                    : kind === ShapeKind.Capsule
                      ? 22
                      : kind === ShapeKind.Hull
                        ? 33
                        : kind === ShapeKind.Mesh
                          ? 44
                          : 55,
            );
        if (kind === ShapeKind.HeightField)
            world.storage(Shape).materialSet.set(
                eid,
                world.resource(ShapeMaterials).add({
                    name: "query materials",
                    materials: [111n, 222n, 333n].map((userMaterialId) => ({
                        friction: 0,
                        restitution: 0,
                        rollingResistance: 0,
                        tangentVelocity: [0, 0, 0],
                        userMaterialId,
                        customColor: 0,
                    })),
                }),
            );
        if ((index & 7) === 7) s.ignored.push(eid);
        s.bodyIndex = (index + 1) % 64;
    }
}
const castControls: Controls = {
    "Cast Type": choice("Cast Type", 0, ["Ray", "Sphere", "Capsule", "Box"], (w, v) => {
        w.resource(CollisionState).castType = v;
    }),
    Radius: {
        ...row(
            "Radius",
            0.5,
            (w, v) => {
                w.resource(CollisionState).radius = v;
            },
            0.1,
            2,
            0.1,
        ),
        enabled: (w) => [1, 2].includes(w.resource(CollisionState).castType),
    },
    Mode: choice("Mode", 1, ["Any", "Closest", "Multiple", "Sorted"], (w, v) => {
        w.resource(CollisionState).mode = v;
    }),
    "Initial Overlap": row("Initial Overlap", 0, (w, v) => {
        w.resource(CollisionState).initialOverlap = !!v;
    }),
};
for (const [label, kind, count] of [
    ["Spheres", ShapeKind.Sphere, 10],
    ["Capsules", ShapeKind.Capsule, 10],
    ["Hulls", ShapeKind.Hull, 10],
    ["Meshes", ShapeKind.Mesh, 1],
    ["Height Field", ShapeKind.HeightField, 1],
] as const)
    castControls[label] = { ...row(label, 0, (w) => createShapes(w, kind, count)), button: true };
castControls["Destroy Shape"] = {
    ...row("Destroy Shape", 0, (w) => {
        const s = w.resource(CollisionState),
            index = s.bodies.findIndex((eid) => !!eid);
        if (index >= 0) {
            w.destroy(s.bodies[index]!);
            s.bodies[index] = 0;
        }
    }),
    button: true,
};
for (let i = 0; i < 3; i++)
    for (const [label, field] of [
        ["Material", "userMaterialId"],
        ["Triangle", "triangleIndex"],
    ] as const)
        castControls[`${label} ${i + 1}`] = {
            label: `${label} ${i + 1}`,
            value: 0,
            read: (w) => {
                const hit = w.resource(CollisionState).sweeps[0]?.hits[i];
                return hit ? Number(hit[field]) : null;
            },
            set() {},
        };
collisionScenes.castWorld.controls = castControls;
const meshControls: Controls = {};
for (const [i, axis] of ["X", "Y", "Z"].entries())
    meshControls[`Scale ${axis}`] = row(
        `Scale ${axis}`,
        1,
        (w, v) => {
            const s = w.resource(CollisionState),
                shapes = w.storage(Shape),
                scale = shapes.scale;
            const p: Point = [
                scale.x.get(s.bodies[0]!),
                scale.y.get(s.bodies[0]!),
                scale.z.get(s.bodies[0]!),
            ];
            p[i] = v;
            scale.set(s.bodies[0]!, ...p);
        },
        -2,
        2,
        0.1,
    );
for (const [label, axis] of [
    ["Start Y", "y"],
    ["Start Z", "z"],
] as const)
    meshControls[label] = row(
        label,
        0,
        (w, v) => {
            w.resource(CollisionState).start[axis] = v;
        },
        -2,
        2,
        0.1,
    );
meshControls["sphere Cast"] = row("sphere Cast", 1, (w, v) => {
    w.resource(CollisionState).sphereCast = !!v;
});
collisionScenes.meshScale.controls = meshControls;
for (const scene of ["shapeCast", "initialOverlap"] as const) {
    const label = scene === "shapeCast" ? "Initial Overlap" : "initial overlap";
    collisionScenes[scene].controls = {
        [label]: row(label, Number(scene === "initialOverlap"), (w, v) => {
            w.resource(CollisionState).initialOverlap = !!v;
        }),
    };
}
collisionScenes.longRayCast.controls = {
    "Ray Length": {
        ...row(
            "Ray Length, km",
            1,
            (w, v) => {
                w.resource(CollisionState).rayLength = v;
            },
            1,
            10000,
            1,
        ),
        logarithmic: true,
    },
    "Cone Angle": row(
        "Cone Angle, deg",
        5,
        (w, v) => {
            w.resource(CollisionState).coneAngle = v;
        },
        0,
        12,
        0.1,
    ),
};
for (const [i, name] of ["Sphere", "Capsule", "Hull", "Mesh", "Height Field"].entries()) {
    collisionScenes.longRayCast.controls[`${name} error`] = {
        label: `${name} error, m`,
        value: 0,
        precision: 4,
        read: (w) => w.resource(CollisionState).errors[i]!,
        set() {},
    };
    collisionScenes.longRayCast.controls[`${name} failures`] = {
        label: `${name} failures, %`,
        value: 0,
        decimals: 0,
        read: (w) => 100 * w.resource(CollisionState).failures[i]!,
        set() {},
    };
}
