import { type Plugin, recoverResource, type World } from "@dylanebert/shallot";
import {
    type DistanceOutput,
    PhysicsWorldDefinition,
    type ShapeCastOutput,
    type SimplexCache,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    type Sweep,
    shapeCast,
    shapeDistance,
    type TOIOutput,
    timeOfImpact,
} from "@dylanebert/shallot/standard/physics";
import type { Controls } from "./scenes";

const f = Math.fround;
export type Vec = { x: number; y: number; z: number };
export const vec = (x = 0, y = 0, z = 0): Vec => ({ x: f(x), y: f(y), z: f(z) });
export const identity = () => ({ p: vec(), q: { v: vec(), s: 1 } });
export const box = (x: number, y: number, z: number, cy = 0) => [
    vec(x, y + cy, z),
    vec(-x, y + cy, z),
    vec(-x, -y + cy, z),
    vec(x, -y + cy, z),
    vec(x, y + cy, -z),
    vec(-x, y + cy, -z),
    vec(-x, -y + cy, -z),
    vec(x, -y + cy, -z),
];
const cache = (): SimplexCache => ({
    metric: 0,
    count: 0,
    indexA: [0, 0, 0, 0],
    indexB: [0, 0, 0, 0],
});
export const debugNames = [
    "shapeCastDebug",
    "distanceDebug",
    "shapeDistance",
    "timeOfImpact",
] as const;
type Scene = (typeof debugNames)[number];
export const CollisionDebug = {
    create: () => ({
        scene: "" as Scene | "",
        typeA: 2,
        typeB: 3,
        radiusA: 0,
        radiusB: 0,
        showIndices: false,
        useCache: false,
        drawSimplex: false,
        index: 0,
        transform: identity(),
        cache: cache(),
        distance: null as DistanceOutput | null,
        cast: null as ShapeCastOutput | null,
        toi: null as TOIOutput | null,
        pointsA: [] as Vec[],
        pointsB: [] as Vec[],
        sweepA: null as Sweep | null,
        sweepB: null as Sweep | null,
        tracking: false,
        dragOrigin: vec(),
        dragBase: vec(),
    }),
};
export const debugTriangle = [vec(-1.5, 0, 0), vec(1.5, 0, 0), vec(0, 0, 2)];
export const toiTriangle = [vec(-4, 0, -4), vec(-4, 0, -8), vec(-8, 0, -8)];
export const castTriangle = [
    vec(),
    vec(0, f(f(0.01) * -6400), 0),
    vec(f(f(0.01) * 6400), 0, f(f(0.01) * 22.609375)),
];
export const castCapsule = [
    vec(f(f(0.01) * f(43616.2109375)), f(f(0.01) * -100213), f(f(0.01) * f(132631.8125))),
    vec(f(f(0.01) * f(342231.96875)), f(f(0.01) * f(359711.6875)), f(f(0.01) * f(132631.8125))),
];
export const castTranslation = vec(f(f(0.01) * f(0.008614914)), 0, f(f(0.01) * f(72267.1171875)));
function points(type: number, toi: boolean): Vec[] {
    if (toi)
        return type === 0
            ? box(f(0.02), f(0.2), f(0.04))
            : type === 1
              ? [vec(0, -0.2, 0), vec(0, 0.2, 0)]
              : toiTriangle;
    return type === 0
        ? [vec()]
        : type === 1
          ? [vec(-0.5, 0, 0), vec(0.5, 0, 0)]
          : type === 2
            ? debugTriangle
            : box(0.125, 0.25, 0.5);
}
export function debugQuery(world: World): void {
    const s = world.resource(CollisionDebug);
    if (!s.scene) return;
    if (s.scene === "shapeCastDebug") {
        s.cast = shapeCast(world, {
            proxyA: { points: castTriangle, count: 3, radius: 0 },
            proxyB: { points: castCapsule, count: 2, radius: f(f(0.01) * 1) },
            transform: s.transform,
            translationB: castTranslation,
            maxFraction: f(0.970617533),
            canEncroach: false,
        });
    } else if (s.scene === "timeOfImpact") {
        s.pointsA = points(s.typeA, true);
        s.pointsB = points(s.typeB, true);
        s.toi = timeOfImpact(world, {
            proxyA: {
                points: s.pointsA,
                count: s.pointsA.length,
                radius: s.typeA === 1 ? f(0.02) : 0,
            },
            proxyB: {
                points: s.pointsB,
                count: s.pointsB.length,
                radius: s.typeB === 1 ? f(0.02) : 0,
            },
            sweepA: s.sweepA!,
            sweepB: s.sweepB!,
            maxFraction: 1,
        });
    } else {
        const debug = s.scene === "distanceDebug";
        s.pointsA = debug ? box(40, 1, 40) : points(s.typeA, false);
        s.pointsB = debug ? box(0.5, 10, 0.5, 10) : points(s.typeB, false);
        if (!s.useCache || debug) s.cache.count = 0;
        s.distance = shapeDistance(
            world,
            {
                proxyA: { points: s.pointsA, count: s.pointsA.length, radius: s.radiusA },
                proxyB: { points: s.pointsB, count: s.pointsB.length, radius: s.radiusB },
                transform: s.transform,
                useRadii: !debug && (s.radiusA > 0 || s.radiusB > 0),
            },
            s.cache,
            debug ? 32 : 20,
        );
    }
}
function construct(world: World, scene: Scene): number[] {
    const s = world.resource(CollisionDebug);
    Object.assign(s, CollisionDebug.create());
    s.scene = scene;
    world.resource(PhysicsWorldDefinition).enableAutomaticStep = scene === "shapeCastDebug";
    if (scene === "shapeCastDebug")
        s.transform.p = vec(f(f(0.01) * -115200), f(f(0.01) * -19200), f(f(0.01) * -202755));
    if (scene === "distanceDebug")
        s.transform = {
            p: vec(-1.64657831e-6, 1.00989532471, 0),
            q: { v: vec(0, 0, 0.004947796), s: f(0.999987781) },
        };
    if (scene === "shapeDistance") s.transform.p = vec(0, 1, 0);
    if (scene === "timeOfImpact") {
        s.typeA = 2;
        s.typeB = 1;
        s.sweepA = { localCenter: vec(), c1: vec(), c2: vec(), q1: identity().q, q2: identity().q };
        s.sweepB = {
            localCenter: vec(),
            c1: vec(-4.0651207, 0.101333618, -7.87591267),
            c2: vec(-4.15895557, 0.0356027633, -7.69682646),
            q1: { v: vec(-0.860495985, -0.272824734, 0.0724888667), s: f(0.424097389) },
            q2: { v: vec(-0.604184389, -0.424355596, 0.0457959622), s: f(0.672894001) },
        };
    }
    return [];
}
export const CollisionDebugPlugin: Plugin = {
    name: "CollisionDebug",
    dependencies: [StandardPhysicsPlugin],
    recovery: recoverResource(CollisionDebug),
    systems: [
        {
            name: "collision-debug-query",
            group: "fixed",
            after: [StepPhysicsSystem],
            update: debugQuery,
        },
    ],
};
const cameras = {
    shapeCastDebug: {
        yaw: 0,
        pitch: 15,
        distance: 9000,
        phoneDistance: 14000,
        target: [800, 800, -300],
    },
    distanceDebug: {
        yaw: 20,
        pitch: 15,
        distance: 115,
        phoneDistance: 180,
        target: [0, 3, 0],
        phoneTarget: [0, 60, 0],
    },
    shapeDistance: {
        yaw: 20,
        pitch: 25,
        distance: 7,
        phoneDistance: 11,
        target: [0, 0.5, 0.5],
        phoneTarget: [0, 4.5, 0.5],
    },
    timeOfImpact: {
        yaw: 20,
        pitch: 25,
        distance: 10,
        phoneDistance: 16,
        target: [-6, 0, -6],
        phoneTarget: [-6, 5, -6],
    },
};
export const debugScenes = Object.fromEntries(
    debugNames.map((scene) => [
        scene,
        {
            construct: (world: World) => construct(world, scene),
            camera: cameras[scene],
            gridExtent: scene === "distanceDebug" ? 40 : scene === "shapeCastDebug" ? 4000 : 10,
            gridSpacing: scene === "shapeCastDebug" ? 500 : scene === "distanceDebug" ? 5 : 1,
            controls: scene === "shapeCastDebug" ? undefined : ({} as Controls),
        },
    ]),
) as Record<
    Scene,
    {
        construct: (world: World) => number[];
        camera: (typeof cameras)[Scene];
        gridExtent: number;
        gridSpacing: number;
        controls: Controls | undefined;
    }
>;
function slider(
    label: string,
    value: number,
    set: (s: ReturnType<typeof CollisionDebug.create>, value: number) => void,
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
        decimals: step === 0.01 ? 2 : 0,
        set(w, _b, v) {
            set(w.resource(CollisionDebug), f(v));
        },
    };
}
for (const scene of ["shapeDistance", "timeOfImpact"] as const) {
    const controls = debugScenes[scene].controls!;
    for (const side of ["A", "B"] as const) {
        const field = side === "A" ? "typeA" : "typeB";
        controls[`shape ${side}`] = {
            ...slider(
                `shape ${side}`,
                side === "A" ? 2 : scene === "shapeDistance" ? 3 : 1,
                (s, v) => {
                    s[field] = v;
                },
            ),
            select: true,
            options: (scene === "shapeDistance"
                ? ["point", "segment", "triangle", "box"]
                : ["box", "capsule", "triangle"]
            ).map((label, value) => ({ label, value })),
        };
        if (scene === "shapeDistance") {
            const radius = side === "A" ? "radiusA" : "radiusB";
            controls[`radius ${side}`] = {
                ...slider(
                    `radius ${side}, m`,
                    0,
                    (s, v) => {
                        s[radius] = v;
                    },
                    0,
                    0.5,
                    0.01,
                ),
                enabled: (w) => w.resource(CollisionDebug)[field] < 2,
            };
        }
    }
}
const controls = debugScenes.shapeDistance.controls!;
for (const [label, field] of [
    ["show indices", "showIndices"],
    ["use cache", "useCache"],
    ["draw simplex", "drawSimplex"],
] as const)
    controls[label] = slider(label, 0, (s, v) => {
        s[field] = !!v;
        if (field === "drawSimplex") s.index = 0;
    });
for (const scene of ["shapeDistance", "distanceDebug"] as const) {
    const c = debugScenes[scene].controls!,
        label = scene === "shapeDistance" ? "index" : "simplex index";
    c[label] = {
        ...slider(
            label,
            0,
            (s, v) => {
                s.index = v;
            },
            0,
            31,
            1,
        ),
        bounds: (_v, w) => [
            0,
            Math.max(0, (w?.resource(CollisionDebug).distance?.simplexCount ?? 1) - 1),
        ],
        ...(scene === "shapeDistance" ? { when: ["draw simplex"] } : {}),
    };
}
function read(
    scene: Scene,
    name: string,
    get: (s: ReturnType<typeof CollisionDebug.create>) => number | string | null,
    decimals?: number,
) {
    debugScenes[scene].controls![name] = {
        label: name,
        value: 0,
        decimals,
        precision: 6,
        read: (w) => get(w.resource(CollisionDebug)),
        set() {},
    };
}
for (const scene of ["shapeDistance", "distanceDebug"] as const) {
    read(
        scene,
        "Distance, m",
        (s) => s.distance?.distance ?? null,
        scene === "shapeDistance" ? 4 : undefined,
    );
    read(scene, "Iterations", (s) => s.distance?.iterations ?? null);
    read(scene, "Simplex count", (s) => s.distance?.simplexCount ?? null);
    read(scene, "Cache A : B", (s) =>
        s.cache.count
            ? `${s.cache.indexA.slice(0, s.cache.count).join("/")} : ${s.cache.indexB.slice(0, s.cache.count).join("/")}`
            : null,
    );
    if (scene === "distanceDebug") {
        for (const axis of ["x", "y", "z"] as const)
            read(scene, `Normal ${axis.toUpperCase()}`, (s) => s.distance?.normal[axis] ?? null);
        read(
            scene,
            "Current distance, m",
            (s) => {
                const simplex = s.distance?.simplexes[s.index];
                if (!simplex) return null;
                let x = 0,
                    y = 0,
                    z = 0;
                for (let i = 0; i < simplex.count; i++) {
                    const v = simplex.vertices[i]!;
                    x = f(x + f(v.a * v.w.x));
                    y = f(y + f(v.a * v.w.y));
                    z = f(z + f(v.a * v.w.z));
                }
                if (simplex.count === 1) ({ x, y, z } = simplex.vertices[0]!.w);
                return f(Math.sqrt(f(f(f(x * x) + f(y * y)) + f(z * z))));
            },
            5,
        );
        for (let i = 0; i < 32; i++)
            read(scene, `Simplex ${i} A : B`, (s) => {
                if (i >= (s.distance?.simplexCount ?? 0)) return null;
                const vertices = s.distance!.simplexes[i]!.vertices.slice(
                    0,
                    s.distance!.simplexes[i]!.count,
                );
                return `${vertices.map((v) => v.indexA).join("/")} : ${vertices.map((v) => v.indexB).join("/")}`;
            });
    }
}
read("timeOfImpact", "State", (s) =>
    s.toi ? ["unknown", "failed", "overlapped", "hit", "separated"][s.toi.state]! : null,
);
read("timeOfImpact", "Fraction", (s) => s.toi?.fraction ?? null);
for (const field of ["distanceIterations", "pushBackIterations", "rootIterations"] as const)
    read(
        "timeOfImpact",
        field === "distanceIterations"
            ? "Iterations"
            : field === "pushBackIterations"
              ? "Push iterations"
              : "Root iterations",
        (s) => s.toi?.[field] ?? null,
    );
export function debugDrag(
    world: World,
    origin: Vec,
    translation: Vec,
    press: boolean,
    held: boolean,
): boolean {
    const s = world.resource(CollisionDebug);
    if (s.scene !== "shapeDistance") return false;
    const length = Math.sqrt(translation.x ** 2 + translation.y ** 2 + translation.z ** 2);
    const at = vec(
        origin.x + (10 * translation.x) / length,
        origin.y + (10 * translation.y) / length,
        origin.z + (10 * translation.z) / length,
    );
    if (press && held) {
        const local = vec(
            origin.x - s.transform.p.x,
            origin.y - s.transform.p.y,
            origin.z - s.transform.p.z,
        );
        s.tracking = shapeCast(world, {
            proxyA: {
                points: points(s.typeB, false),
                count: points(s.typeB, false).length,
                radius: s.radiusB,
            },
            proxyB: { points: [local], count: 1, radius: 0 },
            transform: identity(),
            translationB: translation,
            maxFraction: 1,
            canEncroach: false,
        }).hit;
        if (s.tracking) {
            Object.assign(s.dragOrigin, at);
            Object.assign(s.dragBase, s.transform.p);
        }
    }
    if (s.tracking)
        s.transform.p = vec(
            s.dragBase.x + f(at.x - s.dragOrigin.x),
            s.dragBase.y + f(at.y - s.dragOrigin.y),
            s.dragBase.z + f(at.z - s.dragOrigin.z),
        );
    if (!held) s.tracking = false;
    return s.tracking;
}
