import { type Plugin, recoverResource, type World } from "@dylanebert/shallot";
import {
    collideCapsuleAndSphere,
    collideCapsules,
    collideHullAndCapsule,
    collideHullAndSphere,
    collideHulls,
    collideSpheres,
    collideTriangleAndCapsule,
    collideTriangleAndHull,
    collideTriangleAndSphere,
    type HullData,
    type LocalManifold,
    makeBoxHull,
    makeScaledBoxHull,
    PhysicsWorldDefinition,
    rotationFromAxisAngle,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
} from "@dylanebert/shallot/standard/physics";
import type { Controls } from "./scenes";

const f = Math.fround;
export const manifoldNames = [
    "sphereVsSphere",
    "capsuleVsSphere",
    "hullVsSphere",
    "triangleVsSphere",
    "capsuleVsCapsule",
    "capsuleVsHull",
    "triangleVsCapsule",
    "hullVsHull",
    "triangleVsHull",
] as const;
export type ManifoldScene = (typeof manifoldNames)[number];
export const manifoldPointLimits = [1, 1, 1, 1, 2, 2, 2, 4, 7] as const;
export const vector = (x = 0, y = 0, z = 0) => ({ x: f(x), y: f(y), z: f(z) });
type Vec = ReturnType<typeof vector>;
export type ManifoldPose = { p: Vec; q: { v: Vec; s: number } };
const pose = (x = 0, y = 0, z = 0): ManifoldPose => ({
    p: vector(x, y, z),
    q: { v: vector(), s: 1 },
});
const simplex = () => ({ metric: 0, count: 0, indexA: [0, 0, 0, 0], indexB: [0, 0, 0, 0] });
const sat = () => ({ separation: 0, type: 0, indexA: 0, indexB: 0, hit: 0 });
export const ManifoldSample = {
    create: () => ({
        scene: "" as ManifoldScene | "",
        a: pose(),
        b: pose(),
        sphere: { center: vector(), radius: 1 },
        capsule: { center1: vector(-2), center2: vector(2), radius: 1 },
        hullA: null as HullData | null,
        hullB: null as HullData | null,
        triangle: [vector(), vector(), vector()] as [Vec, Vec, Vec],
        simplex: simplex(),
        sat: sat(),
        useCache: false,
        manualFeature: 0,
        manifold: null as LocalManifold | null,
        tracking: false,
        origin: vector(),
        base: vector(),
    }),
};
// Box3D's quaternion rotation, with float intermediates.
export function manifoldRotate(q: ManifoldPose["q"], p: Vec): Vec {
    const { x, y, z } = q.v,
        w = q.s;
    const tx = f(f(f(y * p.z) - f(z * p.y)) + f(w * p.x)),
        ty = f(f(f(z * p.x) - f(x * p.z)) + f(w * p.y)),
        tz = f(f(f(x * p.y) - f(y * p.x)) + f(w * p.z));
    return vector(
        f(p.x + f(2 * f(f(y * tz) - f(z * ty)))),
        f(p.y + f(2 * f(f(z * tx) - f(x * tz)))),
        f(p.z + f(2 * f(f(x * ty) - f(y * tx)))),
    );
}
export function manifoldTransform(t: ManifoldPose, p: Vec): Vec {
    const r = manifoldRotate(t.q, p);
    return vector(f(t.p.x + r.x), f(t.p.y + r.y), f(t.p.z + r.z));
}
function relative(a: ManifoldPose, b: ManifoldPose): ManifoldPose {
    const av = a.q.v,
        bv = b.q.v,
        p = vector(b.p.x - a.p.x, b.p.y - a.p.y, b.p.z - a.p.z),
        w = a.q.s;
    const tx = f(f(f(av.y * p.z) - f(av.z * p.y)) - f(w * p.x)),
        ty = f(f(f(av.z * p.x) - f(av.x * p.z)) - f(w * p.y)),
        tz = f(f(f(av.x * p.y) - f(av.y * p.x)) - f(w * p.z));
    return {
        p: vector(
            f(p.x + f(2 * f(f(av.y * tz) - f(av.z * ty)))),
            f(p.y + f(2 * f(f(av.z * tx) - f(av.x * tz)))),
            f(p.z + f(2 * f(f(av.x * ty) - f(av.y * tx)))),
        ),
        q: {
            v: vector(
                f(f(f(f(bv.y * av.z) - f(bv.z * av.y)) + f(w * bv.x)) - f(b.q.s * av.x)),
                f(f(f(f(bv.z * av.x) - f(bv.x * av.z)) + f(w * bv.y)) - f(b.q.s * av.y)),
                f(f(f(f(bv.x * av.y) - f(bv.y * av.x)) + f(w * bv.z)) - f(b.q.s * av.z)),
            ),
            s: f(f(w * b.q.s) + f(f(f(av.x * bv.x) + f(av.y * bv.y)) + f(av.z * bv.z))),
        },
    };
}
export function stepManifold(w: World): void {
    const s = w.resource(ManifoldSample);
    if (!s.scene) return;
    if (!s.useCache) {
        s.simplex = simplex();
        s.sat = sat();
    } else if (s.manualFeature && (s.scene === "hullVsHull" || s.scene === "triangleVsHull"))
        s.sat.type = 5 + s.manualFeature;
    const t = relative(s.a, s.b),
        c = s.capsule,
        h = s.hullA!,
        sp = s.sphere;
    const triangle = () =>
        s.triangle.map((p) => manifoldTransform(relative(s.b, s.a), p)) as [Vec, Vec, Vec];
    switch (s.scene) {
        case "sphereVsSphere":
            s.manifold = collideSpheres(w, sp, sp, t);
            break;
        case "capsuleVsSphere":
            s.manifold = collideCapsuleAndSphere(w, c, sp, t);
            break;
        case "hullVsSphere":
            s.manifold = collideHullAndSphere(w, h, sp, t, s.simplex);
            break;
        case "triangleVsSphere":
            s.manifold = collideTriangleAndSphere(w, triangle(), sp, 8);
            break;
        case "capsuleVsCapsule":
            s.manifold = collideCapsules(w, c, c, t);
            break;
        case "capsuleVsHull":
            s.manifold = collideHullAndCapsule(w, h, c, t, s.simplex);
            break;
        case "triangleVsCapsule":
            s.manifold = collideTriangleAndCapsule(w, triangle(), c, s.simplex, 8);
            break;
        case "hullVsHull":
            s.manifold = collideHulls(w, h, s.hullB!, t, s.sat);
            break;
        case "triangleVsHull":
            s.manifold = collideTriangleAndHull(w, triangle(), 0, s.hullB!, s.sat, true, 8);
            break;
    }
}
export const ManifoldPlugin: Plugin = {
    name: "ManifoldSamples",
    dependencies: [StandardPhysicsPlugin],
    recovery: recoverResource(ManifoldSample),
    systems: [
        {
            name: "manifold-samples",
            group: "fixed",
            after: [StepPhysicsSystem],
            update: stepManifold,
        },
    ],
};
function construct(w: World, scene: ManifoldScene): number[] {
    const s = w.resource(ManifoldSample);
    Object.assign(s, ManifoldSample.create(), { scene });
    w.resource(PhysicsWorldDefinition).enableAutomaticStep = false;
    if (scene === "sphereVsSphere") {
        s.a = pose(3.5, 0.5, 0);
        const q = rotationFromAxisAngle(w, vector(0, 1), f(0.5 * f(Math.PI)));
        s.a.q = { v: vector(q[0], q[1], q[2]), s: q[3]! };
        s.b = pose(0, 1.5, 3.5);
        s.sphere = { center: vector(0.5, 0, -0.25), radius: 2 };
    } else if (scene === "capsuleVsSphere") {
        s.b = pose(-4);
        s.sphere.radius = 2;
    } else if (scene === "hullVsSphere") {
        s.b = pose(1.5);
        s.hullA = makeBoxHull(2, 0.5, 0.5, w);
    } else if (scene === "triangleVsSphere") {
        s.triangle = [vector(), vector(4, 0, 4), vector(4)];
        s.b = pose(2, 0.5, 1);
        s.sphere.radius = 0.25;
    } else if (scene === "capsuleVsCapsule") {
        s.a = pose(1, 1);
        s.b = pose(-4, 1);
    } else if (scene === "capsuleVsHull") {
        s.hullA = makeBoxHull(1, 0.5, 0.5, w);
        s.capsule = { center1: vector(-1), center2: vector(1), radius: f(0.15) };
        s.b = {
            p: vector(1.58523774, 0.729615569, 0.451690674),
            q: { v: vector(-0.00256555085, -0.0201825816, 0.126076236), s: f(0.991811991) },
        };
    } else if (scene === "triangleVsCapsule") {
        s.triangle = [vector(-4, 0, -4), vector(-4), vector()];
        s.b = pose(-1, 0, -1);
        s.capsule = { center1: vector(-0.5), center2: vector(0.5), radius: f(0.05) };
    } else if (scene === "hullVsHull") {
        s.hullA = makeScaledBoxHull(vector(0.5, 1, 1), pose(1, 0.5), vector(1, 1, 1), w);
        s.hullB = makeBoxHull(0.5, 0.5, 0.5, w);
    } else {
        s.triangle = [
            vector(0.299769998, -1.01549578, -0.744717002),
            vector(0.299769998, -1.01549578, 1.28728306),
            vector(0.299769998, -0.913895786, 0.271283031),
        ];
        s.hullB = makeBoxHull(0.304800004, 0.914399981, 0.304800004, w);
    }
    return [];
}
export const manifoldScenes = Object.fromEntries(
    manifoldNames.map((scene, index) => [
        scene,
        {
            construct: (w: World) => construct(w, scene),
            camera: {
                yaw: 25,
                pitch: 25,
                distance: [12, 13, 8, 8, 14, 7, 8, 5, 4][index]!,
                phoneDistance: [20, 21, 13, 13, 23, 11, 13, 8.5, 9][index]!,
                target: [
                    [1.5, 0.5, 1.5],
                    [-1, 0, 0],
                    [0, 0, 0],
                    [2, 0, 2],
                    [-1.5, 1, 0],
                    [0.6, 0.2, 0],
                    [-2, 0, -2],
                    [0.5, 0.25, 0],
                    [0.15, -0.2, 0.25],
                ][index]!,
                phoneTarget: [
                    [1.5, 6.5, 1.5],
                    [-1, 6, 0],
                    [0, 3.5, 0],
                    [2, 3.5, 2],
                    [-1.5, 7, 0],
                    [0.6, 3.5, 0],
                    [-2, 3.5, -2],
                    [0.5, 3.5, 0],
                    [0.15, 3.3, 0.25],
                ][index]!,
            },
            gridExtent: 12,
            gridSpacing: 1,
            gridHeight:
                index === 0
                    ? -1.5
                    : index === 1
                      ? -2
                      : index === 2 || index === 5 || index === 7
                        ? -0.5
                        : index === 8
                          ? f(-1.01549578)
                          : 0,
            controls: {
                "Use cache": {
                    label: "Use cache",
                    value: 0,
                    set(w, _b, v) {
                        w.resource(ManifoldSample).useCache = Boolean(v);
                    },
                },
                feature: {
                    label: "Feature",
                    value: 0,
                    select: true,
                    when: ["Use cache"],
                    options: [
                        { label: "auto", value: 0 },
                        { label: "faceA", value: 1 },
                        { label: "faceB", value: 2 },
                        { label: "edgePair", value: 3 },
                    ],
                    set(w, _b, v) {
                        w.resource(ManifoldSample).manualFeature = v;
                    },
                },
                count: {
                    label: "Point count",
                    value: 0,
                    decimals: 0,
                    read: (w) => w.resource(ManifoldSample).manifold?.pointCount ?? 0,
                    set() {},
                },
                ...Object.fromEntries(
                    Array.from({ length: manifoldPointLimits[index]! }, (_, i) => [
                        `separation${i}`,
                        {
                            label: `Separation ${i + 1}, ${scene.startsWith("triangle") ? "cm" : "m"}`,
                            value: 0,
                            decimals: scene.startsWith("triangle") ? 2 : 3,
                            read: (w: World) => {
                                const m = w.resource(ManifoldSample).manifold;
                                return m && i < m.pointCount
                                    ? scene.startsWith("triangle")
                                        ? f(100 * m.points[i]!.separation)
                                        : m.points[i]!.separation
                                    : null;
                            },
                            set() {},
                        },
                    ]),
                ),
            } satisfies Controls as Controls,
        },
    ]),
) as Record<
    ManifoldScene,
    {
        construct: (w: World) => number[];
        camera: {
            yaw: number;
            pitch: number;
            distance: number;
            phoneDistance: number;
            target: number[];
            phoneTarget: number[];
        };
        gridExtent: number;
        gridSpacing: number;
        gridHeight: number;
        controls: Controls;
    }
>;
