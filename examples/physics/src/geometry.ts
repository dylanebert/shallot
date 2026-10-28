import { type Plugin, recoverResource, type World } from "@dylanebert/shallot";
import {
    cloneAndTransformHull,
    computeCapsuleMass,
    computeCosSin,
    computeHullMass,
    createCylinder,
    createHull,
    type HullData,
    type MassData,
    makeBoxHull,
    makeScaledBoxHull,
    PhysicsWorldDefinition,
    rotationFromAxisAngle,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
} from "@dylanebert/shallot/standard/physics";
import hullPoints from "./hull-points.json" with { type: "json" };
import type { Controls } from "./scenes";

const f = Math.fround;
type Vec = { x: number; y: number; z: number };
const vec = (x = 0, y = 0, z = 0): Vec => ({ x: f(x), y: f(y), z: f(z) });
export const geometryNames = [
    "boxHull",
    "geometryHull",
    "hullReduction",
    "hullTransform",
    "capsuleMass",
] as const;
type Scene = (typeof geometryNames)[number];
export const Geometry = {
    create: () => ({
        scene: "" as Scene | "",
        halfWidths: vec(1, 0.5, 0.25),
        scale: vec(1, 1, 1),
        angles: vec(),
        offset: vec(),
        type: 1,
        count: 16,
        sides: 6,
        hulls: [] as (HullData | null)[],
        masses: [] as MassData[],
    }),
};
function multiply(a: number[], b: number[]): number[] {
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
function transform(world: World) {
    const s = world.resource(Geometry);
    const angles = [s.angles.x, s.angles.y, s.angles.z].map((a) =>
        s.scene === "boxHull" ? f(f(f(Math.PI) / 180) * a) : f(f(a * f(Math.PI)) / 180),
    );
    const q = multiply(
        rotationFromAxisAngle(world, vec(0, 0, 1), angles[2]!),
        multiply(
            rotationFromAxisAngle(world, vec(0, 1, 0), angles[1]!),
            rotationFromAxisAngle(world, vec(1, 0, 0), angles[0]!),
        ),
    );
    return { p: s.offset, q: { v: vec(q[0], q[1], q[2]), s: q[3]! } };
}
function reductionPoints(world: World, type: number): Vec[] {
    let seed = 42;
    const random = (lo: number, hi: number) => {
        seed ^= seed << 13;
        seed ^= seed >>> 17;
        seed ^= seed << 5;
        return f(f(f(hi - lo) * f(((seed >>> 0) % 32768) / 32767)) + lo);
    };
    return Array.from({ length: 128 }, () => {
        if (type === 0) {
            const p = vec(random(-2, 2), random(-2, 2), random(-2, 2));
            return vec(
                f(Math.max(-1, Math.min(1, p.x)) + random(f(-0.001), f(0.001))),
                f(Math.max(-1, Math.min(1, p.y)) + random(f(-0.001), f(0.001))),
                f(Math.max(-1, Math.min(1, p.z)) + random(f(-0.001), f(0.001))),
            );
        }
        const u1 = random(0, 1),
            u2 = random(0, f(2 * f(Math.PI))),
            u3 = random(0, f(2 * f(Math.PI)));
        const a = f(Math.sqrt(f(1 - u1))),
            b = f(Math.sqrt(u1));
        const cs2 = computeCosSin(world, u2),
            cs3 = computeCosSin(world, u3);
        return vec(f(a * cs2[1]), f(a * cs2[0]), f(b * cs3[1]));
    });
}
export function rebuildGeometry(world: World): void {
    const s = world.resource(Geometry);
    if (s.scene === "boxHull") {
        const t = transform(world),
            h = s.halfWidths,
            scale = s.scale;
        const points: Vec[] = [];
        for (const x of [h.x, -h.x])
            for (const y of [h.y, -h.y])
                for (const z of [h.z, -h.z]) {
                    const p = rotate([t.q.v.x, t.q.v.y, t.q.v.z, t.q.s], vec(x, y, z));
                    const safe = (v: number) =>
                        Math.abs(v) < f(0.01) ? (v < 0 ? f(-0.01) : f(0.01)) : v;
                    points.push(
                        vec(
                            f(safe(scale.x) * f(p.x + t.p.x)),
                            f(safe(scale.y) * f(p.y + t.p.y)),
                            f(safe(scale.z) * f(p.z + t.p.z)),
                        ),
                    );
                }
        s.hulls = [createHull(points, 8, world), makeScaledBoxHull(h, t, scale, world)];
    } else if (s.scene === "geometryHull") {
        s.hulls = [
            createHull(
                hullPoints.map(([x, y, z]) =>
                    vec(f(f(0.01) * f(x!)), f(f(0.01) * f(y!)), f(f(0.01) * f(z!))),
                ),
                16,
                world,
            ),
        ];
    } else if (s.scene === "hullReduction") {
        s.hulls = [createHull(reductionPoints(world, s.type), s.count, world)];
    } else if (s.scene === "hullTransform") {
        const original = s.hulls[0] ?? createCylinder(1, 0.5, 0, 9, world);
        s.hulls = [original, cloneAndTransformHull(original!, transform(world), s.scale, world)];
    } else if (s.scene === "capsuleMass") {
        const points: Vec[] = [],
            d = f(f(Math.PI) / f(s.sides - 1));
        for (const side of [1, -1]) {
            let angle1 = f(side * f(-0.5 * f(Math.PI)));
            for (let i = 0; i < s.sides; i++, angle1 = f(angle1 + d)) {
                const sin = f(Math.sin(angle1)),
                    cos = f(Math.cos(angle1));
                let angle2 = f(-0.5 * f(Math.PI));
                for (let j = 0; j < s.sides; j++, angle2 = f(angle2 + d))
                    points.push(
                        vec(
                            f(side + cos),
                            f(sin * f(Math.cos(angle2))),
                            f(sin * f(Math.sin(angle2))),
                        ),
                    );
            }
        }
        s.hulls = [makeBoxHull(2, 1, 1, world), createHull(points, points.length, world)];
    }
}
export function geometryMasses(world: World): void {
    const s = world.resource(Geometry);
    if (s.scene === "capsuleMass" && s.hulls[1])
        s.masses = [
            computeHullMass(s.hulls[1], 1, world),
            computeCapsuleMass({ center1: vec(-1), center2: vec(1), radius: 1 }, 1, world),
            computeHullMass(s.hulls[0]!, 1, world),
        ];
}
export const GeometryPlugin: Plugin = {
    name: "GeometrySamples",
    dependencies: [StandardPhysicsPlugin],
    recovery: recoverResource(Geometry),
    systems: [
        {
            name: "geometry-masses",
            group: "fixed",
            after: [StepPhysicsSystem],
            update: geometryMasses,
        },
    ],
};
export const geometryScenes = Object.fromEntries(
    geometryNames.map((scene) => [
        scene,
        {
            construct(world: World): number[] {
                Object.assign(world.resource(Geometry), Geometry.create(), { scene });
                world.resource(PhysicsWorldDefinition).enableAutomaticStep = false;
                rebuildGeometry(world);
                return [];
            },
            camera: {
                yaw: 25,
                pitch: 25,
                distance: scene === "hullTransform" ? 11 : scene === "geometryHull" ? 2 : 8,
                phoneDistance: scene === "hullTransform" ? 17 : scene === "geometryHull" ? 3 : 12,
                target: scene === "geometryHull" ? [-0.18, 1, 0.15] : [0, 0, 0],
                phoneTarget:
                    scene === "geometryHull"
                        ? [-0.18, 1, 0.15]
                        : [0, scene === "hullTransform" ? 5.8 : 4.5, 0],
            },
            gridExtent: scene === "geometryHull" ? 2 : 10,
            gridSpacing: scene === "geometryHull" ? 0.25 : 1,
            controls: scene === "geometryHull" ? undefined : ({} as Controls),
        },
    ]),
) as Record<
    Scene,
    {
        construct(w: World): number[];
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
        controls: Controls | undefined;
    }
>;
function slider(
    scene: Scene,
    name: string,
    label: string,
    value: number,
    min: number,
    max: number,
    step: number,
    update: (s: ReturnType<typeof Geometry.create>, v: number) => void,
) {
    geometryScenes[scene].controls![name] = {
        label,
        value,
        min,
        max,
        step,
        decimals: step === 0.1 ? 1 : 0,
        set(w, _b, v) {
            update(w.resource(Geometry), f(v));
            rebuildGeometry(w);
        },
    };
}
for (const scene of ["boxHull", "hullTransform"] as const) {
    if (scene === "boxHull")
        for (const axis of ["x", "y", "z"] as const)
            slider(
                scene,
                `h ${axis.toUpperCase()}`,
                `h ${axis.toUpperCase()}, m`,
                Geometry.create().halfWidths[axis],
                0.1,
                2,
                0.1,
                (s, v) => {
                    s.halfWidths[axis] = v;
                },
            );
    const fields =
        scene === "boxHull"
            ? ([
                  ["c ", "offset", "Center", -2, 2, 0.1],
                  ["r ", "angles", "Rotation", -180, 180, 1],
                  ["s ", "scale", "Scale", -2, 2, 0.1],
              ] as const)
            : ([
                  ["s", "scale", "Scale", -2, 2, 0.1],
                  ["r", "angles", "Rotation", -180, 180, 1],
                  ["p", "offset", "Center", -1, 1, 0.1],
              ] as const);
    for (const [prefix, field, label, min, max, step] of fields) {
        for (const axis of ["x", "y", "z"] as const) {
            const name = prefix + (scene === "boxHull" ? axis.toUpperCase() : axis);
            slider(
                scene,
                name,
                `${label} ${axis.toUpperCase()}${field === "offset" ? ", m" : field === "angles" ? ", °" : ""}`,
                field === "scale" ? 1 : 0,
                min,
                max,
                step,
                (s, v) => {
                    s[field][axis] = v;
                },
            );
        }
    }
}
geometryScenes.boxHull.controls!.Refresh = {
    label: "Refresh",
    value: 0,
    button: true,
    set: rebuildGeometry,
};
geometryScenes.hullReduction.controls!.type = {
    label: "Shape",
    value: 1,
    select: true,
    options: [
        { label: "Box", value: 0 },
        { label: "Sphere", value: 1 },
    ],
    set(w, _b, v) {
        w.resource(Geometry).type = v;
        rebuildGeometry(w);
    },
};
slider("hullReduction", "count", "count", 16, 4, 128, 1, (s, v) => {
    s.count = v;
});
slider("capsuleMass", "sides", "sides", 6, 3, 6, 1, (s, v) => {
    s.sides = v;
});
function read(
    scene: Scene,
    name: string,
    get: (s: ReturnType<typeof Geometry.create>) => number | null,
) {
    geometryScenes[scene].controls![name] = {
        label: name,
        value: 0,
        precision: 6,
        read: (w) => get(w.resource(Geometry)),
        set() {},
    };
}
for (const [field, label] of [
    ["vertexCount", "Vertices"],
    ["faceCount", "Faces"],
    ["edgeCount", "Edges"],
] as const)
    read("hullReduction", label, (s) =>
        s.hulls[0] ? s.hulls[0][field] / (field === "edgeCount" ? 2 : 1) : null,
    );
for (const [i, side] of ["Original", "Transformed"].entries())
    for (const [field, label] of [
        ["surfaceArea", "Area, m²"],
        ["volume", "Volume, m³"],
        ["innerRadius", "Radius, m"],
    ] as const)
        read("hullTransform", `${side} ${label.toLowerCase()}`, (s) => s.hulls[i]?.[field] ?? null);
for (const [i, name] of ["Hull", "Capsule", "Box"].entries()) {
    read("capsuleMass", `${name} mass, kg`, (s) => s.masses[i]?.mass ?? null);
    for (const axis of ["x", "y", "z"] as const)
        read(
            "capsuleMass",
            `${name} I${axis}${axis}, kg·m²`,
            (s) => s.masses[i]?.inertia[`c${axis}`][axis] ?? null,
        );
}
