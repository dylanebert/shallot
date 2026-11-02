import type { World } from "@dylanebert/shallot";
import {
    collideHulls,
    createCylinder,
    createHull,
    type HullData,
    makeBoxHull,
    PhysicsWorldDefinition,
    rotationFromAxisAngle,
} from "@dylanebert/shallot/standard/physics";
import {
    add,
    cross,
    dot,
    invRotate,
    invTransform,
    length,
    manifoldRotate,
    manifoldTransform,
    mul,
    neg,
    normalize,
    pose,
    relative,
    sub,
    type Vec,
    vector,
} from "./manifold-math";
import { ManifoldSample, type ManifoldState, sat } from "./manifold-state";
import type { Controls } from "./scenes";

const f = Math.fround,
    MAX = f(3.4028234663852886e38);
export interface CullingState {
    cylinders: boolean;
    height: number;
    radius: number;
    sides: number;
    showSpheres: boolean;
    showCulled: boolean;
    showNormals: boolean;
    edgeBound: number;
    facesA: number[];
    facesB: number[];
    edgesA: number[];
    edgesB: number[];
    seedA: number;
    seedB: number;
    probeA: number;
    probeB: number;
    keptA: number;
    keptB: number;
    sphereKeptA: number;
    sphereKeptB: number;
    separationA: number;
    separationB: number;
    gap: number;
    separatedFeature: number;
    probeApplied: boolean;
}
export interface EdgeAxis {
    normal: Vec;
    separation: number;
    edgeA: number;
    edgeB: number;
}
export interface Arc {
    u: Vec;
    v: Vec;
    peak: Vec;
    a: number;
    b: number;
    c: number;
    separationU: number;
    separationV: number;
    outsideDegrees: number;
    inside: boolean;
}
export interface WedgeState {
    hullType: number;
    seed: number;
    searchCount: number;
    found: boolean;
    edge: EdgeAxis;
    arcA: Arc | null;
    arcB: Arc | null;
    faceA: number;
    faceB: number;
    edgeWins: boolean;
    centerOffset: Vec;
    radius: number;
    gaussCenter: Vec;
    gaussRadius: number;
}
function complexPoints(): Vec[] {
    let seed = 42;
    const random = () => {
        seed = (Math.imul(seed, 2147001325) + 715136305) >>> 0;
        return f(f((seed & 0xffff) / 65535) - 0.5);
    };
    return Array.from({ length: 32 }, () => {
        let p: Vec;
        do {
            p = vector(random(), random(), random());
        } while (dot(p, p) > 0.25);
        return mul(2, normalize(p));
    });
}
function complex(world: World): HullData {
    const points = complexPoints();
    return createHull(points, 32, world)!;
}
function cullingHulls(world: World): void {
    const s = world.resource(ManifoldSample),
        c = s.culling!;
    s.hullA = c.cylinders
        ? createCylinder(c.height, c.radius, f(-0.5 * c.height), c.sides, world)
        : complex(world);
    s.hullB = c.cylinders
        ? createCylinder(c.height, c.radius, f(-0.5 * c.height), c.sides, world)
        : complex(world);
}
export function faceSeparation(
    a: HullData,
    b: HullData,
    t: ReturnType<typeof pose>,
    face: number,
    sideB = false,
): { separation: number; index: number } {
    const plane = (sideB ? b : a).planes[face]!;
    let separation = MAX,
        index = 0;
    const points = (sideB ? a : b).points;
    for (let i = 0; i < points.length; i++) {
        const point = sideB ? invTransform(t, points[i]!) : manifoldTransform(t, points[i]!);
        const value = f(dot(plane.normal, point) - plane.offset);
        if (value < separation) {
            separation = value;
            index = i;
        }
    }
    return { separation, index };
}
export function arcCanReach(a: number, b: number, c: number, bound: number): boolean {
    const hi = Math.max(a, b),
        lo = Math.min(a, b),
        u = f(lo - f(c * hi)),
        s = f(1 - f(c * c));
    return (
        hi >= bound ||
        (u >= 0 && (f(u * u) >= f(f(f(bound - hi) * f(bound + hi)) * s) || s < f(1e-4)))
    );
}
function cullEdges(h: HullData, point: Vec, bound: number, states: number[]): number {
    let kept = 0;
    for (let i = 0; i < h.edgeCount; i += 2) {
        if (states[i / 2] === 1) continue;
        const p = h.planes[h.edges[i]!.face]!,
            q = h.planes[h.edges[i + 1]!.face]!;
        const keep = arcCanReach(
            f(dot(p.normal, point) - p.offset),
            f(dot(q.normal, point) - q.offset),
            h.edgeCosines[i / 2]!,
            bound,
        );
        states[i / 2] = keep ? 2 : 1;
        kept += Number(keep);
    }
    return kept;
}
function computeCulling(s: ManifoldState): void {
    const c = s.culling!,
        a = s.hullA!,
        b = s.hullB!,
        t = relative(s.a, s.b);
    Object.assign(c, {
        facesA: Array(a.faceCount).fill(0),
        facesB: Array(b.faceCount).fill(0),
        edgesA: Array(a.edgeCount / 2).fill(0),
        edgesB: Array(b.edgeCount / 2).fill(0),
        seedA: 0,
        seedB: 0,
        probeA: 0,
        probeB: 0,
        keptA: 0,
        keptB: 0,
        sphereKeptA: 0,
        sphereKeptB: 0,
        separationA: -MAX,
        separationB: -MAX,
        gap: 0,
        probeApplied: false,
        separatedFeature: 0,
    });
    const delta = sub(manifoldTransform(t, b.center), a.center),
        distance = length(delta),
        radius = f(a.innerRadius + b.innerRadius),
        radiusBound = f(radius - f(f(0.005) + f(f(0.001) * f(distance + radius))));
    c.gap = f(distance - radius);
    const dotsA = a.planes.map((p) => dot(p.normal, delta));
    for (let i = 0; i < a.faceCount; i++) if (dotsA[i]! > dotsA[c.seedA]!) c.seedA = i;
    const seedA = faceSeparation(a, b, t, c.seedA);
    c.probeB = seedA.index;
    const floorA = Math.min(seedA.separation, f(0.02));
    for (let i = 0; i < a.faceCount; i++) {
        if (f(dotsA[i]! - radiusBound) < Math.max(floorA, c.separationA)) {
            c.facesA[i] = 1;
            continue;
        }
        c.facesA[i] = 2;
        const result = faceSeparation(a, b, t, i);
        if (result.separation > c.separationA) {
            c.separationA = result.separation;
            c.probeB = result.index;
            if (result.separation > f(0.02)) {
                c.separatedFeature = 2;
                return;
            }
        }
    }
    const deltaB = neg(invRotate(t.q, delta)),
        dotsB = b.planes.map((p) => dot(p.normal, deltaB));
    for (let i = 0; i < b.faceCount; i++) if (dotsB[i]! > dotsB[c.seedB]!) c.seedB = i;
    const seedB = faceSeparation(a, b, t, c.seedB, true);
    c.probeA = seedB.index;
    const floorB = Math.min(Math.max(seedB.separation, c.separationA), f(0.02));
    for (let i = 0; i < b.faceCount; i++) {
        if (f(dotsB[i]! - radiusBound) < Math.max(floorB, c.separationB)) {
            c.facesB[i] = 1;
            continue;
        }
        c.facesB[i] = 2;
        const result = faceSeparation(a, b, t, i, true);
        if (result.separation > c.separationB) {
            c.separationB = result.separation;
            c.probeA = result.index;
            if (result.separation > f(0.02)) {
                c.separatedFeature = 3;
                return;
            }
        }
    }
    const useSphere = c.edgeBound === 1 || c.edgeBound === 3,
        useSupport = c.edgeBound === 2 || c.edgeBound === 3,
        maxFace = Math.max(c.separationA, c.separationB);
    if (useSphere) {
        const slack = f(radius - radiusBound);
        c.keptA = cullEdges(
            a,
            manifoldTransform(t, b.center),
            f(f(maxFace + b.innerRadius) - slack),
            c.edgesA,
        );
        c.keptB = cullEdges(
            b,
            invTransform(t, a.center),
            f(f(maxFace + a.innerRadius) - slack),
            c.edgesB,
        );
    } else {
        c.keptA = a.edgeCount / 2;
        c.keptB = b.edgeCount / 2;
        c.edgesA.fill(2);
        c.edgesB.fill(2);
    }
    c.sphereKeptA = c.keptA;
    c.sphereKeptB = c.keptB;
    c.probeApplied = useSupport && c.keptB * ((c.keptA + 3) >> 2) >= 10;
    if (c.probeApplied) {
        const bound = f(
            maxFace - f(f(f(0.1) * f(0.005)) + f(f(0.001) * Math.abs(f(distance + radius)))),
        );
        c.keptA = cullEdges(a, manifoldTransform(t, b.points[c.probeB]!), bound, c.edgesA);
        c.keptB = cullEdges(b, invTransform(t, a.points[c.probeA]!), bound, c.edgesB);
    }
}
export function bestEdgeAxis(a: HullData, b: HullData, t: ReturnType<typeof pose>): EdgeAxis {
    let best: EdgeAxis = { normal: vector(), separation: -MAX, edgeA: -1, edgeB: -1 };
    const epsilon = f(-0.0001),
        squaredTolerance = f(f(0.005) * f(0.005));
    for (let j = 0; j < b.edgeCount; j += 2) {
        const C = neg(manifoldRotate(t.q, b.planes[b.edges[j]!.face]!.normal)),
            D = neg(manifoldRotate(t.q, b.planes[b.edges[j + 1]!.face]!.normal)),
            bv0 = neg(manifoldTransform(t, b.points[b.edges[j]!.origin]!)),
            bv1 = neg(manifoldTransform(t, b.points[b.edges[j + 1]!.origin]!)),
            Dc = sub(bv1, bv0);
        for (let i = 0; i < a.edgeCount; i += 2) {
            const n0 = a.planes[a.edges[i]!.face]!.normal,
                n1 = a.planes[a.edges[i + 1]!.face]!.normal,
                av0 = a.points[a.edges[i]!.origin]!,
                edge = sub(a.points[a.edges[i + 1]!.origin]!, av0);
            const Cba = dot(C, edge),
                Dba = dot(D, edge),
                Adc = dot(n0, Dc),
                Bdc = dot(n1, Dc);
            if (f(Cba * Dba) >= epsilon || f(Adc * Bdc) >= epsilon || f(Cba * Bdc) >= epsilon)
                continue;
            if (Math.max(f(Cba * Cba), f(Dba * Dba)) <= f(squaredTolerance * dot(edge, edge)))
                continue;
            const fraction = f(-Cba / f(Dba - Cba)),
                normal = normalize(add(C, mul(fraction, sub(D, C)))),
                separation = f(-dot(normal, add(av0, bv0)));
            if (separation > best.separation) best = { normal, separation, edgeA: i, edgeB: j };
        }
    }
    return best;
}
export function makeArc(u: Vec, v: Vec, d: Vec, separationU: number, separationV: number): Arc {
    const a = dot(u, d),
        b = dot(v, d),
        c = dot(u, v),
        inside = a >= f(c * b) && b >= f(c * a),
        axis = normalize(cross(u, v)),
        inPlane = sub(d, mul(dot(d, axis), axis)),
        peak = dot(inPlane, inPlane) > f(1e-12) ? normalize(inPlane) : u;
    const acos = (x: number) => f(Math.acos(Math.max(-1, Math.min(1, x))));
    const arcAngle = acos(c),
        angleU = acos(dot(u, peak)),
        angleV = acos(dot(v, peak)),
        nearest = f(f(Math.min(angleU, angleV) * 180) / f(Math.PI));
    return {
        u,
        v,
        peak,
        a,
        b,
        c,
        separationU,
        separationV,
        outsideDegrees: angleU <= arcAngle && angleV <= arcAngle ? -nearest : nearest,
        inside,
    };
}
function maxSeparation(
    s: ManifoldState,
    t: ReturnType<typeof pose>,
): { faceA: number; faceB: number; edge: EdgeAxis; maximum: number } {
    const a = s.hullA!,
        b = s.hullB!;
    let faceA = -MAX,
        faceB = -MAX;
    for (let i = 0; i < a.faceCount; i++)
        faceA = Math.max(faceA, faceSeparation(a, b, t, i).separation);
    for (let i = 0; i < b.faceCount; i++)
        faceB = Math.max(faceB, faceSeparation(a, b, t, i, true).separation);
    const edge = bestEdgeAxis(a, b, t);
    return { faceA, faceB, edge, maximum: Math.max(edge.separation, Math.max(faceA, faceB)) };
}
function computeWedge(s: ManifoldState): void {
    const w = s.wedge!,
        t = relative(s.a, s.b),
        a = s.hullA!,
        b = s.hullB!,
        result = maxSeparation(s, t);
    w.faceA = result.faceA;
    w.faceB = result.faceB;
    w.edge = result.edge;
    w.edgeWins = w.edge.edgeA !== -1 && w.edge.separation > Math.max(w.faceA, w.faceB);
    w.centerOffset = sub(manifoldTransform(t, b.center), a.center);
    w.radius = f(a.innerRadius + b.innerRadius);
    if (w.edge.edgeA === -1) return;
    const faceA1 = a.edges[w.edge.edgeA]!.face,
        faceA2 = a.edges[w.edge.edgeA + 1]!.face,
        faceB1 = b.edges[w.edge.edgeB]!.face,
        faceB2 = b.edges[w.edge.edgeB + 1]!.face;
    w.arcA = makeArc(
        a.planes[faceA1]!.normal,
        a.planes[faceA2]!.normal,
        w.centerOffset,
        faceSeparation(a, b, t, faceA1).separation,
        faceSeparation(a, b, t, faceA2).separation,
    );
    w.arcB = makeArc(
        neg(manifoldRotate(t.q, b.planes[faceB1]!.normal)),
        neg(manifoldRotate(t.q, b.planes[faceB2]!.normal)),
        w.centerOffset,
        faceSeparation(a, b, t, faceB1, true).separation,
        faceSeparation(a, b, t, faceB2, true).separation,
    );
}
function quaternion(world: World, axis: Vec, angle: number): ReturnType<typeof pose>["q"] {
    const q = rotationFromAxisAngle(world, axis, angle);
    return { v: vector(q[0], q[1], q[2]), s: q[3]! };
}
function search(world: World): void {
    const s = world.resource(ManifoldSample),
        w = s.wedge!;
    w.found = false;
    w.searchCount = 0;
    const random = () => {
        w.seed = (Math.imul(w.seed, 1664525) + 1013904223) >>> 0;
        return f((w.seed >>> 8) / 16777216);
    };
    const direction = () => {
        let p: Vec, squared: number;
        do {
            p = vector(f(f(2 * random()) - 1), f(f(2 * random()) - 1), f(f(2 * random()) - 1));
            squared = dot(p, p);
        } while (squared > 1 || squared < f(0.01));
        return normalize(p);
    };
    for (let trial = 0; trial < 500 && !w.found; trial++) {
        w.searchCount++;
        const d = direction(),
            axis = direction(),
            angle = f(f(2 * f(Math.PI)) * random()),
            q = quaternion(world, axis, angle);
        const extents = (h: HullData) => mul(0.5, sub(h.aabb.upperBound, h.aabb.lowerBound));
        let lower = 0,
            upper = f(f(1.1) * f(length(extents(s.hullA!)) + length(extents(s.hullB!))));
        for (let iteration = 0; iteration < 24; iteration++) {
            const mid = f(0.5 * f(lower + upper)),
                t = { p: mul(mid, d), q };
            if (maxSeparation(s, t).maximum < f(-0.02)) lower = mid;
            else upper = mid;
        }
        s.b = { p: add(s.a.p, mul(upper, d)), q };
        computeWedge(s);
        const margin = f(w.edge.separation - Math.max(w.faceA, w.faceB)),
            outside = Math.max(w.arcA?.outsideDegrees ?? -MAX, w.arcB?.outsideDegrees ?? -MAX);
        w.found = w.edgeWins && outside >= 10 && margin >= f(0.02);
    }
}
function wedgeHull(world: World, type: number): void {
    const s = world.resource(ManifoldSample),
        w = s.wedge!;
    s.hullA =
        type === 0
            ? makeBoxHull(2, 0.25, 1, world)
            : type === 1
              ? createHull(
                    [vector(1, 1, 1), vector(1, -1, -1), vector(-1, 1, -1), vector(-1, -1, 1)],
                    4,
                    world,
                )!
              : complex(world);
    s.hullB = s.hullA;
    w.hullType = type;
    w.seed = 12345;
    search(world);
}
export function stepManifoldDiagnostics(world: World): void {
    const s = world.resource(ManifoldSample);
    if (s.culling) {
        s.sat = sat();
        s.manifold = collideHulls(world, s.hullA!, s.hullB!, relative(s.a, s.b), s.sat);
        computeCulling(s);
    } else if (s.wedge) computeWedge(s);
}
function construct(world: World, scene: "hullCulling" | "edgeAxisWedge"): number[] {
    const s = world.resource(ManifoldSample);
    Object.assign(s, ManifoldSample.create(), { scene, a: pose(), b: pose(0.4, 3.6, 0.2) });
    world.resource(PhysicsWorldDefinition).enableAutomaticStep = false;
    if (scene === "hullCulling") {
        s.culling = {
            cylinders: false,
            height: 2,
            radius: 2,
            sides: 32,
            showSpheres: true,
            showCulled: true,
            showNormals: true,
            edgeBound: 3,
            facesA: [],
            facesB: [],
            edgesA: [],
            edgesB: [],
            seedA: 0,
            seedB: 0,
            probeA: 0,
            probeB: 0,
            keptA: 0,
            keptB: 0,
            sphereKeptA: 0,
            sphereKeptB: 0,
            separationA: -MAX,
            separationB: -MAX,
            gap: 0,
            separatedFeature: 0,
            probeApplied: false,
        };
        s.b.q = quaternion(world, normalize(vector(1, 0, 1)), f(0.3));
        cullingHulls(world);
    } else {
        s.b = pose(0, 3.6, 0);
        s.wedge = {
            hullType: 0,
            seed: 12345,
            searchCount: 0,
            found: false,
            edge: { normal: vector(), separation: -MAX, edgeA: -1, edgeB: -1 },
            arcA: null,
            arcB: null,
            faceA: -MAX,
            faceB: -MAX,
            edgeWins: false,
            centerOffset: vector(),
            radius: 0,
            gaussCenter: vector(-10, 1, 0),
            gaussRadius: 2,
        };
        wedgeHull(world, 0);
    }
    stepManifoldDiagnostics(world);
    return [];
}
const cullingControls: Controls = {
    Cache: {
        label: "Use cache",
        value: 0,
        set(w, _b, v) {
            w.resource(ManifoldSample).useCache = Boolean(v);
        },
    },
    Feature: {
        label: "Feature",
        value: 0,
        select: true,
        when: ["Cache"],
        options: ["auto", "faceA", "faceB", "edgePair"].map((label, value) => ({ label, value })),
        set(w, _b, v) {
            w.resource(ManifoldSample).manualFeature = v;
        },
    },
    Points: {
        label: "Draw Points",
        value: 1,
        set(w, _b, v) {
            w.resource(ManifoldSample).drawPoints = Boolean(v);
        },
    },
};
for (const [key, label] of [
    ["showSpheres", "Inscribed spheres"],
    ["showCulled", "Culled features"],
    ["showNormals", "Face normals"],
] as const)
    cullingControls[key] = {
        label,
        value: 1,
        set(w, _b, v) {
            w.resource(ManifoldSample).culling![key] = Boolean(v);
        },
    };
cullingControls.Bound = {
    label: "Edge bound",
    value: 3,
    select: true,
    options: ["None", "Sphere", "Support", "Sphere + Support"].map((label, value) => ({
        label,
        value,
    })),
    set(w, _b, v) {
        w.resource(ManifoldSample).culling!.edgeBound = v;
    },
};
cullingControls.Cylinders = {
    label: "Cylinders",
    value: 0,
    set(w, _b, v) {
        w.resource(ManifoldSample).culling!.cylinders = Boolean(v);
        cullingHulls(w);
    },
};
for (const [key, label, value, min, max, step] of [
    ["height", "Height", 2, 0.1, 8, 0.01],
    ["radius", "Radius", 2, 0.1, 4, 0.01],
    ["sides", "Sides", 32, 3, 32, 1],
] as const)
    cullingControls[key] = {
        label,
        value,
        min,
        max,
        step,
        when: ["Cylinders"],
        set(w, _b, v) {
            w.resource(ManifoldSample).culling![key] = f(v);
            cullingHulls(w);
        },
    };
const wedgeControls: Controls = {
    Hull: {
        label: "Hull",
        value: 0,
        select: true,
        options: ["Slab", "Tetrahedron", "Complex Hull"].map((label, value) => ({ label, value })),
        set(w, _b, v) {
            wedgeHull(w, v);
        },
    },
    Find: {
        label: "Find Example",
        value: 0,
        button: true,
        set(w) {
            search(w);
        },
    },
};
wedgeControls.Winner = {
    label: "Winner",
    value: 0,
    read: (w) => {
        const edge = w.resource(ManifoldSample).wedge;
        if (!edge) return null;
        return edge.edge.edgeA === -1 ? "none" : edge.edgeWins ? "edge" : "face";
    },
    set() {},
};
wedgeControls.Advantage = {
    label: "Edge advantage, m",
    value: 0,
    decimals: 4,
    read: (w) => {
        const edge = w.resource(ManifoldSample).wedge;
        if (!edge) return null;
        return f(edge.edge.separation - Math.max(edge.faceA, edge.faceB));
    },
    set() {},
};
wedgeControls.Search = {
    label: "Search",
    value: 0,
    read: (w) => {
        const edge = w.resource(ManifoldSample).wedge;
        if (!edge) return null;
        return `${edge.found ? "found" : "not found"} / ${edge.searchCount} poses`;
    },
    set() {},
};
export const manifoldDiagnosticScenes = {
    hullCulling: {
        construct: (w: World) => construct(w, "hullCulling"),
        gridExtent: 12,
        gridSpacing: 1,
        gridHeight: -2,
        camera: {
            yaw: 20,
            pitch: 25,
            distance: 14,
            phoneDistance: 30,
            target: [0, 1.8, 0],
            phoneTarget: [0, 12, 0],
        },
        controls: cullingControls,
    },
    edgeAxisWedge: {
        construct: (w: World) => construct(w, "edgeAxisWedge"),
        gridExtent: 16,
        gridSpacing: 1,
        gridHeight: -3,
        camera: {
            yaw: 0,
            pitch: 25,
            distance: 18,
            phoneDistance: 35,
            target: [-6.5, 1.5, 0],
            phoneTarget: [-6.5, 7, 0],
        },
        controls: wedgeControls,
    },
} as const;
