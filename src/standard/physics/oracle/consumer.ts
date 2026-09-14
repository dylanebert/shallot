import { readFileSync } from "node:fs";
import { join } from "node:path";
import { emptyCache, shapeDistance } from "../collision/distance";
import { collideSpheres, makeLocalManifold } from "../collision/manifold";
import { type CollisionPlane, clipVector, solvePlanes } from "../collision/mover";
import { createProxy, createTree, query } from "../collision/tree";
import {
    type AABB,
    computeCosSin,
    pointToSegmentDistance,
    type Quat,
    type Transform,
    type Vec3,
    vec3,
} from "../common/math";
import {
    type Capsule,
    computeCapsuleMass,
    computeSphereAABB,
    rayCastSphere,
    type Sphere,
} from "../shapes/geometry";
import { loadScenarioCorpus, runScenario, type ScenarioOutput } from "./scenario";

export type OracleCase = {
    id: string;
    family: string;
    symbol: string;
    input: unknown;
    output: unknown;
    configuration?: string;
};

const identity: Quat = { v: { x: 0, y: 0, z: 0 }, s: 1 };
const f32 = (hex: string): number => {
    if (!/^0x[0-9a-f]{8}$/.test(hex)) throw new Error(`invalid f32 bits: ${hex}`);
    const view = new DataView(new ArrayBuffer(4));
    view.setUint32(0, Number.parseInt(hex.slice(2), 16), false);
    return view.getFloat32(0, false);
};
const bits = (value: number): string => {
    const view = new DataView(new ArrayBuffer(4));
    view.setFloat32(0, value, false);
    return `0x${view.getUint32(0, false).toString(16).padStart(8, "0")}`;
};
const u64 = (hex: string): bigint => BigInt(hex);
const vec = (values: unknown): Vec3 => {
    if (!Array.isArray(values) || values.length !== 3) throw new Error("expected f32 vec3");
    return { x: f32(String(values[0])), y: f32(String(values[1])), z: f32(String(values[2])) };
};
const transform = (translation: unknown): Transform => ({ p: vec(translation), q: identity });
const outVec = (value: Vec3): string[] => [bits(value.x), bits(value.y), bits(value.z)];
const aabb = (value: AABB): { lower: string[]; upper: string[] } => ({
    lower: outVec(value.lowerBound),
    upper: outVec(value.upperBound),
});

function runBaseCase(item: OracleCase): unknown {
    const input = item.input as Record<string, unknown>;
    switch (item.id) {
        case "math.add.v1.scalar":
        case "math.add.v1.simd": {
            const a = vec(input.a);
            const b = vec(input.b);
            return { value: outVec(vec3.add(a, b)) };
        }
        case "math.dot.v1.scalar":
        case "math.dot.v1.simd":
            return { value: bits(vec3.dot(vec(input.a), vec(input.b))) };
        case "math.cos-sin.v1.scalar":
        case "math.cos-sin.v1.simd": {
            const value = awaitableMath(input.radians);
            return value;
        }
        case "geometry.sphere-aabb.v1.scalar":
        case "geometry.sphere-aabb.v1.simd": {
            const shape: Sphere = { center: vec(input.center), radius: f32(String(input.radius)) };
            return { aabb: aabb(computeSphereAABB(shape, transform(input.translation))) };
        }
        case "geometry.capsule-mass.v1.scalar":
        case "geometry.capsule-mass.v1.simd": {
            const shape: Capsule = {
                center1: vec(input.center1),
                center2: vec(input.center2),
                radius: f32(String(input.radius)),
            };
            const mass = computeCapsuleMass(shape, f32(String(input.density)));
            return {
                mass: bits(mass.mass),
                center: outVec(mass.center),
                // The upstream v6 capsule-mass adapter serializes the matrix's first column.
                inertia: outVec(mass.inertia.cx),
            };
        }
        case "distance.point-segment.v1.scalar":
        case "distance.point-segment.v1.simd":
            return {
                closest: outVec(pointToSegmentDistance(vec(input.a), vec(input.b), vec(input.q))),
            };
        case "distance.shape.v1.scalar":
        case "distance.shape.v1.simd": {
            const cache = emptyCache();
            const result = shapeDistance(
                {
                    proxyA: {
                        points: (input.proxyA as unknown[]).map(vec),
                        count: (input.proxyA as unknown[]).length,
                        radius: 0,
                    },
                    proxyB: {
                        points: (input.proxyB as unknown[]).map(vec),
                        count: (input.proxyB as unknown[]).length,
                        radius: 0,
                    },
                    transform: transform(input.translationB),
                    useRadii: input.useRadii === true,
                },
                cache,
            );
            return {
                pointA: outVec(result.pointA),
                pointB: outVec(result.pointB),
                normal: outVec(result.normal),
                distance: bits(result.distance),
                iterations: `0x${result.iterations.toString(16).padStart(8, "0")}`,
                simplexCount: `0x${result.simplexCount.toString(16).padStart(8, "0")}`,
            };
        }
        case "tree.query.v1.scalar":
        case "tree.query.v1.simd": {
            const tree = createTree(2);
            const hits: { proxyId: number; userData: bigint }[] = [];
            (input.proxies as Record<string, unknown>[]).map((proxy) => {
                const values = (proxy.aabb as string[]).map(f32);
                const box: AABB = {
                    lowerBound: { x: values[0], y: values[1], z: values[2] },
                    upperBound: { x: values[3], y: values[4], z: values[5] },
                };
                const category = u64(String(proxy.category));
                return createProxy(
                    tree,
                    box,
                    Number((category >> 32n) & 0xffffffffn),
                    Number(category & 0xffffffffn),
                    u64(String(proxy.userData)),
                );
            });
            const q = (input.queryAabb as string[]).map(f32);
            const stats = query(
                tree,
                {
                    lowerBound: { x: q[0], y: q[1], z: q[2] },
                    upperBound: { x: q[3], y: q[4], z: q[5] },
                },
                0,
                Number(u64(String(input.mask))),
                false,
                (proxyId, userData) => {
                    hits.push({ proxyId, userData });
                    return true;
                },
                true,
            );
            return {
                stats: {
                    nodeVisits: `0x${stats.nodeVisits.toString(16).padStart(8, "0")}`,
                    leafVisits: `0x${stats.leafVisits.toString(16).padStart(8, "0")}`,
                },
                hits: hits.map((hit) => ({
                    proxyId: `0x${hit.proxyId.toString(16).padStart(8, "0")}`,
                    userData: `0x${hit.userData.toString(16).padStart(16, "0")}`,
                })),
            };
        }
        case "manifold.spheres.v1.scalar":
        case "manifold.spheres.v1.simd": {
            const manifold = makeLocalManifold(4);
            const sphereA = input.sphereA as Record<string, unknown>;
            const sphereB = input.sphereB as Record<string, unknown>;
            collideSpheres(
                manifold,
                4,
                { center: vec(sphereA.center), radius: f32(String(sphereA.radius)) },
                { center: vec(sphereB.center), radius: f32(String(sphereB.radius)) },
                {
                    p: vec((input.transformBtoA as Record<string, unknown>).translation),
                    q: identity,
                },
            );
            return {
                normal: outVec(manifold.normal),
                pointCount: `0x${manifold.pointCount.toString(16).padStart(8, "0")}`,
                points: manifold.points.slice(0, manifold.pointCount).map((point) => ({
                    point: outVec(point.point),
                    separation: bits(point.separation),
                })),
            };
        }
        case "query.ray-sphere.v1.scalar":
        case "query.ray-sphere.v1.simd": {
            const sphere = input.sphere as Record<string, unknown>;
            const result = rayCastSphere(
                { center: vec(sphere.center), radius: f32(String(sphere.radius)) },
                {
                    origin: vec(input.origin),
                    translation: vec(input.translation),
                    maxFraction: f32(String(input.maxFraction)),
                },
            );
            return {
                hit: result.hit,
                normal: outVec(result.normal),
                point: outVec(result.point),
                fraction: bits(result.fraction),
                iterations: `0x${result.iterations.toString(16).padStart(8, "0")}`,
            };
        }
        case "mover.solve-planes.v1.scalar":
        case "mover.solve-planes.v1.simd": {
            const planes = (input.planes as Record<string, unknown>[]).map((value) => ({
                plane: { normal: vec(value.normal), offset: f32(String(value.offset)) },
                pushLimit: f32(String(value.pushLimit)),
                push: 0,
                clipVelocity: value.clipVelocity === true,
            })) as CollisionPlane[];
            const result = solvePlanes(vec(input.targetDelta), planes, planes.length);
            return {
                delta: outVec(result.delta),
                iterationCount: `0x${result.iterationCount.toString(16).padStart(8, "0")}`,
            };
        }
        case "mover.clip-vector.v1.scalar":
        case "mover.clip-vector.v1.simd": {
            const planes = Array.from(
                { length: Number(u64(String(input.planeCount))) },
                (_, index) => ({
                    plane: {
                        normal: index === 0 ? { x: 1, y: 0, z: 0 } : { x: 0, y: 1, z: 0 },
                        offset: 0,
                    },
                    pushLimit: Number.POSITIVE_INFINITY,
                    push: 1,
                    clipVelocity: true,
                }),
            );
            return { value: outVec(clipVector(vec(input.vector), planes, planes.length)) };
        }
        default:
            throw new Error(`no common-input adapter for ${item.id}`);
    }
}

// Kept as a separate function so the vector dispatch has no inline output arithmetic.
function awaitableMath(value: unknown): { cosine: string; sine: string } {
    const result = computeCosSin(f32(String(value)));
    return { cosine: bits(result.cosine), sine: bits(result.sine) };
}

export type ConsumerCorpus = { schema: string; cases: OracleCase[] };
export function loadConsumerCorpus(
    path = join(
        import.meta.dir,
        "box3d",
        "47d7f7cc7e091142c08d11dc7d2e493c5d34f536",
        "v6",
        "cases.json",
    ),
): ConsumerCorpus {
    const value = JSON.parse(readFileSync(path, "utf8")) as ConsumerCorpus;
    if (
        value.schema !== "box3d-oracle/v6" ||
        !Array.isArray(value.cases) ||
        value.cases.length !== 111
    )
        throw new Error("immutable v6 consumer corpus is not the exact 111-case population");
    return value;
}

export function runCommonInput(item: OracleCase): unknown {
    if (item.family === "scenario") {
        const scenarioInput = item.input as { name: string };
        const { corpus, digest } = loadScenarioCorpus();
        const scenario = corpus.scenarios.find(
            (candidate) => candidate.name === scenarioInput.name,
        );
        if (!scenario)
            throw new Error(`scenario ${scenarioInput.name} is absent from the command corpus`);
        return runScenario(scenario, digest) satisfies ScenarioOutput;
    }
    return runBaseCase(item);
}
