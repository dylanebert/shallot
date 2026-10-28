import type { World } from "../../engine/ecs";
import {
    type DistanceInput,
    type DistanceOutput,
    type ShapeProxy,
    type SimplexCache,
    type Sweep,
    shapeCast,
    shapeDistance,
    type TOIOutput,
    timeOfImpact,
} from ".";

const v = (x = 0, y = 0, z = 0) => ({ x, y, z });
const q = { v: v(), s: 1 };
const point: ShapeProxy = { points: [v()], count: 1, radius: 0.25 };
const box: ShapeProxy = {
    points: [-1, 1].flatMap((x) => [-1, 1].flatMap((y) => [-1, 1].map((z) => v(x, y, z)))),
    count: 8,
    radius: 0,
};
const triangle: ShapeProxy = {
    points: [v(-2, 0, -2), v(2, 0, -2), v(0, 0, 2)],
    count: 3,
    radius: 0,
};
const capsule: ShapeProxy = { points: [v(0, -0.5), v(0, 0.5)], count: 2, radius: 0.2 };
export const collisionCases = [point, box, triangle, capsule].flatMap((proxyA, a) =>
    [point, box, triangle, capsule].flatMap((proxyB, b) =>
        [0, 0.5, 2, 5].map((x) => ({
            label: `${a}-${b}-${x}`,
            input: {
                proxyA,
                proxyB,
                transform: { p: v(x, 0.1, 0.2), q },
                useRadii: true,
            } satisfies DistanceInput,
        })),
    ),
);
for (const [label, proxyA, proxyB, p, rotation] of [
    ["touching", point, point, v(0.5), q],
    ["deep", box, box, v(), q],
    ["miss", point, capsule, v(5, 10), q],
    ["rotation", triangle, capsule, v(2, 1, 0), { v: v(0, 0, 0.6), s: 0.8 }],
] as const) {
    collisionCases.push({
        label,
        input: { proxyA, proxyB, transform: { p, q: rotation }, useRadii: true },
    });
}
export function emptyCache(): SimplexCache {
    return { metric: 0, count: 0, indexA: [0, 0, 0, 0], indexB: [0, 0, 0, 0] };
}
const scratch = new Float32Array(1);
const words = new Uint32Array(scratch.buffer);
export function bits(n: number): number {
    scratch[0] = n;
    return words[0];
}
export const hex = (n: number) => (n >>> 0).toString(16).padStart(8, "0");
export const vf = (p: { x: number; y: number; z: number }) => [p.x, p.y, p.z].map(bits);
function sweep(c1: { x: number; y: number; z: number }, c2 = c1): Sweep {
    return { localCenter: v(), c1, c2, q1: q, q2: q };
}
export function nativeCollisionInput(): string {
    const lines: string[] = [];
    for (const { input } of collisionCases)
        for (const operation of [7, 4, 5]) {
            const r = Array<number>(834).fill(0);
            r[0] = input.proxyA.count;
            r[1] = bits(input.proxyA.radius);
            r[2] = input.proxyB.count;
            r[3] = bits(input.proxyB.radius);
            r.splice(4, 3, ...vf(input.transform.p));
            r.splice(7, 3, ...vf(input.transform.q.v));
            r[10] = bits(input.transform.q.s);
            r[11] = 1;
            r[12] = bits(-10);
            r[15] = bits(1);
            r[26] = 32;
            input.proxyA.points.forEach((p, i) => {
                r.splice(32 + 3 * i, 3, ...vf(p));
            });
            input.proxyB.points.forEach((p, i) => {
                r.splice(416 + 3 * i, 3, ...vf(p));
            });
            r[812] = r[816] = r[833] = bits(1);
            r.splice(826, 3, ...vf(input.transform.q.v));
            r[829] = bits(input.transform.q.s);
            r.splice(820, 3, ...vf(input.transform.p));
            r.splice(
                823,
                3,
                ...vf(v(input.transform.p.x - 10, input.transform.p.y, input.transform.p.z)),
            );
            lines.push(`${operation} ${r.length} ${r.map(hex).join(" ")}`);
        }
    return `${lines.join("\n")}\n`;
}
/** The public records serialized in the native harness's word order. */
export function publicCollisionOutput(world: World): string {
    const lines: string[] = [];
    for (const { input } of collisionCases) {
        const cache = emptyCache();
        const d = shapeDistance(world, input, cache, 32);
        const a = distanceWords(d, cache);
        lines.push(`7 ${a.map(hex).join(" ")}`);
        const c = shapeCast(world, {
            ...input,
            translationB: v(-10),
            maxFraction: 1,
            canEncroach: true,
        });
        const cw = [Number(c.hit), c.iterations];
        if (c.hit) cw.push(bits(c.fraction), ...vf(c.point), ...vf(c.normal), -1, 0, 0);
        lines.push(`4 ${cw.map(hex).join(" ")}`);
        const t = timeOfImpact(world, {
            ...input,
            sweepA: sweep(v()),
            sweepB: {
                ...sweep(
                    input.transform.p,
                    v(input.transform.p.x - 10, input.transform.p.y, input.transform.p.z),
                ),
                q1: input.transform.q,
            },
            maxFraction: 1,
        });
        lines.push(`5 ${toiWords(t).map(hex).join(" ")}`);
    }
    return `${lines.join("\n")}\n`;
}
export function distanceWords(d: DistanceOutput, cache: SimplexCache): number[] {
    const a = [
        ...vf(d.pointA),
        ...vf(d.pointB),
        ...vf(d.normal),
        bits(d.distance),
        d.iterations,
        bits(cache.metric),
        cache.count,
        ...cache.indexA,
        ...cache.indexB,
        d.simplexCount,
    ];
    for (let i = 0; i < d.simplexCount; i++) {
        const s = d.simplexes[i];
        a.push(s.count);
        for (let j = 0; j < s.count; j++) {
            const vertex = s.vertices[j];
            a.push(
                ...vf(vertex.wA),
                ...vf(vertex.wB),
                ...vf(vertex.w),
                bits(vertex.a),
                vertex.indexA,
                vertex.indexB,
            );
        }
    }
    return a;
}
export function toiWords(t: TOIOutput): number[] {
    return [
        t.state,
        bits(t.fraction),
        bits(t.distance),
        ...vf(t.point),
        ...vf(t.normal),
        t.distanceIterations,
        t.pushBackIterations,
        t.rootIterations,
        0,
    ];
}
