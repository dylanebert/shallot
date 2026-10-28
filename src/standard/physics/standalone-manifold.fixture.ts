import type { World } from "../../engine";
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
    type LocalManifold,
    makeBoxHull,
    type SATCache,
} from ".";
import { emptyCache } from "./standalone-collision.fixture";

const v = (x = 0, y = 0, z = 0) => ({ x, y, z });
const bitsBuffer = new ArrayBuffer(4),
    floats = new Float32Array(bitsBuffer),
    words = new Uint32Array(bitsBuffer);
const bits = (n: number): number => {
    floats[0] = n;
    return words[0];
};
const hex = (n: number): string => (n >>> 0).toString(16).padStart(8, "0");
export function manifoldSubject(world: World) {
    const a = makeBoxHull(1, 1, 1),
        b = makeBoxHull(0.6, 1.2, 0.8);
    const sphere = { center: v(), radius: 0.5 };
    const capsule = { center1: v(-0.5), center2: v(0.5), radius: 0.5 };
    const transform = { p: v(), q: { v: v(), s: 1 } };
    const triangle: [ReturnType<typeof v>, ReturnType<typeof v>, ReturnType<typeof v>] = [
        v(-4, 0, -4),
        v(-4, 0, 4),
        v(4, 0, 0),
    ];
    const simplex = emptyCache(),
        sat: SATCache = { separation: 0, type: 0, indexA: 0, indexB: 0, hit: 0 };
    function call(op: number): LocalManifold {
        switch (op) {
            case 10:
                return collideSpheres(world, sphere, sphere, transform);
            case 11:
                return collideCapsuleAndSphere(world, capsule, sphere, transform);
            case 12:
                return collideHullAndSphere(world, a, sphere, transform, simplex);
            case 13:
                return collideCapsules(world, capsule, capsule, transform);
            case 14:
                return collideHullAndCapsule(world, a, capsule, transform, simplex);
            case 15:
                return collideHulls(world, a, b, transform, sat);
            case 16:
                return collideTriangleAndSphere(world, triangle, sphere);
            case 17:
                return collideTriangleAndCapsule(world, triangle, capsule, simplex);
            case 18:
                return collideTriangleAndHull(world, triangle, 0, a, sat);
            default:
                throw new Error("Invalid operation");
        }
    }
    return { call, transform, triangle, simplex, sat, sphere, capsule };
}
export function manifoldRecords(world: World): { input: string; output: string } {
    const s = manifoldSubject(world);
    let input = "",
        output = "";
    for (let op = 10; op <= 18; op++) {
        // Deep overlap, touching, speculative, separated, then back through the warm cache.
        const boundary =
            op === 15 ? 2.2 : op === 12 || op === 14 ? 1.5 : op === 18 ? 1 : op >= 16 ? 0.5 : 1;
        for (const pose of op >= 16 ? [0, 1, 2] : [0, 1]) {
            s.transform.p.x = pose * 0.3;
            s.transform.p.z = pose * 0.2;
            s.transform.q.v.z = pose ? 0.24740396 : 0;
            s.transform.q.s = pose ? 0.9689124 : 1;
            for (const [step, delta] of [-0.25, 0, 0.01, 4, 0.01, 0, -0.25, 0, 0, 0].entries()) {
                if (delta === -0.25) {
                    s.simplex.count = 0;
                    s.sat.type = 0;
                }
                if (step >= 7 && (op === 15 || op === 18)) s.sat.type = step - 1;
                const y = boundary + delta;
                s.transform.p.y = y;
                for (const p of s.triangle) p.y = (pose === 2 ? y : -y) + pose * 0.1 * p.x;
                const r = new Uint32Array(64),
                    f = new Float32Array(r.buffer);
                r[0] = 64;
                f[4] = s.transform.p.x;
                f[5] = y;
                f[6] = s.transform.p.z;
                f[9] = s.transform.q.v.z;
                f[10] = s.transform.q.s;
                f[15] = 0.5;
                f[19] = 0.5;
                f[20] = -0.5;
                f[23] = 0.5;
                f[26] = 0.5;
                f[27] = -0.5;
                f[30] = 0.5;
                f[33] = 0.5;
                for (let i = 0; i < 3; i++) {
                    const p = s.triangle[i];
                    f.set([p.x, p.y, p.z], 34 + 3 * i);
                }
                f[44] = s.sat.separation;
                r[45] = s.sat.type;
                r[46] = s.sat.indexA;
                r[47] = s.sat.indexB;
                r[48] = s.sat.hit;
                f[50] = s.simplex.metric;
                r[51] = s.simplex.count;
                r.set(s.simplex.indexA, 52);
                r.set(s.simplex.indexB, 56);
                input += `${op} 64 ${Array.from(r, hex).join(" ")}\n`;
                const m = s.call(op);
                const row = [
                    m.pointCount,
                    ...[m.normal.x, m.normal.y, m.normal.z].map(bits),
                    m.feature,
                    bits(m.squaredDistance),
                ];
                for (let i = 0; i < m.pointCount; i++) {
                    const p = m.points[i],
                        pair = p.pair;
                    row.push(
                        bits(p.point.x),
                        bits(p.point.y),
                        bits(p.point.z),
                        bits(p.separation),
                        (pair.owner1 << 24) |
                            (pair.index1 << 16) |
                            (pair.owner2 << 8) |
                            pair.index2,
                        p.triangleIndex,
                    );
                }
                row.push(
                    1,
                    bits(m.triangleNormal.x),
                    bits(m.triangleNormal.y),
                    bits(m.triangleNormal.z),
                    m.triangleIndex,
                    ...m.vertexIndices,
                    m.triangleFlags,
                    bits(s.simplex.metric),
                    s.simplex.count,
                    ...s.simplex.indexA,
                    ...s.simplex.indexB,
                    bits(s.sat.separation),
                    s.sat.type,
                    s.sat.indexA,
                    s.sat.indexB,
                    s.sat.hit,
                );
                output += `${op} ${row.map(hex).join(" ")}\n`;
            }
        }
    }
    return { input, output };
}
