import type { CollisionDebug } from "../src/collision-debug";

const buffer = new DataView(new ArrayBuffer(4));
function hex(v: number): string {
    buffer.setFloat32(0, v, true);
    return buffer.getUint32(0, true).toString(16).padStart(8, "0");
}
const vector = (v: { x: number; y: number; z: number }) => [v.x, v.y, v.z].map(hex);
export function debugRecords(s: ReturnType<typeof CollisionDebug.create>): string {
    if (s.cast) {
        const d = s.cast;
        return `cast ${[...vector(d.point), ...vector(d.normal), hex(d.fraction), d.iterations, Number(d.hit)].join(" ")}\n`;
    }
    if (s.toi) {
        const d = s.toi;
        return `toi ${[d.state, hex(d.fraction), hex(d.distance), ...vector(d.point), ...vector(d.normal), d.distanceIterations, d.pushBackIterations, d.rootIterations].join(" ")}\n`;
    }
    const d = s.distance;
    if (!d) return "";
    const words: (string | number)[] = [
        ...vector(d.pointA),
        ...vector(d.pointB),
        ...vector(d.normal),
        hex(d.distance),
        d.iterations,
        hex(s.cache.metric),
        s.cache.count,
        ...s.cache.indexA,
        ...s.cache.indexB,
        d.simplexCount,
    ];
    for (let i = 0; i < d.simplexCount; i++) {
        const simplex = d.simplexes[i]!;
        words.push(simplex.count);
        for (let j = 0; j < simplex.count; j++) {
            const v = simplex.vertices[j]!;
            words.push(
                ...vector(v.wA),
                ...vector(v.wB),
                ...vector(v.w),
                hex(v.a),
                v.indexA,
                v.indexB,
            );
        }
    }
    return `distance ${words.join(" ")}\n`;
}
