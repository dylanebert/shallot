import type { Vec3 } from "../common/math";
import type { HullData } from "./hull";

export type HullImage = HullData;
const align8 = (n: number): number => (n + 7) & ~7;

export function hullByteCount(h: HullImage): number {
    return (
        144 +
        align8(h.vertexCount) +
        align8(12 * h.vertexCount) +
        align8(4 * h.edgeCount) +
        align8(16 * h.faceCount) +
        align8(h.faceCount) +
        12 * ((h.vertexCount + 3) & ~3) +
        12 * ((h.faceCount + 3) & ~3)
    );
}

function vector(f: Float32Array, o: number, p: Vec3): void {
    f[o] = p.x;
    f[o + 1] = p.y;
    f[o + 2] = p.z;
}

/** Writes the zero-padded b3HullData byte image, including its relative offsets. */
export function writeHullImage(
    h: HullImage,
    bytes: Uint8Array,
    u: Uint32Array,
    f: Float32Array,
    hashes: BigUint64Array,
    base: number,
): void {
    const size = hullByteCount(h);
    bytes.fill(0, base, base + size);
    const r = base >>> 2;
    u[r] = 0xde57485c;
    u[r + 1] = 0x4a4c9587;
    hashes[(base + 8) >>> 3] = h.hash;
    vector(f, r + 4, h.aabb.lowerBound);
    vector(f, r + 7, h.aabb.upperBound);
    f[r + 10] = h.surfaceArea;
    f[r + 11] = h.volume;
    f[r + 12] = h.innerRadius;
    vector(f, r + 13, h.center);
    vector(f, r + 16, h.centralInertia.cx);
    vector(f, r + 19, h.centralInertia.cy);
    vector(f, r + 22, h.centralInertia.cz);
    let off = 144;
    const vertices = base + off;
    u[r + 25] = h.vertexCount;
    u[r + 26] = off;
    off += align8(h.vertexCount);
    const points = (base + off) >>> 2;
    u[r + 27] = off;
    off += align8(12 * h.vertexCount);
    const edges = base + off;
    u[r + 28] = h.edgeCount;
    u[r + 29] = off;
    off += align8(4 * h.edgeCount);
    const planes = (base + off) >>> 2;
    u[r + 30] = h.faceCount;
    u[r + 31] = off;
    off += align8(16 * h.faceCount);
    const faces = base + off;
    u[r + 32] = off;
    off += align8(h.faceCount);
    const nv = (h.vertexCount + 3) & ~3;
    const nf = (h.faceCount + 3) & ~3;
    const soa = (base + off) >>> 2;
    u[r + 33] = off;
    u[r + 34] = off + 12 * nv;
    u[r + 35] = size;
    for (let p = 0; p < nv; ++p) {
        const pt = h.points[p < h.vertexCount ? p : 0];
        f[soa + p] = pt.x;
        f[soa + nv + p] = pt.y;
        f[soa + 2 * nv + p] = pt.z;
    }
    for (let n = 0; n < h.faceCount; ++n) {
        const normal = h.planes[n].normal;
        f[soa + 3 * nv + n] = normal.x;
        f[soa + 3 * nv + nf + n] = normal.y;
        f[soa + 3 * nv + 2 * nf + n] = normal.z;
        bytes[faces + n] = h.faces[n].edge;
        vector(f, planes + 4 * n, normal);
        f[planes + 4 * n + 3] = h.planes[n].offset;
    }
    for (let p = 0; p < h.vertexCount; ++p) {
        vector(f, points + 3 * p, h.points[p]);
        bytes[vertices + p] = h.vertices[p].edge;
    }
    for (let e = 0; e < h.edgeCount; ++e) {
        const ed = h.edges[e];
        const o = edges + 4 * e;
        bytes[o] = ed.next;
        bytes[o + 1] = ed.twin;
        bytes[o + 2] = ed.origin;
        bytes[o + 3] = ed.face;
    }
}

export function hullImage(h: HullImage): Uint8Array {
    const buffer = new ArrayBuffer(hullByteCount(h));
    const bytes = new Uint8Array(buffer);
    writeHullImage(
        h,
        bytes,
        new Uint32Array(buffer),
        new Float32Array(buffer),
        new BigUint64Array(buffer),
        0,
    );
    return bytes;
}

const secrets = [
    0x2d358dccaa6c78a5n,
    0x8bb84b93962eacc9n,
    0x4b33a62ed433d4a3n,
    0x4d5a2da51de1aa47n,
    0xa0761d6478bd642fn,
    0xe7037ed1a0b428dbn,
    0x90ed1765281c388cn,
    0xaaaaaaaaaaaaaaaan,
];
const low = (n: bigint): bigint => BigInt.asUintN(64, n);
function mix(a: bigint, b: bigint): bigint {
    const product = a * b;
    return low(product) ^ (product >> 64n);
}

/** Box3D b3Hash64NonZero, using its rapidhash V3 default seed and secrets. */
export function hash64NonZero(bytes: Uint8Array): bigint {
    const len = bytes.length;
    if (len === 0) return 1n;
    const view = new DataView(bytes.buffer, bytes.byteOffset, len);
    const read64 = (p: number): bigint => view.getBigUint64(p, true);
    let seed = mix(secrets[2], secrets[1]);
    let a = 0n,
        b = 0n,
        p = 0,
        i = len;
    if (len <= 16) {
        if (len >= 4) {
            seed ^= BigInt(len);
            if (len >= 8) {
                a = read64(0);
                b = read64(len - 8);
            } else {
                a = BigInt(view.getUint32(0, true));
                b = BigInt(view.getUint32(len - 4, true));
            }
        } else {
            a = (BigInt(bytes[0]) << 45n) | BigInt(bytes[len - 1]);
            b = BigInt(bytes[len >> 1]);
        }
    } else {
        if (len > 112) {
            const seeds = [seed, seed, seed, seed, seed, seed, seed];
            do {
                for (let lane = 0; lane < 7; ++lane) {
                    seeds[lane] = mix(
                        read64(p + 16 * lane) ^ secrets[lane],
                        read64(p + 16 * lane + 8) ^ seeds[lane],
                    );
                }
                p += 112;
                i -= 112;
            } while (i > 112);
            seed = seeds[0] ^ seeds[1] ^ seeds[2] ^ seeds[3] ^ seeds[4] ^ seeds[5] ^ seeds[6];
        }
        const tailSecrets = [2, 2, 1, 1, 2, 1];
        for (let lane = 0; lane < 6 && i > 16 * (lane + 1); ++lane) {
            seed = mix(
                read64(p + 16 * lane) ^ secrets[tailSecrets[lane]],
                read64(p + 16 * lane + 8) ^ seed,
            );
        }
        a = read64(p + i - 16) ^ BigInt(i);
        b = read64(p + i - 8);
    }
    const product = (a ^ secrets[1]) * (b ^ seed);
    const hash = mix(low(product) ^ secrets[7], (product >> 64n) ^ secrets[1] ^ BigInt(i));
    return hash === 0n ? 1n : hash;
}
