import { describe, expect, test } from "bun:test";
import { vec2f, vec3f, vec4f } from "typegpu/data";
import { pack2x16unorm } from "typegpu/std";
import {
    octEncode,
    octEncodeNormal,
    packColor4,
    packLdrColor,
    packQuatSnorm16x4,
    packUnorm2,
    unpackQuatSnorm16x4,
} from "./encode";

// The plain-JS mirrors are the only duplicated logic in encode.ts, so each is pinned against the TGSL
// function it mirrors: bit-identical on the packed word, over a sweep dense enough to reach both
// octahedral hemispheres and both segments of the sRGB transfer.
describe("each plain-JS mirror packs the same word as its TGSL source", () => {
    let rng = 0x2545f491;
    const next = () => {
        rng = (rng * 1664525 + 1013904223) >>> 0;
        return rng / 0xffffffff;
    };

    test("octEncode matches octEncodeNormal", () => {
        const edges: [number, number, number][] = [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
            // a zero vector and a -0 lane reach the degenerate-denominator guard; a non-unit input
            // reaches the fold on an unnormalized value.
            [0, 0, 0],
            [0, -0, 0],
            [2, 0, 0],
            [0.3, 0.3, -0.3],
        ];
        for (const [x, y, z] of edges) {
            expect(octEncode(x, y, z)).toBe(octEncodeNormal(vec3f(x, y, z)));
        }
        for (let i = 0; i < 4096; i++) {
            const v = [next() * 2 - 1, next() * 2 - 1, next() * 2 - 1];
            const l = Math.hypot(v[0], v[1], v[2]) || 1;
            const [x, y, z] = [v[0] / l, v[1] / l, v[2] / l];
            expect(octEncode(x, y, z)).toBe(octEncodeNormal(vec3f(x, y, z)));
        }
    });

    test("packUnorm2 matches pack2x16unorm", () => {
        for (const [x, y] of [
            [0, 0],
            [1, 1],
            [2, -1],
        ]) {
            expect(packUnorm2(x, y)).toBe(pack2x16unorm(vec2f(x, y)));
        }
        for (let i = 0; i < 4096; i++) {
            const [x, y] = [next(), next()];
            expect(packUnorm2(x, y)).toBe(pack2x16unorm(vec2f(x, y)));
        }
    });

    test("packColor4 matches packLdrColor", () => {
        for (const [r, g, b, a] of [
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [0.0031308, 0.0031308, 0.0031308, 0], // the sRGB segment boundary
            [2, -1, 0.5, 0.5],
        ]) {
            expect(packColor4(r, g, b, a)).toBe(packLdrColor(vec3f(r, g, b), a));
        }
        for (let i = 0; i < 2048; i++) {
            const [r, g, b, a] = [next(), next(), next(), next()];
            expect(packColor4(r, g, b, a)).toBe(packLdrColor(vec3f(r, g, b), a));
        }
    });
});

// The snorm16x4 quaternion codec, as encode.ts states it: identity and the six 180° axis-aligned
// rotations sit on lattice rails (0 ↔ 0, ±1 ↔ ±32767), per-component error stays within one LSB plus
// the decode's renormalize, and the worst-case angle stays small.
describe("snorm16x4 quaternion codec", () => {
    type Quat = [number, number, number, number];

    function roundTrip(q: Quat): Quat {
        const r = unpackQuatSnorm16x4(packQuatSnorm16x4(vec4f(q[0], q[1], q[2], q[3])));
        return [r.x, r.y, r.z, r.w];
    }

    // q and -q are the same rotation: fold both through the absolute dot.
    function angleDeg(a: Quat, b: Quat): number {
        const dot = Math.min(1, Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]));
        return (2 * Math.acos(dot) * 180) / Math.PI;
    }

    const cardinals: [Quat, string][] = [
        [[0, 0, 0, 1], "identity"],
        [[1, 0, 0, 0], "180° about +x"],
        [[-1, 0, 0, 0], "180° about -x"],
        [[0, 1, 0, 0], "180° about +y"],
        [[0, -1, 0, 0], "180° about -y"],
        [[0, 0, 1, 0], "180° about +z"],
        [[0, 0, -1, 0], "180° about -z"],
    ];
    for (const [q, name] of cardinals) {
        test(`${name} round-trips bit-exact`, () => {
            expect(roundTrip(q)).toEqual(q);
        });
    }

    test("per-component error stays under two LSB (1/32767 each)", () => {
        // one LSB of quantization, plus up to one from the decode's renormalize.
        let rng = 0x12345678;
        const next = () => {
            rng = (rng * 1664525 + 1013904223) >>> 0;
            return rng / 0xffffffff;
        };
        let maxErr = 0;
        for (let i = 0; i < 1024; i++) {
            const raw = [next() * 2 - 1, next() * 2 - 1, next() * 2 - 1, next() * 2 - 1];
            const len = Math.hypot(raw[0], raw[1], raw[2], raw[3]) || 1;
            const q = raw.map((c) => c / len) as Quat;
            const r = roundTrip(q);
            for (let j = 0; j < 4; j++) maxErr = Math.max(maxErr, Math.abs(r[j] - q[j]));
        }
        expect(maxErr).toBeLessThan(2 / 32767);
    });

    test("worst-case angular error stays under 0.06° on uniform unit quats", () => {
        // The f32 renormalize dominates: the decoded length is 1 only to ~2⁻²⁴ ≈ 6e-8, which lands on
        // 1 − q·q' directly, so angle = 2·acos(q·q') ≤ 2·√(2 × 6.2e-8) ≈ 0.040° (measured max 0.036°).
        let rng = 0xdeadbeef;
        const next = () => {
            rng = (rng * 1664525 + 1013904223) >>> 0;
            return rng / 0xffffffff;
        };
        let maxDeg = 0;
        for (let i = 0; i < 4096; i++) {
            // Marsaglia (1972): uniform on the 3-sphere.
            let s1: number;
            let s2: number;
            do {
                s1 = next() * 2 - 1;
                s2 = next() * 2 - 1;
            } while (s1 * s1 + s2 * s2 >= 1);
            let s3: number;
            let s4: number;
            do {
                s3 = next() * 2 - 1;
                s4 = next() * 2 - 1;
            } while (s3 * s3 + s4 * s4 >= 1);
            const f = Math.sqrt((1 - s1 * s1 - s2 * s2) / (s3 * s3 + s4 * s4));
            const q: Quat = [s1, s2, s3 * f, s4 * f];
            maxDeg = Math.max(maxDeg, angleDeg(q, roundTrip(q)));
        }
        expect(maxDeg).toBeLessThan(0.06);
    });
});
