import { describe, expect, test } from "bun:test";
import {
    octDecodeNormal,
    octEncodeNormal,
    pack2x16unorm,
    pack4x8unorm,
    packLdrColor,
    packQuatSnorm16x4,
    unpack2x16unorm,
    unpackQuatSnorm16x4,
} from "./encode";

// Round-trip asserts on the production CPU codecs that pair with the WGSL
// emitters in encode.ts (OCT_ENCODE_WGSL, QUAT_SNORM16X4_WGSL). The CPU and WGSL
// halves must stay bit-identical, so the lattice properties asserted here — exact
// rails for cardinals, a derived per-component bound — are the spec the shaders
// decode against. The 2026-05-08 oct-encode bug was a cardinal decoding to a
// deterministic non-zero offset under the older unorm16 lattice; snorm16 puts
// 0 ↔ 0 exactly, so cardinals must round-trip exactly.

describe("oct-encoded normal (snorm16x2)", () => {
    const cardinals: [number, number, number, string][] = [
        [1, 0, 0, "+x"],
        [-1, 0, 0, "-x"],
        [0, 1, 0, "+y"],
        [0, -1, 0, "-y"],
        [0, 0, 1, "+z"],
        [0, 0, -1, "-z"],
    ];
    for (const [x, y, z, name] of cardinals) {
        test(`${name} round-trips exactly`, () => {
            // 0 ↔ 0 and ±1 ↔ ±32767 are exact rails, and decode normalizes a unit
            // cardinal back to itself (length already 1), so each component is exact.
            const d = octDecodeNormal(octEncodeNormal({ x, y, z }));
            expect(d.x).toBe(x);
            expect(d.y).toBe(y);
            expect(d.z).toBe(z);
        });
    }
});

describe("snorm16x4 quaternion codec", () => {
    // The engine's quaternion storage codec: four components packed into two u32
    // via pack2x16snorm, decode renormalizes to a unit quat. Per-component error is
    // bounded by one LSB (1/32767 ≈ 3.05e-5); identity and the six 180° axis-aligned
    // rotations sit on lattice rails (0 ↔ 0, ±1 ↔ ±32767) and round-trip bit-exact.

    function roundTrip(q: [number, number, number, number]): [number, number, number, number] {
        const [lo, hi] = packQuatSnorm16x4(q[0], q[1], q[2], q[3]);
        return unpackQuatSnorm16x4(lo, hi);
    }

    function quatNormalize(q: [number, number, number, number]): [number, number, number, number] {
        const len = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
        return [q[0] / len, q[1] / len, q[2] / len, q[3] / len];
    }

    // Quaternion double-cover: q and -q are the same rotation. Fold both branches
    // via the signed dot, then convert to an angle.
    function quatAngleDeg(
        a: [number, number, number, number],
        b: [number, number, number, number],
    ): number {
        const dot = Math.min(1, Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]));
        return (2 * Math.acos(dot) * 180) / Math.PI;
    }

    const cardinals: [number, number, number, number, string][] = [
        [0, 0, 0, 1, "identity"],
        [1, 0, 0, 0, "180° about +x"],
        [-1, 0, 0, 0, "180° about -x"],
        [0, 1, 0, 0, "180° about +y"],
        [0, -1, 0, 0, "180° about -y"],
        [0, 0, 1, 0, "180° about +z"],
        [0, 0, -1, 0, "180° about -z"],
    ];
    for (const [x, y, z, w, name] of cardinals) {
        test(`${name} round-trips bit-exact`, () => {
            const [dx, dy, dz, dw] = roundTrip([x, y, z, w]);
            expect(dx).toBe(x);
            expect(dy).toBe(y);
            expect(dz).toBe(z);
            expect(dw).toBe(w);
        });
    }

    test("per-component error bounded by 1 LSB (≈ 3.05e-5)", () => {
        // Sweep random unit quats; the bound is a property of pack2x16snorm and must
        // hold uniformly across the unit sphere. Allow the decode-side normalize to
        // add up to one LSB on top of the raw quantization.
        const OneLsb = 1 / 32767;
        let rng = 0x12345678;
        const next = () => {
            rng = (rng * 1664525 + 1013904223) >>> 0;
            return rng / 0xffffffff;
        };
        let maxComponentErr = 0;
        for (let i = 0; i < 1024; i++) {
            const q = quatNormalize([
                next() * 2 - 1,
                next() * 2 - 1,
                next() * 2 - 1,
                next() * 2 - 1,
            ]);
            const r = roundTrip(q);
            for (let j = 0; j < 4; j++) {
                maxComponentErr = Math.max(maxComponentErr, Math.abs(r[j] - q[j]));
            }
        }
        expect(maxComponentErr).toBeLessThan(2 * OneLsb);
    });

    test("worst-case angular error stays under 0.015° on random unit quats", () => {
        // For a unit quat with per-component perturbation δ = 1 LSB ≈ 3.05e-5,
        // |q·q'| ≥ 1 − 2δ², so the angle ≤ 4δ ≈ 1.22e-4 rad ≈ 0.007°. Allow 2×
        // margin (0.015°) for the renormalize step.
        let rng = 0xdeadbeef;
        const next = () => {
            rng = (rng * 1664525 + 1013904223) >>> 0;
            return rng / 0xffffffff;
        };
        let maxAngleDeg = 0;
        for (let i = 0; i < 4096; i++) {
            // Marsaglia (1972) — sample uniformly on the 3-sphere.
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
            const factor = Math.sqrt((1 - s1 * s1 - s2 * s2) / (s3 * s3 + s4 * s4));
            const q: [number, number, number, number] = [s1, s2, s3 * factor, s4 * factor];
            maxAngleDeg = Math.max(maxAngleDeg, quatAngleDeg(q, roundTrip(q)));
        }
        expect(maxAngleDeg).toBeLessThan(0.015);
    });
});

// The CPU twins of WGSL `pack4x8unorm` + `packLdrColor` (LDR_COLOR_PACK_WGSL), the `color`/`material`
// slab GPU-mirror packers. They must match the intrinsic bit-for-bit so the slab scatter (a lossless
// u32 copy) lands what a reader shader unpacks — the gym `render` transport assert pins the CPU↔GPU
// equality end to end; these pin the byte layout (lane → byte) + the sRGB-on-rgb / linear-on-alpha split.
describe("pack4x8unorm", () => {
    const byte = (p: number, i: number) => (p >>> (i * 8)) & 0xff;

    test("lane → byte order (x is the low byte)", () => {
        const p = pack4x8unorm(1, 0, 0.5, 1);
        expect(byte(p, 0)).toBe(255);
        expect(byte(p, 1)).toBe(0);
        expect(byte(p, 2)).toBe(128); // round(0.5 * 255) = 128
        expect(byte(p, 3)).toBe(255);
    });

    test("clamps out-of-range lanes to [0,1]", () => {
        const p = pack4x8unorm(2, -1, 0, 0);
        expect(byte(p, 0)).toBe(255);
        expect(byte(p, 1)).toBe(0);
    });
});

// The CPU twin of WGSL `pack2x16unorm` + the AABB-relative position round-trip the quantized
// vertex stream stores (POS_QUANT_WGSL `decodePos`). unorm16 maps [0,1] → [0,65535] uniformly,
// so a value quantized over an AABB axis of extent E has per-axis spacing E/65535 (gpu.md rule 6,
// 1.5e-5 × E). That bound is *derived*, not tuned — the spec the shader's `decodePos` decodes
// against. `roundTripAxis` mirrors the encode (normalize → pack) + decode (min + u·extent), with
// the same extent-0 guard the CPU quantizer + the WGSL decode share for a degenerate axis.
describe("unorm16 position (AABB-relative)", () => {
    const roundTripAxis = (p: number, min: number, ext: number): number => {
        const n = ext === 0 ? 0 : (p - min) / ext;
        const [u] = unpack2x16unorm(pack2x16unorm(n, 0));
        return min + u * ext;
    };

    test("lane → word order (x is the low 16 bits)", () => {
        const p = pack2x16unorm(1, 0);
        expect(p & 0xffff).toBe(65535);
        expect(p >>> 16).toBe(0);
    });

    test("clamps out-of-range lanes to [0,1]", () => {
        const p = pack2x16unorm(2, -1);
        expect(p & 0xffff).toBe(65535);
        expect(p >>> 16).toBe(0);
    });

    test("endpoints round-trip exactly (min/max sit on rails)", () => {
        // min → 0 → min and max → 65535 → max are exact lattice rails.
        expect(roundTripAxis(-2, -2, 5)).toBe(-2);
        expect(roundTripAxis(3, -2, 5)).toBe(3);
    });

    test("midpoint stays within extent/65535", () => {
        const min = -3;
        const ext = 8;
        const mid = min + ext / 2;
        expect(roundTripAxis(mid, min, ext)).toBeCloseTo(mid, 3); // 8/65535 ≈ 1.2e-4
    });

    test("per-axis error < extent/65535 over a deterministic sweep", () => {
        // A non-symmetric AABB; the bound is a property of the lattice and must hold
        // uniformly. Deterministic LCG spread (no Math.random — reload-safe, like the oct sweep),
        // sampling between lattice points so the rounding is actually exercised.
        const min = [-2, 0.5, -10];
        const max = [3, 1.5, 7];
        const maxErr = [0, 0, 0];
        let rng = 0x9e3779b9;
        const next = () => {
            rng = (rng * 1664525 + 1013904223) >>> 0;
            return rng / 0xffffffff;
        };
        for (let i = 0; i < 1024; i++) {
            for (let a = 0; a < 3; a++) {
                const ext = max[a] - min[a];
                const p = min[a] + next() * ext;
                maxErr[a] = Math.max(maxErr[a], Math.abs(roundTripAxis(p, min[a], ext) - p));
            }
        }
        for (let a = 0; a < 3; a++) {
            expect(maxErr[a]).toBeLessThan((max[a] - min[a]) / 65535);
        }
    });

    test("degenerate axis (extent 0) decodes to the offset exactly", () => {
        expect(roundTripAxis(5, 5, 0)).toBe(5);
        expect(roundTripAxis(99, 5, 0)).toBe(5); // any input on a flat axis → the min
    });
});

describe("packLdrColor", () => {
    const byte = (p: number, i: number) => (p >>> (i * 8)) & 0xff;

    test("rgb is sRGB-encoded, alpha stays linear", () => {
        // linear 0.5 → sRGB ≈ 0.7354 → 188; alpha 0.5 stays linear → 128
        const p = packLdrColor(0.5, 0.5, 0.5, 0.5);
        expect(byte(p, 0)).toBe(188);
        expect(byte(p, 1)).toBe(188);
        expect(byte(p, 2)).toBe(188);
        expect(byte(p, 3)).toBe(128);
    });

    test("endpoints round-trip exactly (0 and 1 sit on byte rails)", () => {
        expect(packLdrColor(1, 1, 1, 1) >>> 0).toBe(0xffffffff);
        expect(packLdrColor(0, 0, 0, 1) >>> 0).toBe(0xff000000);
    });
});
