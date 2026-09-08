import { describe, expect, test } from "bun:test";
import { vec2f, vec3f, vec4f } from "typegpu/data";
import {
    hdrColorPackWgsl,
    hdrColorUnpackWgsl,
    ldrColorPackWgsl,
    ldrColorUnpackWgsl,
    meshIdOf,
    octDecodeNormal,
    octEncode,
    octEncodeNormal,
    octEncodeWgsl,
    packColor4,
    packLdrColor,
    packQuatSnorm16x4,
    packUnorm2,
    posQuantPackWgsl,
    posQuantWgsl,
    quatSnorm16x4Wgsl,
    smallest3Wgsl,
    unpackQuatSnorm16x4,
    xformQuat,
    xformWgsl,
} from "./encode";
import { packUnorm2x16, packUnorm4x8, unpackUnorm2x16 } from "./tgsl";

// Round-trip asserts on the storage codecs in encode.ts, called here on the CPU. Each is ONE TGSL
// function — the same source the shader resolves — so these lattice properties (exact rails for
// cardinals, a derived per-component bound) are the spec both sides share by construction, not a pair
// held in step by hand. The 2026-05-08 oct-encode bug was a cardinal decoding to a
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
            const d = octDecodeNormal(octEncodeNormal(vec3f(x, y, z)));
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
        const p = packQuatSnorm16x4(vec4f(q[0], q[1], q[2], q[3]));
        const r = unpackQuatSnorm16x4(p);
        return [r.x, r.y, r.z, r.w];
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

    test("worst-case angular error stays under 0.06° on random unit quats", () => {
        // Two terms, and the second dominates because `acos` near 1 is sqrt-amplified:
        //   quantization — per-component δ ≤ 1 LSB ≈ 3.05e-5 gives 1 − q·q' ≤ 2δ² ≈ 1.9e-9;
        //   f32 storage — the codec returns f32 (it IS the shader function, run on the CPU), so the
        //   decoded quat's length is 1 only to within the renormalize's own f32 rounding, ~2⁻²⁴ ≈ 6e-8,
        //   which lands on 1 − q·q' directly.
        // angle = 2·acos(q·q') ≤ 2·√(2 × 6.2e-8) ≈ 7.0e-4 rad ≈ 0.040° → bound 0.06° with margin
        // (measured max 0.036°).
        // The f32 term is the honest one: the old hand-written CPU twin computed the renormalize in
        // f64 and measured 0.003° here, an error the shader never actually achieved.
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
        expect(maxAngleDeg).toBeLessThan(0.06);
    });
});

// The CPU twins of WGSL `pack4x8unorm` (the `packUnorm4x8` leaf) + `packLdrColor` (ldrColorPackWgsl()), the `color`/`material`
// slab GPU-mirror packers. They must match the intrinsic bit-for-bit so the slab scatter (a lossless
// u32 copy) lands what a reader shader unpacks — the gym `render` transport assert pins the CPU↔GPU
// equality end to end; these pin the byte layout (lane → byte) + the sRGB-on-rgb / linear-on-alpha split.
describe("packUnorm4x8", () => {
    const byte = (p: number, i: number) => (p >>> (i * 8)) & 0xff;

    test("lane → byte order (x is the low byte)", () => {
        const p = packUnorm4x8(vec4f(1, 0, 0.5, 1));
        expect(byte(p, 0)).toBe(255);
        expect(byte(p, 1)).toBe(0);
        expect(byte(p, 2)).toBe(128); // round(0.5 * 255) = 128
        expect(byte(p, 3)).toBe(255);
    });

    test("clamps out-of-range lanes to [0,1]", () => {
        const p = packUnorm4x8(vec4f(2, -1, 0, 0));
        expect(byte(p, 0)).toBe(255);
        expect(byte(p, 1)).toBe(0);
    });
});

// The CPU twin of WGSL `pack2x16unorm` + the AABB-relative position round-trip the quantized
// vertex stream stores (posQuantWgsl() `decodePos`). unorm16 maps [0,1] → [0,65535] uniformly,
// so a value quantized over an AABB axis of extent E has per-axis spacing E/65535 (gpu.md rule 6,
// 1.5e-5 × E). That bound is *derived*, not tuned — the spec the shader's `decodePos` decodes
// against. `roundTripAxis` mirrors the encode (normalize → pack) + decode (min + u·extent), with
// the same extent-0 guard the CPU quantizer + the WGSL decode share for a degenerate axis.
describe("unorm16 position (AABB-relative)", () => {
    const roundTripAxis = (p: number, min: number, ext: number): number => {
        const n = ext === 0 ? 0 : (p - min) / ext;
        const u = unpackUnorm2x16(packUnorm2(n, 0)).x;
        return min + u * ext;
    };

    test("lane → word order (x is the low 16 bits)", () => {
        const p = packUnorm2(1, 0);
        expect(p & 0xffff).toBe(65535);
        expect(p >>> 16).toBe(0);
    });

    test("clamps out-of-range lanes to [0,1]", () => {
        const p = packUnorm2(2, -1);
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
        const p = packColor4(0.5, 0.5, 0.5, 0.5);
        expect(byte(p, 0)).toBe(188);
        expect(byte(p, 1)).toBe(188);
        expect(byte(p, 2)).toBe(188);
        expect(byte(p, 3)).toBe(128);
    });

    test("endpoints round-trip exactly (0 and 1 sit on byte rails)", () => {
        expect(packColor4(1, 1, 1, 1) >>> 0).toBe(0xffffffff);
        expect(packColor4(0, 0, 0, 1) >>> 0).toBe(0xff000000);
    });
});

describe("meshIdOf", () => {
    test("reads the high half unsigned, like the WGSL shift does", () => {
        // JS `>>` is arithmetic and WGSL's `>>` on u32 is logical, so a word with bit 31 set is where
        // a bare shift would give the CPU arm a wrapped negative and the shader the right id.
        expect(meshIdOf(0x0007_0000)).toBe(7);
        expect(meshIdOf(0x8000_1234)).toBe(0x8000);
        expect(meshIdOf(0xffff_ffff)).toBe(0xffff);
    });
});

// The quaternion rotation every transform path runs through (`xformPoint` / `xformNormal` / the physics
// compose all bottom out here). Checked against the rotation matrix R(q) built independently from the
// quaternion — a different formulation, so an algebra slip in the cross-product form shows up.
describe("xformQuat", () => {
    const rotateByMatrix = (
        q: [number, number, number, number],
        v: [number, number, number],
    ): [number, number, number] => {
        const [x, y, z, w] = q;
        return [
            (1 - 2 * (y * y + z * z)) * v[0] +
                2 * (x * y - w * z) * v[1] +
                2 * (x * z + w * y) * v[2],
            2 * (x * y + w * z) * v[0] +
                (1 - 2 * (x * x + z * z)) * v[1] +
                2 * (y * z - w * x) * v[2],
            2 * (x * z - w * y) * v[0] +
                2 * (y * z + w * x) * v[1] +
                (1 - 2 * (x * x + y * y)) * v[2],
        ];
    };

    test("a 90° turn about +z carries +x to +y", () => {
        const s = Math.SQRT1_2;
        const r = xformQuat(vec4f(0, 0, s, s), vec3f(1, 0, 0));
        expect(r.x).toBeCloseTo(0, 6);
        expect(r.y).toBeCloseTo(1, 6);
        expect(r.z).toBeCloseTo(0, 6);
    });

    test("matches R(q)·v over a deterministic sweep", () => {
        // f32 throughout (the codec IS the shader function), so the bound is a handful of f32 ulps on
        // a unit-scale result: 8 rounded ops × 2⁻²⁴ ≈ 5e-7, rounded to 1e-6.
        let rng = 0x5bd1e995;
        const next = () => {
            rng = (rng * 1664525 + 1013904223) >>> 0;
            return rng / 0xffffffff;
        };
        for (let i = 0; i < 512; i++) {
            const raw = [next() * 2 - 1, next() * 2 - 1, next() * 2 - 1, next() * 2 - 1];
            const len = Math.hypot(raw[0], raw[1], raw[2], raw[3]) || 1;
            const q: [number, number, number, number] = [
                raw[0] / len,
                raw[1] / len,
                raw[2] / len,
                raw[3] / len,
            ];
            const v: [number, number, number] = [next() * 4 - 2, next() * 4 - 2, next() * 4 - 2];
            const got = xformQuat(vec4f(q[0], q[1], q[2], q[3]), vec3f(v[0], v[1], v[2]));
            const want = rotateByMatrix(q, v);
            expect(got.x).toBeCloseTo(want[0], 6);
            expect(got.y).toBeCloseTo(want[1], 6);
            expect(got.z).toBeCloseTo(want[2], 6);
        }
    });
});

// The chunk contract. Each `*Wgsl()` is a splice-ready WGSL string, and a shader concatenates several,
// so two properties are load-bearing and neither is visible from a CPU round-trip: the emitted function
// names must be the authored ones (a raw-WGSL splice site calls them by name), and no two chunks a real
// shader splices together may define the same thing twice — a duplicate `fn` or `struct` is a WGSL
// compile error the unit tier can catch far cheaper than `bun bench` can.
describe("WGSL chunks", () => {
    const defs = (code: string): string[] =>
        [...code.matchAll(/^\s*(?:fn|struct)\s+([A-Za-z_]\w*)/gm)].map((m) => m[1]);

    test("chunks emit their documented function names", () => {
        expect(defs(octEncodeWgsl())).toEqual(
            expect.arrayContaining(["octEncodeNormal", "octDecodeNormal"]),
        );
        expect(defs(quatSnorm16x4Wgsl())).toEqual(
            expect.arrayContaining(["packQuatSnorm16x4", "unpackQuatSnorm16x4"]),
        );
        expect(defs(smallest3Wgsl())).toEqual(
            expect.arrayContaining(["packQuatSmallest3", "unpackQuatSmallest3"]),
        );
        expect(defs(hdrColorUnpackWgsl())).toEqual(expect.arrayContaining(["unpackHdrColor"]));
        expect(defs(hdrColorPackWgsl())).toEqual(expect.arrayContaining(["packHdrColor"]));
        expect(defs(posQuantWgsl())).toEqual(
            expect.arrayContaining(["MeshQuant", "meshIdOf", "decodePos", "decodeUv"]),
        );
        expect(defs(posQuantPackWgsl())).toEqual(expect.arrayContaining(["encodePos", "encodeUv"]));
        expect(defs(xformWgsl())).toEqual(
            expect.arrayContaining(["Xform", "xformQuat", "xformPoint", "xformNormal", "xformMat"]),
        );
        expect(defs(ldrColorUnpackWgsl())).toEqual(
            expect.arrayContaining(["srgbToLinear1", "unpackLdrColor"]),
        );
        expect(defs(ldrColorPackWgsl())).toEqual(
            expect.arrayContaining(["linearToSrgb1", "packLdrColor"]),
        );
    });

    // Pairwise over every chunk, which covers every splice: a definition emitted by two chunks of a
    // larger combination is a duplicate in one of its pairs. This is what catches a shared dependency
    // (the snorm leaves, `MeshQuant`) landing in both halves of a pair — a WGSL compile error a live
    // shader would hit at pipeline creation and an unspliced chunk would hide until someone used it.
    const chunks: [string, () => string][] = [
        ["octEncodeWgsl", octEncodeWgsl],
        ["quatSnorm16x4Wgsl", quatSnorm16x4Wgsl],
        ["smallest3Wgsl", smallest3Wgsl],
        ["posQuantWgsl", posQuantWgsl],
        ["posQuantPackWgsl", posQuantPackWgsl],
        ["xformWgsl", xformWgsl],
        ["ldrColorUnpackWgsl", ldrColorUnpackWgsl],
        ["ldrColorPackWgsl", ldrColorPackWgsl],
        ["hdrColorUnpackWgsl", hdrColorUnpackWgsl],
        ["hdrColorPackWgsl", hdrColorPackWgsl],
    ];
    for (let i = 0; i < chunks.length; i++) {
        for (let j = i + 1; j < chunks.length; j++) {
            const [aName, a] = chunks[i];
            const [bName, b] = chunks[j];
            test(`${aName} + ${bName} splice without a duplicate definition`, () => {
                const all = [...defs(a()), ...defs(b())];
                const dupes = all.filter((n, k) => all.indexOf(n) !== k);
                expect(dupes).toEqual([]);
            });
        }
    }

    test("a shared dependency is emitted once, into the base half, whatever resolves first", () => {
        // The two documented pairs: the position quantizers share `MeshQuant` and the snorm16 codecs
        // share the pack/unpack leaves. Each dependent half resolves its base half first, so the split
        // is the same however a consumer asks — and the base half is the one carrying the definition.
        expect(defs(posQuantPackWgsl())).not.toContain("MeshQuant");
        expect(defs(posQuantWgsl())).toContain("MeshQuant");
        expect(defs(quatSnorm16x4Wgsl())).not.toContain("packSnorm2x16");
        expect(defs(octEncodeWgsl())).toContain("packSnorm2x16");
    });
});

// The hot-path mirrors carry the only duplicated logic left in this file, so they get a differential
// against the TGSL function each mirrors — bit-identical on the packed word, over a sweep dense enough
// to exercise both octahedral hemispheres and the sRGB transfer's two segments. Drift here is the exact
// failure the old hand-paired CPU/WGSL twins could hide.
describe("hot-path mirrors match their TGSL source", () => {
    let rng = 0x2545f491;
    const next = () => {
        rng = (rng * 1664525 + 1013904223) >>> 0;
        return rng / 0xffffffff;
    };

    test("octEncode === octEncodeNormal", () => {
        const cardinals: [number, number, number][] = [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
            // the edges where the two could take different branches: a zero vector and a -0 lane hit
            // the degenerate-denominator guard, and a non-unit input exercises the fold on an
            // unnormalized value. NaN is unreachable here by construction — typegpu's schemas refuse a
            // non-finite value (the WGSL finite-math assumption), so the TGSL half can't be called with
            // one; the mirror still takes the same branch the shader would (encode.ts).
            [0, 0, 0],
            [0, -0, 0],
            [2, 0, 0],
            [0.3, 0.3, -0.3],
        ];
        for (const [x, y, z] of cardinals) {
            expect(octEncode(x, y, z)).toBe(octEncodeNormal(vec3f(x, y, z)));
        }
        for (let i = 0; i < 4096; i++) {
            const v = [next() * 2 - 1, next() * 2 - 1, next() * 2 - 1];
            const l = Math.hypot(v[0], v[1], v[2]) || 1;
            const [x, y, z] = [v[0] / l, v[1] / l, v[2] / l];
            expect(octEncode(x, y, z)).toBe(octEncodeNormal(vec3f(x, y, z)));
        }
    });

    test("packUnorm2 === packUnorm2x16", () => {
        for (const [x, y] of [
            [0, 0],
            [1, 1],
            [2, -1],
        ]) {
            expect(packUnorm2(x, y)).toBe(packUnorm2x16(vec2f(x, y)));
        }
        for (let i = 0; i < 4096; i++) {
            const [x, y] = [next(), next()];
            expect(packUnorm2(x, y)).toBe(packUnorm2x16(vec2f(x, y)));
        }
    });

    test("packColor4 === packLdrColor", () => {
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
