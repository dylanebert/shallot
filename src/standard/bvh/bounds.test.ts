import { describe, expect, test } from "bun:test";
import { body, flat, IDIV_LEAF, integerDiscipline, noDivision } from "../../../tests/wgsl";
import { boundsWgsl, orderU32, unorderU32 } from "./bounds";

// The ordered-u32 codec is the reduction's whole correctness argument — WebGPU has integer atomics
// only, so the six axis extremes reduce as order-preserving u32 and decode back. It's pure, so the CPU
// arm of the same TGSL source proves the bijection + the ordering here; the real-GPU truth (the scene
// AABB itself, both arms) is the `accel` gym gate against tests/bvh/oracle.ts `sceneBounds`.

const wgsl = boundsWgsl();
const arms = [wgsl.subgroup, wgsl.lds];

const bits = new DataView(new ArrayBuffer(4));
const asF32 = (v: number): number => {
    bits.setFloat32(0, v);
    return bits.getFloat32(0);
};

const SAMPLES = [
    0, 1, -1, 1e-38, -1e-38, 3.4028234663852886e38, -3.4028234663852886e38, 0.5, -0.5, 123.456,
    -123.456, 1e-7, -1e-7,
].map(asF32);

describe("ordered-u32 codec", () => {
    test("round-trips every sample exactly — the decode is the encode's inverse", () => {
        for (const v of SAMPLES) expect(unorderU32(orderU32(v))).toBe(v);
    });

    test("encodes into u32 range, never a negative JS int", () => {
        for (const v of SAMPLES) {
            const o = orderU32(v);
            expect(o).toBeGreaterThanOrEqual(0);
            expect(o).toBeLessThanOrEqual(0xffffffff);
        }
    });

    test("integer ordering agrees with float ordering across the whole range", () => {
        // the property atomicMin/atomicMax rely on: the encoded u32s must sort into the same order as
        // the floats, so an integer atomic computes the float extreme
        const sorted = [...SAMPLES].sort((a, b) => a - b);
        for (let i = 1; i < sorted.length; i++)
            expect(orderU32(sorted[i])).toBeGreaterThan(orderU32(sorted[i - 1]));
    });

    test("negatives sort below positives — the sign-bit flip's whole job", () => {
        expect(orderU32(-1e-38)).toBeLessThan(orderU32(0));
        expect(orderU32(0)).toBeLessThan(orderU32(1e-38));
        expect(orderU32(-3.4028234663852886e38)).toBe(0x00800000);
        expect(orderU32(3.4028234663852886e38)).toBe(0xff7fffff);
    });
});

describe("emitted WGSL", () => {
    test("neither arm divides anything but through `idiv`, and every local stays u32", () => {
        for (const src of [...arms, wgsl.finalize]) {
            noDivision(src);
            integerDiscipline(src);
        }
    });

    test("the two workgroup-size divides go through idiv", () => {
        // `tid / sgsize` and `WG / sgsize` are the builder's only integer divisions — the construct
        // TGSL transpiles to a fractional f32 quotient
        expect(flat(wgsl.subgroup)).toContain(IDIV_LEAF);
        expect(flat(wgsl.subgroup)).toContain("idiv(tid, sgsize)");
        expect(flat(wgsl.subgroup)).toContain("idiv(256u, sgsize)");
        expect(wgsl.lds).not.toContain("idiv");
    });

    test("both arms share the fold and the ordered publish verbatim", () => {
        for (const fn of ["fn foldSlice(", "fn publishExtremes(", "fn orderU32("])
            expect(body(wgsl.lds, fn)).toBe(body(wgsl.subgroup, fn));
    });

    test("the subgroup arm declares the subgroup builtins and the LDS arm declares none", () => {
        expect(wgsl.subgroup).toContain("@builtin(subgroup_invocation_id)");
        expect(wgsl.subgroup).toContain("@builtin(subgroup_size)");
        expect(wgsl.subgroup).toContain("subgroupMin(");
        expect(wgsl.lds).not.toContain("subgroup");
    });

    test("each arm sizes its workgroup scratch to what its reduce needs", () => {
        expect(wgsl.subgroup).toMatch(/var<workgroup> \w+: array<vec3f, 64>/);
        expect(wgsl.lds).toMatch(/var<workgroup> \w+: array<vec3f, 256>/);
        // the LDS tree halves from WG/2 and barriers every step
        expect(flat(wgsl.lds)).toContain("var s = 128u");
        expect(flat(wgsl.lds)).toContain("s = (s >> 1u)");
    });

    test("the level-2 reduce runs on every lane, outside the lane-0 publish guard", () => {
        // a subgroup op inside `if (tid == 0)` is non-uniform control flow — Tint rejects the module,
        // so this is a compile-time failure, not a wrong result
        const main = flat(body(wgsl.subgroup, "@compute"));
        expect(main).toMatch(/let tmin = subgroupMin\(vmin\); let tmax = subgroupMax\(vmax\);/);
        expect(main.indexOf("subgroupMax(vmax)")).toBeLessThan(main.indexOf("if ((tid == 0u))"));
    });

    test("exactly one atomic per axis per workgroup, gated on lane 0", () => {
        const publish = body(wgsl.subgroup, "fn publishExtremes(");
        expect(publish.match(/atomicMin\(/g)).toHaveLength(3);
        expect(publish.match(/atomicMax\(/g)).toHaveLength(3);
        for (const src of arms) expect(flat(src)).toContain("if ((tid == 0u))");
    });

    test("the emitted codec carries the same sign-bit flip the CPU arm proves", () => {
        // the CPU arm above proves the property; this pins the GPU arm to the same formula, since the
        // dual's two branches are separate source and only one of them ships
        expect(flat(body(wgsl.subgroup, "fn orderU32("))).toContain(
            "select(~u, (u | 2147483648u), ((u >> 31u) == 0u))",
        );
        expect(flat(body(wgsl.finalize, "fn unorderU32("))).toContain(
            "select(~o, (o ^ 2147483648u), ((o >> 31u) == 1u))",
        );
    });

    test("the neutral element folds to the exact f32 max — the bitcast is comptime", () => {
        // `bitcastU32toF32(0x7f7fffff)` has a constant argument, so typegpu folds it at shader-gen; the
        // decimal it prints is the exact f32 max, so WGSL parses back the same bit pattern
        for (const src of arms) expect(src).toContain("const fmax = 3.4028234663852886e+38f;");
        expect(Math.fround(3.4028234663852886e38)).toBe(3.4028234663852886e38);
    });

    test("finalize writes both AABB corners and reads the scratch non-atomically", () => {
        expect(wgsl.finalize).toMatch(/var<storage, read> scratch: array<u32>/);
        expect(wgsl.finalize).toMatch(/var<storage, read_write> bounds: array<vec4f, 2>/);
        expect(wgsl.finalize).not.toContain("atomic");
        integerDiscipline(wgsl.finalize);
    });
});
