import { describe, expect, test } from "bun:test";
import * as d from "typegpu/data";
import { allFixtures, primMax, primMin } from "../../../tests/bvh/fixtures";
import {
    mortonCodes,
    interleaveBits32 as oracleInterleave,
    mortonCode as oracleQuantize,
    sceneBounds,
    type Vec3,
} from "../../../tests/bvh/oracle";
import { flat, integerDiscipline, noIntegerDivision } from "../../../tests/wgsl";
import { interleaveBits32, mortonCode, mortonWgsl } from "./morton";

// Two seams, no device. The Morton math is a pair of TGSL functions, so the CPU arm runs the *same
// source* the shader does — differentially checked against the independent NexusBVH port in
// tests/bvh/oracle.ts (which is also what the `accel` gym gate compares the real GPU against). The
// emitted WGSL is the second seam: the shape assertions that catch TGSL's silent-wrong classes.

const CAP = 4096;
const wgsl = mortonWgsl(CAP);

describe("interleaveBits32", () => {
    test("agrees with the oracle bit-for-bit across the 10-bit domain", () => {
        for (let x = 0; x < 1024; x++) expect(interleaveBits32(x)).toBe(oracleInterleave(x));
    });

    test("masks to 10 bits and spreads to every third", () => {
        expect(interleaveBits32(0)).toBe(0);
        expect(interleaveBits32(1)).toBe(1);
        expect(interleaveBits32(0x3ff)).toBe(0x9249249);
        // bits above the low 10 are dropped, not folded in
        expect(interleaveBits32(0xfffffc00)).toBe(0);
    });
});

describe("mortonCode", () => {
    test("normalizes and quantizes exactly like the oracle over every fixture", () => {
        for (const { name, prims } of allFixtures()) {
            const bounds = sceneBounds(prims);
            const expected = mortonCodes(prims, bounds);
            const ext: Vec3 = [
                bounds.max[0] - bounds.min[0],
                bounds.max[1] - bounds.min[1],
                bounds.max[2] - bounds.min[2],
            ];
            const bmin = d.vec3f(...bounds.min);
            const extent = d.vec3f(...ext);
            for (let i = 0; i < prims.count; i++) {
                const mn = primMin(prims, i);
                const mx = primMax(prims, i);
                const centroid = d.vec3f(...([0, 1, 2].map((a) => (mn[a] + mx[a]) * 0.5) as Vec3));
                expect(mortonCode(centroid, bmin, extent), `${name} prim ${i}`).toBe(expected[i]);
            }
        }
    });

    test("a zero-extent axis codes as 0, not NaN — the coplanar-scene requirement", () => {
        // extent zero on every axis: the divide is discarded, so the code is 0
        expect(mortonCode(d.vec3f(5, 5, 5), d.vec3f(5, 5, 5), d.vec3f(0, 0, 0))).toBe(0);
        // y flat, x/z live at the top of the range
        expect(mortonCode(d.vec3f(1, 7, 1), d.vec3f(0, 7, 0), d.vec3f(1, 0, 1))).toBe(
            oracleQuantize(1, 0, 1),
        );
    });

    test("out-of-range centroids clamp instead of wrapping the quantizer", () => {
        expect(mortonCode(d.vec3f(-10, -10, -10), d.vec3f(0, 0, 0), d.vec3f(1, 1, 1))).toBe(0);
        expect(mortonCode(d.vec3f(10, 10, 10), d.vec3f(0, 0, 0), d.vec3f(1, 1, 1))).toBe(
            oracleQuantize(1, 1, 1),
        );
    });
});

describe("emitted WGSL", () => {
    test("the only division is the float normalize — no integer `/` slipped through", () => {
        noIntegerDivision(wgsl);
        integerDiscipline(wgsl);
    });

    test("maxPrims and the sentinel fold to literals — neither costs a binding", () => {
        expect(flat(wgsl)).toContain(`i < ${CAP}u`);
        expect(wgsl).toContain("4294967295u");
        expect(wgsl).not.toContain("MAX_PRIMS");
    });

    test("the grid-stride loop reads the count on the GPU, so it never crosses to the CPU", () => {
        expect(wgsl).toMatch(/var<storage, read> countBuf: array<u32>/);
        expect(flat(wgsl)).toContain("countBuf[0i]");
        expect(flat(wgsl)).toContain("nwg.x * 256u");
    });

    test("keys and payload are the only writable bindings", () => {
        for (const name of ["keys", "payload"])
            expect(wgsl).toMatch(new RegExp(`var<storage, read_write> ${name}:`));
        for (const name of ["prims", "bounds", "countBuf"])
            expect(wgsl).toMatch(new RegExp(`var<storage, read> ${name}:`));
    });
});
