// The GPU vertex-stage sampler (`vertex-displacement.ts`'s `catmullRom1D`/`wrapIndex`) and the
// plain-TS reference (`reconstruction.ts`'s `catmullRom1D`/`wrap`) are two hand-authored expressions
// of the same textbook formula, not one derivation under two names — this file is the arm both
// modules' own docblocks promise: it drives the SAME inputs through both and asserts the outputs
// match, so a divergence localizes to whichever side changed rather than being explained away as "an
// equivalent formula" (both docblocks' own words).
//
// `wrapIndex`/`catmullRom1D` are called DIRECTLY from JS, not dispatched on a GPU. This is sound for
// exactly these two functions and no others in this package: both are pure add/mul/select/mod with no
// transcendental (`cos`/`sin`), so typegpu's CPU execution of a `tgpu.fn` reproduces the WGSL runtime
// exactly — verified by `gpu-fft.ts`'s own docblock for `twiddleAngle` ("typegpu's CPU execution of a
// `tgpu.fn` only rounds its RETURN value once ... which cannot represent what a real WGSL runtime's
// `cos`/`sin` intrinsics compute"). `bicubicSample0`/`bicubicSample1` themselves are NOT covered here:
// their `std.textureLoad` calls resolve against a bound texture at pipeline creation and have no
// CPU-callable form — this file's own scope is the two shared helpers, not the full vertex-stage
// kernel (`vertex-displacement.ts`'s own header names this boundary).
import { describe, expect, test } from "bun:test";
import * as d from "typegpu/data";
import { catmullRom1D as refCatmullRom1D, wrap } from "../src/ocean/reconstruction";
import { catmullRom1D, wrapIndex } from "../src/ocean/vertex-displacement";

describe("wrapIndex vs reconstruction.ts's wrap — same formula, called directly from JS", () => {
    test("agrees on every offset in a texture's practical index range, both signs", () => {
        const n = 128;
        for (let i = -2 * n; i <= 2 * n; i++) {
            expect(wrapIndex(i, n)).toBe(wrap(i, n));
        }
    });

    test("RED-WITNESS — a wrap widened to `i % n` (drops the negative-index wraparound) diverges", () => {
        const brokenWrap = (i: number, n: number) => i % n;
        let anyDivergence = false;
        for (let i = -5; i < 0; i++) {
            if (brokenWrap(i, 8) !== wrapIndex(i, 8)) anyDivergence = true;
        }
        expect(
            anyDivergence,
            "the mutation must diverge from wrapIndex on at least one negative i",
        ).toBe(true);
    });
});

describe("vertex-displacement.ts's catmullRom1D vs reconstruction.ts's — same formula, per channel", () => {
    // a position-encoded fixture: every source texel and channel a distinct value, so a swapped tap
    // or dropped channel is visible (`checks.md`'s own recorded shape for this class of arm).
    const p0 = [1.1, 2.2, 3.3, 4.4];
    const p1 = [5.5, 6.6, 7.7, 8.8];
    const p2 = [9.9, 10.1, 11.2, 12.3];
    const p3 = [13.4, 14.5, 15.6, 16.7];

    function vec4(row: number[]): d.v4f {
        return d.vec4f(row[0], row[1], row[2], row[3]);
    }

    test("agrees at several off-grid t, all four channels", () => {
        for (const t of [0, 0.13, 0.5, 0.87, 1]) {
            const gpuResult = catmullRom1D(vec4(p0), vec4(p1), vec4(p2), vec4(p3), t);
            for (let channel = 0; channel < 4; channel++) {
                const refResult = refCatmullRom1D(
                    p0[channel],
                    p1[channel],
                    p2[channel],
                    p3[channel],
                    t,
                );
                const gpuChannel = [gpuResult.x, gpuResult.y, gpuResult.z, gpuResult.w][channel];
                expect(gpuChannel).toBeCloseTo(refResult, 5);
            }
        }
    });

    test("RED-WITNESS — reconstruction.ts's own c-coefficient mutation (0.5*p2 -> 0.35*p2) diverges from the GPU side", () => {
        function mutatedRefCatmullRom1D(
            m0: number,
            m1: number,
            m2: number,
            m3: number,
            t: number,
        ): number {
            const a = -0.5 * m0 + 1.5 * m1 - 1.5 * m2 + 0.5 * m3;
            const b = m0 - 2.5 * m1 + 2 * m2 - 0.5 * m3;
            const c = -0.5 * m0 + 0.35 * m2; // mutated
            return m1 + t * (c + t * (b + t * a));
        }
        const t = 0.37;
        const gpuResult = catmullRom1D(vec4(p0), vec4(p1), vec4(p2), vec4(p3), t);
        const mutated = mutatedRefCatmullRom1D(p0[0], p1[0], p2[0], p3[0], t);
        expect(gpuResult.x).not.toBeCloseTo(mutated, 5);
    });
});
