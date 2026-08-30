import { describe, expect, test } from "bun:test";
import { body, flat } from "../../../../../packages/shallot/tests/wgsl";
import { GROUND_LEVEL, HFREQ, makePermutation, noiseWgsl, RELIEF } from "./noise";

describe("makePermutation", () => {
    // the seed → identical terrain determinism (validated end-to-end on the GPU in the roads gate) rests
    // on the permutation table being deterministic in its seed. These pin that CPU-side foundation —
    // The default-suite verdict is device-independent even where engine setup binds the preloaded adapter.

    test("is deterministic in the seed", () => {
        expect(makePermutation(1337)).toEqual(makePermutation(1337));
    });

    test("differs across seeds", () => {
        expect(makePermutation(1)).not.toEqual(makePermutation(2));
    });

    test("first half is a permutation of 0..255, second half doubles it", () => {
        const perm = makePermutation(42);
        expect(perm.length).toBe(512);
        const seen = new Set<number>();
        for (let i = 0; i < 256; i++) {
            expect(perm[i]).toBeGreaterThanOrEqual(0);
            expect(perm[i]).toBeLessThan(256);
            expect(perm[i + 256]).toBe(perm[i]); // doubled so lattice hashes never wrap
            seen.add(perm[i]);
        }
        expect(seen.size).toBe(256); // every value 0..255 appears exactly once
    });
});

describe("emitted WGSL", () => {
    test("preserves the f32 gradient evaluation and FBM recurrence", () => {
        const wgsl = noiseWgsl();
        const grad = flat(body(wgsl, "fn grad2"));
        expect(grad).toBe(
            "fn grad2(hash: u32, x: f32, y: f32) -> f32 { let h = (hash & 7u); " +
                "var gx = 0f; var gy = 0f; " +
                "if ((h == 0u)) { gx = 1f; gy = 0f; } " +
                "if ((h == 1u)) { gx = -1f; gy = 0f; } " +
                "if ((h == 2u)) { gx = 0f; gy = 1f; } " +
                "if ((h == 3u)) { gx = 0f; gy = -1f; } " +
                "if ((h == 4u)) { gx = 0.7071067690849304f; gy = 0.7071067690849304f; } " +
                "if ((h == 5u)) { gx = -0.7071067690849304f; gy = 0.7071067690849304f; } " +
                "if ((h == 6u)) { gx = 0.7071067690849304f; gy = -0.7071067690849304f; } " +
                "if ((h == 7u)) { gx = -0.7071067690849304f; gy = -0.7071067690849304f; } " +
                "return ((gx * x) + (gy * y)); }",
        );

        const perlin = flat(body(wgsl, "fn perlin2"));
        expect(perlin).toBe(
            "fn perlin2(p: vec2f) -> f32 { " +
                "let x = p.x; let y = p.y; let fx = floor(x); let fy = floor(y); " +
                "let X = u32((i32(fx) & 255i)); let Y = u32((i32(fy) & 255i)); " +
                "let xf = (x - fx); let yf = (y - fy); " +
                "let u = (((xf * xf) * xf) * ((xf * ((xf * 6f) - 15f)) + 10f)); " +
                "let v = (((yf * yf) * yf) * ((yf * ((yf * 6f) - 15f)) + 10f)); " +
                "let perm = (&perm_1); let A = ((*perm)[X] + Y); " +
                "let B = ((*perm)[(X + 1u)] + Y); let v00 = grad2((*perm)[A], xf, yf); " +
                "let v10 = grad2((*perm)[B], (xf - 1f), yf); " +
                "let v01 = grad2((*perm)[(A + 1u)], xf, (yf - 1f)); " +
                "let v11 = grad2((*perm)[(B + 1u)], (xf - 1f), (yf - 1f)); " +
                "return mix(mix(v00, v10, u), mix(v01, v11, u), v); }",
        );

        const fbm = flat(body(wgsl, "fn fbm2"));
        expect(fbm).toBe(
            "fn fbm2(p: vec2f) -> f32 { var amp = 1f; var freq = 1f; var sum = 0f; " +
                "var norm = 0f; var i = 0u; for (; (i < 5u); i = (i + 1u)) { " +
                "sum = (sum + (perlin2((p * freq)) * amp)); norm = (norm + amp); " +
                "amp = (amp * 0.5f); freq = (freq * 2f); } return (sum / norm); }",
        );
    });

    test("heightAt bakes this project's own frequency/relief/ground constants", () => {
        const wgsl = noiseWgsl();
        const height = flat(body(wgsl, "fn heightAt"));
        // HFREQ round-trips through an f32 literal (0.003 has no exact binary fraction), so the emitted
        // WGSL carries the f32-rounded value, not the JS source literal — read, not guessed.
        expect(height).toBe(
            `fn heightAt(x: f32, z: f32) -> f32 { return (${GROUND_LEVEL}f + ` +
                `(fbm2((vec2f(x, z) * ${Math.fround(HFREQ)}f)) * ${RELIEF}f)); }`,
        );
    });
});
