import { describe, expect, test } from "bun:test";
import { makePermutation, solidFractionBand } from "./noise";

describe("makePermutation", () => {
    // the seed → identical grid determinism (validated end-to-end on the GPU in the voxel gate) rests on
    // the permutation table being deterministic in its seed. These pin that CPU-side foundation.

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

describe("solidFractionBand", () => {
    // the derived band the generated grid's solid fraction must fall in. A bound, not a tuned threshold:
    // the surface sits at GROUND ± RELIEF, so it brackets 0.5 (the zero-mean field centred at GROUND) and
    // excludes all-air / all-solid.
    test("brackets the mid-height surface and excludes the degenerate extremes", () => {
        const [lo, hi] = solidFractionBand();
        expect(lo).toBeGreaterThan(0);
        expect(hi).toBeLessThan(1);
        expect(lo).toBeLessThan(0.5);
        expect(hi).toBeGreaterThan(0.5);
    });
});
