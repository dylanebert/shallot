// Wires `checkContinuity`/`finestCascadeTexel` (`clipmap.ts`) into a gate — before this file they
// had no reader beyond the package barrel, so a future edit to `OCEAN_CLIP_CONFIG` or the shipped
// cascade bands could break the near-spacing Nyquist bound this spec's Locked decision leans on with
// nothing redding. Every check here is structural (inequality/ordering over the shipped config's own
// numbers), so nothing is authored: `checkContinuity`'s findings compare `OCEAN_CLIP_LEVELS` against
// `CASCADE_CONFIGS` (both already-locked, shipped values), and the "represented band half-extent"
// bound is `OCEAN_CLIP_CONFIG.totalHalfExtent` itself — the mesh's own declared extent, not a second
// authored number — so this is a regression guard on the shipped config's own self-consistency, not a
// live decision sweep (`mesh-inversion-sweep.oracle.ts`'s header states the same distinction).
//
// `buildClipLevels`'s two throw branches (non-power-of-two extent ratio, non-integral core/spacing
// ratio) are refusal-capable guards with no arm before this file — armed below.
import { describe, expect, test } from "bun:test";
import {
    buildClipLevels,
    checkContinuity,
    finestCascadeTexel,
    OCEAN_CLIP_CONFIG,
    OCEAN_CLIP_LEVELS,
} from "../src/clipmap";
import { CASCADE_CONFIGS } from "../src/spectrum";

describe("checkContinuity on the shipped clipmap config", () => {
    test("every finding holds for OCEAN_CLIP_LEVELS against CASCADE_CONFIGS", () => {
        const findings = checkContinuity(
            OCEAN_CLIP_LEVELS,
            CASCADE_CONFIGS,
            OCEAN_CLIP_CONFIG.totalHalfExtent,
        );
        for (const f of findings) expect(f.ok, f.message).toBe(true);
    });

    test("finestCascadeTexel reads the smallest cascade texel size", () => {
        const expected = Math.min(...CASCADE_CONFIGS.map((c) => c.L / c.N));
        expect(finestCascadeTexel(CASCADE_CONFIGS)).toBeCloseTo(expected, 12);
    });

    test("RED-WITNESS — a near spacing above the Nyquist bound breaks the near-field finding", () => {
        // re-runs the guarded arm's own comparison with only the subject (the near-field level's
        // spacing) mutated, doubled past the Nyquist bound `checkContinuity`'s first finding checks.
        const mutatedLevels = OCEAN_CLIP_LEVELS.map((lvl, i) =>
            i === 0 ? { ...lvl, spacing: lvl.spacing * 20 } : lvl,
        );
        const findings = checkContinuity(
            mutatedLevels,
            CASCADE_CONFIGS,
            OCEAN_CLIP_CONFIG.totalHalfExtent,
        );
        expect(findings[0].ok, findings[0].message).toBe(false);
    });
});

describe("buildClipLevels's refusal branches", () => {
    test("throws when totalHalfExtent/coreHalfExtent is not a power of two", () => {
        expect(() =>
            buildClipLevels({ coreHalfExtent: 10, nearSpacing: 1, totalHalfExtent: 30 }),
        ).toThrow(/not a power of 2/);
    });

    test("throws when coreHalfExtent/nearSpacing is not integral", () => {
        expect(() =>
            buildClipLevels({ coreHalfExtent: 10, nearSpacing: 3, totalHalfExtent: 40 }),
        ).toThrow(/not integral/);
    });
});
