// CPU spec for the GPU body coloring (step.ts COLORING_PASS_WGSL) + the joint hard-conflict repair
// (REPAIR_PASS_WGSL, Phase 6.2). Deterministic integer logic — hardware-invariant, so `bun test` is the
// home (the GPU reproduces it; the gym carries the real-device + observable-invariant gates). The
// greedy avoids ALL constraint neighbors but tolerates a folded same-color pair; a SOFT spring survives
// that, a HARD joint must not (degrading to same-color Jacobi destabilizes a hard constraint), so the
// repair recolors the lower-eid endpoint of any same-color joint pair. The invariant: no dynamic joint
// pair ends same-color when a free color exists within the cap.

import { describe, expect, test } from "bun:test";
import { countConflicts, repairHardColors } from "./coloring";

describe("AVBD coloring — joint hard-conflict repair (Phase 6.2)", () => {
    test("a forced same-color joint pair is repaired to different colors", () => {
        // the greedy folded both endpoints to color 0 (a tolerated soft outcome the greedy can produce past
        // the cap); the repair must split them because the edge is HARD. Body 0 (lower eid) moves.
        const mass = [1, 1];
        const greedy = [0, 0]; // forced conflict
        const out = repairHardColors([], [[0, 1]], greedy, mass, 8);
        expect(out[0]).not.toBe(out[1]);
        expect(countConflicts([[0, 1]], out, mass)).toBe(0);
        expect(out[1]).toBe(0); // the higher-id endpoint stays fixed; the lower one moves
    });

    test("an already-separated joint pair is left untouched", () => {
        const out = repairHardColors([], [[0, 1]], [0, 1], [1, 1], 8);
        expect(out).toEqual([0, 1]);
    });

    test("a soft (spring) same-color pair is NOT repaired — only joints trigger", () => {
        // the spring edge is `soft`: it's avoided but a folded same-color pair is tolerated (clean Jacobi).
        const out = repairHardColors([[0, 1]], [], [0, 0], [1, 1], 8);
        expect(out).toEqual([0, 0]);
    });

    test("a static-anchored joint never conflicts (the static endpoint is uncolored)", () => {
        // pendulum shape: static anchor 0 + dynamic bob 1. The static body imposes no scheduling constraint,
        // so the bob keeps its color and the repair is a no-op — matching the GPU (statics are 0xffffffff).
        const out = repairHardColors([], [[0, 1]], [0xffffffff, 0], [0, 1], 8);
        expect(out[1]).toBe(0);
    });

    test("the repair avoids recoloring onto another neighbor's color", () => {
        // body 1 has a hard joint to body 2 (same color 0) AND a soft edge to body 0 (color 1). It must move
        // off 0 (the hard conflict) but NOT onto 1 (body 0's color) — so it lands on 2.
        const mass = [1, 1, 1];
        const out = repairHardColors(
            [[0, 1]], // soft: 1 avoids 0 (color 1)
            [[1, 2]], // hard: 1 conflicts with 2 (both color 0); 1 < 2 so 1 moves
            [1, 0, 0],
            mass,
            8,
        );
        expect(out[1]).not.toBe(out[2]); // hard conflict resolved
        expect(out[1]).not.toBe(out[0]); // didn't move onto the soft neighbor's color
        expect(out[1]).toBe(2);
    });
});
