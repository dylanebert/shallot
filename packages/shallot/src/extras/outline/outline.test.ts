import { describe, expect, test } from "bun:test";
import { groupByMesh, jfaSteps, maskCode } from "./passes";

// The JFA step ladder decides how many fullscreen passes the outline runs (the screen × log(width) cost),
// and the mesh grouping decides the scoped instanced draws. Both are pure CPU logic the GPU passes build
// on; the band itself is validated visually in `bun bench --scenario outline`.
describe("outline jfaSteps", () => {
    test("a power-of-two width is the start step, halving to 1", () => {
        expect(jfaSteps(4)).toEqual([4, 2, 1]);
        expect(jfaSteps(64)).toEqual([64, 32, 16, 8, 4, 2, 1]);
    });

    test("a non-power-of-two width rounds the start up to the next power of two", () => {
        // ceil(log2(5)) = 3 → start 8 → 4 passes (a pixel within 5px is within the 8px first jump)
        expect(jfaSteps(5)).toEqual([8, 4, 2, 1]);
        expect(jfaSteps(3)).toEqual([4, 2, 1]);
    });

    test("width 1 is a single step", () => {
        expect(jfaSteps(1)).toEqual([1]);
    });

    test("width clamps to [1, 64] — zero/negative floors at one pass, oversized caps at 64", () => {
        expect(jfaSteps(0)).toEqual([1]);
        expect(jfaSteps(-5)).toEqual([1]);
        expect(jfaSteps(100)[0]).toBe(64); // clamped to MAX_WIDTH before the ladder
    });

    test("every ladder ends at step 1 and strictly halves", () => {
        for (const w of [2, 7, 16, 33, 64]) {
            const steps = jfaSteps(w);
            expect(steps[steps.length - 1]).toBe(1);
            expect(steps[0]).toBeGreaterThanOrEqual(Math.min(64, w));
            for (let i = 1; i < steps.length; i++) expect(steps[i]).toBe(steps[i - 1] / 2);
        }
    });
});

describe("outline groupByMesh", () => {
    test("groups eids by mesh id, preserving insertion order within each group", () => {
        const meshOf = (eid: number) => ({ 10: 1, 11: 2, 12: 1, 13: 2 })[eid] ?? 0;
        const groups = groupByMesh([10, 11, 12, 13], meshOf);
        expect(groups.get(1)).toEqual([10, 12]);
        expect(groups.get(2)).toEqual([11, 13]);
        expect(groups.size).toBe(2);
    });

    test("a single mesh collapses to one group; empty input is an empty map", () => {
        expect(groupByMesh([5, 6, 7], () => 3).get(3)).toEqual([5, 6, 7]);
        expect(groupByMesh([], () => 0).size).toBe(0);
    });
});

describe("outline maskCode", () => {
    // the occlusion gate discards a highlighted fragment that's behind the visible scene. The engine is
    // reverse-Z (near→1/far→0, depthCompare greater), so "behind" = the fragment's depth is LESS than the
    // nearest stored scene depth. A `>` here (forward-Z) never fires, so an occluded outline draws over the
    // wall — the regression this guards (a reverse-Z site silently inverted).
    test("the occlude variant discards with the reverse-Z comparison (clip.z < scene)", () => {
        const occ = maskCode(true);
        expect(occ).toContain("textureLoad(sceneDepth");
        expect(occ).toMatch(/in\.clip\.z\s*<\s*scene/); // reverse-Z: behind = lesser depth
        expect(occ).not.toMatch(/in\.clip\.z\s*>\s*scene/); // forward-Z comparison = the bug
    });

    test("the plain variant has no depth gate (always-on-top)", () => {
        const plain = maskCode(false);
        expect(plain).not.toContain("sceneDepth");
        expect(plain).not.toContain("discard");
    });
});
