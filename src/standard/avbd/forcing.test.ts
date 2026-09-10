import { expect, test } from "bun:test";
import { hullCoreWgsl, hullSatWgsl, roundedPolyWgsl } from "./collide";

// Base-forcing, in a file of its own because the property is **resolution-order dependent** and each
// chunk memoizes: the shared `spliceNs` emits a shared definition into whichever chunk resolves FIRST,
// so a dependent chunk that forgot to force its base would swallow the base's definitions — and a
// consumer that splices both then gets a duplicate declaration, which Dawn rejects at pipeline creation
// with nothing pointing at the cause. Once `collide.test.ts` has resolved the chunks in dependency
// order the property is unfalsifiable in that process, so the adversarial order lives here: resolve the
// DEEPEST chunk first and assert the bases still own their definitions.

const defs = (wgsl: string): string[] =>
    [...wgsl.matchAll(/^(?:fn|struct)\s+([A-Za-z0-9_]+)/gm)].map((m) => m[1]);

test("the deepest chunk resolved first still leaves its bases their definitions", () => {
    // roundedPoly → hullCore → helpers is the longest chain in the module
    const deepest = defs(roundedPolyWgsl());
    expect(deepest).toContain("collideRoundedPolytope");
    for (const base of [
        "SatResult", // helpers
        "addContact",
        "closestPointBox",
        "Convex", // hullCore
        "projectPoly",
        "hVertL",
    ])
        expect(deepest, `${base} leaked into roundedPolyWgsl`).not.toContain(base);

    // and they are where they belong, resolved after the chunk that depends on them
    expect(defs(hullCoreWgsl())).toContain("projectPoly");
    // the sibling hull SAT then adds only its own
    expect(defs(hullSatWgsl())).toContain("collideHull");
    expect(defs(hullSatWgsl())).not.toContain("Convex");
});
