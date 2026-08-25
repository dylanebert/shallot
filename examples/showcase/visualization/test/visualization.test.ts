// S3 arm — examples/showcase/visualization/test/visualization.playwright.ts derived-DEMOS empty-guard
//
// Invariant: an empty derived population reads red. The S2 fix derived the demo list from the
// built index.html iframes (instead of a hand list) and added an empty-guard:
//   expect(demos.length, "...").toBeGreaterThan(0)
// Without the guard, an empty iframe list (missing index, broken parse) would vacuously pass
// by skipping every demo.
//
// The DEMOS derivation is extracted into demos.ts (deriveDemosFromIframeSrcs) so this arm can
// exercise it behaviorally without a browser. The arm asserts:
//   - an empty iframe list produces an empty demo array (the condition the guard reds on)
//   - the guard itself reds on that empty array (expect(0).toBeGreaterThan(0) throws)
//   - a populated iframe list produces the expected demo names (the derivation is correct)

import { expect, test } from "bun:test";
import { deriveDemosFromIframeSrcs } from "./demos";

test("deriveDemosFromIframeSrcs — an empty iframe list produces no demos (the empty-guard reds)", () => {
    // An index.html with no iframes → empty srcs → empty demos. This is the condition the
    // spec's empty-guard `expect(demos.length).toBeGreaterThan(0)` reds on.
    const demos = deriveDemosFromIframeSrcs([]);
    expect(demos).toEqual([]);
    expect(demos.length).toBe(0);
    // The guard itself reds: expect(0).toBeGreaterThan(0) throws.
    expect(() => expect(demos.length).toBeGreaterThan(0)).toThrow();
});

test("deriveDemosFromIframeSrcs — non-matching srcs produce no demos", () => {
    // Iframes that don't point at /demos/<name>.html (e.g. ads, analytics) are filtered out.
    // If every iframe is non-matching, the result is empty — the guard reds.
    const demos = deriveDemosFromIframeSrcs([
        "https://example.com/ad.html",
        "https://example.com/analytics.html",
    ]);
    expect(demos).toEqual([]);
    expect(() => expect(demos.length).toBeGreaterThan(0)).toThrow();
});

test("deriveDemosFromIframeSrcs — matching srcs produce the demo names", () => {
    // The derivation correctly extracts demo names from iframe srcs.
    const demos = deriveDemosFromIframeSrcs([
        "http://localhost:3118/demos/voxel.html",
        "http://localhost:3118/demos/particles.html",
        "http://localhost:3118/not-a-demo.html",
    ]);
    expect(demos).toEqual(["voxel", "particles"]);
    expect(demos.length).toBeGreaterThan(0);
});
