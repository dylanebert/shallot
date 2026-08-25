// S3 arm — examples/showcase/visualization/test/visualization.spec.ts derived-DEMOS empty-guard
//
// Invariant: an empty derived population reads red. The S2 fix derived the demo list from the
// built index.html iframes (instead of a hand list) and added an empty-guard:
//   expect(demos.length, "...").toBeGreaterThan(0)
// Without the guard, an empty iframe list (missing index, broken parse) would vacuously pass
// by skipping every demo. The spec is a Playwright test (needs a browser + built site), so this
// arm reads the source and asserts the empty-guard is present, pinning the invariant structurally.

import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const src = readFileSync(resolve(import.meta.dir, "visualization.spec.ts"), "utf8");

test("visualization spec — derives the demo list from index.html iframes, not a hand list", () => {
    // The fix replaced a hand-maintained DEMOS array with a derivation from the built index.html
    // iframes, so a partially-added demo can't ship unverified.
    expect(src).toMatch(/iframe.*evaluateAll/);
    expect(src).toMatch(/demos\/([^/]+)\.html/);
});

test("visualization spec — an empty derived population reads red (the empty-guard)", () => {
    // The empty-guard: if the iframe derivation produces zero demos, the gate fails — never
    // vacuous-green by skipping every demo. This is the invariant the S2 fix added.
    expect(src).toMatch(/demos\.length.*toBeGreaterThan\(0\)/);
});
