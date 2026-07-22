// The floor-vanish break rule, proven red-first (spec tumble-inline stage 6b / F1). The auto-dump watcher and
// the trusted-input Playwright driver both decide "the floor vanished this frame" from one pure predicate,
// {@link detectBreach} — a non-finite pose/transform, or a draw count that fell below the distinct-mesh count.
// The GPU readback and the scan that feed it need a real device; the DECISION doesn't, so it's factored out
// and pinned here without one. A break rule that never went red on the very condition it guards pins nothing.
//
// Outside bunfig's `bun test` scope (rooted at `packages/shallot`) — run via `bun run test:gym`, or:
//   bun test ./examples/gym/src/tumble-watch.test.ts

import { expect, test } from "bun:test";
import {
    detectBreach,
    type NonFiniteHit,
    type PatchLum,
    pixelBreach,
    type StaticState,
    staticBreach,
} from "./tumble-watch";

test("a clean frame is no break — every mesh draws, nothing non-finite", () => {
    // joints-bridge dedupes to 4 draw pairs; all 4 drawing, no non-finite → no dump.
    expect(detectBreach(4, 4, [])).toBeNull();
});

test("a draw-drop is a break — a pair stopped rasterizing while the derivation still counts it", () => {
    // the reported symptom: the floor (its own pair) + the boxes (their pair) vanish GPU-side → 2 of 4 draw.
    const breach = detectBreach(2, 4, []);
    expect(breach).not.toBeNull();
    expect(breach?.kind).toBe("draw-drop");
    expect(breach?.drawing).toBe(2);
    expect(breach?.meshes).toBe(4);
});

test("a non-finite pose is a break, and ranks ahead of the draw count", () => {
    // the suspected root: a poisoned transform. It's a break even when the draw count still looks whole, and
    // it's reported as `non-finite`, not `draw-drop`, so the trace points at the cause not the symptom.
    const hits: NonFiniteHit[] = [
        { source: "part-pos", index: 12, values: [Number.NaN, Number.NaN, Number.NaN] },
    ];
    const breach = detectBreach(4, 4, hits);
    expect(breach?.kind).toBe("non-finite");
    expect(breach?.nonFinite).toHaveLength(1);
});

test("a failed readback is not a break — `drawing < 0` means the GPU sample couldn't run", () => {
    // `drawingPairs()` returns -1 before a device/pack exists; that must never latch a false dump.
    expect(detectBreach(-1, 4, [])).toBeNull();
});

// ── F1′ pixel breach (spec 6b/F1′): the layer the user SEES ──────────────────────────────────────────────
// the driver samples several pixel patches over each static surface (the ground, the two end posts) and decides
// "this surface went black" from one pure predicate, pixelBreach — proven red-first here without a screenshot.

const G = (lums: number[]): PatchLum[] => [...lums.map((lum) => ({ surface: "ground", lum }))];

test("a lit scene is no pixel breach — every patch near its reference", () => {
    const ref = G([120, 130, 110]);
    const sample = G([118, 132, 108]);
    expect(pixelBreach(ref, sample)).toBeNull();
});

test("a whole static surface going black is a BREACH — the reported floor-vanish, at the pixel layer", () => {
    // every ground patch drops near-black in the same sample: not one body occluding one patch, the surface gone.
    const ref = G([120, 130, 110]);
    const breach = pixelBreach(ref, G([6, 4, 5]));
    expect(breach).not.toBeNull();
    expect(breach?.surface).toBe("ground");
    expect(breach?.kind).toBe("dark");
    expect(breach?.patches).toBe(3);
});

test("one patch going dark is NOT a breach — a body passed in front of it, the surface still shows", () => {
    // the legitimate-occlusion guard: a plank crossing one ground patch must never trip the whole-surface rule.
    const ref = G([120, 130, 110]);
    expect(pixelBreach(ref, G([120, 3, 110]))).toBeNull();
});

test("a surface with fewer than two reliably-lit reference patches is skipped", () => {
    // a single post patch (or dim refs below MIN_REF_LUM) can't distinguish occlusion from black-out — skip it.
    const ref: PatchLum[] = [{ surface: "post", lum: 90 }];
    expect(pixelBreach(ref, [{ surface: "post", lum: 2 }])).toBeNull();
});

test("a bright wash of a whole surface is a BREACH — a NaN colour blow-out reads as vanished too", () => {
    const ref = G([80, 90, 85]);
    const breach = pixelBreach(ref, G([250, 255, 248]));
    expect(breach?.kind).toBe("wash");
});

// ── F3′ registered-static invariant (spec 6b/F3′): the world-truth source drawArgs can't corrupt ──────────
// The floor-vanish was a broadphase-tree corruption that dropped the static bodies from `overlapAABB` /
// `world.draw` — so every detector derived from those (the mesh count above) fell WITH them and read whole.
// staticBreach checks the live broadphase against the static set captured at build, keyed to live body handles
// so a legitimate destroy (`--inject statics`) is distinguished from the corruption. Proven red-first here.

const S = (id: number, valid: boolean, present: boolean): StaticState => ({ id, valid, present });

test("every registered static present is no breach — the broadphase is intact", () => {
    expect(staticBreach([S(0, true, true), S(1, true, true), S(2, true, true)])).toBeNull();
});

test("a valid static gone from the broadphase IS a breach — the corruption the drawArgs check is blind to", () => {
    // the confirmed floor-vanish: the static tree dropped ground + posts while they were still live bodies.
    const missing = staticBreach([S(0, true, false), S(1, true, false), S(2, true, false)]);
    expect(missing).toEqual([0, 1, 2]);
});

test("a legitimately-destroyed static is NOT a breach — the live-handle distinction", () => {
    // `--inject statics` / a re-typed body destroys it: absent from the query AND `!valid` → dropped from the
    // set, not flagged. Without keying to live handles this would false-positive on every legitimate destroy.
    expect(staticBreach([S(0, false, false), S(1, true, true), S(2, true, true)])).toBeNull();
});

test("one valid static missing among present ones is still a breach — the partial-drop case", () => {
    expect(staticBreach([S(0, true, true), S(1, true, false), S(2, true, true)])).toEqual([1]);
});
