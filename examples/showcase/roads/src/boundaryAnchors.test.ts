import { describe, expect, test } from "bun:test";
import { heightMidpointAnchor, sideSlopeWindow, worldEdgeAnchors } from "./boundaryAnchors";
import { detectEdgeOffset } from "./straightness";
import { computeFalloff, flattenHeight, SIDE_SLOPE_LIMIT } from "./terrain/flatten";

// Stage 10 repair, blocker 1: `heightSilhouette`'s pre-fix reading anchored the height-axis crossing
// detector at the road edge itself (`WorldEdgeAnchor.ex`/`ez`, `coreDist = 0`). `detectEdgeOffset`
// (`straightness.ts`) reports where the sampled signal crosses the *midpoint* between its `lo`/`hi`
// brackets, and `terrain/flatten.ts`'s `flattenHeight` cosine ease (`0.5 - 0.5·cos(π·t)`) reaches that
// midpoint at `t = 0.5`, i.e. `coreDist = falloff / 2` — not at the road edge. So a perfectly straight
// edge, anchored at the edge, reads a constant `falloff / 2` bias, mistaken for raggedness.
// `heightMidpointAnchor` moves the anchor to that ease midpoint instead — this file proves both halves on
// a synthetic straight edge built from the real `flattenHeight` formula, pure math, no device.
const FALLOFF = 14.026; // historical (stage 11b), same order as computeFalloff's own range; not today's
// network's falloff (6.7876) — kept as a fixed anchor for the synthetic edge, do NOT re-anchor
const NATURAL = 100;
const TARGET = 10;
const STEPS_PER_M = 20;

describe("heightMidpointAnchor — the height-axis instrument's own ground-truth zero point", () => {
    // a real production anchor (`worldEdgeAnchors`'s road-frame derivation), fed a synthetic sampler that
    // is exactly `flattenHeight` evaluated at the signed distance from that anchor along its own outward
    // normal — the same geometry `heightSilhouette` samples, minus the device mesh readback.
    const a = worldEdgeAnchors()[0];
    const sampleAt = (x: number, z: number): number => {
        const coreDist = (x - a.ex) * a.nx + (z - a.ez) * a.nz;
        return flattenHeight(NATURAL, TARGET, coreDist, FALLOFF);
    };

    test("anchoring at the road edge (coreDist = 0) reads a falloff/2 bias on a perfectly straight edge — the pre-fix defect", () => {
        const offset = detectEdgeOffset(
            sampleAt,
            a.ex,
            a.ez,
            a.nx,
            a.nz,
            TARGET,
            NATURAL,
            FALLOFF,
            STEPS_PER_M,
        );
        expect(offset).not.toBeNull();
        expect(offset as number).toBeCloseTo(FALLOFF / 2, 1);
    });

    test("anchoring at heightMidpointAnchor reads ~0 on the same straight edge — the fix", () => {
        const { mx, mz } = heightMidpointAnchor(a, FALLOFF);
        const offset = detectEdgeOffset(
            sampleAt,
            mx,
            mz,
            a.nx,
            a.nz,
            TARGET,
            NATURAL,
            FALLOFF,
            STEPS_PER_M,
        );
        expect(offset).not.toBeNull();
        expect(offset as number).toBeCloseTo(0, 1);
    });

    test("mutation: a genuinely ragged edge still reads ragged through heightMidpointAnchor — the fix corrects the anchor, not the raggedness signal", () => {
        const raggedSampleAt = (x: number, z: number): number => {
            const coreDist = (x - a.ex) * a.nx + (z - a.ez) * a.nz;
            // the real crossing sits FALLOFF/4 past the ease midpoint — a genuine deviation, not the
            // constant bias the two tests above isolate.
            return flattenHeight(NATURAL, TARGET, coreDist - FALLOFF / 4, FALLOFF);
        };
        const { mx, mz } = heightMidpointAnchor(a, FALLOFF);
        const offset = detectEdgeOffset(
            raggedSampleAt,
            mx,
            mz,
            a.nx,
            a.nz,
            TARGET,
            NATURAL,
            FALLOFF,
            STEPS_PER_M,
        );
        expect(offset).not.toBeNull();
        expect(offset as number).toBeCloseTo(FALLOFF / 4, 1);
    });
});

describe("worldEdgeAnchors — the analytic road-edge anchors the height-axis instrument walks", () => {
    test("returns 16 anchors, each with a unit outward search normal", () => {
        const anchors = worldEdgeAnchors();
        expect(anchors.length).toBe(16);
        for (const a of anchors) {
            expect(Math.hypot(a.nx, a.nz)).toBeCloseTo(1, 5);
        }
    });
});

// Stage 11b: stage 10's `heightSilhouette` scanned and anchored on `computeFalloff(cutDepth)` directly —
// the AASHTO side-slope term maxed with a mesh-sampling floor (`SPACING`) — so a wider floor widened the
// search window into raw terrain *and* moved the threshold anchor, with no change on the ground either was
// supposed to measure. `sideSlopeWindow` is the AASHTO term alone, upstream of any floor.
describe("sideSlopeWindow — the decoupled measurement window", () => {
    test("agrees exactly with computeFalloff on today's real network — the floor is inert here", () => {
        // 2.976 is a historical measurement point (stage 11b's pre-route-selection network, cutDepth
        // 2.976 m). Stage 22's route selection moved cutDepth to 1.4404 m (falloff 6.7876 m), so this is
        // no longer today's network's real cut depth. The value is kept as a historical anchor for the
        // test's synthetic edge — do NOT update it to track today's network (re-anchoring an instrument's
        // literals is the fitted-constant inversion stage 24a paid for). computeFalloff's floor is bare
        // SPACING (4 m, stage 13's revert), well below the AASHTO term (14.026 m) at this cut depth, so
        // the floor never binds and the two functions read the same number.
        const cutDepth = 2.976;
        expect(computeFalloff(cutDepth)).toBeCloseTo(14.026, 2);
        expect(sideSlopeWindow(cutDepth)).toBeCloseTo(14.026, 2); // spec's recorded falloffM/windowM
        expect(sideSlopeWindow(cutDepth)).toBeCloseTo(computeFalloff(cutDepth), 6);
    });

    test("is exactly the AASHTO formula, with no floor term at all", () => {
        const cutDepth = 5;
        expect(sideSlopeWindow(cutDepth)).toBeCloseTo(
            ((Math.PI / 2) * cutDepth) / SIDE_SLOPE_LIMIT,
            9,
        );
    });

    test("zero cut depth gives a zero window — no floor to fall back on", () => {
        expect(sideSlopeWindow(0)).toBe(0);
    });
});

// The flat-across-an-inert-treatment arm, run CPU-side against a synthetic straight edge. Predates 11a
// (written on 11b's branch, when 11a's own floor derivation was still unshipped) and outlives it (11a's
// concrete floor was removed at stage 13) — kept deliberately generic: it models the *class* of defect
// (any floor derivation binding against `computeFalloff`'s `max(floor, AASHTO(cutDepth))` shape), which is
// the durable statement, not any one floor's own number. `coupled` below reconstructs that shape, generic
// in `floor`. The synthetic edge's true transition is built from the real `flattenHeight` at a *fixed*
// embedded width (`TRUE_WINDOW`, matching production's `sideSlopeWindow(2.976)`) — the "boundary" here is
// a fixed array of sampled heights, so a floor value that only feeds `couple`'s own `Math.max` (never the
// sampler) provably cannot move it. Any reading swing as `floor` varies is the instrument, not the road.
describe("flat-across-an-inert-treatment: a floor change moves the coupled window, not the boundary", () => {
    const TrueCutDepth = 2.976; // historical (stage 11b) — not today's network's cutDepth (1.4404); kept
    // as a fixed anchor for the synthetic edge, do NOT re-anchor onto today's network
    const TrueWindow = sideSlopeWindow(TrueCutDepth); // 14.026 — historical (stage 11b), the synthetic
    // edge's real, fixed width; not today's network's falloff (6.7876)
    const Natural = 100;
    const Target = 10;
    const StepsPerM = 20;
    const a = worldEdgeAnchors()[0];
    const sampleAt = (x: number, z: number): number => {
        const coreDist = (x - a.ex) * a.nx + (z - a.ez) * a.nz;
        return flattenHeight(Natural, Target, coreDist, TrueWindow);
    };

    /** stage 10's instrument shape, generalized: `computeFalloff`'s own `max(floor, AASHTO)`, with
     *  `floor` a free parameter standing in for whatever mesh-sampling derivation produced it — the
     *  treatment this arm varies. Never reads the sampler or `TRUE_CUT_DEPTH`'s embedding, only
     *  `sideSlopeWindow`'s own output, exactly mirroring `computeFalloff`'s two-term max. */
    function coupled(floor: number): number {
        return Math.max(floor, sideSlopeWindow(TrueCutDepth));
    }

    function readAt(window: number): number | null {
        const { mx, mz } = heightMidpointAnchor(a, window);
        return detectEdgeOffset(sampleAt, mx, mz, a.nx, a.nz, Target, Natural, window, StepsPerM);
    }

    test("red-first: the coupled (stage-10-shaped) window reads a floor-sized bias on a perfectly straight edge", () => {
        const inflatedFloor = 20; // > TRUE_WINDOW (14.026) — the treatment: a wider floor, real geometry untouched
        const window = coupled(inflatedFloor);
        expect(window).toBeCloseTo(inflatedFloor, 6); // the floor won the max — this is what "coupled" means
        const offset = readAt(window);
        expect(offset).not.toBeNull();
        // biased by -(inflatedFloor - TRUE_WINDOW) / 2: the ease-midpoint anchor moves *outward* with the
        // window (past the true crossing), so the detector finds it back toward the anchor's own origin —
        // even though the sampled edge never moved (the same mechanism stage 10's own repair fixed, one
        // level up: the wrong quantity feeding heightMidpointAnchor, not a wrong constant inside it).
        expect(offset as number).toBeCloseTo(-(inflatedFloor - TrueWindow) / 2, 1);
        expect(Math.abs(offset as number)).toBeGreaterThan(1); // not noise — metre-scale, same order as the real defect
    });

    test("the decoupled window reads flat across the same floor treatment — 0, 20, and 100", () => {
        const readings = [0, 20, 100].map((floor) => {
            const window = sideSlopeWindow(TrueCutDepth); // floor never enters this call at all
            expect(window).toBeCloseTo(TrueWindow, 9); // literally the same number regardless of `floor`
            return { floor, offset: readAt(window) };
        });
        for (const { floor, offset } of readings) {
            expect(offset, `floor ${floor}`).not.toBeNull();
            expect(Math.abs(offset as number), `floor ${floor}`).toBeLessThan(0.01);
        }
    });

    test("the coupled window, by contrast, moves across the same three floor values — not flat", () => {
        const offsets = [0, 20, 100].map((floor) => readAt(coupled(floor)));
        const values = offsets.map((o) => o as number);
        // 0 and 100 floor: AASHTO term (14.026) wins at floor=0, the floor wins at floor=100 — the spread
        // across the swept treatment is the coupling, in the same units the real defect was measured in.
        expect(Math.max(...values) - Math.min(...values)).toBeGreaterThan(1);
    });
});
