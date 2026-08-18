import { describe, expect, test } from "bun:test";
import { heightMidpointAnchor, worldEdgeAnchors } from "./boundaryAnchors";
import { detectEdgeOffset } from "./straightness";
import { flattenHeight } from "./terrain/flatten";

// Stage 10 repair, blocker 1: `heightSilhouette`'s pre-fix reading anchored the height-axis crossing
// detector at the road edge itself (`WorldEdgeAnchor.ex`/`ez`, `coreDist = 0`). `detectEdgeOffset`
// (`straightness.ts`) reports where the sampled signal crosses the *midpoint* between its `lo`/`hi`
// brackets, and `terrain/flatten.ts`'s `flattenHeight` cosine ease (`0.5 - 0.5·cos(π·t)`) reaches that
// midpoint at `t = 0.5`, i.e. `coreDist = falloff / 2` — not at the road edge. So a perfectly straight
// edge, anchored at the edge, reads a constant `falloff / 2` bias, mistaken for raggedness.
// `heightMidpointAnchor` moves the anchor to that ease midpoint instead — this file proves both halves on
// a synthetic straight edge built from the real `flattenHeight` formula, pure math, no device.
const FALLOFF = 14.026; // an arbitrary representative falloff, same order as computeFalloff's own range
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
