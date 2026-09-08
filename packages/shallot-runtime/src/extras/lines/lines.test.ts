import { describe, expect, test } from "bun:test";
import * as d from "typegpu/data";
import { arrow, box, Lines, segment } from "./segments";
import { lineQuad } from "./surface";

// The immediate API's box/arrow expansion is the arithmetic that decides how many segments draw. It's
// pure — the GPU draw itself is validated in `bun bench`; the color codec (packColor) in color.test.ts.
describe("shallot lines immediate API", () => {
    test("segment appends one, box twelve edges, arrow shaft + four fins", () => {
        const a = Lines.count;
        segment([0, 0, 0], [1, 0, 0], 0xffffff);
        expect(Lines.count - a).toBe(1);
        const b = Lines.count;
        box([0, 0, 0], [1, 1, 1], 0xffffff);
        expect(Lines.count - b).toBe(12);
        const c = Lines.count;
        arrow([0, 0, 0], [0, 1, 0], 0xffffff);
        expect(Lines.count - c).toBe(5);
    });

    test("a zero-length arrow emits the shaft but no head", () => {
        const a = Lines.count;
        arrow([1, 1, 1], [1, 1, 1], 0xffffff);
        expect(Lines.count - a).toBe(1);
    });
});

// `lineQuad` is the surface's whole projection → near-clip → constant-pixel-expansion → sub-pixel-fade
// kernel, authored as a pure TGSL fn so the same source the shader runs is callable here (testing.md's
// CPU-kernel tier — the draw itself is gated by the `accel` framebuffer probe in `bun bench`).
const RES = d.vec2f(200, 100);
// a segment lying along screen +x at NDC y = 0, both endpoints in front (w = 1)
const AHEAD_S = d.vec4f(-0.5, 0, 0.25, 1);
const AHEAD_E = d.vec4f(0.5, 0, 0.25, 1);

describe("lines screen-space quad expansion", () => {
    test("the quad's two edge corners sit a constant pixel width apart, independent of the segment", () => {
        // width 2 px + 1 px of AA pad each side = a half-extent of 2 px, so the corners span 4 px.
        // Pixels → NDC is × 2/resolution, so on a 100 px-tall viewport that is 0.08 of NDC y
        const lo = lineQuad(AHEAD_S, AHEAD_E, RES, 0, -1, 2);
        const hi = lineQuad(AHEAD_S, AHEAD_E, RES, 0, 1, 2);
        expect(hi.clip.y - lo.clip.y).toBeCloseTo(0.08, 6);
        // and the perpendicular offset is the whole difference — the corners share the start endpoint
        expect(lo.clip.x).toBeCloseTo(hi.clip.x, 6);
        expect(lo.clip.x).toBeCloseTo(-0.5, 6);
        // t > 0.5 picks the end endpoint instead
        expect(lineQuad(AHEAD_S, AHEAD_E, RES, 1, 1, 2).clip.x).toBeCloseTo(0.5, 6);
        expect(lineQuad(AHEAD_S, AHEAD_E, RES, 1, 1, 2).useEnd).toBe(1);
    });

    test("a sub-pixel width clamps the geometry to 1 px and fades the alpha to keep its energy", () => {
        const thin = lineQuad(AHEAD_S, AHEAD_E, RES, 0, 1, 0.25);
        expect(thin.edge.y).toBe(0.5); // half-width floors at 1 px / 2
        expect(thin.tint.w).toBe(0.25); // …and the lost coverage comes off the alpha
        expect(thin.edge.x).toBeCloseTo(1.5, 6); // the AA distance still spans halfW + 1 px of pad
        // at or above 1 px the geometry carries the width and the alpha is untouched
        const fat = lineQuad(AHEAD_S, AHEAD_E, RES, 0, 1, 4);
        expect(fat.edge.y).toBe(2);
        expect(fat.tint.w).toBe(1);
    });

    test("an endpoint behind the camera is pulled onto the near plane, not projected through it", () => {
        const behind = d.vec4f(-2, 0, 0.5, -1); // w < 0 — the raw divide would flip it across the origin
        const front = d.vec4f(1, 0, 0.5, 1);
        const q = lineQuad(behind, front, RES, 0, 1, 2);
        // clip.z is the chosen endpoint's z/w. Unclipped that is 0.5 / -1 = -0.5 (behind the viewer);
        // pulled onto the near plane its w is ~1e-5, so the ratio is large and positive
        expect(q.clip.z).toBeGreaterThan(1000);
        // the endpoint that was already in front is untouched: 0.5 / 1
        expect(lineQuad(behind, front, RES, 1, 1, 2).clip.z).toBeCloseTo(0.5, 6);
    });

    test("both endpoints behind the camera collapse the vertex offscreen with a zeroed tint", () => {
        const q = lineQuad(d.vec4f(1, 2, 3, -1), d.vec4f(4, 5, 6, -2), RES, 0, 1, 2);
        expect([q.clip.x, q.clip.y, q.clip.z, q.clip.w]).toEqual([0, 0, -1, 1]); // z < 0 clips
        expect([q.tint.x, q.tint.y, q.tint.z, q.tint.w]).toEqual([0, 0, 0, 0]);
        expect([q.edge.x, q.edge.y]).toEqual([0, 0]);
        expect(q.useEnd).toBe(0); // world falls back to the start endpoint
    });
});
