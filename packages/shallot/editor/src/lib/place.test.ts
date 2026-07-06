import { describe, expect, it } from "bun:test";
import { place } from "./place";

const vp = { width: 1000, height: 800 };
const panel = { width: 200, height: 200 };

describe("place", () => {
    it("opens below an anchor, left-aligned, when it fits", () => {
        // swatch 40x22 high in the inspector; room below for the whole panel
        const anchor = { left: 100, top: 100, right: 140, bottom: 122 };
        // top = bottom + gap(4) = 126; left = anchor.left = 100
        expect(place(panel, anchor, vp)).toEqual({ left: 100, top: 126 });
    });

    it("flips above when there is no room below", () => {
        // swatch near the bottom: 800 - 722 = 78px below, less than the 200px panel
        const anchor = { left: 100, top: 700, right: 140, bottom: 722 };
        // flip: top = anchor.top - gap(4) - height(200) = 496 (more room above than below)
        expect(place(panel, anchor, vp)).toEqual({ left: 100, top: 496 });
    });

    it("clamps horizontally so a right-edge anchor stays on screen", () => {
        const anchor = { left: 950, top: 100, right: 990, bottom: 122 };
        // left would be 950; clamped to viewport.width - width - margin = 1000 - 200 - 8 = 792
        expect(place(panel, anchor, vp).left).toBe(792);
    });

    it("flips both axes for a point anchor in the bottom-right corner (context menu)", () => {
        const cursor = { left: 990, top: 780, right: 990, bottom: 780 };
        const menu = { width: 140, height: 60 };
        // below clips (800 - 784 = 16 < 60) → flip up: 780 - 4 - 60 = 716
        // right clips (990 + 140 > 1000) → clamp: 1000 - 140 - 8 = 852
        expect(place(menu, cursor, vp)).toEqual({ left: 852, top: 716 });
    });

    it("pins to the margin when the panel is taller than the viewport", () => {
        const tall = { width: 200, height: 900 };
        const anchor = { left: 100, top: 400, right: 140, bottom: 422 };
        expect(place(tall, anchor, vp).top).toBe(8);
    });

    it("aligns the panel's far edge to the anchor when align is end", () => {
        const anchor = { left: 800, top: 100, right: 900, bottom: 122 };
        // end: panel right edge meets anchor right edge → left = 900 - 200 = 700
        expect(place(panel, anchor, vp, { align: "end" }).left).toBe(700);
    });
});
