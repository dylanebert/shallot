import { test } from "@playwright/test";
import type { PNG } from "pngjs";
import { type Assertion, boot, diff, emit, region, shot } from "../../harness/lib";

// Positive claim: the cube's surface carries a procedural pattern (bands, not one flat colour) and the
// pattern animates on its own with the camera untouched. Two discriminators: a single-frame brightness
// profile across the cube's face must oscillate (a flat face, even a corner-on cube showing two shaded
// faces, gives at most one light/dark step; stripes give several), and frames sampled over ~1s must
// differ (a static scene is bit-static, delta 0.00).

// average background brightness from a screen corner — the void the cube sits in
function bgLevel(png: PNG): number {
    const c = region(png, 0, 0, 0.12, 0.12);
    return (c.r + c.g + c.b) / 3;
}

// brightness across a strip of cells along one axis over a wide central window, then trimmed to the
// contiguous cube run (cells brighter than the background). Trimming drops the cube's silhouette against
// the background — that outer edge would otherwise read as a band the surface never drew — while dark
// bands *inside* the cube stay, so a genuine stripe still registers.
//
// The scan axis is sampled at ~pixel resolution: cell count is derived from the scan window's pixel span,
// not a fixed n. A stripe is only visible to a human once each half-band spans ≥ ~2px post-antialiasing
// (below that, adjacent light/dark bands blur into one gray). So a cell of ≤ 2px resolves any stripe a
// human can see — with wider cells, one cell straddles a light and a dark band and averages them out (the
// n=40 false-negative: a 0.4-of-720p window is 7.2px/cell, wider than an 11px stripe's 5.5px half-band).
// The perpendicular band (0.46..0.54) still averages across the scan for noise reduction — that width is
// not the aliasing source and stays fixed.
function faceProfile(png: PNG, axis: "x" | "y", bg: number): number[] {
    const spanPx = 0.4 * (axis === "x" ? png.width : png.height); // the 0.3..0.7 scan window, in pixels
    const n = Math.ceil(spanPx / 2); // ≤ 2px per cell — the visibility floor for a resolvable half-band
    const vals: number[] = [];
    for (let i = 0; i < n; i++) {
        const t0 = 0.3 + (0.4 * i) / n;
        const t1 = 0.3 + (0.4 * (i + 1)) / n;
        const c = axis === "x" ? region(png, t0, 0.46, t1, 0.54) : region(png, 0.46, t0, 0.54, t1);
        vals.push((c.r + c.g + c.b) / 3);
    }
    const cut = bg + 25;
    let lo = 0;
    while (lo < vals.length && vals[lo] < cut) lo++;
    let hi = vals.length - 1;
    while (hi > lo && vals[hi] < cut) hi--;
    // pull in a couple of cells at each end: the antialiased silhouette ramps from background to the face
    // over ~1 cell, and a dark ramp at both ends would read as a dark-bright-dark band on a flat face.
    // Interior dark stripes are untouched.
    const edge = 2;
    lo = Math.min(lo + edge, hi);
    hi = Math.max(hi - edge, lo);
    return vals.slice(lo, hi + 1);
}

// peak-to-peak spread and the number of light/dark band crossings in a brightness profile. A margin
// gates out antialiasing wobble so only real bands count; a single face-to-face step is one crossing,
// alternating stripes are several.
function bands(vals: number[]): { pp: number; crossings: number } {
    if (vals.length < 2) return { pp: 0, crossings: 0 };
    const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
    const pp = Math.max(...vals) - Math.min(...vals);
    const margin = Math.max(8, pp * 0.2);
    let crossings = 0;
    let sign = 0;
    for (const v of vals) {
        const d = v - mean;
        if (Math.abs(d) < margin) continue;
        const s = d > 0 ? 1 : -1;
        if (sign !== 0 && s !== sign) crossings++;
        sign = s;
    }
    return { pp, crossings };
}

test("striped-material", async ({ page }) => {
    test.setTimeout(80_000);
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("console", (m) => {
        if (m.type() === "error") errors.push(m.text());
    });

    const { booted, rendered } = await boot(page);
    const a: Assertion[] = [];

    if (rendered) {
        // three frames at two different gaps: a pattern whose period happens to alias one interval (it
        // returns to its start after exactly that gap) can't alias both, so the max pairwise delta stays
        // a faithful measure of motion at any animation speed
        const s0 = await shot(page);
        await page.waitForTimeout(500);
        const s1 = await shot(page);
        await page.waitForTimeout(600);
        const s2 = await shot(page);

        const bg = bgLevel(s0);
        const hx = bands(faceProfile(s0, "x", bg));
        const vy = bands(faceProfile(s0, "y", bg));
        const pp = Math.max(hx.pp, vy.pp);
        const crossings = Math.max(hx.crossings, vy.crossings);
        // within the cube's own span a flat face is near-constant and even a corner-on cube shows at most
        // one face-to-face step (crossings ≤ 1). Requiring ≥ 2 crossings with a real peak-to-peak rejects
        // both while a striped surface (several bands across the face) clears it comfortably.
        a.push({
            name: "the surface shows a pattern, not a flat colour",
            ok: pp > 30 && crossings >= 2,
            detail: `peak-to-peak ${pp.toFixed(0)}, band crossings ${crossings}`,
        });

        // camera untouched, so a settled scene is bit-static (delta 0.00); a moving surface pattern
        // repaints pixels frame to frame. > 0.5 sits above the settled floor other gates measure at 0.00.
        const motion = Math.max(diff(s0, s1), diff(s1, s2), diff(s0, s2));
        a.push({
            name: "the pattern moves on its own",
            ok: motion > 0.5,
            detail: `max frame delta ${motion.toFixed(2)}`,
        });
    }

    emit({
        ok: booted && rendered && a.every((x) => x.ok),
        booted,
        rendered,
        assertions: a,
        errors: errors.length ? errors : undefined,
    });
});
