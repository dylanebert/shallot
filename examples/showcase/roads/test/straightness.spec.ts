import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, type Page, test } from "@playwright/test";
import { PNG } from "pngjs";
import { detectEdgeOffset, raggedness } from "../src/straightness";

// Stage 9's discriminator, deliverable 1: does the road boundary's raggedness scale with SPACING (mesh
// quantization) or with TILE_RES/DIST_RANGE (distance-field resolution)? Not part of the standard device
// gate (`bun run gate`/`test/roads.spec.ts`) — it's a measurement run, not a pass/fail criterion, since
// picking a tolerance is the fix stage's job once the mechanism is known (the spec's own "no fix chosen
// here"). Opt in with `ROADS_STRAIGHTNESS=1 bun run gate` (or `playwright test test/straightness.spec.ts`)
// so a normal gate run doesn't pay for it. The two-resolution comparison itself is manual: this file reads
// whatever SPACING/TILE_RES/DIST_RANGE the source currently has (`window.__roadsMeshParams`, `boot.ts`)
// and reports the raggedness for *that* build — the reading in the spec's Live log is two separate runs
// of this file, one per source edit, never asserted against each other in-process.

// arm-labeled so re-running under a different source edit (SPACING/TILE_RES/DIST_RANGE) doesn't
// overwrite the previous arm's capture — needed to diff two arms' frames against each other as a
// positive witness that a treatment actually reached the rendered image (`ROADS_ARM_LABEL`, default
// "run", never read by the assertion below — purely a filename for cross-arm comparison).
const ARM_LABEL = process.env.ROADS_ARM_LABEL ?? "run";
const CAPTURE_PATH = join(
    dirname(fileURLToPath(import.meta.url)),
    "..",
    "test-results",
    `straightness-capture-${ARM_LABEL}.png`,
);

interface GrazingAnchor {
    x: number;
    y: number;
    dirX: number;
    dirY: number;
}

const SOFTWARE = /swiftshader|llvmpipe|lavapipe|warp|basic render/i;

const adapterName = (page: Page): Promise<string> =>
    page.evaluate(async () => {
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) return "";
        const { vendor, architecture, device, description } = adapter.info;
        return [vendor, architecture, device, description].filter(Boolean).join(" ");
    });

// how far either side of each analytic anchor to search for the real edge.
const SEARCH_RADIUS_PX = 30;
const STEPS_PER_PX = 3;
// an anchor's own ±SEARCH_RADIUS_PX endpoint samples must differ by at least this much to count as a real
// road/terrain contrast, not two off-road (or two on-road) samples the search radius failed to straddle
// the boundary with — a low-contrast anchor is excluded rather than fed a near-zero lo/hi that would turn
// any pixel noise into a spurious "crossing".
const MIN_CONTRAST = 15;

test("boundary straightness — grazing-view discriminator (opt-in, not a gate)", async ({
    page,
}) => {
    test.skip(
        !process.env.ROADS_STRAIGHTNESS,
        "opt-in: set ROADS_STRAIGHTNESS=1 to run stage 9's boundary-raggedness reading",
    );

    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));

    await page.goto("/");

    const adapter = await adapterName(page);
    console.log(`straightness probe adapter: ${adapter || "none offered"}`);
    test.skip(
        adapter === "" || SOFTWARE.test(adapter),
        `no real-GPU adapter (${adapter || "none offered"})`,
    );

    await page.waitForFunction(() => typeof window.__roadsGate === "function", null, {
        timeout: 120_000,
    });
    await page.waitForFunction(
        () => (window as unknown as { __roadsOverlayIdle: () => boolean }).__roadsOverlayIdle(),
        null,
        { timeout: 10_000 },
    );

    const meshParams = await page.evaluate(
        () => (window as unknown as { __roadsMeshParams: unknown }).__roadsMeshParams,
    );

    // reposition the camera to the fixed grazing pose *before* anything reads screen-space position — the
    // on-screen anchors and the eventual screenshot must both reflect the moved camera, not the scene's
    // default orbit pose (`public/scenes/roads.scene`).
    const { anchors } = (await page.evaluate(() =>
        (
            window as unknown as {
                __roadsGrazingCapture: () => Promise<{ anchors: GrazingAnchor[] }>;
            }
        ).__roadsGrazingCapture(),
    )) as { anchors: GrazingAnchor[] };

    const screenshot = await page.screenshot();
    mkdirSync(dirname(CAPTURE_PATH), { recursive: true });
    writeFileSync(CAPTURE_PATH, screenshot);

    const png = PNG.sync.read(screenshot);
    const luminanceAt = (fracX: number, fracY: number): number => {
        const x = Math.min(png.width - 1, Math.max(0, Math.round(fracX)));
        const y = Math.min(png.height - 1, Math.max(0, Math.round(fracY)));
        const at = (y * png.width + x) * 4;
        return 0.3 * png.data[at] + 0.59 * png.data[at + 1] + 0.11 * png.data[at + 2];
    };

    // per-anchor local plateau: sample the two ends of *this anchor's own* search segment (the anchor sits
    // analytically on the boundary, so its ±radius endpoints should straddle road and terrain) rather than
    // a shared on/off-road pair from elsewhere on screen — the grazing camera's framing gives no guarantee
    // stage 4/7's own probe points land anywhere near this view.
    // diagnostic per anchor — kept alongside the offset itself so a null is legible (excluded for low
    // contrast vs. searched and found nothing) rather than a bare `null` an evidence table can't explain.
    const diagnostics = anchors.map((a, i) => {
        const ax = a.x * png.width;
        const ay = a.y * png.height;
        const s0 = luminanceAt(ax - a.dirX * SEARCH_RADIUS_PX, ay - a.dirY * SEARCH_RADIUS_PX);
        const s1 = luminanceAt(ax + a.dirX * SEARCH_RADIUS_PX, ay + a.dirY * SEARCH_RADIUS_PX);
        const lo = Math.min(s0, s1);
        const hi = Math.max(s0, s1);
        const contrast = hi - lo;
        if (contrast < MIN_CONTRAST) {
            return { i, offset: null, contrast, reason: "low-contrast" as const };
        }
        const offset = detectEdgeOffset(
            (x, y) => luminanceAt(x, y),
            ax,
            ay,
            a.dirX,
            a.dirY,
            lo,
            hi,
            SEARCH_RADIUS_PX,
            STEPS_PER_PX,
        );
        return {
            i,
            offset,
            contrast,
            reason: offset === null ? ("no-crossing" as const) : ("found" as const),
        };
    });
    const offsets = diagnostics.map((d) => d.offset);
    const r = raggedness(offsets);

    console.log(
        `STRAIGHTNESS_READING ${JSON.stringify({
            armLabel: ARM_LABEL,
            meshParams,
            anchorCount: anchors.length,
            foundCount: r.n,
            rmsPx: r.rms,
            maxPx: r.max,
            offsets,
            diagnostics,
        })}`,
    );

    expect(errors, errors.join("\n")).toEqual([]);
    // the only assertion this opt-in probe makes: the instrument itself worked (found most of its
    // anchors) — never a straightness pass/fail, which is the fix stage's own criterion once the
    // mechanism is known.
    expect(r.n, `found ${r.n}/${anchors.length} edge crossings`).toBeGreaterThanOrEqual(
        Math.floor(anchors.length * 0.6),
    );
});

// Stage 10: stage 9's mechanism finding (the spec's Approach) — the road boundary staircase a
// person reads as metre-scale lives in the flattened corridor's *height*, not its albedo edge, so a
// screen-space luminance probe (above) is structurally blind to it (0.74 px rms genuine spread against a
// complaint a person reads as metres). This probe re-points the same anchor machinery at the world-space
// heightfield directly (`window.__roadsHeightSilhouette`, `grazingCapture.ts`) — no screen projection, no
// luminance, no camera dependence for the reading itself. `window.__roadsGrazingCapture` is still called
// first, purely to move the camera and save an evidence screenshot a person can look at alongside the
// number; the PNG is decoded only for a non-blank sanity check, never fed into the height reading.
// a network whose deepest cut is below this reads as "no meaningful cut" (the flat-ground/zeroed-RELIEF
// control) — derived from the same MIN_CONTRAST_M floor `grazingCapture.ts` uses per-anchor (100x the
// height quantization step), so the same noise floor gates both the per-anchor and the whole-run read.
const MIN_MEANINGFUL_CUT_M = 0.12;

test("boundary straightness — height-axis world-space discriminator (opt-in, not a gate)", async ({
    page,
}) => {
    test.skip(
        !process.env.ROADS_STRAIGHTNESS,
        "opt-in: set ROADS_STRAIGHTNESS=1 to run the height-axis straightness reading",
    );

    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));

    await page.goto("/");

    const adapter = await adapterName(page);
    console.log(`height-silhouette probe adapter: ${adapter || "none offered"}`);
    test.skip(
        adapter === "" || SOFTWARE.test(adapter),
        `no real-GPU adapter (${adapter || "none offered"})`,
    );

    await page.waitForFunction(() => typeof window.__roadsGate === "function", null, {
        timeout: 120_000,
    });
    await page.waitForFunction(
        () => (window as unknown as { __roadsOverlayIdle: () => boolean }).__roadsOverlayIdle(),
        null,
        { timeout: 10_000 },
    );

    const meshParams = await page.evaluate(
        () => (window as unknown as { __roadsMeshParams: unknown }).__roadsMeshParams,
    );

    // evidence only — the height reading below needs no camera pose at all.
    await page.evaluate(() =>
        (
            window as unknown as {
                __roadsGrazingCapture: () => Promise<unknown>;
            }
        ).__roadsGrazingCapture(),
    );
    const evidenceScreenshot = await page.screenshot();
    const evidencePath = join(
        dirname(fileURLToPath(import.meta.url)),
        "..",
        "test-results",
        `height-silhouette-capture-${ARM_LABEL}.png`,
    );
    mkdirSync(dirname(evidencePath), { recursive: true });
    writeFileSync(evidencePath, evidenceScreenshot);
    const png = PNG.sync.read(evidenceScreenshot);
    const nonBlack = png.data.some((v, i) => i % 4 !== 3 && v > 0);
    expect(nonBlack, "evidence screenshot is entirely black — camera pose likely wrong").toBe(true);

    const result = (await page.evaluate(() =>
        (
            window as unknown as {
                __roadsHeightSilhouette: () => Promise<{
                    falloffM: number;
                    windowM: number;
                    cutDepthM: number;
                    anchorCount: number;
                    readings: unknown[];
                    rmsM: number;
                    maxM: number;
                    foundCount: number;
                }>;
            }
        ).__roadsHeightSilhouette(),
    )) as {
        falloffM: number;
        windowM: number;
        cutDepthM: number;
        anchorCount: number;
        readings: unknown[];
        rmsM: number;
        maxM: number;
        foundCount: number;
    };

    console.log(
        `HEIGHT_SILHOUETTE_READING ${JSON.stringify({
            armLabel: ARM_LABEL,
            meshParams,
            ...result,
        })}`,
    );

    expect(errors, errors.join("\n")).toEqual([]);
    // same validity bar as the screen-space probe: the instrument itself has to work (find most of its
    // anchors, when there's a real cut to find) before its reading means anything — never a straightness
    // pass/fail, which stays stage 11's job.
    if (result.cutDepthM > MIN_MEANINGFUL_CUT_M) {
        expect(
            result.foundCount,
            `found ${result.foundCount}/${result.anchorCount} height crossings`,
        ).toBeGreaterThanOrEqual(Math.floor(result.anchorCount * 0.6));
        // a magnitude validity bound, not a straightness tolerance (stage 11's own job): rmsM must sit
        // comfortably above sub-instrument-resolution noise and can never structurally exceed windowM,
        // the search radius `heightSilhouette` actually walks each anchor over as of stage 11b
        // (`grazingCapture.ts`, `boundaryAnchors.ts`'s `sideSlopeWindow`) — a reading outside either bound
        // means the instrument itself regressed, not that the road did. `falloffM` (the real, floor-
        // inclusive `computeFalloff` output) rides along for evidence only and is no longer the bound.
        // RMS_FLOOR_M: two orders of magnitude below the mesh's own vertex spacing (SPACING) — the same
        // "100x the physical noise floor" margin `grazingCapture.ts`'s own MIN_CONTRAST_M uses against
        // height quantization, applied here to the mesh's spatial resolution. A reading this much smaller
        // than the grid itself is sub-millimetre axis-blindness (stage 9's own screen-space defect, 0.74
        // px rms against a metre-scale complaint), not a real quantized boundary.
        const spacingM = (meshParams as { spacing: number }).spacing;
        const rmsFloorM = spacingM / 100;
        expect(
            result.rmsM,
            `rmsM ${result.rmsM} is sub-instrument-resolution against SPACING ${spacingM}`,
        ).toBeGreaterThan(rmsFloorM);
        expect(
            result.rmsM,
            `rmsM ${result.rmsM} exceeds windowM ${result.windowM}, the instrument's own search radius`,
        ).toBeLessThanOrEqual(result.windowM);
    } else {
        // the control arm (zeroed RELIEF, no cut): with no meaningful cut there is no height silhouette
        // to find, so a "found" reading here can only be MIN_CONTRAST_M's per-anchor gate tripping on
        // something other than a real transition — a regression in the instrument, not a road. The bound
        // is derived, not fit to today's 0: MIN_CONTRAST_M (`grazingCapture.ts`) is built with a 100x
        // margin over raw quantization noise specifically so that margin can't be crossed by noise alone;
        // treating that same 1-in-100 margin as a per-anchor false-positive tolerance and scaling it by
        // the anchor count (rounded up to a whole anchor) gives the largest count consistent with "still
        // just noise" rather than a real crossing. At 16 anchors this is 1 — today's control reads 0,
        // comfortably inside it, but the bound doesn't move if the anchor set does.
        const controlFoundBound = Math.ceil(result.anchorCount * (1 / 100));
        expect(
            result.foundCount,
            `found ${result.foundCount}/${result.anchorCount} height crossings on the zeroed-RELIEF, ` +
                `no-cut control — expected at most ${controlFoundBound} (noise floor), since there is no ` +
                "cut for a genuine height silhouette to cross",
        ).toBeLessThanOrEqual(controlFoundBound);
    }
});
