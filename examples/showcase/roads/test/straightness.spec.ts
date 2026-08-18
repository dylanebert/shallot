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

const CAPTURE_PATH = join(
    dirname(fileURLToPath(import.meta.url)),
    "..",
    "test-results",
    "straightness-capture.png",
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
    const offsets = anchors.map((a) => {
        const ax = a.x * png.width;
        const ay = a.y * png.height;
        const s0 = luminanceAt(ax - a.dirX * SEARCH_RADIUS_PX, ay - a.dirY * SEARCH_RADIUS_PX);
        const s1 = luminanceAt(ax + a.dirX * SEARCH_RADIUS_PX, ay + a.dirY * SEARCH_RADIUS_PX);
        const lo = Math.min(s0, s1);
        const hi = Math.max(s0, s1);
        if (hi - lo < MIN_CONTRAST) return null; // no real road/terrain contrast straddled — exclude
        return detectEdgeOffset(
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
    });
    const r = raggedness(offsets);

    console.log(
        `STRAIGHTNESS_READING ${JSON.stringify({
            meshParams,
            anchorCount: anchors.length,
            foundCount: r.n,
            rmsPx: r.rms,
            maxPx: r.max,
            offsets,
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
