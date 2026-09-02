import { pixelProbePass, probePixels } from "@dylanebert/shallot/harness/pixels";
import { expect, test } from "@playwright/test";
import { PNG } from "pngjs";
import { adapterName, SOFTWARE } from "./gpu-adapter";

// Criterion 5 (`shallot-tui` spec): "the web sink reaches the compositor" — a masked pixel probe through
// the existing `harness/pixels` classifier, so a blank canvas cannot pass on green draw counts. Runs by
// path — `cd examples/showcase/ascii && bunx playwright test test/pixel-probe.playwright.ts` — display-
// gated (WSL bridge, `playwright.global-setup.ts`), never part of `bun run test`.
//
// Two probes, because the first alone doesn't discriminate the Cells system from a bare scene render.
//
// `boxSwatch` targets the cell grid's own background swatch — every visible cell paints its bg as the
// scene's own tonemapped, gamma-encoded average color (`select.ts`'s `avgKernel` + `packCell`'s sRGB
// pack), so a warm hue band over the box's own albedo (`specs/shallot-tui.md`'s locked shading — a
// per-face lit color that stays constant per face regardless of viewing angle) is a property of *this*
// render succeeding, not an assumption about glyph shapes criterion 8 alone is responsible for. Measured
// directly (`shallot verify examples/recipes/render-to-a-terminal --screenshot`, the same scene's frozen
// sibling, nvidia/lovelace): the box's swatch reads rgb(172, 151, 126) at a sampled interior point,
// comfortably inside the band below with real margin on every channel. But this probe alone is
// satisfiable by the *scene* alone — the box paints roughly the same warm hue whether or not Cells ever
// runs (`"Cells": true` deleted from `shallot.json`, the scene's own draw still fills that hue region) —
// so it discriminates blank-canvas from something-rendered, one rung below what criterion 5 names.
//
// `glyphInk` is the cells-only signal: `select.ts`'s `selectKernel` always draws its glyph ink in pure
// grayscale — `fg = select(vec3f(0,0,0), vec3f(1,1,1), ownLuma < 0.5)`, i.e. exactly black or exactly
// white, never the scene's own warm hue — and every visible cell (background or box) that selects any
// glyph but the blank space glyph paints some of that grayscale ink. The scene's own box albedo
// (rgb ~0.85 0.55 0.35, warm) and the background clear color are both chromatic (r ≠ g ≠ b well outside
// this probe's per-channel bands), so no un-celled pixel in this scene can land inside a band demanding
// all three channels simultaneously high (near-white ink) — a signal the raw scene cannot produce, only
// the Cells system's glyph draw can. Deleting `"Cells": true` removes every ink pixel from the frame,
// which is what `glyphInk`'s own two-sidedness rests on (this file's second test).
//
// The console-clean check (`shallot-tui` spec, s3r item 3, added 2026-09-01): a `⚠️ [implicit-conversion]`
// tgpu warning reached this page's live console with no gate reading for it — this probe already drives
// the page, so it now asserts every console warning too, not only `console.error`. One warning is a
// known, pre-existing exclusion rather than a fix: `Implicit conversions from [r: struct:vertexVsOut] to
// struct` — verified (this file's own diagnostic run, both directions) to reproduce **identically** with
// `"Cells": true` deleted from `shallot.json`, i.e. with `extras/cells`'s own draw pipeline never built at
// all, so it cannot be `extras/cells`'s emission; every `createRenderPipeline({ vertex, fragment })` call
// across `standard/sear`'s several pipelines shares that same literal `vertex` property key, which is what
// tgpu derives the generic `vertexVsOut` struct name from regardless of which pipeline it belongs to, so
// the name alone can't attribute it further than "some pipeline outside this stage's footprint" — sear's
// shading pipeline is exactly what `Boundaries` (`specs/shallot-tui.md`'s s3r brief) forbids retuning in
// this stage. `extras/cells`'s own draw pipeline was independently re-shaped to declare its fragment `in`
// as its vertex `out` verbatim (`draw.ts`'s `cellVsOut`) rather than a narrower varyings-only subset — the
// shape that triggers this exact warning class — so a future regression there is not shielded by this
// exclusion: only this one exact, narrow signature is excluded, and any other warning (including a
// differently-worded implicit-conversion from a cells shape change) still fails this check.
const KNOWN_PREEXISTING_WARNING = /\[implicit-conversion\][\s\S]*struct:vertexVsOut/;

test("ascii showcase — the cell grid reaches the compositor", async ({ page }) => {
    const errors: string[] = [];
    const warnings: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));
    page.on("console", (m) => {
        if (m.type() === "error") errors.push(`[console.error] ${m.text()}`);
        if (m.type() === "warning" && !KNOWN_PREEXISTING_WARNING.test(m.text())) {
            warnings.push(`[console.warning] ${m.text()}`);
        }
    });

    await page.goto("/");

    const adapter = await adapterName(page);
    console.log(`ascii pixel-probe adapter: ${adapter || "none offered"}`);
    test.skip(
        adapter === "" || SOFTWARE.test(adapter),
        `no real-GPU adapter (${adapter || "none offered"})`,
    );

    const canvas = page.locator("canvas").first();
    await expect(canvas).toBeVisible();
    await page.waitForFunction(() => {
        const c = document.querySelector("canvas");
        return c instanceof HTMLCanvasElement && c.width > 0 && c.height > 0;
    });
    // the atlas font load (`CellsPlugin.warm`) is async and gates the first cell draw — a fixed settle
    // rather than a harness hook, since this project ships no `window.__harness` (no smoke concept to
    // assert, only that the sink painted)
    await page.waitForTimeout(1000);

    const shot = await canvas.screenshot({ timeout: 5_000 });
    const png = PNG.sync.read(shot);

    const boxSwatch = {
        name: "cell background reaches the compositor",
        minPixels: 500,
        minSpan: 40,
        r: [90, 255] as [number, number],
        g: [40, 220] as [number, number],
        b: [10, 180] as [number, number],
    };
    const boxResult = probePixels(png.data, png.width, png.height, boxSwatch);
    expect(
        pixelProbePass(boxResult, boxSwatch),
        `matched ${boxResult.pixels} px, span ${boxResult.width}x${boxResult.height} (want >= ${boxSwatch.minPixels} px, >= ${boxSwatch.minSpan} px span)`,
    ).toBe(true);

    // the cells-only signal (module doc above): grayscale glyph ink, which the scene's own chromatic
    // colors (warm box, dark background) cannot produce — a probe `boxSwatch` alone can't discriminate,
    // since deleting `"Cells": true` from `shallot.json` still leaves the box's own warm swatch on screen.
    const glyphInk = {
        name: "cell glyph ink reaches the compositor",
        minPixels: 80,
        minSpan: 20,
        r: [210, 255] as [number, number],
        g: [210, 255] as [number, number],
        b: [210, 255] as [number, number],
    };
    const inkResult = probePixels(png.data, png.width, png.height, glyphInk);
    expect(
        pixelProbePass(inkResult, glyphInk),
        `matched ${inkResult.pixels} px, span ${inkResult.width}x${inkResult.height} (want >= ${glyphInk.minPixels} px, >= ${glyphInk.minSpan} px span) — a pass here is the signal only the Cells system's glyph draw can produce`,
    ).toBe(true);

    expect(errors, `page errors: ${errors.join("\n")}`).toEqual([]);
    expect(warnings, `page console warnings: ${warnings.join("\n")}`).toEqual([]);
});
