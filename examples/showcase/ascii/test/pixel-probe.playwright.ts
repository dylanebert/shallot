import { pixelProbePass, probePixels } from "@dylanebert/shallot/harness/pixels";
import { expect, test } from "@playwright/test";
import { PNG } from "pngjs";
import { adapterName, SOFTWARE } from "./gpu-adapter";

// Criterion 5 (`shallot-tui` spec): "the web sink reaches the compositor" — a masked pixel probe through
// the existing `harness/pixels` classifier, so a blank canvas cannot pass on green draw counts. Runs by
// path — `cd examples/showcase/ascii && bunx playwright test test/pixel-probe.playwright.ts` — display-
// gated (WSL bridge, `playwright.global-setup.ts`), never part of `bun run test`.
//
// The probe targets the cell grid's own background swatch — every visible cell paints its bg as the
// scene's own tonemapped, gamma-encoded average color (`select.ts`'s `avgKernel` + `packCell`'s sRGB
// pack), so a warm hue band over the box's flat, unlit color (`specs/shallot-tui.md`'s locked shading —
// near-constant regardless of viewing angle) is a property of *this* render succeeding, not an assumption
// about glyph shapes criterion 8 alone is responsible for. Measured directly (`shallot verify
// examples/recipes/render-to-a-terminal --screenshot`, the same scene's frozen sibling, nvidia/lovelace):
// the box's swatch reads rgb(172, 151, 126) at a sampled interior point, comfortably inside the band
// below with real margin on every channel — the band is wide enough to tolerate AA/edge pixels and a
// differently-sized canvas, never so wide it would pass a near-black blank frame.

test("ascii showcase — the cell grid reaches the compositor", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));
    page.on("console", (m) => {
        if (m.type() === "error") errors.push(`[console.error] ${m.text()}`);
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

    const probe = {
        name: "cell background reaches the compositor",
        minPixels: 500,
        minSpan: 40,
        r: [90, 255] as [number, number],
        g: [40, 220] as [number, number],
        b: [10, 180] as [number, number],
    };
    const result = probePixels(png.data, png.width, png.height, probe);
    expect(
        pixelProbePass(result, probe),
        `matched ${result.pixels} px, span ${result.width}x${result.height} (want >= ${probe.minPixels} px, >= ${probe.minSpan} px span)`,
    ).toBe(true);

    expect(errors, `page errors: ${errors.join("\n")}`).toEqual([]);
});
