import { pixelProbePass, probePixels } from "@dylanebert/shallot/harness/pixels";
import { expect, test } from "@playwright/test";
import { PNG } from "pngjs";
import { adapterName, SOFTWARE } from "./gpu-adapter";

const KNOWN_PREEXISTING_WARNING = /\[implicit-conversion\][\s\S]*struct:vertexVsOut/;

// Criterion 5: the Cells system publishes a non-empty grid on the canvas and the compositor presents
// the cube's glyph-colored pixels. The two signals are independent: metadata alone cannot prove a
// framebuffer reached presentation, while the warm swatch alone is also available from an uncelled cube.
test("ascii showcase — the cell grid reaches the compositor", async ({ page }) => {
    const errors: string[] = [];
    const warnings: string[] = [];
    page.on("pageerror", (error) => errors.push(String(error)));
    page.on("console", (message) => {
        if (message.type() === "error") errors.push(`[console.error] ${message.text()}`);
        if (message.type() === "warning" && !KNOWN_PREEXISTING_WARNING.test(message.text())) {
            warnings.push(`[console.warning] ${message.text()}`);
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
    await expect
        .poll(() =>
            canvas.evaluate((element) => {
                const cells = element as HTMLCanvasElement & {
                    cellCols?: number;
                    cellRows?: number;
                };
                return (cells.cellCols ?? 0) * (cells.cellRows ?? 0);
            }),
        )
        .toBeGreaterThan(0);

    const swatch = {
        name: "glyph-colored cube reaches the compositor",
        minPixels: 500,
        minSpan: 40,
        r: [70, 255] as [number, number],
        g: [30, 230] as [number, number],
        b: [5, 190] as [number, number],
    };
    let result = { pixels: 0, width: 0, height: 0 };
    await expect
        .poll(
            async () => {
                const png = PNG.sync.read(await canvas.screenshot({ timeout: 5_000 }));
                result = probePixels(png.data, png.width, png.height, swatch);
                return pixelProbePass(result, swatch);
            },
            { message: "the composited cell grid should contain the cube swatch", timeout: 15_000 },
        )
        .toBe(true);

    expect(
        pixelProbePass(result, swatch),
        `matched ${result.pixels} px, span ${result.width}x${result.height}`,
    ).toBe(true);
    expect(errors, `page errors: ${errors.join("\n")}`).toEqual([]);
    expect(warnings, `page console warnings: ${warnings.join("\n")}`).toEqual([]);
});
