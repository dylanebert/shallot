import { isDegradedBootMessage } from "@dylanebert/shallot/harness";
import { type Browser, expect, test } from "@playwright/test";
import { PNG } from "pngjs";
import { adapterName, SOFTWARE } from "./gpu-adapter";

interface Capture {
    readonly cssWidth: number;
    readonly cssHeight: number;
    readonly framebufferWidth: number;
    readonly framebufferHeight: number;
    readonly cols: number;
    readonly rows: number;
}

async function capture(
    browser: Browser,
    baseURL: string,
    width: number,
    height: number,
    deviceScaleFactor: number,
): Promise<Capture | null> {
    const context = await browser.newContext({
        baseURL,
        viewport: { width, height },
        deviceScaleFactor,
    });
    const page = await context.newPage();
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(String(error)));
    page.on("console", (message) => {
        if (message.type() === "error" || isDegradedBootMessage(message.text())) {
            errors.push(`[console.${message.type()}] ${message.text()}`);
        }
    });
    await page.goto("/");

    const adapter = await adapterName(page);
    if (adapter === "" || SOFTWARE.test(adapter)) {
        await context.close();
        return null;
    }

    const canvas = page.locator("canvas").first();
    await expect(canvas).toBeVisible();
    await expect
        .poll(() =>
            canvas.evaluate((element) => {
                const cells = element as HTMLCanvasElement & {
                    cellCols?: number;
                    cellRows?: number;
                };
                return cells.cellCols && cells.cellRows
                    ? `${cells.cellCols}x${cells.cellRows}`
                    : "";
            }),
        )
        .not.toBe("");

    const geometry = await canvas.evaluate((element) => {
        const canvas = element as HTMLCanvasElement & { cellCols: number; cellRows: number };
        const rect = canvas.getBoundingClientRect();
        return {
            cssWidth: rect.width,
            cssHeight: rect.height,
            cols: canvas.cellCols,
            rows: canvas.cellRows,
        };
    });
    const png = PNG.sync.read(await canvas.screenshot());
    expect(errors, `page errors: ${errors.join("\n")}`).toEqual([]);
    await context.close();
    return {
        ...geometry,
        framebufferWidth: png.width,
        framebufferHeight: png.height,
    };
}

// Criterion 11: one CSS surface at DPR 1 and 2 keeps its grid and apparent cell size, while a
// different CSS surface changes the population. The screenshot dimensions independently prove that
// the framebuffer still scales in device pixels rather than being pinned to CSS pixels.
test("ascii showcase — CSS-sized cells are invariant to device scale", async ({
    browser,
    baseURL,
}) => {
    expect(baseURL).toBeTruthy();
    const one = await capture(browser, baseURL!, 640, 360, 1);
    test.skip(one === null, "no real-GPU adapter");
    const two = await capture(browser, baseURL!, 640, 360, 2);
    const larger = await capture(browser, baseURL!, 800, 440, 1);
    expect(two).not.toBeNull();
    expect(larger).not.toBeNull();
    if (!one || !two || !larger) return;

    expect(Math.abs(one.cols - two.cols)).toBeLessThanOrEqual(1);
    expect(Math.abs(one.rows - two.rows)).toBeLessThanOrEqual(1);
    for (const result of [one, two, larger]) {
        expect(result.cssWidth / result.cols).toBeGreaterThanOrEqual(10);
        expect(result.cssWidth / result.cols).toBeLessThanOrEqual(12);
        expect(result.cssHeight / result.rows).toBeGreaterThanOrEqual(10);
        expect(result.cssHeight / result.rows).toBeLessThanOrEqual(12);
    }

    expect(two.framebufferWidth / one.framebufferWidth).toBeCloseTo(2, 1);
    expect(two.framebufferHeight / one.framebufferHeight).toBeCloseTo(2, 1);
    expect(larger.cols).not.toBe(one.cols);
    expect(larger.rows).not.toBe(one.rows);
});
