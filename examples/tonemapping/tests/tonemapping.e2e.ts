import { mkdir } from "node:fs/promises";
import { holdFrames, pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, type Page, test } from "playwright/test";
import { methods } from "../src/settings";
import { bodyHasFocus, gameKeyMovesCamera, pointerControl } from "./pointer";

test.use(pipelineWarnings);

test("the first frame has no late-pipeline warning", async ({ page }) => {
    await page.goto("/");
    await page.waitForFunction(() =>
        performance
            .getEntriesByType("measure")
            .some((entry) => entry.name.endsWith(":first-update")),
    );
});

test("panel controls retain orbit and the pattern stays framed while orbiting", async ({
    page,
}) => {
    // Each step queues a full draw that the first capture waits behind; at 1280 px a software
    // adapter spends about a second on each. The desktop panel and the framing hold at a quarter of it.
    await page.setViewportSize({ width: 640, height: 360 });
    await page.goto("http://127.0.0.1:4297/tests/stepped.html");
    await page.waitForFunction(() => "controlsKey" in window);
    await gameKeyMovesCamera(page);
    await pointerControl(page, page.locator("summary"));
    await gameKeyMovesCamera(page);
    await pointerControl(page, page.locator("summary"));
    await gameKeyMovesCamera(page);
    for (const [name, value] of methods) {
        await page.locator("#method").focus();
        await page.locator("#method").selectOption(String(value));
        await bodyHasFocus(page);
        await gameKeyMovesCamera(page);
        expect(await page.locator("#method").inputValue(), name).toBe(String(value));
    }
    await page.locator("#method").selectOption("0");
    for (const scene of ["pattern", "basic"]) {
        await page.locator("#scene").focus();
        await page.locator("#scene").selectOption(scene);
        await bodyHasFocus(page);
        await gameKeyMovesCamera(page);
    }
    await page.locator("#scene").selectOption("pattern");
    const before = await capture(page);
    for (let i = 0; i < 8; i++) await gameKeyMovesCamera(page);
    expect(await difference(page, before, await capture(page))).toBeLessThan(0.01);
    await page.locator("#scene").selectOption("basic");
});

type CaptureWindow = Window & { captureScene: () => Promise<number[]>; captures: number[][] };
async function capture(page: Page): Promise<number> {
    return page.evaluate(async () => {
        const win = window as unknown as CaptureWindow;
        win.captures ??= [];
        win.captures.push(await win.captureScene());
        return win.captures.length - 1;
    });
}
async function difference(page: Page, a: number, b: number) {
    return page.evaluate(
        ([a, b]) => {
            const captures = (window as unknown as CaptureWindow).captures;
            let sum = 0;
            for (let i = 0; i < captures[a].length; i++)
                if (i % 4 !== 3) sum += Math.abs(captures[a][i] - captures[b][i]);
            return sum / (captures[a].length * 0.75);
        },
        [a, b],
    );
}

test("each method is distinct on both scenes and the scene picker changes the frame", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    // Each capture waits on a full draw, which a software adapter spends most of the test on. The
    // difference is a per-pixel mean, so the claim holds at a quarter of the 1280 px frame.
    await page.setViewportSize({ width: 640, height: 360 });
    await page.goto("http://127.0.0.1:4297/tests/stepped.html");
    await page.waitForFunction(() => "captureScene" in window);
    const basic = await capture(page);
    for (const scene of ["basic", "pattern"]) {
        await page.locator("#scene").selectOption(scene);
        const frames: number[] = [];
        for (const [name, value] of methods) {
            await page.locator("#method").selectOption(String(value));
            const frame = await capture(page);
            for (const previous of frames)
                expect(
                    await difference(page, frame, previous),
                    `${scene}: ${name}`,
                ).toBeGreaterThan(0.001);
            frames.push(frame);
        }
        if (scene === "pattern")
            expect(await difference(page, basic, frames[0])).toBeGreaterThan(1);
    }
    await page.locator("#method").selectOption("0");
    await page.locator("#scene").selectOption("basic");
    expect(await difference(page, basic, await capture(page))).toBeLessThan(0.01);
    expect(errors).toEqual([]);
});

for (const viewport of [
    { width: 1280, height: 720 },
    { width: 390, height: 844 },
]) {
    test(`captures at ${viewport.width}px`, async ({ page }) => {
        await page.setViewportSize(viewport);
        await page.goto("http://127.0.0.1:4297/tests/stepped.html");
        await page.waitForFunction(() => "captureScene" in window);
        await page.evaluate(() => document.fonts.ready);
        // Reset every control moved by the frame checks before capturing.
        await page.locator("#method").selectOption("0");
        await page.locator("#scene").selectOption("basic");
        const panel = page.getByRole("group", { name: "Tonemapping controls" });
        const bounds = (await panel.boundingBox())!;
        expect(bounds.x).toBeGreaterThanOrEqual(0);
        expect(bounds.x + bounds.width).toBeLessThanOrEqual(viewport.width);
        expect(bounds.y + bounds.height).toBeLessThanOrEqual(viewport.height);
        for (const scene of ["basic", "pattern"])
            for (const method of ["0", "5"]) {
                await page.locator("#scene").selectOption(scene);
                await page.locator("#method").selectOption(method);
                await capture(page);
                if (process.env.SHALLOT_CAPTURE_DIR) {
                    await mkdir(process.env.SHALLOT_CAPTURE_DIR, { recursive: true });
                    await page.screenshot({
                        path: `${process.env.SHALLOT_CAPTURE_DIR}/${scene}-${method === "0" ? "TonyMcMapface" : "AgX"}-${viewport.width}.png`,
                    });
                }
            }
    });
}

test("phone touch opens the panel and both pickers remain usable", async ({ browser }) => {
    const context = await browser.newContext({
        hasTouch: true,
        viewport: { width: 390, height: 844 },
    });
    try {
        const page = await context.newPage();
        await holdFrames(page, "[aria-label='Tonemapping controls']");
        await page.goto("/");
        const panel = page.getByRole("group", { name: "Tonemapping controls" });
        await panel.waitFor();
        await page.locator("summary").tap();
        await expect(panel).not.toHaveAttribute("open", "");
        await page.locator("summary").tap();
        await expect(panel).toHaveAttribute("open", "");
        await page.locator("#method").selectOption("5");
        await expect(page.locator("#method")).toHaveValue("5");
        await page.locator("#scene").selectOption("pattern");
        await expect(page.locator("#scene")).toHaveValue("pattern");
    } finally {
        await context.close();
    }
});
