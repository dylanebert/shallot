import { mkdir } from "node:fs/promises";
import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, type Page, test } from "playwright/test";
import { settings } from "../src/settings";
import { dragRange, gameKeyMovesCamera, pointerControl } from "./pointer";

test.use(pipelineWarnings);

test("the first frame has no late-pipeline warning", async ({ page }) => {
    await page.goto("/");
    await page.waitForFunction(() =>
        performance
            .getEntriesByType("measure")
            .some((entry) => entry.name.endsWith(":first-update")),
    );
});

test("each panel control leaves the orbit key working", async ({ page }) => {
    await page.goto("http://127.0.0.1:4296/tests/stepped.html?controls");
    await page.waitForFunction(() => "controlsKey" in window);
    await gameKeyMovesCamera(page);
    await pointerControl(page, page.locator("summary"));
    await gameKeyMovesCamera(page);
    await pointerControl(page, page.locator("summary"));
    await gameKeyMovesCamera(page);
    for (const [index, control] of settings.entries()) {
        await dragRange(page, page.locator(`#control-${index}`));
        await gameKeyMovesCamera(page);
        await page.locator(`#control-${index}`).fill(String(control.value));
    }
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

test("each control changes the frame and identity returns the ungraded frame", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    // Each capture waits on a full draw, which a software adapter spends most of the test on. The
    // difference is a per-pixel mean, so the claim holds at a quarter of the 1280 px frame.
    await page.setViewportSize({ width: 640, height: 360 });
    await page.goto("http://127.0.0.1:4296/tests/stepped.html");
    await page.waitForFunction(() => "captureScene" in window);
    await page.evaluate(() =>
        (window as unknown as { setGrading: (enabled: boolean) => void }).setGrading(false),
    );
    const identity = await capture(page);
    await page.evaluate(() =>
        (window as unknown as { setGrading: (enabled: boolean) => void }).setGrading(true),
    );
    expect(await difference(page, await capture(page), identity)).toBeLessThan(0.01);
    for (const [index, control] of settings.entries()) {
        const slider = page.locator(`#control-${index}`);
        await slider.fill(String(control.max));
        expect(
            await difference(page, await capture(page), identity),
            `${control.group} ${control.label}`,
        ).toBeGreaterThan(0.01);
        await slider.fill(String(control.value));
        expect(await difference(page, await capture(page), identity)).toBeLessThan(0.01);
    }
    expect(errors).toEqual([]);
});

for (const viewport of [
    { width: 1280, height: 720 },
    { width: 390, height: 844 },
]) {
    test(`default and strong grade at ${viewport.width}px`, async ({ page }) => {
        await page.setViewportSize(viewport);
        await page.goto("/");
        const panel = page.getByRole("region", { name: "Color grading controls" });
        await panel.waitFor();
        await page.evaluate(() => document.fonts.ready);
        for (const [index, control] of settings.entries())
            await page.locator(`#control-${index}`).fill(String(control.value));
        await panel.evaluate((el) => {
            el.scrollTop = 0;
        });
        const bounds = (await panel.boundingBox())!;
        expect(bounds.x).toBeGreaterThanOrEqual(0);
        expect(bounds.y + bounds.height).toBeLessThanOrEqual(viewport.height);
        expect(bounds.x + bounds.width).toBeLessThanOrEqual(viewport.width);
        for (const label of await panel.locator(".row > label").all())
            expect(await label.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(true);
        if (process.env.SHALLOT_CAPTURE_DIR) {
            await mkdir(process.env.SHALLOT_CAPTURE_DIR, { recursive: true });
            await page.screenshot({
                path: `${process.env.SHALLOT_CAPTURE_DIR}/default-${viewport.width}.png`,
            });
        }
        await page.locator("#control-0").fill("1");
        await page.locator("#control-1").fill("0.05");
        await page.locator("#control-3").fill("0.6");
        const last = page.getByRole("slider").last();
        await last.scrollIntoViewIfNeeded();
        await last.fill("0.05");
        await expect(last).toHaveValue("0.05");
        await last.fill("0");
        expect(await panel.boundingBox()).toEqual(bounds);
        await panel.evaluate((el) => {
            el.scrollTop = 0;
        });
        if (process.env.SHALLOT_CAPTURE_DIR)
            await page.screenshot({
                path: `${process.env.SHALLOT_CAPTURE_DIR}/strong-${viewport.width}.png`,
            });
    });
}

test("phone touch opens the panel and adjusts a grade", async ({ browser }) => {
    const context = await browser.newContext({
        hasTouch: true,
        viewport: { width: 390, height: 844 },
    });
    try {
        const page = await context.newPage();
        await page.goto("/");
        const panel = page.getByRole("region", { name: "Color grading controls" });
        const summary = page.locator("summary");
        await summary.tap();
        await expect(panel).not.toHaveAttribute("open", "");
        await summary.tap();
        await expect(panel).toHaveAttribute("open", "");
        const slider = page.getByRole("slider").first();
        const box = (await slider.boundingBox())!;
        const session = await context.newCDPSession(page);
        await session.send("Input.dispatchTouchEvent", {
            type: "touchStart",
            touchPoints: [{ id: 1, x: box.x + box.width * 0.25, y: box.y + box.height / 2 }],
        });
        const down = await slider.inputValue();
        await session.send("Input.dispatchTouchEvent", {
            type: "touchMove",
            touchPoints: [{ id: 1, x: box.x + box.width * 0.75, y: box.y + box.height / 2 }],
        });
        await session.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
        expect(await slider.inputValue()).not.toBe(down);
        await expect(page.locator("output").first()).toHaveText(
            Number(await slider.inputValue()).toFixed(2),
        );
    } finally {
        await context.close();
    }
});
