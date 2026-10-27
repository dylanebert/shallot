import { mkdir } from "node:fs/promises";
import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, type Page, test } from "playwright/test";
import { bodyHasFocus, dragRange, gameKeyMovesCamera, pointerControl } from "./pointer";

type CaptureWindow = Window & { captureScene: () => Promise<number[]>; captures: number[][] };
async function capture(page: Page): Promise<number> {
    return page.evaluate(async () => {
        const win = window as unknown as CaptureWindow;
        win.captures ??= [];
        win.captures.push(await win.captureScene());
        return win.captures.length - 1;
    });
}
async function difference(page: Page, a: number, b: number): Promise<number> {
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

test.use(pipelineWarnings);

test("the first frame has no late-pipeline warning", async ({ page }) => {
    await page.goto("/");
    await page.waitForFunction(() =>
        performance
            .getEntriesByType("measure")
            .some((entry) => entry.name.endsWith(":first-update")),
    );
});

test("every bloom control changes the captured frame and off restores the unbloomed frame", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("http://127.0.0.1:4276/tests/stepped.html");
    await page.waitForFunction(() => "captureScene" in window);
    const toggle = page.getByRole("checkbox");
    const panel = page.getByRole("region", { name: "Bloom controls" });
    const bounds = await panel.boundingBox();
    const bloom = await capture(page);
    await toggle.uncheck();
    const off = await capture(page);
    expect(await difference(page, bloom, off)).toBeGreaterThan(0.1);
    expect(await panel.boundingBox()).toEqual(bounds);
    for (const slider of await page.getByRole("slider").all()) await expect(slider).toBeDisabled();
    await toggle.check();
    expect(await difference(page, await capture(page), bloom)).toBeLessThan(0.1);
    const values = [0.8, 0, 0.1, 0.1, 50, 1, 8];
    for (const [index, slider] of (await page.getByRole("slider").all()).entries()) {
        // Softness filters only a nonzero threshold.
        if (index === 5)
            await page.getByRole("slider", { name: "Threshold", exact: true }).fill("50");
        const before = await capture(page);
        const original = await slider.inputValue();
        await slider.fill(String(values[index]));
        expect(
            await difference(page, await capture(page), before),
            (await slider.getAttribute("id"))!,
        ).toBeGreaterThan(0.01);
        await slider.fill(original);
        if (index === 5)
            await page.getByRole("slider", { name: "Threshold", exact: true }).fill("0");
    }
    await page.getByRole("combobox").selectOption("1");
    expect(await difference(page, await capture(page), bloom)).toBeGreaterThan(0.01);
    await toggle.uncheck();
    expect(await difference(page, await capture(page), off)).toBeLessThan(0.1);
    expect(errors).toEqual([]);
});

for (const viewport of [
    { width: 1280, height: 720 },
    { width: 390, height: 844 },
]) {
    test(`default view and usable controls at ${viewport.width}px`, async ({ page }) => {
        await page.setViewportSize(viewport);
        await page.goto("/");
        const panel = page.getByRole("region", { name: "Bloom controls" });
        await panel.waitFor();
        await page.evaluate(() => document.fonts.ready);
        const bounds = (await panel.boundingBox())!;
        expect(bounds.x).toBeGreaterThanOrEqual(0);
        expect(bounds.y + bounds.height).toBeLessThanOrEqual(viewport.height);
        expect(bounds.x + bounds.width).toBeLessThanOrEqual(viewport.width);
        expect(await panel.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(true);
        for (const label of await panel.locator(".row > label").all())
            expect(await label.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(true);
        if (process.env.SHALLOT_CAPTURE_DIR) {
            await mkdir(process.env.SHALLOT_CAPTURE_DIR, { recursive: true });
            await page.screenshot({
                path: `${process.env.SHALLOT_CAPTURE_DIR}/default-${viewport.width}.png`,
            });
        }
        const toggle = page.getByRole("checkbox");
        await toggle.uncheck();
        expect(await panel.boundingBox()).toEqual(bounds);
        await toggle.check();
        const slider = page.getByRole("slider").last();
        await slider.scrollIntoViewIfNeeded();
        await slider.fill("4");
        await expect(slider).toHaveValue("4");
        await expect(page.locator("output").last()).toHaveText("4.00");
    });
}

test("each pointer control leaves the bound orbit key working", async ({ page }) => {
    // Each step queues a full draw, and every pointer action then waits behind the queue. Focus and the
    // orbit key hold at a quarter of the 1280 px frame, still above the 480 px phone layout.
    await page.setViewportSize({ width: 640, height: 360 });
    await page.goto("http://127.0.0.1:4276/tests/stepped.html");
    await page.waitForFunction(() => "controlsKey" in window);
    await dragRange(page, page.getByRole("slider").first());
    await gameKeyMovesCamera(page);
    const toggle = page.getByRole("checkbox").first();
    await pointerControl(page, toggle);
    await gameKeyMovesCamera(page);
    const summary = page.locator("summary");
    await pointerControl(page, summary);
    await gameKeyMovesCamera(page);
});

test("native touch toggles, summary and range activate without taking game keys", async ({
    browser,
}) => {
    const context = await browser.newContext({
        hasTouch: true,
        viewport: { width: 390, height: 844 },
    });
    try {
        const page = await context.newPage();
        await page.goto("http://127.0.0.1:4276/tests/stepped.html");
        await page.waitForFunction(() => "controlsKey" in window);
        const session = await context.newCDPSession(page);
        const touch = async (type: "touchStart" | "touchMove" | "touchEnd", x = 0, y = 0) => {
            await session.send("Input.dispatchTouchEvent", {
                type,
                touchPoints: type === "touchEnd" ? [] : [{ id: 1, x, y }],
            });
            await bodyHasFocus(page);
        };
        const toggle = page.getByRole("checkbox").first();
        const slider = page.getByRole("slider").first();
        const panel = page.getByRole("region", { name: "Bloom controls" });
        await expect(toggle).toBeChecked();
        const toggleBox = (await toggle.boundingBox())!;
        await touch(
            "touchStart",
            toggleBox.x + toggleBox.width / 2,
            toggleBox.y + toggleBox.height / 2,
        );
        await touch("touchEnd");
        await expect(toggle).not.toBeChecked();
        await expect(slider).toBeDisabled();
        await gameKeyMovesCamera(page);
        await touch(
            "touchStart",
            toggleBox.x + toggleBox.width / 2,
            toggleBox.y + toggleBox.height / 2,
        );
        await touch("touchEnd");
        await expect(toggle).toBeChecked();
        await expect(slider).toBeEnabled();

        const summary = page.locator("summary");
        const summaryBox = (await summary.boundingBox())!;
        await touch(
            "touchStart",
            summaryBox.x + summaryBox.width / 2,
            summaryBox.y + summaryBox.height / 2,
        );
        await touch("touchEnd");
        await expect(panel).not.toHaveAttribute("open", "");
        await expect(slider).toBeHidden();
        await gameKeyMovesCamera(page);
        await touch(
            "touchStart",
            summaryBox.x + summaryBox.width / 2,
            summaryBox.y + summaryBox.height / 2,
        );
        await touch("touchEnd");
        await expect(panel).toHaveAttribute("open", "");
        await expect(slider).toBeVisible();

        await slider.scrollIntoViewIfNeeded();
        const box = (await slider.boundingBox())!;
        const initial = await slider.inputValue();
        await touch("touchStart", box.x + box.width * 0.25, box.y + box.height / 2);
        const down = await slider.inputValue();
        await touch("touchMove", box.x + box.width * 0.75, box.y + box.height / 2);
        const dragged = await slider.inputValue();
        expect(dragged).not.toBe(down);
        expect(dragged).not.toBe(initial);
        await touch("touchEnd");
        await expect(slider).toHaveValue(dragged);
        await expect(page.locator(`output[for='${await slider.getAttribute("id")}']`)).toHaveText(
            Number(dragged).toFixed(2),
        );
        await gameKeyMovesCamera(page);
    } finally {
        await context.close();
    }
});
