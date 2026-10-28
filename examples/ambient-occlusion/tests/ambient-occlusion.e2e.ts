import { mkdir } from "node:fs/promises";
import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, type Page, test } from "playwright/test";
import { settings } from "../src/settings";
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

test.use(pipelineWarnings);

test("the first frame has no late-pipeline warning and F3 toggles the profiler", async ({
    page,
}) => {
    await page.goto("/");
    await page.waitForFunction(() =>
        performance
            .getEntriesByType("measure")
            .some((entry) => entry.name.endsWith(":first-update")),
    );
    await expect(page.getByLabel("GPU time")).toHaveCount(0);
    const profiler = page.locator("[data-shallot-profile]");
    await expect(profiler).toHaveCount(0);
    await page.keyboard.press("F3");
    await expect(profiler).toBeVisible();
    await page.keyboard.press("F3");
    await expect(profiler).toHaveCount(0);
});

test("each control writes its camera field and the toggle restores them", async ({ page }) => {
    await page.goto("http://127.0.0.1:4276/tests/stepped.html?controls");
    await page.waitForFunction(() => "occlusionFields" in window);
    const fields = () =>
        page.evaluate(() =>
            (
                window as unknown as { occlusionFields(): Record<string, number> | null }
            ).occlusionFields(),
        );
    const toggle = page.getByRole("checkbox");
    const quality = page.getByRole("combobox");
    const initial = await fields();
    expect(initial!.qualityLevel).toBe(Number(await quality.inputValue()));
    await toggle.uncheck();
    await toggle.check();
    expect(await fields()).toEqual(initial);
    const values = [8, 4];
    for (const [index, control] of settings.entries()) {
        await page.locator(`#control-${index}`).fill(String(values[index]));
        expect((await fields())![control.field], control.label).toBeCloseTo(values[index]);
    }
    for (const level of [0, 1, 3]) {
        await quality.selectOption(String(level));
        expect((await fields())!.qualityLevel).toBe(level);
    }
    const set = await fields();
    await toggle.uncheck();
    expect(await fields()).toBeNull();
    for (const slider of await page.getByRole("slider").all()) await expect(slider).toBeDisabled();
    await expect(quality).toBeDisabled();
    await toggle.check();
    expect(await fields()).toEqual(set);
});

for (const viewport of [
    { width: 1280, height: 720 },
    { width: 390, height: 844 },
]) {
    test(`default view and usable controls at ${viewport.width}px`, async ({ page }) => {
        await page.setViewportSize(viewport);
        await page.goto("http://127.0.0.1:4276/tests/stepped.html");
        await page.waitForFunction(() => "captureScene" in window);
        await capture(page);
        const panel = page.getByRole("region", { name: "Ambient occlusion controls" });
        await panel.waitFor();
        await page.evaluate(() => document.fonts.ready);
        const bounds = (await panel.boundingBox())!;
        expect(bounds.x).toBeGreaterThanOrEqual(0);
        expect(bounds.y + bounds.height).toBeLessThanOrEqual(viewport.height);
        expect(bounds.x + bounds.width).toBeLessThanOrEqual(viewport.width);
        if (viewport.width === 390) {
            const subject = await page.evaluate(() => {
                const frames = (window as unknown as CaptureWindow).captures;
                const rgba = frames[frames.length - 1];
                let left = 390,
                    right = 0,
                    top = 844,
                    bottom = 0;
                for (let y = 0; y < 844; y++)
                    for (let x = 0; x < 390; x++) {
                        const at = (y * 390 + x) * 4;
                        if (Math.max(rgba[at], rgba[at + 1], rgba[at + 2]) <= 45) continue;
                        left = Math.min(left, x);
                        right = Math.max(right, x);
                        top = Math.min(top, y);
                        bottom = Math.max(bottom, y);
                    }
                return { left, right, top, bottom };
            });
            expect(subject.left).toBeGreaterThanOrEqual(16);
            expect(subject.right).toBeLessThan(374);
            expect(subject.top).toBeGreaterThan(bounds.y + bounds.height + 16);
            expect(subject.bottom).toBeLessThan(828);
            expect(subject.right - subject.left).toBeGreaterThan(200);
        }
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
        await capture(page);
        if (process.env.SHALLOT_CAPTURE_DIR && viewport.width === 1280)
            await page.screenshot({ path: `${process.env.SHALLOT_CAPTURE_DIR}/off-1280.png` });
        expect(await panel.boundingBox()).toEqual(bounds);
        await toggle.check();
        const slider = page.getByRole("slider").last();
        await slider.scrollIntoViewIfNeeded();
        await slider.fill("4");
        await expect(slider).toHaveValue("4");
        await expect(page.locator("output[for]").last()).toHaveText("4.0000");
    });
}

test("each pointer control leaves the bound orbit key working", async ({ page }) => {
    await page.goto("http://127.0.0.1:4276/tests/stepped.html?controls");
    await page.waitForFunction(() => "controlsKey" in window);
    for (const slider of await page.getByRole("slider").all()) {
        await dragRange(page, slider);
        await gameKeyMovesCamera(page);
    }
    await page.getByRole("combobox").click();
    await page.keyboard.press("Escape");
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
        const panel = page.getByRole("region", { name: "Ambient occlusion controls" });
        const panelBox = (await panel.boundingBox())!;
        expect(panelBox.x).toBeGreaterThanOrEqual(16);
        expect(panelBox.x + panelBox.width).toBeLessThanOrEqual(374);
        expect(panelBox.y + panelBox.height).toBeLessThanOrEqual(844);
        for (const row of await panel.locator(".row, .section-title").all())
            expect((await row.boundingBox())!.height).toBeGreaterThanOrEqual(44);
        if (process.env.SHALLOT_CAPTURE_DIR) {
            await page.evaluate(() => document.fonts.ready);
            await capture(page);
            await page.screenshot({ path: `${process.env.SHALLOT_CAPTURE_DIR}/touch-390.png` });
        }
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
            Number(dragged).toFixed(4),
        );
        await gameKeyMovesCamera(page);
    } finally {
        await context.close();
    }
});
