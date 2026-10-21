import { mkdir, writeFile } from "node:fs/promises";
import { expect, type Page, test } from "playwright/test";
import { bodyHasFocus, dragRange, gameKeyMovesCamera, pointerControl } from "../pointer";

type SteppedWindow = Window & {
    captureScene?: () => Promise<number[]>;
    baseline?: number[];
    previous?: number[];
};
const names = ["Lens distortion", "Chromatic aberration", "Vignette"];

test("each pointer control leaves the bound orbit key working", async ({ page }) => {
    await page.goto("http://127.0.0.1:4266/tests/stepped.html?controls");
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
        await page.goto("http://127.0.0.1:4266/tests/stepped.html?controls");
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
        const panel = page.getByRole("region", { name: "Post-processing controls" });
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

async function capture(page: Page, name: string) {
    const save =
        Boolean(process.env.SHALLOT_CAPTURE_DIR) &&
        ["none", "lens-distortion", "chromatic-aberration", "vignette", "all"].includes(name);
    const result = await page.evaluate(
        async ({ save }) => {
            const win = window as SteppedWindow;
            const rgba = await win.captureScene!();
            win.baseline ??= rgba;
            function difference(reference: number[]): number {
                let total = 0;
                for (let i = 0; i < rgba.length; i += 4)
                    for (let channel = 0; channel < 3; channel++)
                        total += Math.abs(rgba[i + channel] - reference[i + channel]);
                return total / (rgba.length * 0.75);
            }
            let png = "";
            if (save) {
                const source = document.querySelector("canvas")!;
                const canvas = document.createElement("canvas");
                canvas.width = source.width;
                canvas.height = source.height;
                canvas
                    .getContext("2d")!
                    .putImageData(
                        new ImageData(new Uint8ClampedArray(rgba), canvas.width, canvas.height),
                        0,
                        0,
                    );
                png = canvas.toDataURL("image/png").split(",")[1];
            }
            const previous = win.previous ? difference(win.previous) : 0;
            win.previous = rgba;
            return {
                previous,
                baseline: difference(win.baseline),
                colors: new Set(rgba).size,
                png,
            };
        },
        { name, save },
    );
    if (save) {
        await mkdir(process.env.SHALLOT_CAPTURE_DIR!, { recursive: true });
        await writeFile(
            `${process.env.SHALLOT_CAPTURE_DIR}/${name}.png`,
            Buffer.from(result.png, "base64"),
        );
    }
    return result;
}

test.describe(() => {
    let page: Page;
    const errors: string[] = [];
    test.beforeAll(async ({ browser }) => {
        page = await browser.newPage({ viewport: { width: 640, height: 480 } });
        page.on("pageerror", (error) => errors.push(error.message));
        await page.goto("http://127.0.0.1:4266/tests/stepped.html");
        await page.waitForFunction(() => (window as SteppedWindow).captureScene !== undefined);
    });
    test.afterAll(() => page.close());
    test.beforeEach(() =>
        page.evaluate(() => {
            const win = window as SteppedWindow;
            win.baseline = win.previous = undefined;
        }),
    );

    test("each HUD toggle changes the all-on scene reversibly without moving or resizing the panel", async () => {
        expect((await capture(page, "all")).colors).toBeGreaterThan(100);
        const panel = page.getByRole("region", { name: "Post-processing controls" });
        const bounds = await panel.boundingBox();
        for (const name of names) {
            const toggle = page.getByRole("checkbox", { name, exact: true });
            await toggle.uncheck();
            await expect(toggle).not.toBeChecked();
            expect(
                await panel.boundingBox(),
                `${name} must preserve the panel bounding box`,
            ).toEqual(bounds);
            expect((await capture(page, "without")).baseline, name).toBeGreaterThan(0.25);
            await toggle.check();
            expect(await panel.boundingBox()).toEqual(bounds);
            // Tonemapping dithering changes sub-byte averages between otherwise identical frames.
            expect((await capture(page, "reset")).baseline).toBeLessThan(0.1);
        }
        for (const name of names) {
            await page.getByRole("checkbox", { name, exact: true }).uncheck();
            expect(await panel.boundingBox()).toEqual(bounds);
            await expect(
                page.getByRole("group", { name, exact: true }).getByRole("slider").first(),
            ).toBeVisible();
        }
        expect(errors).toEqual([]);
    });

    test("a slider for each effect changes captured pixels and the panel stays inside the canvas", async () => {
        for (const name of names) await page.getByRole("checkbox", { name, exact: true }).check();
        for (const name of names) {
            await page.getByRole("checkbox", { name, exact: true }).check();
            // scene.test.ts checks every field write; pixels prove the HUD-to-frame path.
            const slider = page
                .getByRole("group", { name, exact: true })
                .getByRole("slider")
                .first();
            await capture(page, "before-slider");
            const original = await slider.inputValue();
            const min = Number(await slider.getAttribute("min"));
            const max = Number(await slider.getAttribute("max"));
            const value = Number(original) > (min + max) / 2 ? min : max;
            await slider.fill(String(value));
            expect(
                (await capture(page, "slider")).previous,
                (await slider.getAttribute("id"))!,
            ).toBeGreaterThan(0.1);
            await slider.fill(original);
            await page.getByRole("checkbox", { name, exact: true }).uncheck();
        }
        for (const name of names) await page.getByRole("checkbox", { name, exact: true }).check();
        for (const viewport of [
            { width: 1280, height: 720 },
            { width: 390, height: 844 },
            { width: 844, height: 390 },
        ]) {
            await page.setViewportSize(viewport);
            const canvas = (await page.locator("canvas").boundingBox())!;
            const panel = (await page
                .getByRole("region", { name: "Post-processing controls" })
                .boundingBox())!;
            expect(panel.x).toBeGreaterThanOrEqual(canvas.x);
            expect(panel.y).toBeGreaterThanOrEqual(canvas.y);
            expect(panel.x + panel.width).toBeLessThanOrEqual(canvas.x + canvas.width);
            expect(panel.y + panel.height).toBeLessThanOrEqual(canvas.y + canvas.height);
        }
    });

    test("dragging the canvas orbits the camera", async () => {
        await page.setViewportSize({ width: 640, height: 480 });
        await capture(page, "before-orbit");
        await page.mouse.move(20, 350);
        await page.mouse.down();
        await page.mouse.move(240, 400, { steps: 20 });
        await capture(page, "during-orbit");
        await page.mouse.up();
        expect((await capture(page, "after-orbit")).baseline).toBeGreaterThan(0.25);
        if (process.env.SHALLOT_CAPTURE_DIR) {
            await page.goto("/");
            for (const name of names)
                await page.getByRole("checkbox", { name, exact: true }).check();
            await page.screenshot({ path: `${process.env.SHALLOT_CAPTURE_DIR}/panel-all.png` });
            await page.setViewportSize({ width: 390, height: 844 });
            await page.screenshot({
                path: `${process.env.SHALLOT_CAPTURE_DIR}/panel-all-phone.png`,
            });
        }
    });
});
