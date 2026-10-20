import { mkdir, writeFile } from "node:fs/promises";
import { expect, type Page, test } from "playwright/test";

type SteppedWindow = Window & {
    captureScene?: () => Promise<number[]>;
    baseline?: number[];
    previous?: number[];
};
const names = ["Lens distortion", "Chromatic aberration", "Vignette"];

async function capture(page: Page, name: string) {
    const save =
        Boolean(process.env.SHALLOT_CAPTURE_DIR) &&
        ["none", "lens-distortion", "chromatic-aberration", "vignette", "all"].includes(name);
    const result = await page.evaluate(
        async ({ name, save }) => {
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

test("the built page presents the scene and keeps its HUD inside the canvas", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("/");
    const panel = page.getByRole("region", { name: "Post-processing controls" });
    await panel.waitFor();
    const canvas = (await page.locator("canvas").boundingBox())!;
    const bounds = (await panel.boundingBox())!;
    expect(bounds.x).toBeGreaterThanOrEqual(canvas.x);
    expect(bounds.x + bounds.width).toBeLessThanOrEqual(canvas.x + canvas.width);
    expect(bounds.y + bounds.height).toBeLessThanOrEqual(canvas.y + canvas.height);
    for (const name of names)
        await expect(page.getByRole("checkbox", { name, exact: true })).toBeChecked();
    if (process.env.SHALLOT_CAPTURE_DIR) {
        await mkdir(process.env.SHALLOT_CAPTURE_DIR, { recursive: true });
        await page.screenshot({
            path: `${process.env.SHALLOT_CAPTURE_DIR}/panel-start-all-on.png`,
        });
    }
    expect(errors).toEqual([]);
});

test("each HUD toggle changes the all-on scene reversibly without moving or resizing the panel", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("http://127.0.0.1:4266/tests/stepped.html");
    await page.waitForFunction(() => (window as SteppedWindow).captureScene !== undefined);
    expect((await capture(page, "all")).colors).toBeGreaterThan(100);
    const panel = page.getByRole("region", { name: "Post-processing controls" });
    const bounds = await panel.boundingBox();
    for (const name of names) {
        const toggle = page.getByRole("checkbox", { name, exact: true });
        await toggle.uncheck();
        await expect(toggle).not.toBeChecked();
        expect(await panel.boundingBox(), `${name} must preserve the panel bounding box`).toEqual(
            bounds,
        );
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

test("each slider changes captured pixels and the panel stays inside the canvas", async ({
    page,
}) => {
    await page.goto("http://127.0.0.1:4266/tests/stepped.html");
    await page.waitForFunction(() => (window as SteppedWindow).captureScene !== undefined);
    for (const name of names) {
        await page.getByRole("checkbox", { name, exact: true }).check();
        const sliders = page.getByRole("group", { name, exact: true }).getByRole("slider");
        for (const slider of await sliders.all()) {
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
        }
        await page.getByRole("checkbox", { name, exact: true }).uncheck();
    }
    for (const name of names) await page.getByRole("checkbox", { name, exact: true }).check();
    await capture(page, "all");
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

test("dragging the canvas orbits the camera", async ({ page }) => {
    await page.goto("http://127.0.0.1:4266/tests/stepped.html");
    await page.waitForFunction(() => (window as SteppedWindow).captureScene !== undefined);
    await capture(page, "before-orbit");
    await page.mouse.move(300, 350);
    await page.mouse.down();
    await page.mouse.move(550, 400, { steps: 20 });
    await capture(page, "during-orbit");
    await page.mouse.up();
    expect((await capture(page, "after-orbit")).baseline).toBeGreaterThan(0.25);
    if (process.env.SHALLOT_CAPTURE_DIR) {
        await page.goto("/");
        for (const name of names) await page.getByRole("checkbox", { name, exact: true }).check();
        await page.screenshot({ path: `${process.env.SHALLOT_CAPTURE_DIR}/panel-all.png` });
        await page.setViewportSize({ width: 390, height: 844 });
        await page.screenshot({ path: `${process.env.SHALLOT_CAPTURE_DIR}/panel-all-phone.png` });
    }
});
