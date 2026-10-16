import { mkdirSync, writeFileSync } from "node:fs";
import { expect, type Page, test } from "playwright/test";

type Frame = { hour: number; width: number; height: number; rgba: number[]; png: string };
declare global {
    interface Window {
        frame: (hour?: number) => Promise<Frame>;
    }
}

test("the built game's only control floats centred at the canvas's bottom", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("/");
    const slider = page.getByRole("slider", { name: "Hour" });
    await slider.waitFor();
    const canvas = await page.locator("canvas").boundingBox();
    const controls = await page.getByRole("region", { name: "Time of day" }).boundingBox();
    expect(canvas && controls).toBeTruthy();
    expect(controls!.x + controls!.width / 2).toBeCloseTo(canvas!.x + canvas!.width / 2, 0);
    expect(controls!.y).toBeGreaterThan(canvas!.y + canvas!.height / 2);
    expect(controls!.y + controls!.height).toBeLessThanOrEqual(canvas!.y + canvas!.height);
    expect(await page.getByRole("slider").count()).toBe(1);
    expect(errors).toEqual([]);
});

async function open(page: Page, hour: number, evening = false, side = false) {
    await page.goto(
        `http://127.0.0.1:4266/tests/stepped.html?hour=${hour}${evening ? "&evening" : ""}${side ? "&side" : ""}${process.env.SHALLOT_CAPTURE_DIR ? "&capture" : ""}`,
    );
    await page.waitForFunction(() => !!window.frame);
}

function save(name: string, frame: Frame) {
    if (!process.env.SHALLOT_CAPTURE_DIR) return;
    mkdirSync(process.env.SHALLOT_CAPTURE_DIR, { recursive: true });
    writeFileSync(
        `${process.env.SHALLOT_CAPTURE_DIR}/${name}.png`,
        Buffer.from(frame.png.split(",")[1], "base64"),
    );
}

function mean(frame: Frame, x0: number, y0: number, x1: number, y1: number) {
    let sum = 0,
        count = 0;
    for (let y = Math.floor(y0 * frame.height); y < y1 * frame.height; y++) {
        for (let x = Math.floor(x0 * frame.width); x < x1 * frame.width; x++) {
            const i = (y * frame.width + x) * 4;
            sum += 0.2126 * frame.rgba[i] + 0.7152 * frame.rgba[i + 1] + 0.0722 * frame.rgba[i + 2];
            count++;
        }
    }
    return sum / count;
}

test("the slider drives the hour and noon sky is brighter than midnight", async ({ page }) => {
    await open(page, 12);
    const noon = await page.evaluate(() => window.frame());
    await page.getByRole("slider", { name: "Hour" }).fill("0");
    const night = await page.evaluate(() => window.frame());
    expect(night.hour).toBe(0);
    expect(mean(noon, 0.1, 0.05, 0.9, 0.25)).toBeGreaterThan(
        mean(night, 0.1, 0.05, 0.9, 0.25) + 30,
    );
});

test("morning and afternoon cast a shape's shadow on opposite sides", async ({ page }) => {
    await open(page, 9, false, true);
    const morning = await page.evaluate(() => window.frame());
    await open(page, 15, false, true);
    const afternoon = await page.evaluate(() => window.frame());
    save("morning", morning);
    save("afternoon", afternoon);
    const contrast = (f: Frame) =>
        mean(f, 0.44, 0.49, 0.47, 0.505) - mean(f, 0.53, 0.49, 0.56, 0.505);
    expect(contrast(morning) * contrast(afternoon)).toBeLessThan(-4);
});

test("the first frame after changing the light direction has the new shadows", async ({ page }) => {
    test.fail(
        true,
        "Shadow presentation lags a light-direction change by one frame; stage 2 owns the fix.",
    );
    await open(page, 15);
    const reference = await page.evaluate(() => window.frame());
    await open(page, 9);
    await page.evaluate(() => window.frame());
    const changed = await page.evaluate(() => window.frame(15));
    save("changed-sun-first-frame", changed);
    save("changed-sun-reference", reference);
    let error = 0;
    for (let i = 0; i < reference.rgba.length; i++)
        error += Math.abs(reference.rgba[i] - changed.rgba[i]);
    expect(error / reference.rgba.length).toBeLessThan(0.05);
});

test("a fresh scene starts at the requested hour with the low sun's disk in frame", async ({
    page,
}) => {
    await open(page, 7);
    const initial = await page.evaluate(() => window.frame());
    expect(initial.hour).toBe(7);
    expect(mean(initial, 0.798, 0.048, 0.807, 0.067)).toBeGreaterThan(
        mean(initial, 0.7, 0.05, 0.74, 0.1) + 8,
    );
    if (!process.env.SHALLOT_CAPTURE_DIR) return;
    for (const [name, hour] of Object.entries({
        dawn: 6.25,
        "low-morning": 7,
        noon: 12,
        "low-evening": 17,
        dusk: 17.75,
        midnight: 0,
    })) {
        await open(page, hour, hour > 12);
        save(name, await page.evaluate(() => window.frame()));
        await page.screenshot({ path: `${process.env.SHALLOT_CAPTURE_DIR}/page-${name}.png` });
    }
    for (const hour of [7, 17]) {
        await open(page, hour, hour > 12);
        await page.goto(`${page.url()}&no-shafts`);
        await page.waitForFunction(() => !!window.frame);
        save(
            hour === 7 ? "low-morning-no-shafts" : "low-evening-no-shafts",
            await page.evaluate(() => window.frame()),
        );
    }
    await open(page, 7);
    for (let i = 0; i < 12; i++)
        save(
            `low-sun-${String(i).padStart(2, "0")}`,
            await page.evaluate((hour) => window.frame(hour), 7 + i / 150),
        );
});
