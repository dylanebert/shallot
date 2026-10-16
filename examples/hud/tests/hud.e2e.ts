import { expect, type Page, test } from "playwright/test";

type Turn = { angle: number; speed: number };
type SteppedWindow = Window & {
    stepHud?: (steps: number) => Turn;
    captureHud?: () => Promise<number[]>;
    materialHud?: () => number[];
};

const STEPPED = "http://127.0.0.1:4176/tests/stepped.html";

async function step(page: Page, steps: number): Promise<Turn> {
    return page.evaluate((count) => (window as SteppedWindow).stepHud!(count), steps);
}

test.describe("zoomed pointer controls", () => {
    // 721 is the smallest whole-pixel shortest side above the 720px zoom reference.
    test.use({ viewport: { width: 721, height: 721 }, deviceScaleFactor: 1 });
    test("the speed slider follows a pointer drag through the scaled overlay", async ({ page }) => {
        await page.goto("/");
        const controls = page.getByRole("region", { name: "Cube controls" });
        await expect
            .poll(async () =>
                Number(
                    await controls
                        .locator("..")
                        .evaluate((overlay) => getComputedStyle(overlay).zoom),
                ),
            )
            .toBeCloseTo(721 / 720, 5);
        const slider = page.getByRole("slider", { name: "Turn speed" });
        const bounds = (await slider.boundingBox())!;
        const y = bounds.y + bounds.height / 2;
        await page.mouse.move(bounds.x + bounds.width * 0.4, y);
        await page.mouse.down();
        try {
            for (const [x, value] of [
                [bounds.x + 1, "0"],
                [bounds.x + bounds.width / 2, "1.5"],
                [bounds.x + bounds.width - 1, "3"],
            ] as const) {
                await page.mouse.move(x, y, { steps: 5 });
                await expect(slider).toHaveValue(value);
            }
        } finally {
            await page.mouse.up();
        }
    });
});

test("the controls float centred along the canvas's bottom edge", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("/");

    const controls = page.getByRole("region", { name: "Cube controls" });
    await controls.waitFor();
    const canvas = await page.locator("canvas").boundingBox();
    const hud = await controls.boundingBox();
    expect(canvas && hud).toBeTruthy();
    expect(hud!.x + hud!.width / 2).toBeCloseTo(canvas!.x + canvas!.width / 2, 0);
    expect(hud!.y).toBeGreaterThan(canvas!.y + canvas!.height / 2);
    expect(hud!.y + hud!.height).toBeLessThanOrEqual(canvas!.y + canvas!.height);
    expect(errors).toEqual([]);
});

test("the angle the HUD shows is the game's own Turn after each step", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(STEPPED);
    await page.waitForFunction(() => (window as SteppedWindow).stepHud !== undefined);

    let previous: Turn | undefined;
    for (const steps of [1, 29, 300, 37]) {
        const turn = await step(page, steps);
        expect(turn.angle).toBeGreaterThan(previous?.angle ?? 0);
        const degrees = Math.floor((turn.angle * 180) / Math.PI) % 360;
        // `stepHud` flushes Svelte's updates before it returns, so the text is already the step's.
        expect(await page.locator('[data-hud="angle"]').textContent()).toBe(`${degrees}°`);
        previous = turn;
    }
    expect(previous!.angle).toBeGreaterThan(2 * Math.PI);
    expect(errors).toEqual([]);
});

test("DOM controls change stepped speed, material colour and the cube centre hue", async ({
    page,
}) => {
    await page.goto(STEPPED);
    await page.waitForFunction(() => (window as SteppedWindow).stepHud !== undefined);
    const before = await step(page, 1);
    const amber = await page.evaluate(() => (window as SteppedWindow).captureHud!());
    await page.locator("#speed").fill("2.4");
    await page.getByRole("button", { name: "Cyan" }).click();
    await expect(page.getByRole("button", { name: "Cyan" })).toHaveAttribute(
        "aria-pressed",
        "true",
    );
    const after = await step(page, 1);
    expect(after.angle - before.angle).toBeCloseTo(2.4 / 60, 8);
    const color = await page.evaluate(() => (window as SteppedWindow).materialHud!());
    for (const [index, value] of [0.04, 0.43, 0.53, 1].entries())
        expect(color[index]).toBeCloseTo(value, 6);
    const cyan = await page.evaluate(() => (window as SteppedWindow).captureHud!());
    expect(amber[0]).toBeGreaterThan(amber[2]);
    expect(cyan[2]).toBeGreaterThan(cyan[0]);
    if (process.env.SHALLOT_CAPTURE_DIR) {
        await page.screenshot({ path: `${process.env.SHALLOT_CAPTURE_DIR}/hud-blue-2.4.png` });
    }
});
