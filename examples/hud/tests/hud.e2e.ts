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
