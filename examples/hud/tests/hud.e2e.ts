import { expect, type Page, test } from "playwright/test";

type Turn = { angle: number; turns: number };
type SteppedWindow = Window & { stepHud?: (steps: number) => Turn };

const STEPPED = "http://127.0.0.1:4176/tests/stepped.html";

async function step(page: Page, steps: number): Promise<Turn> {
    return page.evaluate((count) => (window as SteppedWindow).stepHud!(count), steps);
}

test("the HUD shows the turntable's live angle over the canvas", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("/");

    const angle = page.locator('[data-hud="angle"]');
    await angle.waitFor();
    const first = await angle.textContent();
    await expect(angle).not.toHaveText(first ?? "");

    const canvas = await page.locator("canvas").boundingBox();
    const hud = await angle.boundingBox();
    expect(canvas && hud).toBeTruthy();
    expect(hud!.x).toBeGreaterThanOrEqual(canvas!.x);
    expect(hud!.y).toBeGreaterThanOrEqual(canvas!.y);
    expect(hud!.x + hud!.width).toBeLessThanOrEqual(canvas!.x + canvas!.width);
    expect(hud!.y + hud!.height).toBeLessThanOrEqual(canvas!.y + canvas!.height);
    expect(errors).toEqual([]);
});

test("each value the HUD shows is the game's own Turn after each step", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(STEPPED);
    await page.waitForFunction(() => (window as SteppedWindow).stepHud !== undefined);

    // Past a whole turn (5.2 s at 1.2 rad/s), so the turn count is seen to change as well as the angle.
    let previous: Turn | undefined;
    for (const steps of [1, 29, 300, 37]) {
        const turn = await step(page, steps);
        expect(turn.angle).toBeGreaterThan(previous?.angle ?? 0);
        const degrees = Math.floor((turn.angle * 180) / Math.PI) % 360;
        // `stepHud` flushes Svelte's updates before it returns, so the text is already the step's.
        expect(await page.locator('[data-hud="angle"]').textContent()).toBe(`${degrees}°`);
        expect(await page.locator('[data-hud="turns"]').textContent()).toBe(`${turn.turns}`);
        previous = turn;
    }
    expect(previous!.turns).toBeGreaterThan(0);
    expect(errors).toEqual([]);
});
