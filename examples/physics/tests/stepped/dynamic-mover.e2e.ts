import { expect, test } from "playwright/test";

type State = { keys: Record<string, number>; balls: number };
const state = () => (window as unknown as { dynamicState(steps: number): State }).dynamicState(1);

for (const width of [1280, 390]) {
    test(`Dynamic Mover shoots with RMB held and LMB at ${width}`, async ({ page }) => {
        await page.setViewportSize({ width, height: width === 390 ? 844 : 720 });
        await page.goto("/tests/controls.html");
        await page.waitForFunction(() => "dynamicState" in window);
        await page.getByRole("button", { name: "Scene", exact: true }).click();
        await page.getByRole("button", { name: "Dynamic Mover", exact: true }).click();
        await page.locator(".controls summary").click();
        await page.mouse.move(30, 500);
        await page.mouse.down({ button: "right" });
        await page.waitForFunction(() => document.pointerLockElement !== null);
        const aimed = await page.evaluate(state);
        expect(aimed.keys.Aim).toBe(1);
        await page.mouse.down({ button: "left" });
        const shot = await page.evaluate(state);
        expect(shot.balls).toBe(aimed.balls + 1);
        await page.mouse.up({ button: "left" });
        await page.mouse.up({ button: "right" });
    });
}

test("the LMB that engages pointer lock does not shoot even while Aim is held", async ({
    page,
}) => {
    await page.goto("/tests/controls.html");
    await page.waitForFunction(() => "dynamicState" in window);
    await page.getByRole("button", { name: "Scene", exact: true }).click();
    await page.getByRole("button", { name: "Dynamic Mover", exact: true }).click();
    await page.locator(".controls summary").click();
    await page.keyboard.down("KeyQ");
    await page.mouse.move(30, 500);
    await page.mouse.down({ button: "left" });
    await page.waitForFunction(() => document.pointerLockElement !== null);
    expect((await page.evaluate(state)).balls).toBe(0);
    await page.mouse.up({ button: "left" });
    await page.evaluate(state);
    await page.mouse.down({ button: "left" });
    expect((await page.evaluate(state)).balls).toBe(1);
    await page.mouse.up({ button: "left" });
    await page.keyboard.up("KeyQ");
});
