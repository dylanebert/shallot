import { expect, test } from "playwright/test";
import { capture } from "./capture";

test.beforeEach(async ({ page }) => {
    const session = await page.context().newCDPSession(page);
    await session.send("Emulation.setHardwareConcurrencyOverride", { hardwareConcurrency: 1 });
});

test("the built page opens a new game with clear controls and Load disabled until Save", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.setViewportSize({ width: 640, height: 360 });
    await page.goto("/");
    await expect(page.getByRole("status", { name: "Coins", exact: true })).toHaveText("0 coins");
    await expect(page.getByRole("button", { name: "Load L" })).toBeDisabled();
    for (const key of ["N", "S", "L"])
        await expect(page.locator("kbd", { hasText: key })).toBeVisible();
    await page.getByRole("button", { name: "Save S" }).click();
    await expect(page.getByRole("button", { name: "Saved S", exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Load L" })).toBeEnabled();
    expect(errors).toEqual([]);
});

test("every control holds still across feedback, disabled, hover and focus", async ({ page }) => {
    await page.goto(
        `http://127.0.0.1:4190/tests/${process.env.SHALLOT_CAPTURE_DIR ? "stepped.html?capture" : "hud.html"}`,
    );
    await page.waitForFunction(() => !!window.saveAndLoad);
    const controls = page.locator(".save-hud button");
    const boxes = () =>
        controls.evaluateAll((elements) =>
            elements.map((element) => {
                const { x, y, width, height } = element.getBoundingClientRect();
                return { x, y, width, height };
            }),
        );
    const initial = await boxes();
    await expect(controls.nth(2)).toBeDisabled();
    await controls.nth(2).hover();
    await expect(controls.nth(2).locator("kbd")).toHaveCSS("transform", "none");
    expect(await boxes()).toEqual(initial);
    await page.mouse.move(0, 0);
    await capture(page, "00-load-disabled");
    for (let i = 0; i < 3; i++) await page.evaluate(() => window.saveAndLoad.click());
    await page.evaluate(() => window.saveAndLoad.advance(60));
    await capture(page, "01-three-coins");
    await controls.nth(1).click();
    await page.evaluate(() => window.saveAndLoad.advance());
    await expect(controls.nth(1)).toHaveText("Saved S");
    expect(await boxes()).toEqual(initial);
    await expect(controls.nth(2)).toBeEnabled();
    await page.mouse.move(0, 0);
    await capture(page, "02-saved");
    await page.evaluate(() => window.saveAndLoad.advance(130));
    expect(await boxes()).toEqual(initial);
    for (const control of await controls.all()) {
        await control.hover();
        await expect(control).toHaveCSS("outline-style", "none");
        await expect(control.locator("kbd")).toHaveCSS(
            "transform",
            "matrix(1.12, 0, 0, 1.12, 0, 0)",
        );
        expect(await boxes()).toEqual(initial);
        await page.mouse.move(0, 0);
        await page.keyboard.press("Tab");
        await control.focus();
        await expect(control).toBeFocused();
        await expect(control).toHaveCSS("outline-style", "none");
        await expect(control.locator("kbd")).toHaveCSS(
            "transform",
            "matrix(1.12, 0, 0, 1.12, 0, 0)",
        );
        expect(await boxes()).toEqual(initial);
    }
    await controls.nth(2).evaluate((element) => (element as HTMLElement).blur());
    await controls.nth(1).hover();
    await expect(controls.nth(1).locator("kbd")).toHaveCSS(
        "transform",
        "matrix(1.12, 0, 0, 1.12, 0, 0)",
    );
    await capture(page, "03-save-hovered");
});
