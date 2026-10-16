import { mkdirSync } from "node:fs";
import { expect, type Page, test } from "playwright/test";
import type { Save } from "../src/save";

declare global {
    interface Window {
        capture(): Promise<string>;
        adapter: { class: string; identity: string };
        saveAndLoad: {
            state(): { count: number; save: Save };
            advance(ticks?: number): Promise<unknown>;
            hover(): void;
            click(): void;
            key(code: string): void;
        };
    }
}
async function capture(page: Page, name: string): Promise<void> {
    const directory = process.env.SHALLOT_CAPTURE_DIR;
    if (!directory) return;
    mkdirSync(directory, { recursive: true });
    const adapter = await page.evaluate(() => window.adapter);
    expect(adapter.class).toBe("real");
    console.log("save-and-load adapter", adapter);
    const image = await page.evaluate(() => window.capture());
    await page.evaluate(async (src) => {
        const img = new Image();
        img.id = "capture";
        img.style.cssText = "position:absolute;left:0;top:0;pointer-events:none";
        img.src = src;
        await img.decode();
        document.body.insertBefore(img, document.querySelector("canvas")!.nextSibling);
    }, image);
    await page.screenshot({ path: `${directory}/${name}.png` });
    await page.evaluate(() => document.getElementById("capture")!.remove());
}
async function count(page: Page, expected: number) {
    expect(await page.evaluate(() => window.saveAndLoad.state().count)).toBe(expected);
    await expect(page.getByRole("status", { name: "Coins", exact: true })).toHaveText(
        `${expected} coins`,
    );
}
async function button(page: Page, name: string) {
    await page.getByRole("button", { name }).click();
    await page.evaluate(() => window.saveAndLoad.advance());
}

test("the built page opens a new game with clear controls and Load disabled until Save", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
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

test("three clicks, Save, New game and Load restore the pile and shown count, also after page reload", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(
        `http://127.0.0.1:4190/tests/stepped.html${process.env.SHALLOT_CAPTURE_DIR ? "?capture" : ""}`,
    );
    await page.waitForFunction(() => !!window.saveAndLoad);
    await count(page, 0);
    await expect(page.getByRole("button", { name: "Load L" })).toBeDisabled();
    await page.evaluate(() => window.saveAndLoad.hover());
    await capture(page, "01-start-hovered");
    for (let i = 1; i <= 3; i++) {
        await page.evaluate(() => window.saveAndLoad.click());
        await count(page, i);
    }
    await page.evaluate(() => window.saveAndLoad.advance(60));
    await capture(page, "02-three-clicks");
    await button(page, "Save S");
    await expect(page.getByRole("button", { name: "Saved S", exact: true })).toBeVisible();
    const saved = await page.evaluate(() => window.saveAndLoad.state().save);
    expect(saved.entities).toHaveLength(3);
    await capture(page, "03-saved");
    await page.evaluate(() => window.saveAndLoad.advance(130));
    await expect(page.getByRole("button", { name: "Save S", exact: true })).toBeVisible();
    await button(page, "New game N");
    await count(page, 0);
    await button(page, "Load L");
    await count(page, 3);
    expect((await page.evaluate(() => window.saveAndLoad.state().save)).entities).toEqual(
        expect.arrayContaining(saved.entities),
    );
    await capture(page, "04-loaded");
    await page.reload();
    await page.waitForFunction(() => !!window.saveAndLoad);
    await count(page, 0);
    await expect(page.getByRole("button", { name: "Load L" })).toBeEnabled();
    await page.evaluate(() => window.saveAndLoad.key("KeyL"));
    await count(page, 3);
    expect((await page.evaluate(() => window.saveAndLoad.state().save)).entities).toEqual(
        expect.arrayContaining(saved.entities),
    );
    for (let i = 4; i <= 10; i++) {
        await page.evaluate(() => window.saveAndLoad.click());
        await count(page, i);
    }
    await page.evaluate(() => window.saveAndLoad.advance(60));
    await capture(page, "05-ten-clicks");
    if (process.env.SHALLOT_CAPTURE_DIR) {
        for (let i = 10; i < 104; i++) await page.evaluate(() => window.saveAndLoad.click());
        await page.evaluate(() => window.saveAndLoad.advance(60));
        await count(page, 104);
        await capture(page, "06-cap");
        await button(page, "Save S");
        await capture(page, "07-saved-cap");
        await button(page, "New game N");
        await button(page, "Load L");
        await capture(page, "08-loaded-cap");
        await page.getByRole("button", { name: "Save S", exact: true }).hover();
        await capture(page, "09-save-hovered");
    }
    await page.evaluate(() => window.saveAndLoad.key("KeyN"));
    await count(page, 0);
    await page.evaluate(() => window.saveAndLoad.key("KeyS"));
    await expect(page.getByRole("button", { name: "Saved S", exact: true })).toBeVisible();
    await page.evaluate(() => window.saveAndLoad.key("KeyL"));
    await count(page, 0);
    expect(errors).toEqual([]);
});
