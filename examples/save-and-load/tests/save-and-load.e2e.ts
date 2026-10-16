import { mkdirSync } from "node:fs";
import { expect, type Page, test } from "playwright/test";
import type { Save } from "../src/save";

type State = { money: number; position: number[]; carried: boolean; save: Save };
declare global {
    interface Window {
        capture(): Promise<string>;
        adapter: { class: string; identity: string };
        saveAndLoad: {
            state: () => State;
            aim: (action: number) => State;
            key: (down: boolean) => void;
            advance: () => State;
        };
    }
}
type Stepped = Window;
async function capture(page: Page, name: string): Promise<void> {
    const directory = process.env.SHALLOT_CAPTURE_DIR;
    if (!directory) return;
    mkdirSync(directory, { recursive: true });
    const adapter = await page.evaluate(() => window.adapter);
    expect(adapter.class).toBe("real");
    console.log("save-and-load adapter", adapter);
    await page.evaluate(() => window.capture());
    await page.screenshot({ path: `${directory}/${name}.png` });
}

async function state(page: Page): Promise<State> {
    const value = await page.evaluate(() => (window as Stepped).saveAndLoad.state());
    await expect(page.getByRole("status", { name: "Money" })).toHaveText(`${value.money} coins`);
    return value;
}
async function use(page: Page, action: number): Promise<State> {
    await page.evaluate((action) => (window as Stepped).saveAndLoad.aim(action), action);
    await state(page);
    await page.evaluate(() => {
        const game = (window as Stepped).saveAndLoad;
        game.key(true);
        game.advance();
    });
    await state(page);
    await page.evaluate(() => {
        const game = (window as Stepped).saveAndLoad;
        game.key(false);
        game.advance();
    });
    return state(page);
}

test("the built page shows money centred at the bottom of the canvas", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("/");
    const money = page.getByRole("status", { name: "Money" });
    await expect(money).toHaveText("0 coins");
    const canvas = (await page.locator("canvas").boundingBox())!;
    const bounds = (await money.boundingBox())!;
    expect(bounds.x + bounds.width / 2).toBeCloseTo(canvas.x + canvas.width / 2, 0);
    expect(bounds.y).toBeGreaterThan(canvas.y + canvas.height / 2);
    expect(bounds.y + bounds.height).toBeLessThan(canvas.y + canvas.height);
    expect(errors).toEqual([]);
});

test("collect, buy, carry and save; page reload restores money, player placement and carried crate", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(
        `http://127.0.0.1:4190/tests/stepped.html${process.env.SHALLOT_CAPTURE_DIR ? "?captures" : ""}`,
    );
    await page.waitForFunction(() => !!(window as Stepped).saveAndLoad);
    expect((await state(page)).money).toBe(0);
    await capture(page, "spawn");
    for (let i = 1; i <= 3; i++) expect((await use(page, 0)).money).toBe(i);
    expect((await use(page, 1)).money).toBe(0);
    expect((await use(page, 2)).carried).toBe(true);
    await capture(page, "carrying");
    const before = await use(page, 3);
    const stored = await page.evaluate(() =>
        JSON.parse(localStorage.getItem("shallot-save-and-load")!),
    );
    expect(stored.version).toBe(2);
    await page.reload();
    await page.waitForFunction(() => !!(window as Stepped).saveAndLoad);
    const after = await state(page);
    await capture(page, "after-reload");
    expect(after.money).toBe(before.money);
    expect(after.carried).toBe(true);
    const savedPlayer = stored.entities.find((entry: Record<string, unknown>) => entry.Player);
    expect(after.position).toEqual(savedPlayer.Transform.translation.slice(0, 3));
    expect(after.save.entities.find((entry) => entry.Player)).toEqual(savedPlayer);
    const carried = after.save.entities[savedPlayer.Wallet.carried];
    expect(carried.Saved.kind).toBe(3);
    expect(errors).toEqual([]);
});
