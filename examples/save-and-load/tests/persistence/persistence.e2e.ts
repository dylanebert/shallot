import { expect, type Page, test } from "playwright/test";
import { capture } from "../capture";

async function measured<T>(phase: string, run: () => Promise<T>): Promise<T> {
    const started = performance.now();
    try {
        return await run();
    } finally {
        console.info("persistence phase ms", phase, performance.now() - started);
    }
}

async function count(page: Page, expected: number) {
    expect(await page.evaluate(() => window.saveAndLoad.state().count)).toBe(expected);
    await expect(page.getByRole("status", { name: "Coins", exact: true })).toHaveText(
        `${expected} coins`,
    );
}
async function button(page: Page, name: string) {
    await measured(`${name} DOM click`, () => page.getByRole("button", { name }).click());
    await measured(`${name} advance`, () => page.evaluate(() => window.saveAndLoad.advance()));
}
async function open(page: Page, errors: string[]) {
    page.on("pageerror", (error) => errors.push(error.message));
    await measured("goto", () =>
        page.goto(
            `http://127.0.0.1:4190/tests/stepped.html${process.env.SHALLOT_CAPTURE_DIR ? "?capture" : ""}`,
        ),
    );
    await measured("ready", () => page.waitForFunction(() => !!window.saveAndLoad));
}
test.describe.configure({ mode: "serial" });
let page: Page;
const errors: string[] = [];
test.beforeAll(async ({ browser }) => {
    page = await browser.newPage();
    const session = await page.context().newCDPSession(page);
    await session.send("Emulation.setHardwareConcurrencyOverride", { hardwareConcurrency: 1 });
    await open(page, errors);
});
test.afterAll(async () => {
    await page?.close();
});
test.beforeEach(async () => {
    await button(page, "New game N");
});

test("three clicks, Save, New game and Load restore the pile and shown count", async () => {
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
    await measured("advance 130", () => page.evaluate(() => window.saveAndLoad.advance(130)));
    await expect(page.getByRole("button", { name: "Save S", exact: true })).toBeVisible();
    await button(page, "New game N");
    await count(page, 0);
    await button(page, "Load L");
    await count(page, 3);
    expect((await page.evaluate(() => window.saveAndLoad.state().save)).entities).toEqual(
        expect.arrayContaining(saved.entities),
    );
    await capture(page, "04-loaded");
    expect(errors).toEqual([]);
});

test("a save survives a page reload, and Load restores its pile and shown count", async () => {
    for (let i = 1; i <= 3; i++) await page.evaluate(() => window.saveAndLoad.click());
    await count(page, 3);
    await page.evaluate(() => window.saveAndLoad.advance(60));
    await button(page, "Save S");
    await expect(page.getByRole("button", { name: "Saved S", exact: true })).toBeVisible();
    const saved = await page.evaluate(() => window.saveAndLoad.state().save);
    expect(saved.entities).toHaveLength(3);
    await measured("real reload", () => page.reload());
    await measured("reload ready", () => page.waitForFunction(() => !!window.saveAndLoad));
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
