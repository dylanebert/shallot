import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, type Page, test } from "playwright/test";
import { bodyHasFocus, dragRange, gameKeyMovesCamera, pointerControl } from "../pointer";

test.use(pipelineWarnings);

type Turn = { angle: number; speed: number };
type SteppedWindow = Window & {
    stepHud?: (steps: number) => Turn;
    captureHud?: () => Promise<number[]>;
    materialHud?: () => number[];
    resetHud?: () => void;
};
const STEPPED = "http://127.0.0.1:4176/tests/stepped.html";
async function step(page: Page, steps: number): Promise<Turn> {
    return page.evaluate((count) => (window as SteppedWindow).stepHud!(count), steps);
}

test.describe(() => {
    let page: Page;
    const errors: string[] = [];
    test.beforeAll(async ({ browser }) => {
        page = await browser.newPage();
        page.on("pageerror", (error) => errors.push(error.message));
        await page.goto(STEPPED);
        await page.waitForFunction(() => (window as SteppedWindow).stepHud !== undefined);
    });
    test.afterAll(() => page.close());
    test.beforeEach(async () => {
        await page.locator("#speed").fill("1.2");
        await page.getByRole("button", { name: "Amber" }).click();
        await page.evaluate(() => (window as SteppedWindow).resetHud!());
    });

    test("each pointer control leaves the bound orbit key working", async () => {
        await dragRange(page, page.getByRole("slider", { name: "Turn speed" }));
        await gameKeyMovesCamera(page);
        await pointerControl(page, page.getByRole("button", { name: "Cyan" }));
        await gameKeyMovesCamera(page);
    });
    test("a swatch reacquires the bound orbit key after window blur", async () => {
        await page.evaluate(() => {
            (window as unknown as { controlsKey(): number }).controlsKey();
            window.dispatchEvent(new Event("blur"));
        });
        await pointerControl(page, page.getByRole("button", { name: "Cyan" }));
        await gameKeyMovesCamera(page);
    });

    test("the angle the HUD shows is the game's own Turn after each step", async () => {
        let previous: Turn | undefined;
        for (const steps of [1, 29, 300, 37]) {
            const turn = await step(page, steps);
            expect(turn.angle).toBeGreaterThan(previous?.angle ?? 0);
            const degrees = Math.floor((turn.angle * 180) / Math.PI) % 360;
            expect(await page.locator('[data-hud="angle"]').textContent()).toBe(`${degrees}°`);
            previous = turn;
        }
        expect(previous!.angle).toBeGreaterThan(2 * Math.PI);
        expect(errors).toEqual([]);
    });

    test("DOM controls change stepped speed, material colour and the cube centre hue", async () => {
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
        if (process.env.SHALLOT_CAPTURE_DIR)
            await page.screenshot({ path: `${process.env.SHALLOT_CAPTURE_DIR}/hud-blue-2.4.png` });
    });
});

test("a touch swatch changes the cube without taking the bound orbit key", async ({ browser }) => {
    const context = await browser.newContext({
        hasTouch: true,
        viewport: { width: 390, height: 844 },
    });
    try {
        const page = await context.newPage();
        await page.goto(STEPPED);
        await page.waitForFunction(() => "controlsKey" in window);
        const material = () => page.evaluate(() => (window as SteppedWindow).materialHud!());
        const before = await material();
        const cyan = page.getByRole("button", { name: "Cyan" });
        const box = (await cyan.boundingBox())!;
        const session = await context.newCDPSession(page);
        await session.send("Input.dispatchTouchEvent", {
            type: "touchStart",
            touchPoints: [{ id: 1, x: box.x + box.width / 2, y: box.y + box.height / 2 }],
        });
        await bodyHasFocus(page);
        await session.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
        await bodyHasFocus(page);
        await expect(cyan).toHaveAttribute("aria-pressed", "true");
        await expect(page.getByRole("button", { name: "Amber" })).toHaveAttribute(
            "aria-pressed",
            "false",
        );
        await gameKeyMovesCamera(page);
        expect(await material()).not.toEqual(before);
    } finally {
        await context.close();
    }
});
