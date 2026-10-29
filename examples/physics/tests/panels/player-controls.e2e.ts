import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";
import { pointerControl } from "../pointer";

test.use(pipelineWarnings);

type State = {
    grabbed: number;
    position: { x: number; y: number; z: number };
    keys: Record<"W" | "Space" | "Shift", number>;
    yaw: number;
    pan: number[];
    distance: number;
};

for (const width of [1280, 390]) {
    test(`player look and controls stay on the follow camera at ${width}`, async ({
        browser,
    }, info) => {
        const context = await browser.newContext({
            viewport: { width, height: width === 390 ? 844 : 720 },
            hasTouch: width === 390,
        });
        const page = await context.newPage();
        try {
            await page.goto(`${info.project.use.baseURL}/tests/controls.html`);
            await page.waitForFunction(() => "playerState" in window);
            await pointerControl(page, page.getByRole("button", { name: "Scene", exact: true }));
            await pointerControl(
                page,
                page.getByRole("button", { name: "Geometric Mover", exact: true }),
            );
            const state = (steps = 0) =>
                page.evaluate(
                    (steps) =>
                        (window as unknown as { playerState(steps: number): State }).playerState(
                            steps,
                        ),
                    steps,
                );
            await state(2);
            const before = await state();
            if (width === 1280) {
                await page.mouse.click(640, 400);
                await expect
                    .poll(() => page.evaluate(() => !!document.pointerLockElement))
                    .toBe(true);
                await page.mouse.move(700, 420);
                await state(1);
                expect((await state()).yaw).not.toBe(before.yaw);
                await page.keyboard.down("w");
                await state(20);
                await page.keyboard.up("w");
                await state(1);
                await page.evaluate(() => document.exitPointerLock());
                await expect
                    .poll(() => page.evaluate(() => !!document.pointerLockElement))
                    .toBe(false);
            } else {
                const session = await context.newCDPSession(page);
                const points = [
                    { id: 1, x: 80, y: 600 },
                    { id: 2, x: 280, y: 450 },
                ];
                await session.send("Input.dispatchTouchEvent", {
                    type: "touchStart",
                    touchPoints: points,
                });
                await state(1);
                points[0]!.y = 530;
                points[1]!.x = 330;
                await session.send("Input.dispatchTouchEvent", {
                    type: "touchMove",
                    touchPoints: points,
                });
                await state(20);
                expect((await state()).yaw).not.toBe(before.yaw);
                expect((await state()).pan).toEqual(before.pan);
                expect((await state()).distance).toBe(before.distance);
                await expect(page.locator("[data-touch-stick]")).toBeVisible();
                const jump = page.getByRole("button", { name: "Jump", exact: true });
                const sprint = page.getByRole("button", { name: "Sprint", exact: true });
                await expect(jump).toBeVisible();
                await expect(sprint).toBeVisible();
                for (const button of [jump, sprint]) {
                    const box = await button.boundingBox();
                    expect(box!.width).toBeGreaterThanOrEqual(44);
                    expect(box!.height).toBeGreaterThanOrEqual(44);
                    expect(box!.x + box!.width).toBeLessThanOrEqual(390);
                    expect(box!.y + box!.height).toBeLessThan(780);
                }
                const jumpBox = (await jump.boundingBox())!;
                const sprintBox = (await sprint.boundingBox())!;
                points.push({ id: 3, x: jumpBox.x + 32, y: jumpBox.y + 32 });
                await session.send("Input.dispatchTouchEvent", {
                    type: "touchStart",
                    touchPoints: points,
                });
                await state(1);
                expect((await state()).keys.Space).toBe(1);
                points.push({ id: 4, x: sprintBox.x + 32, y: sprintBox.y + 32 });
                await session.send("Input.dispatchTouchEvent", {
                    type: "touchStart",
                    touchPoints: points,
                });
                await state(1);
                expect((await state()).keys.Shift).toBe(1);
                await session.send("Input.dispatchTouchEvent", {
                    type: "touchEnd",
                    touchPoints: [],
                });
                await state(1);
                await expect(page.locator("[data-touch-stick]")).toBeHidden();
                expect((await state()).keys.W).toBe(0);
                await state(1);
                expect((await state()).keys.Space).toBe(0);
                expect((await state()).keys.Shift).toBe(0);
                await page.keyboard.down("w");
                await state(1);
                expect((await state()).keys.W).toBe(1);
                await page.keyboard.up("w");
                await state(1);
                await session.detach();
            }
            const after = await state();
            expect(after.grabbed).toBe(0);
            expect(
                Math.hypot(
                    after.position.x - before.position.x,
                    after.position.z - before.position.z,
                ),
            ).toBeGreaterThan(0.1);
            expect(await page.evaluate(() => document.activeElement === document.body)).toBe(true);
        } finally {
            await context.close();
        }
    });
}
