import { holdFrames } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";
import { capture } from "./capture";

for (const width of [1280, 390]) {
    test(`Dynamic Mover aiming look review at ${width}`, async ({ browser }, info) => {
        const context = await browser.newContext({
            viewport: { width, height: width === 390 ? 844 : 720 },
            hasTouch: width === 390,
        });
        const page = await context.newPage();
        try {
            const frames = await holdFrames(page, "[aria-label='Scene navigation']");
            await page.goto(`${info.project.use.baseURL}/?scene=dynamicMover`);
            await page.getByRole("button", { name: "Scene", exact: true }).waitFor();
            await page.locator(".controls summary").click();
            await capture(page, `dynamic-mover-${width}-default`, frames);
            if (width === 1280) {
                await page.mouse.move(30, 500);
                await page.mouse.down({ button: "right" });
                await page.waitForFunction(() => document.pointerLockElement !== null);
            } else {
                await page.touchscreen.tap(30, 500);
                await frames.run();
                const button = page.getByRole("button", { name: "Aim", exact: true });
                const box = (await button.boundingBox())!;
                const session = await context.newCDPSession(page);
                await session.send("Input.dispatchTouchEvent", {
                    type: "touchStart",
                    touchPoints: [{ id: 1, x: box.x + box.width / 2, y: box.y + box.height / 2 }],
                });
                await session.detach();
            }
            await frames.run(15);
            await expect(page.locator(".reticle")).toBeVisible();
            await capture(page, `dynamic-mover-${width}-aim`, frames);
        } finally {
            await context.close();
        }
    });
}
