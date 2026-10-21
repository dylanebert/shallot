import { holdFrames } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";
import {
    bodyHasFocus,
    dragRange,
    gameKeyMovesCamera,
    pointerControl,
    tabSkipsChrome,
} from "./pointer";

test("each pointer control leaves the bound orbit key working", async ({ page }) => {
    await page.goto("http://127.0.0.1:4176/tests/stepped.html");
    await page.waitForFunction(() => "controlsKey" in window);
    await dragRange(page, page.getByRole("slider", { name: "Turn speed" }));
    await gameKeyMovesCamera(page);
    await pointerControl(page, page.getByRole("button", { name: "Cyan" }));
    await gameKeyMovesCamera(page);
});

test("a swatch reacquires the bound orbit key after window blur", async ({ page }) => {
    await page.goto("http://127.0.0.1:4176/tests/stepped.html");
    await page.waitForFunction(() => "controlsKey" in window);
    await page.evaluate(() => {
        (window as unknown as { controlsKey(): number }).controlsKey();
        window.dispatchEvent(new Event("blur"));
    });
    await pointerControl(page, page.getByRole("button", { name: "Cyan" }));
    await gameKeyMovesCamera(page);
});

test("a touch swatch changes the cube without taking the bound orbit key", async ({ browser }) => {
    const context = await browser.newContext({
        hasTouch: true,
        viewport: { width: 390, height: 844 },
    });
    try {
        const page = await context.newPage();
        await page.goto("http://127.0.0.1:4176/tests/stepped.html");
        await page.waitForFunction(() => "controlsKey" in window);
        const material = () =>
            page.evaluate(() => (window as unknown as { materialHud(): number[] }).materialHud());
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

test("dragging compact HUD labels selects no page text", async ({ page }) => {
    await holdFrames(page, "[aria-label='Cube controls']");
    await page.goto("/");
    const label = page.locator(".ec label").first();
    await label.waitFor();
    const box = (await label.boundingBox())!;
    await page.mouse.move(box.x + 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width - 2, box.y + box.height / 2, { steps: 20 });
    await page.mouse.up();
    expect(await page.evaluate(() => getSelection()?.toString())).toBe("");
});

for (const viewport of [
    { width: 1280, height: 720 },
    { width: 390, height: 844 },
]) {
    test(`compact HUD keeps its geometry without keyboard focus at ${viewport.width}px`, async ({
        page,
    }) => {
        await page.setViewportSize(viewport);
        const frames = await holdFrames(page, "[aria-label='Cube controls']");
        await page.goto("/");
        const panel = page.getByRole("region", { name: "Cube controls" });
        await panel.waitFor();
        await page.evaluate(() => document.fonts.ready);
        expect(
            await page.evaluate(() => [
                document.fonts.check('400 13px "IBM Plex Sans"'),
                document.fonts.check('400 12px "JetBrains Mono"'),
            ]),
        ).toEqual([true, true]);
        const cdp = await page.context().newCDPSession(page);
        await cdp.send("DOM.enable");
        await cdp.send("CSS.enable");
        const { root } = await cdp.send("DOM.getDocument");
        for (const selector of ["label", "output", "[data-hud='angle']", "#angle-label"]) {
            const { nodeId } = await cdp.send("DOM.querySelector", {
                nodeId: root.nodeId,
                selector,
            });
            const { fonts } = await cdp.send("CSS.getPlatformFontsForNode", { nodeId });
            expect(fonts.length).toBeGreaterThan(0);
            for (const font of fonts) {
                expect(font.isCustomFont, font.familyName).toBe(true);
                expect(["IBM Plex Sans", "JetBrains Mono"]).toContain(font.familyName);
            }
        }
        await cdp.detach();
        const bounds = (await panel.boundingBox())!;
        expect(bounds.x + bounds.width / 2).toBeCloseTo(viewport.width / 2, 5);
        expect(bounds.y).toBeGreaterThan(viewport.height / 2);
        expect(viewport.height - bounds.y - bounds.height).toBe(viewport.width === 390 ? 16 : 24);
        for (const element of await panel.locator("*").all()) {
            const box = (await element.boundingBox())!;
            expect(box.x).toBeGreaterThanOrEqual(0);
            expect(box.y).toBeGreaterThanOrEqual(0);
            expect(box.x + box.width).toBeLessThanOrEqual(viewport.width);
            expect(box.y + box.height).toBeLessThanOrEqual(viewport.height);
        }
        const controls = panel.locator("button, input");
        const before = await controls.evaluateAll((elements) =>
            elements.map((el) => {
                const { x, y, width, height } = el.getBoundingClientRect();
                return { x, y, width, height };
            }),
        );
        if (viewport.width === 390)
            for (const box of before) {
                expect(box.height).toBeGreaterThanOrEqual(44);
                expect(box.width).toBeGreaterThanOrEqual(44);
            }
        const slider = page.getByRole("slider", { name: "Turn speed" });
        await dragRange(page, slider);
        await expect(panel.locator("output").first()).toHaveText(
            Number(await slider.inputValue()).toFixed(1),
        );
        await pointerControl(page, page.getByRole("button", { name: "Cyan" }));
        await expect(page.getByRole("button", { name: "Cyan" })).toHaveAttribute(
            "aria-pressed",
            "true",
        );
        await expect(page.getByRole("button", { name: "Amber" })).toHaveAttribute(
            "aria-pressed",
            "false",
        );
        await page.mouse.move(0, 0);
        await expect(page.getByRole("button", { name: "Cyan" })).toHaveCSS(
            "border-color",
            "rgb(212, 149, 96)",
        );
        await expect(page.locator('[data-hud="angle"]')).toHaveAttribute(
            "aria-labelledby",
            "angle-label",
        );
        await tabSkipsChrome(page);
        const angle = page.locator('[data-hud="angle"]');
        const shown = await angle.textContent();
        await expect
            .poll(async () => {
                await frames.run();
                return angle.textContent();
            }, "the turning cube must change the angle readout")
            .not.toBe(shown);
        expect(
            await panel
                .locator("output")
                .evaluateAll((elements) => elements.every((el) => el.tabIndex === -1)),
        ).toBe(true);
        expect(
            await controls.evaluateAll((elements) =>
                elements.map((el) => {
                    const { x, y, width, height } = el.getBoundingClientRect();
                    return { x, y, width, height };
                }),
            ),
        ).toEqual(before);
        expect(await panel.boundingBox()).toEqual(bounds);
        await page.mouse.move(0, 0);
        expect(await panel.locator(":focus-visible, :hover").count()).toBe(0);
        if (process.env.SHALLOT_CONTROLS_CAPTURE_DIR)
            await page.screenshot({
                path: `${process.env.SHALLOT_CONTROLS_CAPTURE_DIR}/${viewport.width}.png`,
            });
    });
}
