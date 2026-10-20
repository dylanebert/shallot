import { expect, test } from "playwright/test";

for (const viewport of [
    { width: 1280, height: 720 },
    { width: 390, height: 844 },
]) {
    test(`compact HUD keeps its geometry and keyboard access at ${viewport.width}px`, async ({
        page,
    }) => {
        await page.setViewportSize(viewport);
        await page.goto("/");
        const panel = page.getByRole("region", { name: "Time of day" });
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
        for (const selector of ["label", "output"]) {
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
        const slider = page.getByRole("slider", { name: "Hour" });
        await slider.fill("0");
        await slider.focus();
        await page.keyboard.press("ArrowRight");
        expect(Number(await slider.inputValue())).toBeGreaterThan(0);
        await expect(slider).toBeFocused();
        expect(
            await slider.evaluate((el) => {
                const css = getComputedStyle(el);
                return [
                    el.matches(":focus-visible"),
                    css.outlineWidth,
                    css.outlineColor,
                    css.outlineOffset,
                ];
            }),
        ).toEqual([true, "2px", "rgb(212, 149, 96)", "2px"]);
        await expect(panel.locator("output")).toHaveText(/^\d{2}:\d{2}$/);
        await slider.fill("23.99");
        await expect(panel.locator("output")).toHaveText(/^\d{2}:\d{2}$/);
        expect(
            await controls.evaluateAll((elements) =>
                elements.map((el) => {
                    const { x, y, width, height } = el.getBoundingClientRect();
                    return { x, y, width, height };
                }),
            ),
        ).toEqual(before);
        expect(await panel.boundingBox()).toEqual(bounds);
        await page.evaluate(() => (document.activeElement as HTMLElement)?.blur());
        await page.mouse.move(0, 0);
        expect(await panel.locator(":focus-visible, :hover").count()).toBe(0);
        if (process.env.SHALLOT_CONTROLS_CAPTURE_DIR)
            await page.screenshot({
                path: `${process.env.SHALLOT_CONTROLS_CAPTURE_DIR}/${viewport.width}.png`,
            });
    });
}
