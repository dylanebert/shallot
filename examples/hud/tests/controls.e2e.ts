import { holdFrames, pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, type Page, test } from "playwright/test";
import { dragRange, pointerControl, tabSkipsChrome } from "./pointer";

test.use(pipelineWarnings);

test("the first frame has no late-pipeline warning", async ({ page }) => {
    await page.goto("/");
    await page.waitForFunction(() =>
        performance
            .getEntriesByType("measure")
            .some((entry) => entry.name.endsWith(":first-update")),
    );
});

test.describe(() => {
    let page: Page;
    let frames: Awaited<ReturnType<typeof holdFrames>>;
    const errors: string[] = [];
    test.beforeAll(async ({ browser }) => {
        page = await browser.newPage({ viewport: { width: 1280, height: 720 }, hasTouch: true });
        page.on("pageerror", (error) => errors.push(error.message));
        frames = await holdFrames(page, "[aria-label='Cube controls']");
        await page.goto("/");
        await page.getByRole("region", { name: "Cube controls" }).waitFor();
    });
    test.afterAll(() => page.close());

    test.describe("the built game's overlay", () => {
        test("the controls float centered along the canvas's bottom edge", async () => {
            const controls = page.getByRole("region", { name: "Cube controls" });
            await controls.waitFor();
            const canvas = await page.locator("canvas").boundingBox();
            const hud = await controls.boundingBox();
            expect(canvas && hud).toBeTruthy();
            expect(hud!.x + hud!.width / 2).toBeCloseTo(canvas!.x + canvas!.width / 2, 0);
            expect(hud!.y).toBeGreaterThan(canvas!.y + canvas!.height / 2);
            expect(hud!.y + hud!.height).toBeLessThanOrEqual(canvas!.y + canvas!.height);
            expect(errors).toEqual([]);
        });
        test("every overlay control fits a 390x844 touch viewport", async () => {
            await page.setViewportSize({ width: 390, height: 844 });
            const controls = page.getByRole("region", { name: "Cube controls" });
            await expect(controls).toBeVisible();
            const bounds = await controls.evaluate((element) =>
                [element, ...element.querySelectorAll("*")]
                    .filter((element) => element.checkVisibility())
                    .map((element) => {
                        const { left, top, right, bottom } = element.getBoundingClientRect();
                        return { left, top, right, bottom };
                    }),
            );
            expect(bounds.length).toBeGreaterThan(0);
            for (const rect of bounds) {
                expect(rect.left).toBeGreaterThanOrEqual(0);
                expect(rect.top).toBeGreaterThanOrEqual(0);
                expect(rect.right).toBeLessThanOrEqual(390);
                expect(rect.bottom).toBeLessThanOrEqual(844);
            }
        });
    });

    test("dragging compact HUD labels selects no page text", async () => {
        await page.setViewportSize({ width: 1280, height: 720 });
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
        test(`compact HUD keeps its geometry without keyboard focus at ${viewport.width}px`, async () => {
            await page.setViewportSize(viewport);
            await page.getByRole("button", { name: "Amber" }).click();
            await page.getByRole("slider", { name: "Turn speed" }).fill("1.2");
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
            expect(viewport.height - bounds.y - bounds.height).toBe(
                viewport.width === 390 ? 16 : 24,
            );
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
});
