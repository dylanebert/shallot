import { mkdir } from "node:fs/promises";
import { holdFrames } from "@dylanebert/shallot/testing";
import { expect, type Page, test } from "playwright/test";
import { bodyHasFocus, dragRange, pointerControl, tabSkipsChrome } from "./pointer";

const names = ["Lens distortion", "Chromatic aberration", "Vignette"];

test.describe(() => {
    let page: Page;
    let initialLens: string;
    const errors: string[] = [];
    test.beforeAll(async ({ browser }) => {
        page = await browser.newPage();
        page.on("pageerror", (error) => errors.push(error.message));
        await holdFrames(page, "[aria-label='Post-processing controls']");
        await page.goto("/");
        await page.getByRole("region", { name: "Post-processing controls" }).waitFor();
        initialLens = await page.getByRole("slider").first().inputValue();
    });
    test.afterAll(() => page.close());

    test("the built page presents the scene and keeps its HUD inside the canvas", async () => {
        const panel = page.getByRole("region", { name: "Post-processing controls" });
        await panel.waitFor();
        const canvas = (await page.locator("canvas").boundingBox())!;
        const bounds = (await panel.boundingBox())!;
        expect(bounds.x).toBeGreaterThanOrEqual(canvas.x);
        expect(bounds.x + bounds.width).toBeLessThanOrEqual(canvas.x + canvas.width);
        expect(bounds.y + bounds.height).toBeLessThanOrEqual(canvas.y + canvas.height);
        for (const name of names)
            await expect(page.getByRole("checkbox", { name, exact: true })).toBeChecked();
        if (process.env.SHALLOT_CAPTURE_DIR) {
            await mkdir(process.env.SHALLOT_CAPTURE_DIR, { recursive: true });
            await page.screenshot({
                path: `${process.env.SHALLOT_CAPTURE_DIR}/panel-start-all-on.png`,
            });
        }
        expect(errors).toEqual([]);
    });

    test("dragging panel chrome and labels selects no page text", async () => {
        const label = page.locator(".section-title").first();
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
        test(`controls retain geometry and values without keyboard focus at ${viewport.width}px`, async () => {
            await page.setViewportSize(viewport);
            for (const name of names)
                await page.getByRole("checkbox", { name, exact: true }).check();
            await page.getByRole("slider").first().fill(initialLens);
            const panel = page.getByRole("region", { name: "Post-processing controls" });
            await panel.waitFor();
            await page.evaluate(() => document.fonts.ready);
            expect(
                await page.evaluate(() => [
                    document.fonts.check('400 13px "IBM Plex Sans"'),
                    document.fonts.check('400 12px "JetBrains Mono"'),
                    document.fonts.check('700 15px "JetBrains Mono"'),
                ]),
            ).toEqual([true, true, true]);
            const cdp = await page.context().newCDPSession(page);
            await cdp.send("DOM.enable");
            await cdp.send("CSS.enable");
            const { root } = await cdp.send("DOM.getDocument");
            for (const selector of [".controls", ".section-title input", "summary"]) {
                const { nodeId } = await cdp.send("DOM.querySelector", {
                    nodeId: root.nodeId,
                    selector,
                });
                const { fonts } = await cdp.send("CSS.getPlatformFontsForNode", { nodeId });
                expect(fonts.length, selector).toBeGreaterThan(0);
                for (const font of fonts) {
                    expect(font.isCustomFont, `${selector}: ${font.familyName}`).toBe(true);
                    expect(["IBM Plex Sans", "JetBrains Mono"]).toContain(font.familyName);
                }
            }
            await cdp.detach();
            const bounds = (await panel.boundingBox())!;
            const inset = viewport.width === 390 ? 16 : 24;
            expect(bounds.x).toBe(viewport.width - inset - bounds.width);
            expect(bounds.y).toBe(inset);
            expect(bounds.width).toBe(Math.min(360, viewport.width - 32));
            const rows = panel.locator(".row, .section-title");
            for (const row of await rows.all())
                expect((await row.boundingBox())!.height).toBe(viewport.width === 390 ? 44 : 32);
            expect(await panel.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(true);
            const slider = page
                .getByRole("group", { name: names[0], exact: true })
                .getByRole("slider")
                .first();
            const output = page.locator(`output[for='${await slider.getAttribute("id")}']`);
            await dragRange(page, slider);
            await expect(output).toHaveText(Number(await slider.inputValue()).toFixed(2));
            const retained = await slider.inputValue();
            const toggle = page.getByRole("checkbox", { name: names[0], exact: true });
            await pointerControl(page, toggle);
            await expect(slider).toBeDisabled();
            await slider.click({ force: true });
            await bodyHasFocus(page);
            await expect(slider).toHaveValue(retained);
            await expect(output).toHaveText(Number(retained).toFixed(2));
            expect(await panel.boundingBox()).toEqual(bounds);
            await pointerControl(page, toggle);
            await expect(slider).toBeEnabled();
            await expect(slider).toHaveValue(retained);
            const title = panel.locator("summary");
            await pointerControl(page, title);
            await expect(panel).not.toHaveAttribute("open", "");
            const collapsed = (await panel.boundingBox())!;
            expect({ x: collapsed.x, y: collapsed.y, width: collapsed.width }).toEqual({
                x: bounds.x,
                y: bounds.y,
                width: bounds.width,
            });
            await pointerControl(page, title);
            await expect(panel).toHaveAttribute("open", "");
            await tabSkipsChrome(page);
            if (process.env.SHALLOT_CAPTURE_DIR) {
                await page.evaluate(() => {
                    document.querySelector(".controls")!.scrollTop = 0;
                    (document.activeElement as HTMLElement)?.blur();
                });
                await page.mouse.move(0, 0);
                await expect(title).not.toBeFocused();
                await expect(title).toHaveCSS("border-bottom-color", "rgb(42, 36, 32)");
                await mkdir(process.env.SHALLOT_CAPTURE_DIR, { recursive: true });
                await page.screenshot({
                    path: `${process.env.SHALLOT_CAPTURE_DIR}/${viewport.width}-all-on.png`,
                });
                await toggle.uncheck();
                await toggle.blur();
                await page.mouse.move(0, 0);
                await page.screenshot({
                    path: `${process.env.SHALLOT_CAPTURE_DIR}/${viewport.width}-lens-off.png`,
                });
            }
        });
    }
});
