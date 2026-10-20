import { expect, test } from "playwright/test";
import { formatValue } from "../src/control-format";
import { categories, type Scene, scenes } from "../src/scenes";

for (const viewport of [
    { width: 1280, height: 720 },
    { width: 390, height: 844 },
]) {
    test(`fixed allocations, scrolling and uncropped picker at ${viewport.width}`, async ({
        browser,
    }) => {
        const phone = viewport.width === 390;
        const context = await browser.newContext({ viewport, hasTouch: phone });
        const page = await context.newPage();
        await page.goto("/");
        const trigger = page.getByRole("button", { name: "Scene", exact: true });
        const arrows = page.locator("nav button:not(.name)");
        await expect(trigger).toBeVisible();
        const cdp = await context.newCDPSession(page);
        await cdp.send("DOM.enable");
        await cdp.send("CSS.enable");
        async function shippedFonts() {
            await page.evaluate(() => document.fonts.ready);
            const { root } = await cdp.send("DOM.getDocument");
            const { nodeIds } = await cdp.send("DOM.querySelectorAll", {
                nodeId: root.nodeId,
                selector: ".ec, .ec *",
            });
            for (const nodeId of nodeIds) {
                const { fonts } = await cdp.send("CSS.getPlatformFontsForNode", { nodeId });
                for (const font of fonts) {
                    expect(font.isCustomFont, font.familyName).toBe(true);
                    expect(["IBM Plex Sans", "JetBrains Mono"]).toContain(font.familyName);
                }
            }
        }
        const initial = await arrows.evaluateAll((nodes) =>
            nodes.map((node) => node.getBoundingClientRect().toJSON()),
        );
        for (const sample of categories.flatMap((category) => category.scenes)) {
            await trigger.click();
            await page.getByRole("button", { name: sample.name, exact: true }).click();
            expect(
                await arrows.evaluateAll((nodes) =>
                    nodes.map((node) => node.getBoundingClientRect().toJSON()),
                ),
            ).toEqual(initial);
            await shippedFonts();
            const panel = page.locator(".controls");
            if (!scenes[sample.id as Scene].controls) continue;
            for (const row of await panel.locator(".row").all())
                expect((await row.boundingBox())!.height).toBe(phone ? 44 : 32);
            const box = (await panel.boundingBox())!;
            expect(box.width).toBe(Math.min(360, viewport.width - 32));
            expect(box.y).toBe(phone ? 16 : 24);
            expect(box.x + box.width).toBe(viewport.width - (phone ? 16 : 24));
            expect(box.height).toBeLessThanOrEqual(
                phone ? viewport.height * 0.6 : viewport.height - 112,
            );
            expect(await panel.evaluate((node) => node.scrollWidth <= node.clientWidth)).toBe(true);
            // Measure every label, and every authored numeric endpoint, not only defaults.
            const fits = await panel
                .locator(".row label, .toggle span, output")
                .evaluateAll((nodes) =>
                    nodes.map((node) => {
                        const range = document.createRange();
                        range.selectNodeContents(node);
                        return (
                            range.getBoundingClientRect().width <=
                            node.getBoundingClientRect().width
                        );
                    }),
                );
            expect(fits.every(Boolean), sample.name).toBe(true);
            for (const slider of await panel.locator("input[type=range]").all()) {
                for (const endpoint of ["min", "max"]) {
                    const output = slider.locator("..").locator("output");
                    const original = await output.textContent();
                    await output.evaluate(
                        (node, text) => {
                            node.textContent = text;
                        },
                        formatValue(Number(await slider.getAttribute(endpoint))),
                    );
                    expect(
                        await output.evaluate((node) => node.scrollWidth <= node.clientWidth),
                    ).toBe(true);
                    await output.evaluate((node, text) => {
                        node.textContent = text;
                    }, original);
                }
            }
            if (sample.id === "wheel" && phone) {
                expect(await panel.evaluate((node) => node.scrollHeight > node.clientHeight)).toBe(
                    true,
                );
                await panel.locator("output").last().scrollIntoViewIfNeeded();
                await expect(panel.locator("output").last()).toBeInViewport();
            }
            if (sample.id === "contact") {
                const help = (await page.getByText("WASD to roll", { exact: true }).boundingBox())!;
                expect([help.x, help.y]).toEqual([phone ? 16 : 24, phone ? 16 : 24]);
            }
        }
        await trigger.focus();
        await page.keyboard.press("Enter");
        const list = page.locator("#scene-list");
        await shippedFonts();
        expect(await list.evaluate((node) => node.scrollWidth <= node.clientWidth)).toBe(true);
        const box = (await list.boundingBox())!;
        expect(box.x).toBeGreaterThanOrEqual(phone ? 16 : 24);
        expect(box.y).toBeGreaterThanOrEqual(phone ? 16 : 24);
        expect(box.x + box.width).toBeLessThanOrEqual(viewport.width - (phone ? 16 : 24));
        expect(box.y + box.height).toBeLessThanOrEqual(initial[0].y);
        await list.getByRole("button").last().scrollIntoViewIfNeeded();
        await expect(list.getByRole("button").last()).toBeInViewport();
        await expect(list.locator(".categories")).toHaveCSS(
            "flex-direction",
            phone ? "column" : "row",
        );
        await page.keyboard.press("Escape");
        await expect(list).toBeHidden();
        await expect(trigger).toBeFocused();
        await page.keyboard.press("Enter");
        await page.mouse.click(4, 4);
        await expect(list).toBeHidden();
        await expect(trigger).toBeFocused();
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(
            true,
        );
        // Substitute host insets in the authored CSS to exercise nonzero safe areas.
        await page.addStyleTag({
            content: await page
                .locator("style")
                .allTextContents()
                .then((styles) =>
                    styles
                        .join("\n")
                        .replaceAll("env(safe-area-inset-top)", "12px")
                        .replaceAll("env(safe-area-inset-right)", "8px")
                        .replaceAll("env(safe-area-inset-bottom)", "12px")
                        .replaceAll("env(safe-area-inset-left)", "8px"),
                ),
        });
        await trigger.click();
        await page.getByRole("button", { name: "Revolute", exact: true }).click();
        const safe = (await page.locator(".controls").boundingBox())!;
        expect(safe.y).toBe((phone ? 16 : 24) + 12);
        expect(safe.x + safe.width).toBe(viewport.width - (phone ? 16 : 24) - 8);
        const navigation = (await page.locator(".picker").boundingBox())!;
        expect(navigation.x).toBe((phone ? 16 : 24) + 8);
        expect(navigation.y + navigation.height).toBe(viewport.height - (phone ? 16 : 24) - 12);
        await cdp.detach();
        await context.close();
    });
}
