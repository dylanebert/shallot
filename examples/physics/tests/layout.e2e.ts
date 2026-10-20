import { expect, test } from "playwright/test";
import { formatValue } from "../src/control-format";
import { categories, type Scene, scenes } from "../src/scenes";

test("distinct control kinds render only shipped fonts", async ({ page, context }) => {
    await page.goto("/");
    const trigger = page.getByRole("button", { name: "Scene", exact: true });
    await trigger.click();
    await page.getByRole("button", { name: "Wheel", exact: true }).click();
    await trigger.click();
    await page.evaluate(() => document.fonts.ready);
    const cdp = await context.newCDPSession(page);
    await cdp.send("DOM.enable");
    await cdp.send("CSS.enable");
    async function shippedFont(selector: string) {
        const { root } = await cdp.send("DOM.getDocument");
        const { nodeId } = await cdp.send("DOM.querySelector", { nodeId: root.nodeId, selector });
        expect(nodeId, selector).not.toBe(0);
        const { fonts } = await cdp.send("CSS.getPlatformFontsForNode", { nodeId });
        expect(fonts.length, selector).toBeGreaterThan(0);
        for (const font of fonts) {
            expect(font.isCustomFont, `${selector}: ${font.familyName}`).toBe(true);
            expect(["IBM Plex Sans", "JetBrains Mono"]).toContain(font.familyName);
        }
    }
    for (const selector of [
        ".panel-title",
        ".section-title",
        ".row > label",
        ".toggle span",
        "output",
        "nav button",
        "nav .name",
        ".categories h2",
        ".categories button",
    ])
        await shippedFont(selector);
    await trigger.click();
    for (const [scene, selector] of [
        ["Contact", ".input-help"],
        ["Motor Joint", ".fields button"],
    ]) {
        await trigger.click();
        await page.getByRole("button", { name: scene, exact: true }).click();
        await shippedFont(selector!);
    }
    await cdp.detach();
});

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
            const panel = page.locator(".controls");
            if (!scenes[sample.id as Scene].controls) continue;
            expect(
                await panel
                    .locator(".row")
                    .evaluateAll(
                        (rows, height) =>
                            rows.every((row) => row.getBoundingClientRect().height === height),
                        phone ? 44 : 32,
                    ),
            ).toBe(true);
            const box = (await panel.boundingBox())!;
            expect(box.width).toBe(Math.min(360, viewport.width - 32));
            expect(box.y).toBe(phone ? 16 : 24);
            expect(box.x + box.width).toBe(viewport.width - (phone ? 16 : 24));
            expect(box.height).toBeLessThanOrEqual(
                phone ? viewport.height * 0.6 : viewport.height - 112,
            );
            expect(await panel.evaluate((node) => node.scrollWidth <= node.clientWidth)).toBe(true);
            // Labels retain scene-specific nesting and allocations.
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
            if (sample.id === "wheel") {
                const values = Object.values(scenes).flatMap((scene) =>
                    Object.values(scene.controls ?? {}).flatMap((control) =>
                        control.min === undefined
                            ? []
                            : [control.min, control.max!, control.value].map(formatValue),
                    ),
                );
                expect(
                    await panel.locator("output").evaluateAll((outputs, values) => {
                        const context = document.createElement("canvas").getContext("2d")!;
                        return outputs.every((output) => {
                            context.font = getComputedStyle(output).font;
                            return values.every(
                                (value) =>
                                    context.measureText(value).width <=
                                    output.getBoundingClientRect().width,
                            );
                        });
                    }, values),
                ).toBe(true);
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
        await context.close();
    });
}
