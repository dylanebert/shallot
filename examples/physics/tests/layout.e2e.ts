import { holdFrames, pipelineWarnings } from "@dylanebert/shallot/testing";
import { type Browser, expect, type Page, test } from "playwright/test";
import { formatValue } from "../src/control-format";
import { categories, type Scene, scenes } from "../src/scenes";

test.use(pipelineWarnings);

// Each group's read-only claims share one page whose frames are held: scene switches and DOM reads need no
// draw, and every action on a running scene would otherwise wait on one.
async function heldPage(browser: Browser, options: Parameters<Browser["newContext"]>[0] = {}) {
    const page = await (await browser.newContext(options)).newPage();
    const session = await page.context().newCDPSession(page);
    await session.send("Emulation.setHardwareConcurrencyOverride", { hardwareConcurrency: 1 });
    await holdFrames(
        page,
        "body:not(:has([data-shallot-loading])) [aria-label='Scene navigation']",
    );
    await page.goto("/");
    return page;
}

let desktop: Page | undefined;
async function desktopPage(browser: Browser) {
    return (desktop ??= await heldPage(browser, { viewport: { width: 1280, height: 720 } }));
}
test.afterAll(async () => {
    await desktop?.context().close();
    desktop = undefined;
});

test.describe(() => {
    let page: Page;
    test.beforeAll(async ({ browser }) => {
        page = await desktopPage(browser);
    });
    // The Wheel claim leaves the scene list open.
    test.beforeEach(() => page.keyboard.press("Escape"));
    for (const [scene, selectors] of [
        [
            "Wheel",
            [
                ".panel-title",
                ".section-title",
                ".row > label",
                ".toggle span",
                "output",
                "nav button",
                "nav .name",
                ".categories h2",
                ".categories button",
            ],
        ],
        ["Contact", [".input-help"]],
        ["Motor Joint", [".fields button"]],
    ] as const) {
        test(`${scene} control kinds render only shipped fonts`, async () => {
            const trigger = page.getByRole("button", { name: "Scene", exact: true });
            await trigger.click();
            await page.getByRole("button", { name: scene, exact: true }).click();
            if (scene === "Wheel") await trigger.click();
            await page.evaluate(() => document.fonts.ready);
            const cdp = await page.context().newCDPSession(page);
            await cdp.send("DOM.enable");
            await cdp.send("CSS.enable");
            const { root } = await cdp.send("DOM.getDocument");
            async function shippedFont(selector: string) {
                const { nodeId } = await cdp.send("DOM.querySelector", {
                    nodeId: root.nodeId,
                    selector,
                });
                expect(nodeId, selector).not.toBe(0);
                const { fonts } = await cdp.send("CSS.getPlatformFontsForNode", { nodeId });
                expect(fonts.length, selector).toBeGreaterThan(0);
                for (const font of fonts) {
                    expect(font.isCustomFont, `${selector}: ${font.familyName}`).toBe(true);
                    expect(["IBM Plex Sans", "JetBrains Mono"]).toContain(font.familyName);
                }
            }
            for (const selector of selectors) await shippedFont(selector);
            await cdp.detach();
        });
    }
});

for (const viewport of [
    { width: 1280, height: 720 },
    { width: 390, height: 844 },
]) {
    test.describe(`${viewport.width}`, () => {
        const phone = viewport.width === 390;
        let page: Page;
        test.beforeAll(async ({ browser }) => {
            page = phone
                ? await heldPage(browser, { viewport, hasTouch: true })
                : await desktopPage(browser);
        });
        test.afterAll(async () => {
            if (phone) await page.context().close();
        });
        test.beforeEach(() => page.keyboard.press("Escape"));
        const samples = categories.flatMap((category) => category.scenes);
        for (let start = 0; start < samples.length; start += 2) {
            const group = samples.slice(start, start + 2);
            test(`${group.map((sample) => sample.name).join(", ")}: fixed allocations and label fit`, async () => {
                const trigger = page.getByRole("button", { name: "Scene", exact: true });
                const arrows = page.locator("nav button:not(.name)");
                await expect(trigger).toBeVisible();
                const initial = await arrows.evaluateAll((nodes) =>
                    nodes.map((node) => node.getBoundingClientRect().toJSON()),
                );
                for (const sample of group) {
                    await trigger.click();
                    await page.getByRole("button", { name: sample.name, exact: true }).click();
                    expect(
                        await arrows.evaluateAll((nodes) =>
                            nodes.map((node) => node.getBoundingClientRect().toJSON()),
                        ),
                    ).toEqual(initial);
                    const panel = page.locator(".controls");
                    if (!scenes[sample.id as Scene].controls) continue;
                    const geometry = await panel.evaluate((panel) => {
                        const fits = Array.from(
                            panel.querySelectorAll(".row label, .toggle span, output"),
                            (node) => {
                                const range = document.createRange();
                                range.selectNodeContents(node);
                                return (
                                    range.getBoundingClientRect().width <=
                                    node.getBoundingClientRect().width
                                );
                            },
                        );
                        return {
                            box: panel.getBoundingClientRect().toJSON(),
                            rows: Array.from(
                                panel.querySelectorAll(".row"),
                                (row) => row.getBoundingClientRect().height,
                            ),
                            fits: fits.every(Boolean),
                            overflow: panel.scrollWidth > panel.clientWidth,
                            scrolls: panel.scrollHeight > panel.clientHeight,
                        };
                    });
                    expect(geometry.rows.every((height) => height === (phone ? 44 : 32))).toBe(
                        true,
                    );
                    const box = geometry.box;
                    expect(box.width).toBe(Math.min(360, viewport.width - 32));
                    expect(box.y).toBe(phone ? 16 : 24);
                    expect(box.x + box.width).toBe(viewport.width - (phone ? 16 : 24));
                    expect(box.height).toBeLessThanOrEqual(
                        phone ? viewport.height * 0.6 : viewport.height - 112,
                    );
                    expect(geometry.overflow).toBe(false);
                    expect(geometry.fits, sample.name).toBe(true);
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
                                    const width = output.getBoundingClientRect().width;
                                    return values.every(
                                        (value) => context.measureText(value).width <= width,
                                    );
                                });
                            }, values),
                        ).toBe(true);
                    }
                    if (sample.id === "wheel" && phone) {
                        expect(geometry.scrolls).toBe(true);
                        await panel.locator("output").last().scrollIntoViewIfNeeded();
                        await expect(panel.locator("output").last()).toBeInViewport();
                    }
                    if (sample.id === "contact") {
                        const help = (await page
                            .getByText("WASD to roll", { exact: true })
                            .boundingBox())!;
                        expect([help.x, help.y]).toEqual([phone ? 16 : 24, phone ? 16 : 24]);
                    }
                }
            });
        }
        test("uncropped pointer picker and safe areas", async () => {
            const trigger = page.getByRole("button", { name: "Scene", exact: true });
            await expect(trigger).toBeVisible();
            const initial = await page
                .locator("nav button:not(.name)")
                .evaluateAll((nodes) => nodes.map((node) => node.getBoundingClientRect().toJSON()));
            await trigger.click();
            expect(await page.evaluate(() => document.activeElement === document.body)).toBe(true);
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
            expect(await page.evaluate(() => document.activeElement === document.body)).toBe(true);
            await trigger.click();
            await page.mouse.click(4, 4);
            await expect(list).toBeHidden();
            expect(await page.evaluate(() => document.activeElement === document.body)).toBe(true);
            expect(
                await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
            ).toBe(true);
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
        });
    });
}
