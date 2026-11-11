import { holdFrames, pipelineWarnings } from "@dylanebert/shallot/testing";
import { type ElementHandle, expect, type Page, test } from "playwright/test";
import { categories } from "../src/scenes";
import { scenePixels } from "./scenes";

test.use({ ...pipelineWarnings, deviceScaleFactor: 1 });

for (const [width, height, columns] of [
    [1280, 720, 6],
    [1512, 982, 7],
    [390, 844, 1],
]) {
    test.describe(`${width}x${height}`, () => {
        const phone = width === 390;
        test.use({ viewport: { width, height }, hasTouch: phone, isMobile: phone });
        test("compact categories fit the viewport and present the initial scene", async ({
            page,
        }) => {
            const errors: string[] = [];
            page.on("pageerror", (error) => errors.push(error.message));
            page.on("console", (message) => {
                if (message.type() === "error") errors.push(message.text());
            });
            const frames = await holdFrames(page, "[aria-label='Scene']");
            await page.goto("/");
            const picker = page.getByRole("button", { name: "Scene", exact: true });
            await expect(picker).toHaveText(
                categories.flatMap((category) => category.scenes)[0].name,
            );
            await frames.run(2);
            await expect
                .poll(async () => {
                    expect(errors).toEqual([]);
                    await frames.run();
                    return scenePixels(page);
                })
                .toBeGreaterThan(100);

            if (phone) await picker.tap();
            else await picker.click();
            await expect
                .poll(() =>
                    page
                        .locator(".categories")
                        .evaluate((node) => Number(getComputedStyle(node).columnCount)),
                )
                .toBe(columns);
            const sections = await page.locator(".categories section").evaluateAll((nodes) =>
                nodes.map((node) => ({
                    fragments: node.getClientRects().length,
                    x: Math.round(node.getBoundingClientRect().x),
                    breakInside: getComputedStyle(node).breakInside,
                })),
            );
            expect(
                sections.every(
                    (section) => section.fragments === 1 && section.breakInside === "avoid",
                ),
            ).toBe(true);
            expect(new Set(sections.map((section) => section.x)).size).toBe(columns);
            const category = categories[categories.length - 1]!;
            const scene = category.scenes[0]!;
            const choice = page
                .getByRole("region", { name: category.name, exact: true })
                .getByRole("button", { name: scene.id, exact: true });
            if (phone) await choice.tap();
            else await choice.click();
            await expect(picker).toHaveText(scene.id);
            await expect(picker).toHaveAttribute("aria-expanded", "false");
            await frames.run(2);
            expect(
                await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
            ).toBe(true);
            await expect(page.getByRole("slider")).toHaveCount(0);
            expect(errors).toEqual([]);
        });
    });
}

test.describe("every model switches on the same phone canvas", () => {
    test.describe.configure({ mode: "serial" });
    let page: Page;
    let frames: Awaited<ReturnType<typeof holdFrames>>;
    let canvas: ElementHandle<SVGElement | HTMLElement> | null;
    const errors: string[] = [];
    test.beforeAll(async ({ browser }) => {
        page = await browser.newPage({
            baseURL: test.info().project.use.baseURL,
            viewport: { width: 390, height: 844 },
            deviceScaleFactor: 1,
            hasTouch: true,
            isMobile: true,
        });
        page.on("pageerror", (error) => errors.push(error.message));
        page.on("console", (message) => {
            if (message.type() === "error") errors.push(message.text());
        });
        frames = await holdFrames(page, "[aria-label='Scene']");
        await page.goto("/");
        await expect(page.getByRole("button", { name: "Scene", exact: true })).toHaveText(
            categories.flatMap((category) => category.scenes)[0].name,
        );
        canvas = await page.locator("canvas").elementHandle();
    });
    test.afterAll(() => page.close());
    test.afterEach(() => expect(errors).toEqual([]));

    for (const category of categories) {
        test(`${category.name} models switch on the same canvas without horizontal scroll`, async () => {
            const picker = page.getByRole("button", { name: "Scene", exact: true });
            for (const scene of category.scenes) {
                await picker.tap();
                await page
                    .getByRole("region", { name: category.name, exact: true })
                    .getByRole("button", { name: scene.id, exact: true })
                    .tap();
                await expect(picker).toHaveText(scene.id);
                await expect(picker).toHaveAttribute("aria-expanded", "false");
                await frames.run(2);
                if (scene.id === "TextureCoordinateTest" || scene.id === "CesiumMan") {
                    await expect.poll(() => scenePixels(page, true)).toBeGreaterThan(100);
                }
                expect(
                    await page
                        .locator("canvas")
                        .evaluate((current, original) => current === original, canvas),
                ).toBe(true);
                expect(
                    await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
                ).toBe(true);
                await expect(page.getByRole("slider")).toHaveCount(0);
            }
        });
    }
});
