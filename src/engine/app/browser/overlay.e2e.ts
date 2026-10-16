import { expect, test } from "playwright/test";

declare global {
    interface Window {
        overlayObserverCount: number;
    }
}

for (const removal of ["overlay", "ancestor", "shadow-overlay", "shadow-host-ancestor"]) {
    test(`removing the ${removal} disconnects a world-less overlay's resize observer`, async ({
        page,
    }) => {
        await page.addInitScript(() => {
            const active = new Set<ResizeObserver>();
            const NativeResizeObserver = window.ResizeObserver;
            window.ResizeObserver = class extends NativeResizeObserver {
                override observe(target: Element, options?: ResizeObserverOptions) {
                    active.add(this);
                    super.observe(target, options);
                }
                override disconnect() {
                    active.delete(this);
                    super.disconnect();
                }
            };
            Object.defineProperty(window, "overlayObserverCount", { get: () => active.size });
        });
        await page.goto(`/overlay.html?no-world${removal.startsWith("shadow-") ? "&shadow" : ""}`);
        await expect(page.locator("span")).toBeVisible();
        expect(await page.evaluate(() => window.overlayObserverCount)).toBe(1);
        if (removal === "shadow-host-ancestor") {
            await page.locator("section").evaluate((ancestor) => ancestor.remove());
        } else {
            await page
                .locator(removal.endsWith("overlay") ? "span" : "canvas")
                .evaluate((child) => {
                    child.parentElement!.remove();
                });
        }
        await expect.poll(() => page.evaluate(() => window.overlayObserverCount)).toBe(0);
    });
}

for (const deviceScaleFactor of [1, 1.333]) {
    test.describe(`overlay at device scale ${deviceScaleFactor}`, () => {
        test.use({ deviceScaleFactor });
        test("reference text scales with the shortest side and never shrinks below 14px", async ({
            page,
        }) => {
            await page.setViewportSize({ width: 1280, height: 720 });
            await page.goto("/overlay.html");
            const text = page.locator("span");
            await expect(text).toBeVisible();
            const reference = (await text.boundingBox())!;
            expect(reference.height).toBeCloseTo(20, 1);
            for (const [width, height, scale] of [
                [2880, 1620, 1.5],
                [1620, 2880, 1.5],
                [320, 180, 1],
                [1280, 720, 1],
            ]) {
                await page.setViewportSize({ width, height });
                await expect
                    // CSS zoom can round layout to whole rendered pixels on the host.
                    .poll(async () =>
                        Math.abs((await text.boundingBox())!.height - reference.height * scale),
                    )
                    .toBeLessThanOrEqual(1);
                expect(
                    Math.abs((await text.boundingBox())!.width - reference.width * scale),
                ).toBeLessThanOrEqual(1);
                const overlay = (await text.locator("..").boundingBox())!;
                expect(Math.abs(overlay.width - width)).toBeLessThanOrEqual(1);
                expect(Math.abs(overlay.height - height)).toBeLessThanOrEqual(1);
                if (scale === 1) await expect(text).toHaveCSS("font-size", "14px");
                const bounds = (await text.boundingBox())!;
                expect(Math.abs(bounds.x + bounds.width / 2 - width / 2)).toBeLessThanOrEqual(1);
                expect(
                    Math.abs(bounds.y + bounds.height - (height - 16 * scale)),
                ).toBeLessThanOrEqual(1);
            }
            await page.locator("canvas").evaluate((canvas) => {
                canvas.style.width = "2880px";
                canvas.style.height = "1620px";
            });
            await expect
                .poll(async () =>
                    Math.abs((await text.boundingBox())!.height - reference.height * 1.5),
                )
                .toBeLessThanOrEqual(1);
        });
    });
}

test("an unscaled overlay keeps its CSS size on a large canvas", async ({ page }) => {
    await page.setViewportSize({ width: 2880, height: 1620 });
    await page.goto("/overlay.html?unscaled");
    const text = page.locator("span");
    await expect(text).toBeVisible();
    await expect(text).toHaveCSS("font-size", "14px");
    expect((await text.boundingBox())!.height).toBeCloseTo(20, 1);
});
