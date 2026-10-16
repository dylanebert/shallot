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
                [2880, 1620, 2.25],
                [1620, 2880, 2.25],
                [320, 180, 1],
                [1280, 720, 1],
            ]) {
                await page.setViewportSize({ width, height });
                await expect
                    .poll(async () => (await text.boundingBox())!.height)
                    .toBeCloseTo(reference.height * scale, 1);
                expect((await text.boundingBox())!.width).toBeCloseTo(reference.width * scale, 0);
                const overlay = (await text.locator("..").boundingBox())!;
                expect(overlay.width).toBeCloseTo(width, 0);
                expect(overlay.height).toBeCloseTo(height, 0);
                if (scale === 1) await expect(text).toHaveCSS("font-size", "14px");
                const bounds = (await text.boundingBox())!;
                expect(bounds.x + bounds.width / 2).toBeCloseTo(width / 2, 0);
                expect(bounds.y + bounds.height).toBeCloseTo(height - 16 * scale, 0);
            }
            await page.locator("canvas").evaluate((canvas) => {
                canvas.style.width = "2880px";
                canvas.style.height = "1620px";
            });
            await expect
                .poll(async () => (await text.boundingBox())!.height)
                .toBeCloseTo(reference.height * 2.25, 1);
        });
    });
}
