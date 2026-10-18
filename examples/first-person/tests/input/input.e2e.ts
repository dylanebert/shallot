import { expect, type Page, test } from "playwright/test";

type CaptureWindow = Window & { __capturedFrames?: Record<string, Uint8ClampedArray> };

async function screenshotCanvas(page: Page, key: string): Promise<void> {
    const bounds = await page.locator("#canvas").boundingBox();
    if (!bounds) throw new Error("first-person canvas has no page bounds");
    const png = await page.screenshot({ clip: bounds });
    await page.evaluate(
        async ({ frameKey, encoded }) => {
            const bitmap = await createImageBitmap(
                new Blob([Uint8Array.from(atob(encoded), (value) => value.charCodeAt(0))], {
                    type: "image/png",
                }),
            );
            const surface = document.createElement("canvas");
            surface.width = bitmap.width;
            surface.height = bitmap.height;
            const context = surface.getContext("2d")!;
            context.drawImage(bitmap, 0, 0);
            bitmap.close();
            ((window as CaptureWindow).__capturedFrames ??= {})[frameKey] = context.getImageData(
                0,
                0,
                surface.width,
                surface.height,
            ).data;
        },
        { frameKey: key, encoded: png.toString("base64") },
    );
}
async function changedFraction(page: Page, first: string, second: string): Promise<number> {
    return page.evaluate(
        ({ first, second }) => {
            const frames = (window as CaptureWindow).__capturedFrames ?? {};
            const a = frames[first],
                b = frames[second];
            if (!a || !b || a.length !== b.length) return 1;
            let changed = 0;
            for (let offset = 0; offset < a.length; offset += 4) {
                const delta = Math.max(
                    Math.abs(a[offset]! - b[offset]!),
                    Math.abs(a[offset + 1]! - b[offset + 1]!),
                    Math.abs(a[offset + 2]! - b[offset + 2]!),
                );
                if (delta > 24) changed++;
            }
            return changed / (a.length / 4);
        },
        { first, second },
    );
}
async function holding<T>(page: Page, key: string, body: () => Promise<T>): Promise<T> {
    await page.keyboard.down(key);
    let result: T;
    try {
        result = await body();
    } catch (error) {
        await page.keyboard.up(key).catch(() => {});
        throw error;
    }
    await page.keyboard.up(key);
    return result;
}
async function captureUntilChange(
    page: Page,
    baseline: string,
    threshold: number,
    deadline: number,
): Promise<{ change: number; samples: number }> {
    let change = 0,
        samples = 0;
    while (Date.now() < deadline) {
        const sample = `${baseline}-sample-${samples++}`;
        await screenshotCanvas(page, sample);
        change = await changedFraction(page, baseline, sample);
        await page.evaluate((key) => {
            delete (window as CaptureWindow).__capturedFrames?.[key];
        }, sample);
        if (change > threshold) break;
    }
    return { change, samples };
}
test.beforeEach(async ({ page }) => {
    const session = await page.context().newCDPSession(page);
    await session.send("Emulation.setHardwareConcurrencyOverride", { hardwareConcurrency: 1 });
});

test.describe("at a quarter size", () => {
    test.use({ viewport: { width: 320, height: 180 } });
    test("focused canvas input presents movement, and input after focus leaves does not", async ({
        page,
    }, testInfo) => {
        const started = Date.now();
        await page.goto("/");
        const canvas = page.locator("#canvas");
        await expect(canvas).toBeVisible();
        await page.waitForFunction(() => document.querySelector("canvas")?.width === 320);
        await screenshotCanvas(page, "idle-before");
        await page.waitForTimeout(350);
        await screenshotCanvas(page, "idle-after");
        const idleChange = await changedFraction(page, "idle-before", "idle-after");
        await canvas.evaluate((element) => {
            element.tabIndex = 0;
        });
        await canvas.click();
        await canvas.focus();
        expect(
            await canvas.evaluate((element) => document.activeElement === element),
            "the real first-person canvas receives browser focus",
        ).toBe(true);
        await page.evaluate(() => document.exitPointerLock());
        await page.waitForFunction(() => document.pointerLockElement === null);
        await page.evaluate(() => {
            const outside = document.createElement("button");
            outside.id = "focus-outside-canvas";
            outside.textContent = "outside";
            outside.style.cssText = "position:fixed;right:0;top:0;z-index:99999";
            document.body.append(outside);
        });
        const outside = page.locator("#focus-outside-canvas");
        await outside.click();
        await outside.focus();
        expect(await outside.evaluate((element) => document.activeElement === element)).toBe(true);
        expect(await canvas.evaluate((element) => document.activeElement !== element)).toBe(true);
        await screenshotCanvas(page, "outside-before");
        await holding(page, "w", async () => {
            await page.waitForTimeout(350);
            await screenshotCanvas(page, "outside-after");
        });
        const unfocusedChange = await changedFraction(page, "outside-before", "outside-after");
        const inputSignal = Math.max(idleChange * 3, 0.02);
        expect(
            unfocusedChange,
            "a W press after canvas focus leaves does not move the first-person scene",
        ).toBeLessThan(inputSignal);
        await canvas.click();
        await canvas.focus();
        expect(
            await canvas.evaluate((element) => document.activeElement === element),
            "the canvas regains browser focus before input",
        ).toBe(true);
        await screenshotCanvas(page, "focused-before");
        const { change: focusedChange, samples: focusedSamples } = await holding(page, "w", () => {
            const deadline = started + testInfo.timeout - 5000;
            return captureUntilChange(page, "focused-before", inputSignal, deadline);
        });
        expect(
            focusedChange,
            `a focused W press changes the rendered scene beyond idle motion (idle ${idleChange}, unfocused ${unfocusedChange}, threshold ${inputSignal}, samples ${focusedSamples})`,
        ).toBeGreaterThan(inputSignal);
    });
});
