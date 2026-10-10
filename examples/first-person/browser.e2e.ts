import { execFileSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, type Page, test } from "playwright/test";
import { CEILING } from "../../scripts/test-tiers";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const CAPTURE_ENTRY = resolve(ROOT, ".artifacts/capture-page.js");
execFileSync("bun", ["run", "scripts/build-capture-page.ts"], { cwd: ROOT });

interface BrowserCapture {
    rgba: Uint8ClampedArray;
    width: number;
    height: number;
    identity: {
        width: number;
        height: number;
        deviceScale: number;
        surface: string;
        encoding: string;
    };
}

declare global {
    interface Window {
        shallotCaptureFrame?: (canvas: HTMLCanvasElement) => Promise<BrowserCapture>;
        __capturedFrames?: Record<string, Uint8ClampedArray>;
    }
}

async function captureCanvasFrame(page: Page, key: string): Promise<void> {
    const image = await page.evaluate(async (frameKey) => {
        if (!window.shallotCaptureFrame) throw new Error("captureFrame is not installed");
        const canvas = document.querySelector<HTMLCanvasElement>("#canvas")!;
        const capture = await window.shallotCaptureFrame(canvas);
        (window.__capturedFrames ??= {})[frameKey] = capture.rgba;
        return {
            width: capture.width,
            height: capture.height,
            identity: capture.identity,
        };
    }, key);
    expect(image.width, "captureFrame reads the declared canvas width").toBe(1280);
    expect(image.height, "captureFrame reads the declared canvas height").toBe(720);
    expect(image.identity).toMatchObject({
        width: 1280,
        height: 720,
        deviceScale: 1,
        surface: "final-canvas",
        encoding: "rgba8-tight",
    });
}

async function changedFraction(page: Page, first: string, second: string): Promise<number> {
    return page.evaluate(
        ({ first, second }) => {
            const frames = window.__capturedFrames ?? {};
            const a = frames[first];
            const b = frames[second];
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

test("the browser input adapter fails to record a real key press on the focused canvas or accepts it after focus leaves the canvas", async ({
    page,
}) => {
    const pageErrors: string[] = [];
    page.on("pageerror", (error) => pageErrors.push(error.message));
    await page.goto("/");
    const canvas = page.locator("#canvas");
    await expect(canvas).toBeVisible();
    try {
        await expect(page.locator("[data-recipe-controls]")).toContainText("WASD", {
            timeout: CEILING.browser,
        });
    } catch (cause) {
        throw new Error(
            `first-person did not reach its first draw; page errors: ${JSON.stringify(pageErrors)}`,
            { cause },
        );
    }
    expect(pageErrors, "first-person reaches its first draw without a runtime error").toEqual([]);
    await page.addScriptTag({ path: CAPTURE_ENTRY, type: "module" });
    await canvas.evaluate((element) => {
        element.tabIndex = 0;
    });
    await canvas.click();
    await canvas.focus();
    expect(
        await canvas.evaluate((element) => document.activeElement === element),
        "the real first-person canvas receives browser focus",
    ).toBe(true);

    await captureCanvasFrame(page, "idle-before");
    await page.waitForTimeout(700);
    await captureCanvasFrame(page, "idle-after");
    const idleChange = await changedFraction(page, "idle-before", "idle-after");

    await page.evaluate(() => {
        const outside = document.createElement("button");
        outside.id = "focus-outside-canvas";
        outside.textContent = "outside";
        outside.style.cssText = "position:fixed;right:0;top:0;z-index:99999";
        document.body.append(outside);
    });
    const outside = page.locator("#focus-outside-canvas");
    await page.evaluate(() => document.exitPointerLock());
    await page.waitForFunction(() => document.pointerLockElement === null);
    await outside.click();
    await outside.focus();
    expect(
        await outside.evaluate((element) => document.activeElement === element),
        "focus can leave the real canvas",
    ).toBe(true);
    expect(
        await canvas.evaluate((element) => document.activeElement !== element),
        "the canvas is no longer the focused input target",
    ).toBe(true);

    await captureCanvasFrame(page, "outside-before");
    await page.keyboard.down("w");
    await page.waitForTimeout(700);
    await captureCanvasFrame(page, "outside-after");
    await page.keyboard.up("w");
    const unfocusedChange = await changedFraction(page, "outside-before", "outside-after");
    const inputSignal = Math.max(idleChange * 3, 0.02);
    expect(
        unfocusedChange,
        "a W press after canvas focus leaves does not move the first-person scene",
    ).toBeLessThan(inputSignal);

    await canvas.click();
    await canvas.focus();
    await captureCanvasFrame(page, "focused-before");
    await page.keyboard.down("w");
    await page.waitForTimeout(700);
    await captureCanvasFrame(page, "focused-after");
    await page.keyboard.up("w");
    const focusedChange = await changedFraction(page, "focused-before", "focused-after");
    expect(
        focusedChange,
        "a focused W press changes the rendered first-person scene beyond idle motion",
    ).toBeGreaterThan(inputSignal);
});
