import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { Capture } from "@dylanebert/shallot/rendering";
import { expect, type Page, test } from "playwright/test";

const CAPTURE_ENTRY = resolve(
    fileURLToPath(new URL(".", import.meta.url)),
    ".artifacts/capture-page.js",
);

type BrowserDeviceRecord = {
    info: Pick<GPUAdapterInfo, "vendor" | "architecture" | "device" | "description"> & {
        isFallbackAdapter?: boolean;
    };
    requestedFeatures: string[];
    requestedLimits: Record<string, number | undefined>;
    features: string[];
    maxStorageBuffersPerShaderStage: number;
};

type CaptureWindow = Window & {
    shallotCaptureFrame?: (canvas: HTMLCanvasElement) => Promise<Capture>;
    __capturedFrames?: Record<string, Uint8ClampedArray>;
    __firstPersonDeviceRecord?: BrowserDeviceRecord;
};

type Frame = Pick<Capture, "width" | "height" | "identity"> & { scenePixels: number };

async function captureCanvasFrame(page: Page, key: string): Promise<Frame> {
    return page.evaluate(async (frameKey) => {
        const captureWindow = window as CaptureWindow;
        if (!captureWindow.shallotCaptureFrame) throw new Error("captureFrame is not installed");
        const canvas = document.querySelector<HTMLCanvasElement>("#canvas")!;
        const capture = await captureWindow.shallotCaptureFrame(canvas);
        (captureWindow.__capturedFrames ??= {})[frameKey] = capture.rgba;
        const clear = [capture.rgba[0]!, capture.rgba[1]!, capture.rgba[2]!];
        let scenePixels = 0;
        for (let offset = 0; offset < capture.rgba.length; offset += 4) {
            if (
                Math.max(
                    Math.abs(capture.rgba[offset]! - clear[0]!),
                    Math.abs(capture.rgba[offset + 1]! - clear[1]!),
                    Math.abs(capture.rgba[offset + 2]! - clear[2]!),
                ) > 24
            )
                scenePixels++;
        }
        return {
            width: capture.width,
            height: capture.height,
            identity: capture.identity,
            scenePixels,
        } as Frame;
    }, key);
}

async function changedFraction(page: Page, first: string, second: string): Promise<number> {
    return page.evaluate(
        ({ first, second }) => {
            const frames = (window as CaptureWindow).__capturedFrames ?? {};
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

async function captureUntilChange(
    page: Page,
    baseline: string,
    threshold: number,
    timeout: number,
): Promise<{ change: number; samples: number }> {
    const deadline = Date.now() + timeout;
    let change = 0;
    let samples = 0;
    while (Date.now() < deadline) {
        const sample = `${baseline}-sample-${samples++}`;
        await captureCanvasFrame(page, sample);
        change = await changedFraction(page, baseline, sample);
        await page.evaluate((key) => {
            delete (window as CaptureWindow).__capturedFrames?.[key];
        }, sample);
        if (change > threshold) break;
    }
    return { change, samples };
}

test("the preview serves its isolated page and focused canvas input presents movement", async ({
    page,
}, testInfo) => {
    const pageErrors: string[] = [];
    page.on("pageerror", (error) => pageErrors.push(error.message));
    await page.addInitScript(() => {
        const original = GPUAdapter.prototype.requestDevice;
        GPUAdapter.prototype.requestDevice = async function (descriptor?: GPUDeviceDescriptor) {
            const info = this.info as GPUAdapterInfo & { isFallbackAdapter?: boolean };
            const device = await original.call(this, descriptor);
            (window as CaptureWindow).__firstPersonDeviceRecord = {
                info: {
                    vendor: info.vendor,
                    architecture: info.architecture,
                    device: info.device,
                    description: info.description,
                    isFallbackAdapter: info.isFallbackAdapter,
                },
                requestedFeatures: [...(descriptor?.requiredFeatures ?? [])].sort(),
                requestedLimits: { ...(descriptor?.requiredLimits ?? {}) },
                features: [...device.features]
                    .filter((feature) => feature !== "core-features-and-limits")
                    .sort(),
                maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
            };
            return device;
        };
    });
    const response = await page.goto("/");
    expect(response?.status()).toBe(200);
    expect(response?.headers()["cross-origin-opener-policy"]).toBe("same-origin");
    expect(response?.headers()["cross-origin-embedder-policy"]).toBe("require-corp");
    expect(await page.evaluate(() => crossOriginIsolated)).toBe(true);
    expect(await page.title()).toBe("First Person");
    expect(await page.locator('meta[name="description"]').getAttribute("content")).toContain(
        "first-person ascent",
    );
    const icon = await page.request.get(new URL("icon.svg", page.url()).toString());
    expect(icon.status()).toBe(200);
    expect(icon.headers()["content-type"]).toContain("image/svg+xml");

    const canvas = page.locator("#canvas");
    await expect(canvas).toBeVisible();
    await expect(page.locator("[data-recipe-controls]")).toContainText("WASD");
    expect(pageErrors, "first-person reaches its first draw without a runtime error").toEqual([]);
    const device = await page.evaluate(() => (window as CaptureWindow).__firstPersonDeviceRecord);
    if (!device) throw new Error("first-person browser did not record its presenting device");
    expect(device.info).toHaveProperty("vendor");
    console.info(`First-person browser device: ${JSON.stringify(device)}`);
    const floorFeatures = ["indirect-first-instance", "rg11b10ufloat-renderable"];
    expect(device.requestedFeatures).toEqual(floorFeatures);
    expect(device.requestedLimits).not.toHaveProperty("maxStorageBuffersPerShaderStage");
    expect(device.features).toEqual(floorFeatures);
    expect(device.maxStorageBuffersPerShaderStage).toBe(8);
    await page.addScriptTag({ path: CAPTURE_ENTRY });
    const first = await captureCanvasFrame(page, "idle-before");
    expect(first.width).toBe(1280);
    expect(first.height).toBe(720);
    expect(first.identity).toMatchObject({
        width: 1280,
        height: 720,
        deviceScale: 1,
        surface: "final-canvas",
        encoding: "rgba8-tight",
    });
    expect(
        first.scenePixels,
        "captureFrame includes the rendered scene beyond its clear color",
    ).toBeGreaterThan(0);
    await page.waitForTimeout(700);
    await captureCanvasFrame(page, "idle-after");
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
    await captureCanvasFrame(page, "outside-before");
    await page.keyboard.down("w");
    try {
        await page.waitForTimeout(700);
        await captureCanvasFrame(page, "outside-after");
    } finally {
        await page.keyboard.up("w");
    }
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
    await captureCanvasFrame(page, "focused-before");
    await page.keyboard.down("w");
    let focusedChange = 0;
    let focusedSamples = 0;
    try {
        const remaining = Math.max(0, testInfo.timeout - testInfo.duration);
        const result = await captureUntilChange(page, "focused-before", inputSignal, remaining);
        focusedChange = result.change;
        focusedSamples = result.samples;
    } finally {
        await page.keyboard.up("w");
    }
    expect(
        focusedChange,
        `a focused W press changes the rendered scene beyond idle motion (idle ${idleChange}, unfocused ${unfocusedChange}, threshold ${inputSignal}, samples ${focusedSamples})`,
    ).toBeGreaterThan(inputSignal);
});
