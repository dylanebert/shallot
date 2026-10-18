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
    timestampsAvailable: boolean;
    requestedFeatures: string[];
    requestedLimits: Record<string, number | undefined>;
    features: string[];
    maxStorageBuffersPerShaderStage: number;
};
type CaptureWindow = Window & {
    shallotCaptureFrame?: (canvas: HTMLCanvasElement) => Promise<Capture>;
    __firstPersonDeviceRecord?: BrowserDeviceRecord;
};
type Frame = Pick<Capture, "width" | "height" | "identity"> & { scenePixels: number };
async function captureCanvasFrame(page: Page): Promise<Frame> {
    return page.evaluate(async () => {
        const captureWindow = window as CaptureWindow;
        if (!captureWindow.shallotCaptureFrame) throw new Error("captureFrame is not installed");
        const capture = await captureWindow.shallotCaptureFrame(
            document.querySelector<HTMLCanvasElement>("#canvas")!,
        );
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
    });
}
test.beforeEach(async ({ page }) => {
    const session = await page.context().newCDPSession(page);
    await session.send("Emulation.setHardwareConcurrencyOverride", { hardwareConcurrency: 1 });
});

test("the preview serves its isolated page and presents its first frame", async ({ page }) => {
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
                timestampsAvailable: this.features.has("timestamp-query"),
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
    expect(await page.title()).toBe("first-person");
    expect(await page.locator('meta[name="description"]').getAttribute("content")).toContain(
        "first-person ascent",
    );
    const icon = await page.request.get(new URL("icon.svg", page.url()).toString());
    expect(icon.status()).toBe(200);
    expect(icon.headers()["content-type"]).toContain("image/svg+xml");
    const canvas = page.locator("#canvas");
    await expect(canvas).toBeVisible();
    await page.waitForFunction(() => document.querySelector("canvas")?.width === 1280);
    await page.addScriptTag({ path: CAPTURE_ENTRY });
    const first = await captureCanvasFrame(page);
    await expect(page.locator("[data-recipe-controls]")).toHaveCount(0);
    expect(pageErrors, "first-person reaches its first draw without a runtime error").toEqual([]);
    const device = await page.evaluate(() => (window as CaptureWindow).__firstPersonDeviceRecord);
    if (!device) throw new Error("first-person browser did not record its presenting device");
    expect(device.info).toHaveProperty("vendor");
    console.info(`First-person browser device: ${JSON.stringify(device)}`);
    const floorFeatures = [
        "indirect-first-instance",
        ...(device.timestampsAvailable ? ["timestamp-query"] : []),
    ];
    expect(device.requestedFeatures).toEqual(floorFeatures);
    expect(device.requestedLimits).not.toHaveProperty("maxStorageBuffersPerShaderStage");
    expect(device.features).toEqual(floorFeatures);
    expect(device.maxStorageBuffersPerShaderStage).toBe(8);
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
});
