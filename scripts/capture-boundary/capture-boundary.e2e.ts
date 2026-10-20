import type { Capture } from "@dylanebert/shallot/rendering";
import { expect, type Page, test } from "../browser.fixture";

// One running app at its own output size: 480x270 CSS pixels at pixelRatio 2, a 960x540 backing surface.
// Each claim names what its evidence is. A physical display's presentation has no headless witness; it is
// the display-seat manual oracle's claim, not this check's.

type Frame = { width: number; height: number; scene: number; identity?: Capture["identity"] };
type CaptureWindow = Window & {
    captureFrame(canvas: HTMLCanvasElement): Promise<Capture>;
    sceneFrame(image: ImageBitmap | Capture): Promise<Frame>;
};

async function openApp(page: Page) {
    await page.addInitScript(() => {
        // Pixels that differ from the clear color, so a clear-only frame counts none.
        (window as unknown as CaptureWindow).sceneFrame = async (image) => {
            let rgba: ArrayLike<number>;
            if ("rgba" in image) rgba = image.rgba;
            else {
                const surface = new OffscreenCanvas(image.width, image.height);
                const context = surface.getContext("2d")!;
                context.drawImage(image, 0, 0);
                rgba = context.getImageData(0, 0, image.width, image.height).data;
            }
            const clear = [0x19, 0x18, 0x16];
            let scene = 0;
            for (let i = 0; i < rgba.length; i += 4) {
                if (clear.some((value, channel) => Math.abs(rgba[i + channel]! - value) > 24))
                    scene++;
            }
            const identity = "identity" in image ? image.identity : undefined;
            return { width: image.width, height: image.height, scene, identity };
        };
    });
    await page.goto("/");
    await page.waitForFunction(() => document.querySelector("canvas")!.width === 960);
    return page.evaluate(() => {
        const canvas = document.querySelector("canvas")!;
        return {
            width: canvas.width,
            height: canvas.height,
            cssWidth: canvas.clientWidth,
            cssHeight: canvas.clientHeight,
            deviceScale: window.devicePixelRatio,
        };
    });
}

async function screenshotFrame(page: Page): Promise<Frame> {
    const png = await page.locator("canvas").screenshot({ scale: "device" });
    return page.evaluate(async (base64) => {
        const image = await createImageBitmap(
            new Blob([Uint8Array.from(atob(base64), (value) => value.charCodeAt(0))]),
        );
        try {
            return await (window as unknown as CaptureWindow).sceneFrame(image);
        } finally {
            image.close();
        }
    }, png.toString("base64"));
}

test("a page screenshot is the browser-composited canvas at CSS size times device scale, not its backing surface", async ({
    page,
}) => {
    const geometry = await openApp(page);
    expect(geometry).toEqual({
        width: 960,
        height: 540,
        cssWidth: 480,
        cssHeight: 270,
        deviceScale: 1,
    });
    await expect
        .poll(async () => (await screenshotFrame(page)).scene, {
            message: "the composited page shows the drawn cube",
        })
        .toBeGreaterThan(480 * 270 * 0.05);
    const frame = await screenshotFrame(page);
    expect([frame.width, frame.height]).toEqual([
        geometry.cssWidth * geometry.deviceScale,
        geometry.cssHeight * geometry.deviceScale,
    ]);
});

test("captureFrame reads a running canvas's rendered target at its own backing size and names it", async ({
    page,
}) => {
    const geometry = await openApp(page);
    const frame = await page.evaluate(async () => {
        const captureWindow = window as unknown as CaptureWindow;
        const capture = await captureWindow.captureFrame(document.querySelector("canvas")!);
        if (capture.rgba.length !== capture.width * capture.height * 4)
            throw new Error(`capture is not tight RGBA: ${capture.rgba.length} bytes`);
        return captureWindow.sceneFrame(capture);
    });
    expect([frame.width, frame.height]).toEqual([geometry.width, geometry.height]);
    expect(frame.identity).toMatchObject({
        width: geometry.width,
        height: geometry.height,
        surface: "final-canvas",
        encoding: "rgba8-tight",
    });
    expect(frame.scene, "the rendered target holds the drawn cube").toBeGreaterThan(
        geometry.width * geometry.height * 0.05,
    );
});
