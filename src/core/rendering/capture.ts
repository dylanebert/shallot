// The canvas capture contract. It fixes the viewport, device scale, target surface, presentation
// boundary and tightly packed RGBA semantics, so every consumer reads the same geometry. `captureFrame`
// runs IN THE PAGE; the driver fixes the viewport that makes the geometry hold and never re-implements the read.

interface CaptureIdentity {
    width: number;
    height: number;
    deviceScale: number;
    surface: "final-canvas";
    encoding: "rgba8-tight";
}

/** the one declared capture contract every consumer reads. */
export const CAPTURE_CONTRACT: CaptureIdentity = {
    width: 1280,
    height: 720,
    deviceScale: 1,
    surface: "final-canvas",
    encoding: "rgba8-tight",
};

function captureIdentityLabel(identity: CaptureIdentity): string {
    return `${identity.surface} ${identity.width}x${identity.height}@${identity.deviceScale} ${identity.encoding}`;
}

function assertCaptureGeometry(width: number, height: number): void {
    if (width !== CAPTURE_CONTRACT.width || height !== CAPTURE_CONTRACT.height) {
        throw new Error(
            `capture refused: surface is ${width}x${height}, not the declared contract ${captureIdentityLabel(CAPTURE_CONTRACT)}`,
        );
    }
}

/** one capture: tightly packed RGBA at the declared identity. */
export interface Capture {
    rgba: Uint8ClampedArray;
    width: number;
    height: number;
    identity: CaptureIdentity;
}

/**
 * Capture a running app's final canvas as tightly packed RGBA at the declared contract. A WebGPU canvas
 * reads as transparent black once its frame is presented, so the read runs in the next animation frame,
 * after the engine's callback presents. A held app presents no next frame; capture its page with a
 * Playwright page screenshot.
 *
 * @example
 * ```
 * const shot = await captureFrame(document.querySelector("canvas")!);
 * console.log(shot.width, shot.height, shot.identity);
 * ```
 */
export async function captureFrame(canvas: HTMLCanvasElement): Promise<Capture> {
    assertCaptureGeometry(canvas.width, canvas.height);
    const url = await new Promise<string>((done) =>
        requestAnimationFrame(() => done(canvas.toDataURL("image/png"))),
    );
    const bitmap = await createImageBitmap(await (await fetch(url)).blob());
    try {
        assertCaptureGeometry(bitmap.width, bitmap.height);
        const surface = new OffscreenCanvas(bitmap.width, bitmap.height);
        const context = surface.getContext("2d");
        if (!context) throw new Error("capture refused: no 2d context for the capture surface");
        context.drawImage(bitmap, 0, 0);
        const image = context.getImageData(0, 0, surface.width, surface.height);
        return {
            rgba: image.data,
            width: image.width,
            height: image.height,
            identity: CAPTURE_CONTRACT,
        };
    } finally {
        bitmap.close();
    }
}
