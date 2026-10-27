import type { World } from "../../engine";
import { deviceFailure, observedDeviceError, probeTexture } from "../../engine/runtime";
import { cameraView, canvasFrameBinding, type View } from "./view";

interface CaptureIdentity {
    width: number;
    height: number;
    surface: "final-canvas" | "final-texture";
    encoding: "rgba8-tight";
}

/** The viewport size and device scale used by capture fixtures. */
export const CAPTURE_CONTRACT: {
    width: number;
    height: number;
    deviceScale: number;
} = {
    width: 1280,
    height: 720,
    deviceScale: 1,
};

const FRAME_CAPTURE_LABEL = "canvas frame capture";

function labeledCaptureError(error: unknown): Error {
    const reason = error instanceof Error ? error : new Error(String(error));
    if (reason.message.startsWith(`${FRAME_CAPTURE_LABEL}: `)) return reason;
    return new Error(`${FRAME_CAPTURE_LABEL}: ${reason.message}`, { cause: reason });
}

function watchFrameCaptureDevice(view: View, device: GPUDevice): void {
    if (view.frameCaptureLossWatched) return;
    view.frameCaptureLossWatched = true;
    const viewRef = new WeakRef(view);
    void observedDeviceError(device)?.then(() => {
        const liveView = viewRef.deref();
        if (!liveView) return;
        const error = deviceFailure(device)!;
        for (const capture of liveView.frameCaptures ?? []) capture.reject(error);
        liveView.frameCaptures = undefined;
    });
    void device.lost.then((info) => {
        const liveView = viewRef.deref();
        if (!liveView) return;
        const error = (liveView.frameCaptureLost ??=
            deviceFailure(device) ??
            new Error(`GPU device lost: ${info.message || info.reason || "device lost"}`));
        for (const capture of liveView.frameCaptures ?? []) capture.reject(error);
        liveView.frameCaptures = undefined;
    });
}

/**
 * Read a camera's world-owned rendered texture target at its own size after a submitted presenting
 * frame. Call after stepping; queue order places the copy after that submission, without requiring
 * another frame. Refuses an unbound/non-texture camera or a target never presented to.
 * Returns independent tight RGBA bytes
 * that outlive the target; detachment or disposal before the copy completes may reject the read.
 */
export async function captureTexture(world: World, eid: number): Promise<Capture> {
    const view = cameraView(world, eid);
    if (!view?.texture) throw new Error("captureTexture: camera has no texture target");
    if (!view.presented) throw new Error("captureTexture: no frame has presented to target");
    const snapshot = await probeTexture(world, view.texture);
    const rgba = new Uint8ClampedArray(snapshot.bytes);
    if (snapshot.format === "bgra8unorm") {
        for (let i = 0; i < rgba.length; i += 4) {
            const red = rgba[i + 2];
            rgba[i + 2] = rgba[i];
            rgba[i] = red;
        }
    }
    return {
        rgba,
        width: snapshot.width,
        height: snapshot.height,
        identity: {
            width: snapshot.width,
            height: snapshot.height,
            surface: "final-texture",
            encoding: "rgba8-tight",
        },
    };
}

/** one capture: tightly packed RGBA at the declared identity. */
export interface Capture {
    rgba: Uint8ClampedArray;
    width: number;
    height: number;
    identity: CaptureIdentity;
}

export function encodeFrameCapture(world: World, view: View): void {
    const captures = view.frameCaptures;
    if (!captures?.length) return;
    view.frameCaptures = undefined;
    const texture = view.canvasTexture;
    if (!texture) {
        for (const capture of captures)
            capture.reject(
                new Error("captureFrame refused: no canvas texture in the presenting frame"),
            );
        return;
    }
    if (view.canvasTextureReused) {
        for (const capture of captures)
            capture.reject(
                new Error(
                    "captureFrame refused: canvas texture already used by an earlier frame; await world.frameFence (or yield a task) before the capture frame",
                ),
            );
        return;
    }
    const encoder = world.frameEncoder();
    if (!encoder) {
        for (const capture of captures)
            capture.reject(new Error("captureFrame refused: no encoder for the presenting frame"));
        return;
    }
    try {
        for (const capture of captures) {
            if (capture.width !== texture.width || capture.height !== texture.height) {
                capture.reject(
                    new Error("captureFrame refused: canvas resized before presentation"),
                );
                continue;
            }
            encoder.copyTextureToBuffer(
                { texture },
                {
                    buffer: capture.buffer,
                    bytesPerRow: capture.bytesPerRow,
                    rowsPerImage: capture.height,
                },
                { width: capture.width, height: capture.height, depthOrArrayLayers: 1 },
            );
            capture.resolve(texture.format);
        }
    } catch (error) {
        const reason = error instanceof Error ? error : new Error(String(error));
        for (const capture of captures) capture.reject(reason);
    }
}

/**
 * Read a running canvas's rendered target at its own backing size as tightly packed RGBA, not the
 * composited page or physical display. A bound WebGPU canvas captures the next presenting frame and
 * refuses capture if that frame reuses a texture an earlier frame drew to; await world.frameFence
 * (or yield a task) before the capture frame. An unbound canvas uses its bitmap. A held app presents
 * no next frame; capture its page with a Playwright page screenshot instead.
 */
export async function captureFrame(canvas: HTMLCanvasElement): Promise<Capture> {
    const binding = canvasFrameBinding(canvas);
    if (binding) {
        const failure = deviceFailure(binding.device);
        if (failure)
            throw labeledCaptureError(
                new Error(`captureFrame refused: ${failure.message}`, { cause: failure }),
            );
        if (binding.view.frameCaptureLost) throw labeledCaptureError(binding.view.frameCaptureLost);
        const width = canvas.width;
        const height = canvas.height;
        const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
        let buffer: GPUBuffer | undefined;
        try {
            buffer = binding.device.createBuffer({
                label: FRAME_CAPTURE_LABEL,
                size: bytesPerRow * height,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            });
            const format = await new Promise<GPUTextureFormat>((resolve, reject) => {
                const captures = (binding.view.frameCaptures ??= []);
                captures.push({
                    buffer: buffer!,
                    width,
                    height,
                    bytesPerRow,
                    resolve,
                    reject: (error) => reject(labeledCaptureError(error)),
                });
                watchFrameCaptureDevice(binding.view, binding.device);
            });
            await buffer.mapAsync(GPUMapMode.READ);
            const failure = deviceFailure(binding.device);
            if (failure) throw failure;
            const mapped = new Uint8Array(buffer.getMappedRange());
            const rgba = new Uint8ClampedArray(width * height * 4);
            for (let y = 0; y < height; y++) {
                const sourceStart = y * bytesPerRow;
                rgba.set(mapped.subarray(sourceStart, sourceStart + width * 4), y * width * 4);
            }
            buffer.unmap();
            if (format === "bgra8unorm" || format === "bgra8unorm-srgb") {
                for (let i = 0; i < rgba.length; i += 4) {
                    const red = rgba[i + 2];
                    rgba[i + 2] = rgba[i];
                    rgba[i] = red;
                }
            }
            return {
                rgba,
                width,
                height,
                identity: { width, height, surface: "final-canvas", encoding: "rgba8-tight" },
            };
        } catch (error) {
            throw labeledCaptureError(error);
        } finally {
            buffer?.destroy();
        }
    }

    const url = await new Promise<string>((done) =>
        requestAnimationFrame(() => done(canvas.toDataURL("image/png"))),
    );
    const bitmap = await createImageBitmap(await (await fetch(url)).blob());
    try {
        const surface = new OffscreenCanvas(bitmap.width, bitmap.height);
        const context = surface.getContext("2d");
        if (!context) throw new Error("capture refused: no 2d context for the capture surface");
        context.drawImage(bitmap, 0, 0);
        const image = context.getImageData(0, 0, surface.width, surface.height);
        return {
            rgba: image.data,
            width: image.width,
            height: image.height,
            identity: {
                width: image.width,
                height: image.height,
                surface: "final-canvas",
                encoding: "rgba8-tight",
            },
        };
    } finally {
        bitmap.close();
    }
}
