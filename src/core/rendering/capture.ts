// The canvas capture contract. It fixes the viewport, device scale, target surface, presentation
// boundary and tightly packed RGBA semantics, so every consumer reads the same geometry. `captureFrame`
// runs IN THE PAGE; the driver fixes the viewport that makes the geometry hold and never re-implements the read.

import type { World } from "../../engine";
import { probeTexture } from "../../engine/runtime";
import { canvasFrameBinding, type View, Views } from "./view";

interface CaptureIdentity {
    width: number;
    height: number;
    deviceScale: number;
    surface: "final-canvas" | "final-texture";
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
    void device.lost.then((info) => {
        const liveView = viewRef.deref();
        if (!liveView) return;
        const error = (liveView.frameCaptureLost ??= new Error(
            `GPU device lost: ${info.message || info.reason || "device lost"}`,
        ));
        for (const capture of liveView.frameCaptures ?? []) capture.reject(error);
        liveView.frameCaptures = undefined;
    });
}

/**
 * Read a camera's world-owned final texture after a submitted presenting frame. Call after stepping;
 * queue order places the copy after that submission, without requiring another frame. Refuses an
 * unbound/non-texture camera or a target never presented to. Returns independent tight RGBA bytes
 * that outlive the target; detachment or disposal before the copy completes may reject the read.
 */
export async function captureTexture(world: World, eid: number): Promise<Capture> {
    const view = world.resource(Views).get(eid);
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
            deviceScale: 1,
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

/**
 * Capture a running app's final canvas as tightly packed RGBA at the declared contract. A WebGPU canvas
 * copies its current swapchain texture before the browser presents it; an unbound canvas uses its own
 * bitmap. A held app presents no next frame; capture its page with a Playwright page screenshot.
 */
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

export async function captureFrame(canvas: HTMLCanvasElement): Promise<Capture> {
    assertCaptureGeometry(canvas.width, canvas.height);
    const binding = canvasFrameBinding(canvas);
    if (binding) {
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
            return { rgba, width, height, identity: CAPTURE_CONTRACT };
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
