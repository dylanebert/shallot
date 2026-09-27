import { expect } from "bun:test";
import {
    assertCaptureGeometry,
    CAPTURE_CONTRACT,
    captureArtifact,
    captureFrame,
    captureIdentityLabel,
    captureIdentityMatches,
} from "@dylanebert/shallot/harness/capture";
import { check } from "@dylanebert/shallot/harness/check";

check(
    "the capture contract is one fixed identity that refuses another geometry",
    {
        claim: "a surface at another size or scale captures anyway under the declared identity, so two seats' pixels would be compared as one contract",
        subject: "src/harness/capture.ts",
    },
    () => {
        // The contract is explicit in every field a consumer would otherwise leave implicit.
        expect(CAPTURE_CONTRACT).toEqual({
            width: 1280,
            height: 720,
            deviceScale: 1,
            surface: "final-canvas",
            encoding: "rgba8-tight",
        });
        expect(captureIdentityLabel(CAPTURE_CONTRACT)).toBe("final-canvas 1280x720@1 rgba8-tight");
        expect(() => assertCaptureGeometry(1280, 720)).not.toThrow();
        for (const [width, height] of [
            [1280, 719],
            [1281, 720],
            [640, 360],
            [0, 0],
        ]) {
            expect(() => assertCaptureGeometry(width, height)).toThrow("capture refused");
        }
        // Every field distinguishes an identity: a changed scale, surface or encoding is another contract,
        // not the same one at a different setting.
        expect(captureIdentityMatches(CAPTURE_CONTRACT, { ...CAPTURE_CONTRACT })).toBe(true);
        for (const changed of [
            { width: 640 },
            { height: 360 },
            { deviceScale: 2 },
            { surface: "offscreen" as unknown as typeof CAPTURE_CONTRACT.surface },
            { encoding: "png" as unknown as typeof CAPTURE_CONTRACT.encoding },
        ]) {
            expect(
                captureIdentityMatches(CAPTURE_CONTRACT, { ...CAPTURE_CONTRACT, ...changed }),
            ).toBe(false);
        }
    },
);

check(
    "capture initiates its snapshot in the caller's task",
    {
        claim: "captureFrame and captureArtifact defer their canvas snapshot until after another presentation can replace the requested frame",
        subject: "src/harness/capture.ts",
    },
    async () => {
        const original = {
            requestAnimationFrame: globalThis.requestAnimationFrame,
            fetch: globalThis.fetch,
            createImageBitmap: globalThis.createImageBitmap,
            OffscreenCanvas: globalThis.OffscreenCanvas,
        };
        const calls: string[] = [];
        const contract = { ...CAPTURE_CONTRACT, width: 1, height: 1 };
        const canvas = {
            width: 1,
            height: 1,
            toDataURL() {
                calls.push("snapshot");
                return "data:image/png;base64,AA==";
            },
        } as HTMLCanvasElement;
        try {
            globalThis.requestAnimationFrame = (() => {
                calls.push("animation frame");
                return 1;
            }) as typeof requestAnimationFrame;
            globalThis.fetch = (async () =>
                ({ blob: async () => ({}) }) as Response) as unknown as typeof fetch;
            globalThis.createImageBitmap = (async () => ({
                width: 1,
                height: 1,
                close() {},
            })) as typeof createImageBitmap;
            globalThis.OffscreenCanvas = class {
                getContext() {
                    return {
                        drawImage() {},
                        getImageData: () => ({
                            data: new Uint8ClampedArray(4),
                            width: 1,
                            height: 1,
                        }),
                    };
                }
            } as unknown as typeof OffscreenCanvas;

            const frame = captureFrame(canvas, contract);
            expect(calls).toEqual(["snapshot"]);
            await frame;

            calls.length = 0;
            const artifact = captureArtifact(canvas, contract);
            expect(calls).toEqual(["snapshot"]);
            expect(await artifact).toBe("data:image/png;base64,AA==");
        } finally {
            globalThis.requestAnimationFrame = original.requestAnimationFrame;
            globalThis.fetch = original.fetch;
            globalThis.createImageBitmap = original.createImageBitmap;
            globalThis.OffscreenCanvas = original.OffscreenCanvas;
        }
    },
);
