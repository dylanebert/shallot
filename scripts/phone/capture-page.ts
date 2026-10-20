import { CAPTURE_CONTRACT, captureFrame } from "../../src/core/rendering/capture";

Object.assign(globalThis, {
    shallotCaptureFrame(canvas: HTMLCanvasElement) {
        Object.assign(CAPTURE_CONTRACT, { width: canvas.width, height: canvas.height });
        return captureFrame(canvas);
    },
});
