import { captureFrame } from "../../src/core/rendering/capture";

Object.assign(globalThis, {
    shallotCaptureFrame(canvas: HTMLCanvasElement) {
        return captureFrame(canvas);
    },
});
