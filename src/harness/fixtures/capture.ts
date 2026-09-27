import {
    assertCaptureGeometry,
    CAPTURE_CONTRACT,
    captureFrame,
    captureIdentityMatches,
} from "@dylanebert/shallot/harness/capture";
import { pixelProbePass, probePixels } from "@dylanebert/shallot/harness/pixels";
import {
    observeSubjectDevice,
    type SubjectDeviceState,
    selectSubjectAdapter,
    subjectCaptureFailure,
    subjectDeviceChecks,
    subjectDeviceDiagnostics,
} from "./subject-device";

const canvas = document.querySelector<HTMLCanvasElement>("#canvas")!;
const wrongFrameTag = {
    name: "capture reads the frame presented in its caller task",
    minPixels: 300_000,
    minSpan: 400,
    r: [0, 110] as [number, number],
    g: [100, 255] as [number, number],
    b: [0, 110] as [number, number],
};
const correctFrameTag = {
    name: "the later final canvas carries its color tag",
    minPixels: 300_000,
    minSpan: 400,
    r: [100, 255] as [number, number],
    g: [0, 110] as [number, number],
    b: [100, 255] as [number, number],
};
let ready = false;
let setupError: string | undefined;
let firstCapture: ReturnType<typeof captureFrame> | undefined;
const subjectDevice: SubjectDeviceState = { created: false, errors: [] };

window.__harness = {
    get ready() {
        return ready || setupError !== undefined || subjectDevice.errors.length > 0;
    },
    async run() {
        if (setupError !== undefined || subjectDevice.errors.length > 0 || !subjectDevice.created) {
            const checks = [
                ...subjectDeviceChecks(subjectDevice),
                ...(setupError === undefined
                    ? []
                    : [{ name: "subject WebGPU setup completed", ok: false, detail: setupError }]),
            ];
            const diagnostics = subjectDeviceDiagnostics(subjectDevice);
            return {
                ok: checks.every((check) => check.ok),
                checks,
                ...(subjectDevice.hardware === undefined
                    ? {}
                    : { hardware: subjectDevice.hardware }),
                ...(diagnostics === undefined ? {} : { diagnostics }),
            };
        }
        try {
            if (!firstCapture) throw new Error("the wrong-frame capture was not initiated");
            const first = await firstCapture;
            const second = await captureFrame(canvas);
            const third = await captureFrame(canvas);
            const geometry =
                first.width === CAPTURE_CONTRACT.width &&
                first.height === CAPTURE_CONTRACT.height &&
                captureIdentityMatches(first.identity, CAPTURE_CONTRACT);
            const identical =
                second.rgba.length === third.rgba.length &&
                second.rgba.every((value, index) => value === third.rgba[index]);
            const wrongFrame = probePixels(first.rgba, first.width, first.height, wrongFrameTag);
            const correctFrame = probePixels(
                second.rgba,
                second.width,
                second.height,
                correctFrameTag,
            );
            const checks = [
                ...subjectDeviceChecks(subjectDevice),
                {
                    name: "capture uses the declared contract geometry",
                    ok: geometry,
                    detail: `${first.width}x${first.height}`,
                },
                {
                    name: "two captures of the later state are byte-identical",
                    ok: identical,
                },
                {
                    name: wrongFrameTag.name,
                    ok: pixelProbePass(wrongFrame, wrongFrameTag),
                    data: {
                        pixels: wrongFrame.pixels,
                        width: wrongFrame.width,
                        height: wrongFrame.height,
                    },
                },
                {
                    name: correctFrameTag.name,
                    ok: pixelProbePass(correctFrame, correctFrameTag),
                    data: {
                        pixels: correctFrame.pixels,
                        width: correctFrame.width,
                        height: correctFrame.height,
                    },
                },
            ];
            const diagnostics = subjectDeviceDiagnostics(subjectDevice);
            return {
                ok: checks.every((check) => check.ok),
                checks,
                hardware: subjectDevice.hardware,
                ...(diagnostics === undefined ? {} : { diagnostics }),
            };
        } catch (error) {
            return subjectCaptureFailure(subjectDevice, error);
        }
    },
};

async function start() {
    try {
        assertCaptureGeometry(canvas.width, canvas.height);
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("WebGPU returned no adapter");
        selectSubjectAdapter(adapter, subjectDevice);
        const device = await adapter.requestDevice();
        observeSubjectDevice(device, subjectDevice);
        const context = canvas.getContext("webgpu");
        if (!context) throw new Error("canvas has no WebGPU context");
        context.configure({
            device,
            format: navigator.gpu.getPreferredCanvasFormat(),
            alphaMode: "opaque",
        });

        const present = (clearValue: GPUColor) => {
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: context.getCurrentTexture().createView(),
                        clearValue,
                        loadOp: "clear",
                        storeOp: "store",
                    },
                ],
            });
            pass.end();
            device.queue.submit([encoder.finish()]);
        };
        requestAnimationFrame(() => {
            present({ r: 0.05, g: 0.8, b: 0.1, a: 1 });
            // Queue the next presentation before captureFrame can queue its deferred read.
            requestAnimationFrame(() => present({ r: 0.8, g: 0.05, b: 0.65, a: 1 }));
            firstCapture = captureFrame(canvas);
            ready = true;
        });
    } catch (error) {
        setupError = error instanceof Error ? error.message : String(error);
    }
}

void start();
