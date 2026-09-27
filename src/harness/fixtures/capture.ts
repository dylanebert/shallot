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
const tag = {
    name: "the final canvas carries its color tag",
    minPixels: 300_000,
    minSpan: 400,
    r: [100, 255] as [number, number],
    g: [0, 110] as [number, number],
    b: [100, 255] as [number, number],
};
let ready = false;
let setupError: string | undefined;
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
            const first = await captureFrame(canvas);
            const second = await captureFrame(canvas);
            const geometry =
                first.width === CAPTURE_CONTRACT.width &&
                first.height === CAPTURE_CONTRACT.height &&
                captureIdentityMatches(first.identity, CAPTURE_CONTRACT);
            const identical =
                first.rgba.length === second.rgba.length &&
                first.rgba.every((value, index) => value === second.rgba[index]);
            const tagged = probePixels(first.rgba, first.width, first.height, tag);
            const checks = [
                ...subjectDeviceChecks(subjectDevice),
                {
                    name: "capture uses the declared contract geometry",
                    ok: geometry,
                    detail: `${first.width}x${first.height}`,
                },
                {
                    name: "two captures of one state are byte-identical",
                    ok: identical,
                },
                {
                    name: tag.name,
                    ok: pixelProbePass(tagged, tag),
                    data: { pixels: tagged.pixels, width: tagged.width, height: tagged.height },
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

        const present = () => {
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: context.getCurrentTexture().createView(),
                        clearValue: { r: 0.8, g: 0.05, b: 0.65, a: 1 },
                        loadOp: "clear",
                        storeOp: "store",
                    },
                ],
            });
            pass.end();
            device.queue.submit([encoder.finish()]);
            ready = true;
            requestAnimationFrame(present);
        };
        requestAnimationFrame(present);
    } catch (error) {
        setupError = error instanceof Error ? error.message : String(error);
    }
}

void start();
