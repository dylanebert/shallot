import { expect } from "bun:test";
import { check } from "@dylanebert/shallot/harness/check";
import {
    observeSubjectDevice,
    type SubjectDeviceState,
    selectSubjectAdapter,
    subjectCaptureFailure,
    subjectDeviceChecks,
    subjectDeviceDiagnostics,
} from "./fixtures/subject-device";
import { verdictMetadata } from "./verdict";

check(
    "subject-device controls fail on errors and preserve adapter identity",
    {
        claim: "the subject-device verdict retains selected adapter identity and fails on uncaptured errors or loss",
        subject: ["src/harness/fixtures/subject-device.ts", "src/harness/verdict.ts"],
    },
    async () => {
        const state: SubjectDeviceState = { created: false, errors: [] };
        const adapter = {
            info: {
                vendor: "Example",
                architecture: "ExampleGPU",
                device: "Adapter 1",
                description: "test adapter",
            },
            requestDevice: async () => {
                throw new Error("requestDevice rejected");
            },
        };
        selectSubjectAdapter(adapter as unknown as GPUAdapter, state);
        await expect(adapter.requestDevice()).rejects.toThrow("requestDevice rejected");
        expect(state.hardware).toBe("Example ExampleGPU Adapter 1 test adapter");
        expect(subjectDeviceChecks(state).map(({ name, ok }) => [name, ok])).toEqual([
            ["subject selected a WebGPU adapter", true],
            ["subject created its WebGPU device", false],
            ["subject WebGPU device reported no uncaptured errors or loss", true],
        ]);

        let lose!: (info: { reason: string; message: string }) => void;
        const device = Object.assign(new EventTarget(), {
            lost: new Promise<{ reason: string; message: string }>((resolve) => {
                lose = resolve;
            }),
        });
        observeSubjectDevice(device as unknown as GPUDevice, state);
        const error = new Error("invalid binding");
        Object.defineProperty(error, "constructor", { value: { name: "GPUValidationError" } });
        const event = new Event("uncapturederror");
        Object.defineProperty(event, "error", { value: error });
        device.dispatchEvent(event);
        lose({ reason: "unknown", message: "device removed" });
        await Promise.resolve();

        const checks = subjectDeviceChecks(state);
        const diagnostics = subjectDeviceDiagnostics(state);
        const verdict = {
            ok: checks.every((entry) => entry.ok),
            hardware: state.hardware,
            checks,
            diagnostics,
        };
        expect(verdict.ok).toBe(false);
        expect(diagnostics).toEqual({
            gpuErrors: [
                "GPU uncaptured GPUValidationError: invalid binding",
                "GPU device lost unknown: device removed",
            ],
        });
        expect(verdictMetadata(verdict)).toEqual({
            runtime: undefined,
            hardware: state.hardware,
            reason: undefined,
            reproduction: undefined,
            diagnostics: {
                gpuErrors: [
                    "GPU uncaptured GPUValidationError: invalid binding",
                    "GPU device lost unknown: device removed",
                ],
                checks: [
                    {
                        name: "subject WebGPU device reported no uncaptured errors or loss",
                        detail: "GPU uncaptured GPUValidationError: invalid binding\nGPU device lost unknown: device removed",
                    },
                ],
            },
        });

        const green = { created: false, errors: [] } satisfies SubjectDeviceState;
        selectSubjectAdapter(adapter as unknown as GPUAdapter, green);
        const healthyDevice = Object.assign(new EventTarget(), { lost: new Promise(() => {}) });
        observeSubjectDevice(healthyDevice as unknown as GPUDevice, green);
        expect(subjectDeviceChecks(green).every((entry) => entry.ok)).toBe(true);
        expect(subjectDeviceDiagnostics(green)).toBeUndefined();
        const captureFailure = subjectCaptureFailure(green, new Error("snapshot unavailable"));
        expect(captureFailure.ok).toBe(false);
        expect(captureFailure.checks.map(({ name, ok }) => [name, ok])).toEqual([
            ["subject selected a WebGPU adapter", true],
            ["subject created its WebGPU device", true],
            ["subject WebGPU device reported no uncaptured errors or loss", true],
            ["stage-2 capture and pixel checks completed", false],
        ]);
    },
);
