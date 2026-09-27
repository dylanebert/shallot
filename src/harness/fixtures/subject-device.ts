export interface SubjectDeviceState {
    hardware?: string;
    created: boolean;
    errors: string[];
}

export function selectSubjectAdapter(adapter: GPUAdapter, state: SubjectDeviceState): void {
    const info = adapter.info;
    state.hardware =
        [info.vendor, info.architecture, info.device, info.description]
            .filter((part) => typeof part === "string" && part.trim() !== "")
            .join(" ") || "unidentified";
}

export function observeSubjectDevice(device: GPUDevice, state: SubjectDeviceState): void {
    state.created = true;
    device.addEventListener("uncapturederror", (event) => {
        const error = (event as GPUUncapturedErrorEvent).error;
        state.errors.push(`GPU uncaptured ${error.constructor.name}: ${error.message}`);
    });
    void device.lost.then((info) => {
        state.errors.push(`GPU device lost ${info.reason}: ${info.message}`);
    });
}

export function subjectDeviceChecks(state: SubjectDeviceState) {
    return [
        {
            name: "subject selected a WebGPU adapter",
            ok: state.hardware !== undefined,
            detail: state.hardware,
        },
        {
            name: "subject created its WebGPU device",
            ok: state.created,
        },
        {
            name: "subject WebGPU device reported no uncaptured errors or loss",
            ok: state.errors.length === 0,
            ...(state.errors.length === 0 ? {} : { detail: state.errors.join("\n") }),
        },
    ];
}

export function subjectDeviceDiagnostics(state: SubjectDeviceState) {
    return state.errors.length === 0 ? undefined : { gpuErrors: [...state.errors] };
}

export function subjectCaptureFailure(state: SubjectDeviceState, error: unknown) {
    const diagnostics = subjectDeviceDiagnostics(state);
    const checks = [
        ...subjectDeviceChecks(state),
        {
            name: "stage-2 capture and pixel checks completed",
            ok: false,
            detail: error instanceof Error ? error.message : String(error),
        },
    ];
    return {
        ok: false,
        checks,
        ...(state.hardware === undefined ? {} : { hardware: state.hardware }),
        ...(diagnostics === undefined ? {} : { diagnostics }),
    };
}
