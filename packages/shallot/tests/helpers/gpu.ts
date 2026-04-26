let device: GPUDevice | null = null;
let initPromise: Promise<{ device: GPUDevice }> | null = null;

export function shouldSkipGPU(): string | null {
    if (process.platform === "win32") return "Windows";
    return null;
}

export async function initGPU(): Promise<{ device: GPUDevice }> {
    if (initPromise) return initPromise;

    initPromise = (async () => {
        const skipReason = shouldSkipGPU();
        if (skipReason) {
            throw new Error(`GPU tests skipped: ${skipReason}`);
        }

        const { setupGlobals } = await import("bun-webgpu");
        await setupGlobals();

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error("No GPU adapter available");
        }
        device = await adapter.requestDevice();

        return { device: device! };
    })();

    return initPromise;
}
