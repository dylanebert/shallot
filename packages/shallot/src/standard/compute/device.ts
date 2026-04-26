export async function requestGPU(): Promise<GPUDevice> {
    if (!navigator.gpu)
        throw new Error(
            "This browser doesn't support WebGPU. Use Chrome 113+, Edge 113+, or a recent Firefox Nightly.",
        );

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter)
        throw new Error(
            "No compatible GPU found. WebGPU requires a GPU with Vulkan or DirectX 12 (Feature Level 11.1+) support.",
        );

    const requiredFeatures: string[] = ["indirect-first-instance"];
    if (adapter.features.has("timestamp-query")) requiredFeatures.push("timestamp-query");
    if (adapter.features.has("bgra8unorm-storage")) requiredFeatures.push("bgra8unorm-storage");

    const device = await adapter.requestDevice({
        requiredFeatures: requiredFeatures as any,
        requiredLimits: {
            maxTextureDimension2D: adapter.limits.maxTextureDimension2D,
            maxStorageBuffersPerShaderStage: 10,
        },
    });

    device.lost.then((info) => console.error(`GPU device lost: ${info.reason}`, info.message));
    device.onuncapturederror = (event) => {
        const msg = event.error instanceof GPUValidationError ? event.error.message : event.error;
        console.error("GPU uncaptured error:", msg);
    };

    return device;
}
