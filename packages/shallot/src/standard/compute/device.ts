export interface GPUCapabilities {
    supportsF16: boolean;
    supportsTimestampQuery: boolean;
    supportsBGRA8UnormStorage: boolean;
}

export async function requestGPU(): Promise<{ device: GPUDevice; capabilities: GPUCapabilities }> {
    if (!navigator.gpu)
        throw new Error(
            "This browser doesn't support WebGPU. Use Chrome 113+, Edge 113+, or a recent Firefox Nightly.",
        );

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter)
        throw new Error(
            "No compatible GPU found. WebGPU requires a GPU with Vulkan or DirectX 12 (Feature Level 11.1+) support.",
        );

    const capabilities: GPUCapabilities = {
        supportsF16: adapter.features.has("shader-f16"),
        supportsTimestampQuery: adapter.features.has("timestamp-query"),
        supportsBGRA8UnormStorage: adapter.features.has("bgra8unorm-storage"),
    };

    const requiredFeatures: string[] = ["indirect-first-instance"];
    if (capabilities.supportsTimestampQuery) requiredFeatures.push("timestamp-query");
    if (capabilities.supportsBGRA8UnormStorage) requiredFeatures.push("bgra8unorm-storage");
    if (capabilities.supportsF16) requiredFeatures.push("shader-f16");

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

    console.log("GPU Capabilities:", {
        f16: capabilities.supportsF16 ? "supported" : "not supported (will use f32)",
        timestampQuery: capabilities.supportsTimestampQuery,
        bgra8UnormStorage: capabilities.supportsBGRA8UnormStorage,
    });

    return { device, capabilities };
}
