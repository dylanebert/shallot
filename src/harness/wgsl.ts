/**
 * Compile WGSL through Dawn's native WebGPU implementation under Bun.
 *
 * This is deliberately asynchronous: the optional peer is loaded only by callers that ask for a
 * native compile, so browser consumers can import the harness without carrying the Bun bridge.
 */
export async function compileWgsl(code: string): Promise<string | null> {
    if (typeof Bun === "undefined") {
        throw new Error("compileWgsl requires Bun and the optional bun-webgpu peer");
    }

    let native: { setupGlobals(): Promise<void> };
    try {
        const peer = "bun-webgpu";
        native = (await import(peer)) as unknown as { setupGlobals(): Promise<void> };
    } catch (error) {
        throw new Error(
            `compileWgsl requires the optional bun-webgpu peer: ${error instanceof Error ? error.message : String(error)}`,
        );
    }

    await native.setupGlobals();
    const adapter = await navigator.gpu?.requestAdapter();
    if (adapter === null || adapter === undefined) {
        throw new Error("compileWgsl requires a real GPU seat: no WebGPU adapter is available");
    }

    const device = await adapter.requestDevice();
    if (device === null || device === undefined) {
        throw new Error("compileWgsl requires a real GPU seat: no WebGPU device is available");
    }

    device.pushErrorScope("validation");
    try {
        device.createShaderModule({ code });
        const error = await device.popErrorScope();
        return error?.message ?? null;
    } finally {
        device.destroy();
    }
}
