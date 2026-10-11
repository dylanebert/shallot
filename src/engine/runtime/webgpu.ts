import { loadDawn } from "./dawn-host";

let gpu: GPU | undefined;

/** Installs Dawn's WebGPU globals for headless builds in Bun or Node. */
export async function setupGlobals(): Promise<void> {
    const { create, globals } = await loadDawn(
        process.platform,
        process.execPath,
        () => import("webgpu"),
    );
    Object.assign(globalThis, globals);
    gpu ??= create([]);
    Object.defineProperty(navigator, "gpu", { configurable: true, value: gpu });
}
