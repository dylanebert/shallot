import { create, globals } from "webgpu";

let gpu: GPU | undefined;

/** Installs Dawn's WebGPU globals for headless builds in Bun or Node. */
export async function setupGlobals(): Promise<void> {
    Object.assign(globalThis, globals);
    gpu ??= create([]);
    Object.defineProperty(navigator, "gpu", { configurable: true, value: gpu });
}
