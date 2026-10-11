import { UnsupportedError } from "./gpu";

let gpu: GPU | undefined;

/** Refuse Windows hosts that Dawn cannot load before its native addon is imported. */
export function assertDawnHost(platform: string, execPath: string): void {
    const executable = execPath.split(/[\\/]/).at(-1)?.toLowerCase();
    if (platform === "win32" && executable !== "node.exe") {
        throw new UnsupportedError(
            "On Windows, dawn.node imports N-API directly from node.exe (run 38151144247); " +
                "use Node, or Bun run as a copy named node.exe.",
        );
    }
}

/** Installs Dawn's WebGPU globals for headless builds in Bun or Node. */
export async function setupGlobals(): Promise<void> {
    assertDawnHost(process.platform, process.execPath);
    const { create, globals } = await import("webgpu");
    Object.assign(globalThis, globals);
    gpu ??= create([]);
    Object.defineProperty(navigator, "gpu", { configurable: true, value: gpu });
}
