import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

export const REVISION = "f0a684f11ead064e176775c9b796f82a55910e92";

export function pinnedCheckout(variable: string, revision: string): string {
    const source = process.env[variable];
    if (!source) throw new Error(`set ${variable} to an unmodified checkout at ${revision}`);
    for (const args of [["rev-parse", "HEAD"], ["status", "--porcelain", "--untracked-files=no"]]) {
        const result = spawnSync("git", ["-C", source, ...args]);
        const expected = args[0] === "rev-parse" ? revision : "";
        if (result.status !== 0 || result.stdout.toString().trim() !== expected) {
            throw new Error(`${variable} must be an unmodified checkout at ${revision}`);
        }
    }
    return source;
}

// Exposes the pinned wrapper's private instance only for separating boundary and kernel costs.
export async function referenceInstance(): Promise<WebAssembly.Instance> {
    const source = pinnedCheckout("MESHOPTIMIZER", REVISION);
    const script = readFileSync(resolve(source, "js/meshopt_decoder.mjs"), "utf8");
    const instrumented = script.replace("supported: true,", "supported: true, instance: () => instance,");
    if (instrumented === script) throw new Error("pinned decoder instance hook did not match");
    const { MeshoptDecoder } = await import(`data:text/javascript;base64,${Buffer.from(instrumented).toString("base64")}`);
    await MeshoptDecoder.ready;
    return MeshoptDecoder.instance();
}

export async function referenceDecoder() {
    const source = pinnedCheckout("MESHOPTIMIZER", REVISION);
    const { MeshoptDecoder } = await import(pathToFileURL(resolve(source, "js/meshopt_decoder.mjs")).href);
    await MeshoptDecoder.ready;
    return MeshoptDecoder as {
        decodeGltfBuffer(target: Uint8Array, count: number, stride: number, source: Uint8Array, mode: string, filter: string): void;
    };
}
