import { readFileSync } from "node:fs";

/** Every built Rust kernel, as committed: a base64 TypeScript module or a raw `.wasm`. */
export const KERNELS = [
    "src/extras/gltf/basisu.wasm.ts",
    "src/extras/gltf/gltf.wasm.ts",
    "src/core/mesh/mikktspace.wasm.ts",
    "src/standard/physics/kernel/kernel.wasm.ts",
    "src/standard/physics/kernel/kernel.shared.wasm.ts",
    "crates/audio/pkg/shallot_audio.wasm",
];
/**
 * The one kernel above the cap: the shared physics kernel keeps 4 GiB for Box3D Large World
 * (23def785), and its JavaScript reads every pointer and length unsigned.
 */
export const UNCAPPED = { "src/standard/physics/kernel/kernel.shared.wasm.ts": 2 ** 32 };

export function kernelBytes(path: string): Uint8Array {
    const text = readFileSync(path);
    if (path.endsWith(".wasm")) return new Uint8Array(text);
    const base64 = text.toString("utf8").match(/"([A-Za-z0-9+/=]{64,})"/)?.[1];
    if (!base64) throw new Error(`${path}: no base64 WebAssembly payload`);
    return Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
}

/** The linear memory's declared maximum in bytes, defined or imported; undefined when it has none. */
export function memoryMaximum(wasm: Uint8Array): number | undefined {
    let p = 8;
    const uleb = () => {
        let value = 0;
        let shift = 0;
        for (;;) {
            const byte = wasm[p++];
            value += (byte & 0x7f) * 2 ** shift;
            if (!(byte & 0x80)) return value;
            shift += 7;
        }
    };
    const limits = () => {
        const flags = uleb();
        uleb();
        return flags & 1 ? uleb() * 65536 : undefined;
    };
    while (p < wasm.length) {
        const id = wasm[p++];
        const end = uleb() + p;
        if (id === 5) {
            uleb();
            return limits();
        }
        if (id === 2) {
            for (let i = uleb(); i > 0; i--) {
                // Not `p += uleb()`: that reads `p` before uleb advances it.
                for (let name = 0; name < 2; name++) {
                    const size = uleb();
                    p += size;
                }
                const kind = wasm[p++];
                if (kind === 2) return limits();
                if (kind === 0) uleb();
                else if (kind === 1) {
                    p++;
                    limits();
                } else if (kind === 3) p += 2;
                else throw new Error(`unknown import kind ${kind}`);
            }
        }
        p = end;
    }
    throw new Error("no linear memory");
}
