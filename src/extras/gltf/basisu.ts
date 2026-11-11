import { BASISU_WASM_BASE64 } from "./basisu.wasm";

interface Kernel {
    memory: WebAssembly.Memory;
    alloc(length: number): number;
    open(ptr: number, length: number): number;
    close(handle: number): void;
    info(handle: number, field: number): number;
    transcode(handle: number, level: number, target: number): number;
}
let wasm: Kernel | undefined;
function kernel(): Kernel {
    if (!wasm) {
        const bytes = Uint8Array.from(atob(BASISU_WASM_BASE64), (c) => c.charCodeAt(0));
        wasm = new WebAssembly.Instance(new WebAssembly.Module(bytes)).exports as unknown as Kernel;
    }
    return wasm;
}
export type BasisTarget = "etc1" | "etc2-rgba" | "bc7" | "rgba8" | "astc";
const targets: BasisTarget[] = ["etc1", "etc2-rgba", "bc7", "rgba8", "astc"];

/** Owns a copy of a 2D KTX2 image until disposed; each transcode returns caller-owned bytes. */
export function openBasisImage(source: Uint8Array) {
    const k = kernel();
    if (!source.length || source.length > 0x7fffffff) throw new Error("invalid KTX2 byteLength");
    const ptr = k.alloc(source.length);
    new Uint8Array(k.memory.buffer).set(source, ptr);
    let handle = k.open(ptr, source.length);
    const error = () =>
        new TextDecoder().decode(
            new Uint8Array(k.memory.buffer, k.info(handle, 4), k.info(handle, 5)),
        );
    if (k.info(handle, 5)) {
        const message = error();
        k.close(handle);
        throw new Error(message);
    }
    return {
        width: k.info(handle, 0),
        height: k.info(handle, 1),
        levels: k.info(handle, 2),
        model: k.info(handle, 3) ? ("uastc" as const) : ("etc1s" as const),
        transcode(level: number, target: BasisTarget): Uint8Array {
            if (!handle) throw new Error("KTX2 image is disposed");
            if (!Number.isInteger(level) || level < 0 || level >= k.info(handle, 2))
                throw new Error("invalid KTX2 level");
            const index = targets.indexOf(target);
            if (index < 0) throw new Error("invalid Basis Universal target");
            if (k.transcode(handle, level, index)) throw new Error(error());
            const view = new Uint8Array(k.memory.buffer, k.info(handle, 6), k.info(handle, 7));
            const output = new Uint8Array(view.length);
            output.set(view);
            return output;
        },
        dispose(): void {
            if (handle) k.close(handle);
            handle = 0;
        },
    };
}
