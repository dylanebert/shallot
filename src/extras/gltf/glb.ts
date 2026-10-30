import type { GltfJson } from "./gltf";

const MAGIC = 0x46546c67; // "glTF", little-endian
const JSON_CHUNK = 0x4e4f534a; // "JSON"
const BIN_CHUNK = 0x004e4942; // "BIN\0"

export function isGlb(bytes: ArrayBuffer): boolean {
    return bytes.byteLength >= 12 && new DataView(bytes).getUint32(0, true) === MAGIC;
}

export function parseGlb(bytes: ArrayBuffer): { json: GltfJson; bin?: ArrayBuffer } {
    const dv = new DataView(bytes);
    if (dv.getUint32(0, true) !== MAGIC) throw new Error("[gltf] not a .glb (bad magic)");
    const version = dv.getUint32(4, true);
    if (version !== 2) throw new Error(`[gltf] unsupported .glb version ${version} (expected 2)`);
    const total = Math.min(dv.getUint32(8, true), bytes.byteLength);

    let json: GltfJson | undefined;
    let bin: ArrayBuffer | undefined;

    for (let o = 12; o + 8 <= total; ) {
        const length = dv.getUint32(o, true);
        const type = dv.getUint32(o + 4, true);
        const start = o + 8;
        if (type === JSON_CHUNK) {
            json = JSON.parse(new TextDecoder().decode(new Uint8Array(bytes, start, length)));
        } else if (type === BIN_CHUNK) {
            bin = bytes.slice(start, start + length);
        }
        o = start + length;
    }
    if (!json) throw new Error("[gltf] .glb has no JSON chunk");
    return { json, bin };
}
