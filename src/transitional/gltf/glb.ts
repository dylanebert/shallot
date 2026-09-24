import type { GltfJson } from "./gltf";

// .glb binary-container split — the cheap packed-asset add. A .glb is a 12-byte header
// (magic, version, total length) followed by length-prefixed chunks: a JSON chunk (the same document a
// .gltf holds) and an optional BIN chunk (the bytes buffer 0 references when it has no uri). Splitting it
// hands `parse` the same (json, buffers) shape a .gltf produces — the rest of the importer is unchanged.

const MAGIC = 0x46546c67; // "glTF", little-endian
const JSON_CHUNK = 0x4e4f534a; // "JSON"
const BIN_CHUNK = 0x004e4942; // "BIN\0"

/** true if `bytes` starts with the glTF binary magic: lets {@link loadGltf} branch on content, not extension. */
export function isGlb(bytes: ArrayBuffer): boolean {
    return bytes.byteLength >= 12 && new DataView(bytes).getUint32(0, true) === MAGIC;
}

/**
 * split a `.glb` into its JSON document + optional BIN chunk. The BIN chunk backs every glTF `buffers[]`
 * entry that has no `uri` (a `.glb` puts buffer 0 inline); {@link loadGltf} feeds it to `resolveBuffer`.
 * Throws on a bad magic / unsupported version / missing JSON chunk: a malformed container, not a frame to skip.
 *
 * @example
 * const { json, bin } = parseGlb(await readBinary("model.glb"));
 */
export function parseGlb(bytes: ArrayBuffer): { json: GltfJson; bin?: ArrayBuffer } {
    const dv = new DataView(bytes);
    if (dv.getUint32(0, true) !== MAGIC) throw new Error("[gltf] not a .glb (bad magic)");
    const version = dv.getUint32(4, true);
    if (version !== 2) throw new Error(`[gltf] unsupported .glb version ${version} (expected 2)`);
    const total = Math.min(dv.getUint32(8, true), bytes.byteLength);

    let json: GltfJson | undefined;
    let bin: ArrayBuffer | undefined;
    // chunks are 4-byte aligned and the length field already includes the spec's trailing pad, so the
    // walk is `next = dataStart + length` with no extra rounding
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
