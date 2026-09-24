// the Google Draco glTF decoder, vendored from three.js's prebuilt artifacts (the proven reference build —
// reference/three.js/examples/jsm/libs/draco/gltf). The wrapper is emscripten UMD glue; bun + Vite expose
// its `module.exports` factory as the default import. We feed it the wasm bytes so it never path-resolves.
import DracoDecoderModule from "./codec/draco_wasm_wrapper.js";
import type { DracoExtension } from "./gltf";

// the embind surface (module, Decoder/Mesh handles, heap views) ships no types; one alias names the whole
// untyped emscripten boundary
type Embind = any;

/** one decoded Draco primitive: the same attribute arrays a standard accessor read yields, fed to the
 *  shared vertex packing. `pos` is `count*3`, `normal` `count*3`, `uv` `count*2`; `indices` is widened u32. */
export interface DracoMesh {
    pos: Float32Array;
    normal: Float32Array | null;
    uv: Float32Array | null;
    indices: Uint32Array;
}

// process-wide singleton (the Compute/Audio shape) — lazy because the ~190KB wasm is only needed by a
// Draco-compressed asset. loadDraco instantiates once; decodeDraco is sync after.
let _draco: Embind = null;
let _loading: Promise<void> | null = null;

/** instantiate the Draco decoder wasm. Idempotent + concurrency-safe; call before decoding a Draco asset
 *  ({@link loadGltf} awaits it when a primitive carries `KHR_draco_mesh_compression`). */
export async function loadDraco(): Promise<void> {
    if (_draco) return;
    if (!_loading) {
        _loading = (async () => {
            const url = new URL("./codec/draco_decoder.wasm", import.meta.url);
            const wasmBinary = new Uint8Array(await (await fetch(url)).arrayBuffer());
            _draco = await DracoDecoderModule({ wasmBinary });
        })();
    }
    await _loading;
}

// read one float attribute (by its glTF unique id) into a tight count×components array via the decoder's
// heap. All the attributes we consume (position/normal/uv) are float, so DT_FLOAT32 covers every read.
function readAttr(decoder: Embind, geom: Embind, id: number): Float32Array {
    const draco = _draco!;
    const attr = decoder.GetAttributeByUniqueId(geom, id);
    const comps = attr.num_components();
    const count = geom.num_points();
    const byteLength = count * comps * 4;
    const ptr = draco._malloc(byteLength);
    decoder.GetAttributeDataArrayForAllPoints(geom, attr, draco.DT_FLOAT32, byteLength, ptr);
    const out = new Float32Array(draco.HEAPF32.buffer, ptr, count * comps).slice();
    draco._free(ptr);
    return out;
}

// read the triangle index list widened to u32 (the engine's index format)
function readIndices(decoder: Embind, geom: Embind): Uint32Array {
    const draco = _draco!;
    const numIndices = geom.num_faces() * 3;
    const byteLength = numIndices * 4;
    const ptr = draco._malloc(byteLength);
    decoder.GetTrianglesUInt32Array(geom, byteLength, ptr);
    const out = new Uint32Array(draco.HEAPF32.buffer, ptr, numIndices).slice();
    draco._free(ptr);
    return out;
}

/**
 * decode one `KHR_draco_mesh_compression` primitive's compressed bytes into plain attribute + index arrays.
 * Sync: call {@link loadDraco} first. The geometry's attributes are addressed by the extension's unique ids
 * (glTF always uses unique ids, not the 1:1 semantic mapping `.drc` files use).
 *
 * @example
 * await loadDraco();
 * const { pos, indices } = decodeDraco(compressedBytes, ext.attributes);
 */
export function decodeDraco(
    bytes: Uint8Array,
    attributes: DracoExtension["attributes"],
): DracoMesh {
    const draco = _draco;
    if (!draco) throw new Error("[gltf] Draco decoder not loaded — call loadDraco() first");

    const decoder = new draco.Decoder();
    const geom = new draco.Mesh();
    const array = new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const status = decoder.DecodeArrayToMesh(array, array.byteLength, geom);
    if (!status.ok() || geom.ptr === 0) {
        const msg = status.error_msg();
        draco.destroy(geom);
        draco.destroy(decoder);
        throw new Error(`[gltf] Draco decode failed: ${msg}`);
    }

    const pos = readAttr(decoder, geom, attributes.POSITION);
    const normal =
        attributes.NORMAL !== undefined ? readAttr(decoder, geom, attributes.NORMAL) : null;
    const uv =
        attributes.TEXCOORD_0 !== undefined ? readAttr(decoder, geom, attributes.TEXCOORD_0) : null;
    const indices = readIndices(decoder, geom);

    draco.destroy(geom);
    draco.destroy(decoder);
    return { pos, normal, uv, indices };
}
