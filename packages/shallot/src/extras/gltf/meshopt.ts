// the zeux/meshoptimizer glTF buffer decoder (EXT_meshopt_compression). Imported from the decoder submodule
// directly, not the `meshoptimizer` barrel — the barrel also pulls the encoder/simplifier/clusterizer, and
// only the decoder is needed at runtime. The module embeds its wasm as inline base64 and instantiates it from
// `MeshoptDecoder.ready` (no wasm asset to fetch, unlike draco/basis), so it bundles into the decode worker
// chunk with no `import.meta.url` resolution.
import { MeshoptDecoder } from "meshoptimizer/meshopt_decoder.module.js";
import type { MeshoptExtension } from "./gltf";

// process-wide singleton (the loadDraco / loadBasis shape) — lazy because the decoder is only needed by a
// meshopt-compressed asset. loadMeshopt awaits the wasm instantiate once; decodeMeshopt is sync after.
let _ready = false;
let _loading: Promise<void> | null = null;

/** instantiate the meshopt decoder wasm. Idempotent + concurrency-safe; call before decoding a meshopt asset
 *  ({@link loadGltf} awaits it when a bufferView carries `EXT_meshopt_compression`). */
export async function loadMeshopt(): Promise<void> {
    if (_ready) return;
    if (!_loading) {
        _loading = MeshoptDecoder.ready.then(() => {
            _ready = true;
        });
    }
    await _loading;
}

/**
 * decompress one `EXT_meshopt_compression` bufferView's bytes into the plain (optionally filtered) bytes a
 * standard accessor read consumes. Sync: call {@link loadMeshopt} first. `mode` selects the codec
 * (ATTRIBUTES / TRIANGLES / INDICES) and `filter` the post-decode transform (NONE / OCTAHEDRAL / QUATERNION /
 * EXPONENTIAL), both handled in the one `decodeGltfBuffer` call. The output is `count * size` bytes, tightly
 * packed (stride = `size`), the layout the importer's bufferView normalize step rewrites the view to.
 *
 * @example
 * await loadMeshopt();
 * const bytes = decodeMeshopt(source, ext.count, ext.byteStride, ext.mode, ext.filter);
 */
export function decodeMeshopt(
    source: Uint8Array,
    count: number,
    size: number,
    mode: MeshoptExtension["mode"],
    filter: MeshoptExtension["filter"],
): Uint8Array {
    if (!_ready) throw new Error("[gltf] Meshopt decoder not loaded — call loadMeshopt() first");
    const target = new Uint8Array(count * size);
    MeshoptDecoder.decodeGltfBuffer(target, count, size, source, mode, filter ?? "NONE");
    return target;
}
