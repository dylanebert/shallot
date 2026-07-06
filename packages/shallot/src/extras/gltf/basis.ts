// the Basis Universal transcoder, vendored from three.js's prebuilt artifacts (the proven reference build —
// reference/three.js/examples/jsm/libs/basis). A KTX2 file holds Basis-compressed (ETC1S/UASTC) texels; the
// transcoder decodes them to a GPU-native compressed format (BC7 / ETC2 / ASTC) — the largest single
// bandwidth lever for a textured scene (gpu.md). The wrapper is emscripten UMD glue with an appended ESM
// default export; we feed it the wasm bytes so it never path-resolves.
import BASIS from "./codec/basis_transcoder.js";
import type { TranscodeTarget } from "./target";

// the emscripten/embind surface ships no types; one alias names the whole untyped boundary
type Embind = any;

// the one Basis TranscoderFormat we decode straight to (the mixed-size RGBA fallback); the compressed targets
// live in target.ts, picked main-thread-side from the device features (see pickTargets there).
const RGBA32 = 13;

/** one transcoded mip level: compressed bytes plus the level's texel dimensions. */
export interface Ktx2Mip {
    level: number;
    width: number;
    height: number;
    data: Uint8Array;
}

/** one transcoded KTX2 image: a compressed mip chain in `format`, ready to upload as one texture-array layer. */
export interface Ktx2Image {
    width: number;
    height: number;
    format: GPUTextureFormat;
    blockDim: number;
    mips: Ktx2Mip[];
}

let _basis: Embind = null;
let _loading: Promise<void> | null = null;

/** instantiate + initialize the Basis transcoder wasm. Idempotent + concurrency-safe; call before transcoding
 *  ({@link loadGltf} awaits it when a texture carries `KHR_texture_basisu`). */
export async function loadBasis(): Promise<void> {
    if (_basis) return;
    if (!_loading) {
        _loading = (async () => {
            const url = new URL("./codec/basis_transcoder.wasm", import.meta.url);
            const wasmBinary = new Uint8Array(await (await fetch(url)).arrayBuffer());
            const m = await BASIS({ wasmBinary });
            m.initializeBasis();
            _basis = m;
        })();
    }
    await _loading;
}

/**
 * transcode one KTX2/Basis file to the target's compressed mip chain. Sync: call {@link loadBasis} first.
 * Stops at the 4×4 block floor: mips smaller than one block need partial-block copies WebGPU validates
 * awkwardly, and a 4×4 minimum mip is visually ample.
 *
 * @example
 * await loadBasis();
 * const image = transcodeKtx2(ktx2Bytes, pickTargets(device).albedo); // pickTargets from ./target
 */
export function transcodeKtx2(bytes: Uint8Array, target: TranscodeTarget): Ktx2Image {
    const file = openKtx2(bytes);
    try {
        const width = file.getWidth();
        const height = file.getHeight();
        const levels = file.getLevels();
        const mips: Ktx2Mip[] = [];
        for (let mip = 0; mip < levels; mip++) {
            const info = file.getImageLevelInfo(mip, 0, 0);
            const w = info.origWidth;
            const h = info.origHeight;
            if (w < target.blockDim || h < target.blockDim) break;
            const dst = new Uint8Array(file.getImageTranscodedSizeInBytes(mip, 0, 0, target.basis));
            if (!file.transcodeImage(dst, mip, 0, 0, target.basis, 0, -1, -1)) {
                throw new Error(`[gltf] KTX2 transcodeImage failed at mip ${mip}`);
            }
            mips.push({ level: mip, width: w, height: h, data: dst });
        }
        return { width, height, format: target.gpu, blockDim: target.blockDim, mips };
    } finally {
        file.close();
        file.delete();
    }
}

/**
 * transcode a KTX2 file's base level to raw RGBA8 pixels: the fallback the importer uses when a scene's KTX2
 * baseColor images vary in size (a compressed array can't resize, so they route through the same RGBA resize
 * path the PNG importer uses). Sync: {@link loadBasis} first.
 */
export function transcodeKtx2Rgba(bytes: Uint8Array): {
    width: number;
    height: number;
    rgba: Uint8Array;
} {
    const file = openKtx2(bytes);
    try {
        const width = file.getWidth();
        const height = file.getHeight();
        const rgba = new Uint8Array(file.getImageTranscodedSizeInBytes(0, 0, 0, RGBA32));
        if (!file.transcodeImage(rgba, 0, 0, 0, RGBA32, 0, -1, -1)) {
            throw new Error("[gltf] KTX2 transcodeImage (RGBA) failed");
        }
        return { width, height, rgba };
    } finally {
        file.close();
        file.delete();
    }
}

// open + validate a KTX2 file ready for transcoding — the shared preamble of the two transcode paths. The
// caller owns the returned file (close + delete it, via try/finally); this only cleans up its own failures.
function openKtx2(bytes: Uint8Array): Embind {
    const basis = _basis;
    if (!basis) throw new Error("[gltf] Basis transcoder not loaded — call loadBasis() first");
    const file = new basis.KTX2File(bytes);
    if (!file.isValid()) {
        file.close();
        file.delete();
        throw new Error("[gltf] invalid KTX2 file");
    }
    if (!file.startTranscoding()) {
        file.close();
        file.delete();
        throw new Error("[gltf] KTX2 startTranscoding failed");
    }
    return file;
}
