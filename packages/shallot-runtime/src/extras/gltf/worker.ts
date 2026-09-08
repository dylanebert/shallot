import type { DecodedGltf } from "./assets";
import type { Targets } from "./target";

// The decode worker's wire protocol + transfer-list walker. A decoded
// glTF is deviceless, transferable data (typed arrays + ImageBitmaps), so the shared `decode` runs in a module
// worker (decode.worker.ts) and hands the payload back zero-copy. This file owns the request/reply messages and
// the transfer list `transferables` the worker posts with; the pool (pool.ts) owns the workers + dispatch and
// reuses both. The worker shell that runs `decode` is `decode.worker.ts`.

/** the message the main thread posts to a decode worker. `targets` are the device's per-slot compressed
 *  formats, resolved main-thread-side: the deviceless-decode contract, so the worker never reads the GPU.
 *  Undefined for an untextured / PNG asset; only KTX2 textures need them. */
export interface DecodeRequest {
    url: string;
    clip: number;
    /** the live import option — forces the live joint-palette route over VAT (part of the cache key). */
    live: boolean;
    targets?: Targets;
}

/** a decode worker's reply: the transferred payload on success, an error string on failure. */
export type DecodeReply = { ok: true; decoded: DecodedGltf } | { ok: false; error: string };

/**
 * every transferable in a decoded payload: the typed-array buffers + ImageBitmaps the worker hands back
 * zero-copy (the `postMessage` transfer list). Deduped: a buffer shared by two views must appear once
 * (transferring one ArrayBuffer twice throws). Pure: the worker builds the list before posting.
 */
export function transferables(d: DecodedGltf): Transferable[] {
    const seen = new Set<Transferable>();
    const buf = (a: ArrayBufferView) => seen.add(a.buffer as ArrayBuffer);

    for (const m of d.scene.meshes) {
        buf(m.vertices);
        buf(m.indices);
    }
    const s = d.geometry.static;
    if (s) {
        buf(s.quant.main);
        buf(s.quant.position);
        buf(s.quant.quant);
        buf(s.indices);
    }
    for (const sk of d.geometry.skinned) {
        buf(sk.quant.main);
        buf(sk.quant.position);
        buf(sk.quant.quant);
        buf(sk.indices);
    }
    for (const img of d.textures.albedo) {
        if (img.kind === "bitmap") seen.add(img.bitmap);
        else {
            buf(img.bytes);
            for (const mip of img.image.mips) buf(mip.data);
        }
    }
    buf(d.textures.albedoRef);
    for (const map of Object.values(d.textures.maps)) {
        for (const img of map.images) {
            if (img.kind === "bitmap") seen.add(img.bitmap);
            else {
                buf(img.bytes);
                for (const mip of img.image.mips) buf(mip.data);
            }
        }
        buf(map.layer);
    }
    for (const vat of d.vats) {
        if (vat) {
            buf(vat.positions);
            buf(vat.normals);
        }
    }
    for (const g of d.geometry.live) {
        buf(g.quant.main);
        buf(g.quant.position);
        buf(g.quant.quant);
        buf(g.indices);
    }
    for (const lm of d.liveMeshes) {
        if (lm) {
            buf(lm.joints);
            buf(lm.weights);
        }
    }
    return [...seen];
}
