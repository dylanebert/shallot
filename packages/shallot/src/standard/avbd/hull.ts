// AVBD's GPU packing of the substrate's `Hulls` registry (`../physics/hull`) — the flat `hullData` buffer
// `collide.ts`'s HULL_WGSL reads. The registry + geometry types are backend-neutral; this format (the
// header table + concatenated verts/faces/edges) is specific to this backend's narrowphase layout.

import { type Hull, Hulls } from "../physics/core";

// ── GPU packing ──────────────────────────────────────────────────────
// One flat u32 buffer (`hullData`, bound `array<u32>`; floats bitcast). A header table indexed by hullId,
// then each hull's block concatenated. Offsets are u32-element indices into the same buffer.
//
//   header[id] (HULL_HEADER u32):  [vertBase, vertCount, faceBase, faceCount, edgeBase, edgeCount, faceIdxBase, 0]
//   verts:    vertCount × 3  (f32 xyz)
//   faces:    faceCount × HULL_FACE_STRIDE: [nx, ny, nz, offset] (f32) + [faceVertLocalOff, faceVertCount]
//   faceIdx:  Σ faceVertCount  (u32 vertex indices, addressed faceIdxBase + faceVertLocalOff + j)
//   edges:    edgeCount × 3  (f32 xyz)

/** u32 stride of one hull's header record (collide.ts `HULL_HEADER`) */
export const HULL_HEADER = 8;
/** u32 stride of one packed face: plane (n.xyz, offset) + (faceVertLocalOff, faceVertCount) (collide.ts `HULL_FACE_STRIDE`) */
export const HULL_FACE_STRIDE = 6;

/**
 * Serialize every registered hull into the flat `hullData` GPU buffer (the collide pass reads it, indexed
 * by a body's `hullId`). Rebuilt + re-uploaded whenever the registry changes (hulls are static after that).
 * Returns at least a 1-element buffer (an empty registry still needs a valid binding).
 */
export function packHulls(): Uint32Array {
    const hulls: Hull[] = [];
    for (let id = 0; ; id++) {
        const name = Hulls.name(id);
        if (name === undefined) break;
        const h = Hulls.get(name);
        if (!h) break;
        hulls.push(h);
    }
    const count = hulls.length;
    if (count === 0) return new Uint32Array(1);

    const headerSize = count * HULL_HEADER;
    let total = headerSize;
    for (const h of hulls) {
        total += h.verts.length * 3 + h.faces.length * HULL_FACE_STRIDE;
        for (const f of h.faces) total += f.verts.length;
        total += h.edges.length * 3;
    }

    const buf = new ArrayBuffer(total * 4);
    const u32 = new Uint32Array(buf);
    const f32 = new Float32Array(buf);

    let cursor = headerSize;
    for (let id = 0; id < count; id++) {
        const h = hulls[id];
        const vertBase = cursor;
        for (let i = 0; i < h.verts.length; i++) {
            f32[cursor + 0] = h.verts[i][0];
            f32[cursor + 1] = h.verts[i][1];
            f32[cursor + 2] = h.verts[i][2];
            cursor += 3;
        }
        const faceBase = cursor;
        const faceIdxBase = faceBase + h.faces.length * HULL_FACE_STRIDE;
        let faceIdxCursor = faceIdxBase;
        let localOff = 0;
        for (let fi = 0; fi < h.faces.length; fi++) {
            const f = h.faces[fi];
            const o = faceBase + fi * HULL_FACE_STRIDE;
            f32[o + 0] = f.normal[0];
            f32[o + 1] = f.normal[1];
            f32[o + 2] = f.normal[2];
            f32[o + 3] = f.offset;
            u32[o + 4] = localOff;
            u32[o + 5] = f.verts.length;
            for (let j = 0; j < f.verts.length; j++) u32[faceIdxCursor++] = f.verts[j];
            localOff += f.verts.length;
        }
        cursor = faceIdxCursor;
        const edgeBase = cursor;
        for (let i = 0; i < h.edges.length; i++) {
            f32[cursor + 0] = h.edges[i][0];
            f32[cursor + 1] = h.edges[i][1];
            f32[cursor + 2] = h.edges[i][2];
            cursor += 3;
        }

        const ho = id * HULL_HEADER;
        u32[ho + 0] = vertBase;
        u32[ho + 1] = h.verts.length;
        u32[ho + 2] = faceBase;
        u32[ho + 3] = h.faces.length;
        u32[ho + 4] = edgeBase;
        u32[ho + 5] = h.edges.length;
        u32[ho + 6] = faceIdxBase;
        u32[ho + 7] = 0;
    }

    return u32;
}
