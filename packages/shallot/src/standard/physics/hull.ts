// Convex-hull narrowphase geometry (Phase 6.3) — the CPU side of hull collision: the registry of convex
// hull shapes a `Body` references by id, and the flat GPU packing the collide pass (collide.ts HULL_WGSL)
// reads. The collision algorithm itself is the f64 oracle `tests/avbd/hull.ts` (the spec); collide.ts is
// the WGSL port. This file owns the storage layout — the registry value (verts / polygonal faces / unique
// edge directions, identical in shape to the oracle `Hull`) and `packHulls` (the one `hullData` buffer all
// registered hulls concatenate into, addressed per hull by a header table). No GJK/EPA (collide.ts header);
// no quickhull build here —
// authored hulls come from explicit geometry until the §6.6 authoring layer adds a mesh→hull path.

import { Registry } from "../../engine";

type Vec3 = [number, number, number];

/** one polygonal hull face: outward unit normal + plane offset (`dot(normal, v) = offset` on the face) + CCW vertex indices. Matches the oracle `HullFace`. */
export interface HullFace {
    normal: Vec3;
    offset: number;
    verts: number[];
}

/** convex hull geometry: local vertices, polygonal faces, the unique edge directions for the SAT, and a registry name. Matches the oracle `Hull` (plus `name`). */
export interface Hull {
    name: string;
    verts: Vec3[];
    faces: HullFace[];
    edges: Vec3[];
}

/** the registered convex hulls. A `Body` with `ShapeKind.Hull` references one by id; `packHulls` serializes them all into the `hullData` GPU buffer (collide.ts reads it). */
export const Hulls = new Registry<Hull>();

/** register convex hull geometry under `name`, returning its stable id (the body's `hullId`). Re-registering a name reuses the id. */
export function registerHull(name: string, geom: Omit<Hull, "name">): number {
    return Hulls.register({ name, verts: geom.verts, faces: geom.faces, edges: geom.edges });
}

// the built-in unit cube (full-size 2, verts ±1) reserved at id 0 — a box collider is THIS hull scaled by
// its half-extents, so the hull SAT (collide.ts collideHull) reads box and hull through ONE branch-free
// accessor path (no per-access isBox branch — the legacy narrowphase's scale-unified shape, the lever that
// keeps the collide-pass shader compile down). Vertex/face/edge order matches the oracle `boxHull([2,2,2])`.
const UNIT_CUBE: Omit<Hull, "name"> = {
    verts: [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    faces: [
        { normal: [1, 0, 0], offset: 1, verts: [1, 2, 6, 5] },
        { normal: [-1, 0, 0], offset: 1, verts: [0, 4, 7, 3] },
        { normal: [0, 1, 0], offset: 1, verts: [2, 3, 7, 6] },
        { normal: [0, -1, 0], offset: 1, verts: [0, 1, 5, 4] },
        { normal: [0, 0, 1], offset: 1, verts: [4, 5, 6, 7] },
        { normal: [0, 0, -1], offset: 1, verts: [0, 3, 2, 1] },
    ],
    edges: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
};
/** the reserved hull id of the built-in unit cube (collide.ts `UNIT_CUBE_ID`) — a box reads this hull scaled by its half-extents. */
export const UNIT_CUBE_ID = registerHull("__unit_cube__", UNIT_CUBE);

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
