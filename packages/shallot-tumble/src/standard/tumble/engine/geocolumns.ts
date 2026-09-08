// Upload of interned convex hulls into the kernel's static geometry columns (kernel/src/geo.rs). The
// narrowphase reads hull topology from wasm linear memory; TS owns hull construction (hull.ts) and the
// interning database (world.ts `hullDatabase`), so on any change to the hull set it re-uploads every
// hull compactly through this module. Upload is per hull-set change (shape create/destroy), never per
// step. The strides + record layout MIRROR kernel/src/geo.rs — the wasm side is the contract.

import type { HullData } from "./hull";
import { kernel } from "./kernel";
import type { WorldState } from "./world";

/** u32 words per hull record (RECORD_STRIDE in geo.rs): center.xyz + v/e/f counts + 5 pool offsets. */
const RECORD_STRIDE = 12;

// GEO_LAYOUT header indices (geo.rs), in memory order.
const REC = 0;
const POINTS = 1;
const VERTICES = 2;
const EDGES = 3;
const FACES = 4;
const PLANES = 5;
const N_GEO = 6;

/** The subset of a hull the geometry upload reads (and `geoIndex`, which it writes). `HullData`
 * satisfies it structurally. */
export type UploadHull = Pick<
    HullData,
    | "center"
    | "vertexCount"
    | "edgeCount"
    | "faceCount"
    | "points"
    | "vertices"
    | "edges"
    | "faces"
    | "planes"
    | "geoIndex"
>;

/**
 * Upload `hulls` into the kernel's static geometry columns, laying them out compactly and setting each
 * hull's `geoIndex` to its record index. A full rewrite — the pools are sized to the exact totals and
 * every hull's data is written fresh, so growth and renumbering need no in-place preservation.
 */
export function uploadGeometry(hulls: UploadHull[]): void {
    let verts = 0;
    let edges = 0;
    let faces = 0;
    for (const h of hulls) {
        verts += h.vertexCount;
        edges += h.edgeCount;
        faces += h.faceCount;
    }

    const k = kernel();
    k.reserveGeometry(hulls.length, verts, edges, faces);
    const buf = k.memory.buffer;
    const layout = new Uint32Array(buf, k.geoLayoutPtr(), N_GEO);

    // Two views over the record pool: center is f32 bits, counts + offsets are u32, at disjoint slots.
    const recU = new Uint32Array(buf, layout[REC], hulls.length * RECORD_STRIDE);
    const recF = new Float32Array(buf, layout[REC], hulls.length * RECORD_STRIDE);
    const points = new Float32Array(buf, layout[POINTS], verts * 3);
    const vertices = new Uint32Array(buf, layout[VERTICES], verts);
    const edgeCol = new Uint32Array(buf, layout[EDGES], edges * 4);
    const faceCol = new Uint32Array(buf, layout[FACES], faces);
    const planes = new Float32Array(buf, layout[PLANES], faces * 4);

    // Point and vertex pools share an element offset (one point per vertex); edge/face/plane advance
    // independently.
    let vOff = 0;
    let eOff = 0;
    let fOff = 0;
    for (let i = 0; i < hulls.length; ++i) {
        const h = hulls[i];
        h.geoIndex = i;

        const r = i * RECORD_STRIDE;
        recF[r] = h.center.x;
        recF[r + 1] = h.center.y;
        recF[r + 2] = h.center.z;
        recU[r + 3] = h.vertexCount;
        recU[r + 4] = h.edgeCount;
        recU[r + 5] = h.faceCount;
        recU[r + 6] = vOff; // pointOff
        recU[r + 7] = vOff; // vertexOff
        recU[r + 8] = eOff;
        recU[r + 9] = fOff;
        recU[r + 10] = fOff; // planeOff (one plane per face)

        for (let p = 0; p < h.vertexCount; ++p) {
            const pt = h.points[p];
            const o = (vOff + p) * 3;
            points[o] = pt.x;
            points[o + 1] = pt.y;
            points[o + 2] = pt.z;
            vertices[vOff + p] = h.vertices[p].edge;
        }
        for (let e = 0; e < h.edgeCount; ++e) {
            const ed = h.edges[e];
            const o = (eOff + e) * 4;
            edgeCol[o] = ed.next;
            edgeCol[o + 1] = ed.twin;
            edgeCol[o + 2] = ed.origin;
            edgeCol[o + 3] = ed.face;
        }
        for (let f = 0; f < h.faceCount; ++f) {
            faceCol[fOff + f] = h.faces[f].edge;
            const pl = h.planes[f];
            const o = (fOff + f) * 4;
            planes[o] = pl.normal.x;
            planes[o + 1] = pl.normal.y;
            planes[o + 2] = pl.normal.z;
            planes[o + 3] = pl.offset;
        }

        vOff += h.vertexCount;
        eOff += h.edgeCount;
        fOff += h.faceCount;
    }
}

/** Re-upload every interned hull after a change to the hull database (add of a new content hash, or
 * removal of the last reference). Insertion-ordered; the compact renumbering refreshes every hull's
 * `geoIndex`, which shapes read fresh each step. */
export function rebuildGeometry(world: WorldState): void {
    const hulls: HullData[] = [];
    for (const entry of world.hullDatabase.values()) {
        hulls.push(entry.hull);
    }
    uploadGeometry(hulls);
}
