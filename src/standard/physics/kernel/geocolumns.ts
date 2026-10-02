import type { World } from "../../../engine";
// Upload of the resident world's geometry into kernel/src/geo.rs. TS owns construction; the kernel
// reads hull topology and non-convex query records from linear memory. A geometry-set change, region move
// or residency transfer rewrites the pool. Unchanged single-world steps upload nothing.

import { getCompoundChild } from "../shapes/compound";
import type { HullData } from "../shapes/hull";
import type { WorldState } from "../world/world";
import { geometryUploaded } from "./bodycolumns";
import { kernel } from "./kernel";

/** u32 words per hull record (RECORD_STRIDE in geo.rs): center.xyz + v/e/f counts + 5 pool offsets. */
const RECORD_STRIDE = 12;

// GEO_LAYOUT header indices (geo.rs), in memory order.
const REC = 0;
const POINTS = 1;
const VERTICES = 2;
const EDGES = 3;
const FACES = 4;
const PLANES = 5;
const EXTRA = 6;
const N_GEO = 7;

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
export function uploadGeometry(
    world: World | undefined,
    hulls: UploadHull[],
    extra: readonly number[] = [],
): void {
    let verts = 0;
    let edges = 0;
    let faces = 0;
    for (const h of hulls) {
        verts += h.vertexCount;
        edges += h.edgeCount;
        faces += h.faceCount;
    }

    const k = kernel(world);
    k.reserveGeometry(hulls.length, verts, edges, faces, extra.length);
    const buf = k.memory.buffer;
    const layout = new Uint32Array(buf, k.geoLayoutPtr(), N_GEO);
    new Uint32Array(buf, layout[EXTRA], extra.length).set(extra);

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

/** Rebuild the resident world's geometry after a geometry-set change, region move or owner change.
 * Hull references are record indices; non-convex references are word offsets within EXTRA. Mesh
 * records hold counts and offsets to 11-word nodes, xyz vertices, index triples, flags and materials.
 * Height records hold bounds, quantization, scale, dimensions, winding and array offsets. Compound
 * records hold the tree root, node/child counts and offsets to 12-word tree nodes and 19-word children
 * (kind, transform, four material indices, seven geometry words). All offsets survive relocation. */
export function rebuildGeometry(world: WorldState): void {
    const hullArray = Array.from(world.hullDatabase.values(), (entry) => entry.hull);
    for (let i = 0; i < hullArray.length; ++i) hullArray[i].geoIndex = i;

    // All references in these records are word offsets in EXTRA, never absolute addresses. This
    // keeps the body's relocation chain independent of the kind-specific record layouts.
    const words: number[] = [];
    const append = (values: Iterable<number>): void => {
        for (const value of values) words.push(value);
    };
    const float = new DataView(new ArrayBuffer(4));
    const bits = (x: number): number => {
        float.setFloat32(0, x, true);
        return float.getUint32(0, true);
    };
    const vec = (p: { x: number; y: number; z: number }): void => {
        words.push(bits(p.x), bits(p.y), bits(p.z));
    };
    for (const [m, entry] of world.meshDatabase) {
        const record = words.length;
        entry.geoIndex = record;
        words.push(m.nodes.length, m.vertices.length, m.triangles.length, 0, 0, 0, 0, 0);
        words[record + 3] = words.length;
        for (const n of m.nodes) {
            vec(n.lowerBound);
            vec(n.upperBound);
            words.push(Number(n.leaf), n.axis, n.childOffset, n.triangleCount, n.triangleOffset);
        }
        words[record + 4] = words.length;
        for (const p of m.vertices) vec(p);
        words[record + 5] = words.length;
        for (const t of m.triangles) words.push(t.index1, t.index2, t.index3);
        words[record + 6] = words.length;
        append(m.flags);
        words[record + 7] = words.length;
        append(m.materialIndices);
    }
    for (const [h, entry] of world.heightFieldDatabase) {
        const record = words.length;
        entry.geoIndex = record;
        vec(h.aabb.lowerBound);
        vec(h.aabb.upperBound);
        words.push(bits(h.minHeight), bits(h.maxHeight), bits(h.heightScale));
        vec(h.scale);
        words.push(h.columnCount, h.rowCount, Number(h.clockwise), 0, 0, 0);
        words[record + 15] = words.length;
        append(h.compressedHeights);
        words[record + 16] = words.length;
        append(h.materialIndices);
        words[record + 17] = words.length;
        append(h.flags);
    }
    for (const [c, entry] of world.compoundDatabase) {
        const record = words.length;
        entry.geoIndex = record;
        const count = c.capsules.length + c.hulls.length + c.meshes.length + c.spheres.length;
        words.push(c.tree.root, c.tree.nodeCapacity, count, 0, 0);
        words[record + 3] = words.length;
        // The compound tree retains the exact dynamic-tree node layout and leaf user data.
        append(new Uint32Array(c.tree.ni.buffer, c.tree.ni.byteOffset, c.tree.ni.length));
        words[record + 4] = words.length;
        for (let i = 0; i < count; ++i) {
            const child = getCompoundChild(c, i);
            words.push(child.type);
            vec(child.transform.p);
            vec(child.transform.q.v);
            words.push(bits(child.transform.q.s));
            words.push(...child.materialIndices);
            const start = words.length;
            if (child.capsule) {
                vec(child.capsule.center1);
                vec(child.capsule.center2);
                words.push(bits(child.capsule.radius));
            } else if (child.sphere) {
                vec(child.sphere.center);
                words.push(bits(child.sphere.radius));
            } else if (child.hull)
                words.push(world.hullDatabase.get(child.hull.hash)!.hull.geoIndex);
            else if (child.mesh) {
                words.push(world.meshDatabase.get(child.mesh.data)!.geoIndex);
                vec(child.mesh.scale);
            }
            while (words.length < start + 7) words.push(0);
        }
    }
    uploadGeometry(world.ecsState, hullArray, words);
    world.geometryUploadCount += 1;
    world.shapeStore.refreshViews();
    world.bodyStore.refreshViews();
    world.manifoldStore.refreshViews();
    for (const s of world.shapes) {
        if (s.id < 0) continue;
        world.shapeStore.writeGeometryReference(world, s);
    }
    geometryUploaded(world);
}
