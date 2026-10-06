import type { World } from "../../../engine";
// Upload of the resident world's geometry into kernel/src/geo.rs. TS owns construction; the kernel
// reads hull topology and non-convex query records from linear memory. A geometry-set change, region move
// or residency transfer rewrites the pool. Unchanged single-world steps upload nothing.

import { getCompoundChild } from "../shapes/compound";
import type { HullData } from "../shapes/hull";
import type { WorldState } from "../world/world";
import { kernel } from "./kernel";

/** b3HullData header, followed by its eight-byte-aligned trailing arrays. */
const HULL_HEADER_WORDS = 36;
const align8 = (words: number): number => (words + 1) & ~1;

// GEO_LAYOUT header indices (geo.rs), in memory order.
const REC = 0;

const EXTRA = 6;
const N_GEO = 8;

/** The subset of a hull the geometry upload reads (and `geoIndex`, which it writes). `HullData`
 * satisfies it structurally. */
export type UploadHull = Pick<
    HullData,
    | "center"
    | "aabb"
    | "surfaceArea"
    | "volume"
    | "innerRadius"
    | "centralInertia"
    | "hash"
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
    const size = (h: UploadHull): number =>
        HULL_HEADER_WORDS +
        align8(h.vertexCount) +
        align8(h.vertexCount * 3) +
        align8(h.edgeCount * 4) +
        align8(h.faceCount * 4) +
        align8(h.faceCount) +
        3 * ((h.vertexCount + 3) & ~3) +
        3 * ((h.faceCount + 3) & ~3);
    let total = align8(hulls.length);
    for (const h of hulls) total += size(h);
    const k = kernel(world);
    k.reserveGeometry(total, extra.length);
    const buf = k.memory.buffer;
    const layout = new Uint32Array(buf, k.geoLayoutPtr(), N_GEO);
    new Uint32Array(buf, layout[EXTRA], extra.length).set(extra);
    const recU = new Uint32Array(buf, layout[REC], total);
    const recF = new Float32Array(buf, layout[REC], total);
    recU.fill(0);
    let base = align8(hulls.length);
    for (let i = 0; i < hulls.length; ++i) {
        const h = hulls[i];
        h.geoIndex = i;

        const r = base;
        recU[i] = r * 4;
        recU[r] = 0xde57485c;
        recU[r + 1] = 0x4a4c9587;
        recU[r + 2] = h.hash >>> 0;
        recF.set(
            [
                h.aabb.lowerBound.x,
                h.aabb.lowerBound.y,
                h.aabb.lowerBound.z,
                h.aabb.upperBound.x,
                h.aabb.upperBound.y,
                h.aabb.upperBound.z,
                h.surfaceArea,
                h.volume,
                h.innerRadius,
                h.center.x,
                h.center.y,
                h.center.z,
                h.centralInertia.cx.x,
                h.centralInertia.cx.y,
                h.centralInertia.cx.z,
                h.centralInertia.cy.x,
                h.centralInertia.cy.y,
                h.centralInertia.cy.z,
                h.centralInertia.cz.x,
                h.centralInertia.cz.y,
                h.centralInertia.cz.z,
            ],
            r + 4,
        );
        let off = HULL_HEADER_WORDS;
        const vertices = new Uint32Array(buf, layout[REC] + 4 * (r + off), h.vertexCount);
        recU[r + 25] = h.vertexCount;
        recU[r + 26] = off * 4;
        off += align8(h.vertexCount);
        const points = new Float32Array(buf, layout[REC] + 4 * (r + off), h.vertexCount * 3);
        recU[r + 27] = off * 4;
        off += align8(h.vertexCount * 3);
        const edgeCol = new Uint32Array(buf, layout[REC] + 4 * (r + off), h.edgeCount * 4);
        recU[r + 28] = h.edgeCount;
        recU[r + 29] = off * 4;
        off += align8(h.edgeCount * 4);
        const planes = new Float32Array(buf, layout[REC] + 4 * (r + off), h.faceCount * 4);
        recU[r + 30] = h.faceCount;
        recU[r + 31] = off * 4;
        off += align8(h.faceCount * 4);
        const faceCol = new Uint32Array(buf, layout[REC] + 4 * (r + off), h.faceCount);
        recU[r + 32] = off * 4;
        off += align8(h.faceCount);
        const soa = new Float32Array(buf, layout[REC] + 4 * (r + off), size(h) - off);
        recU[r + 33] = off * 4;
        recU[r + 34] = (off + 3 * ((h.vertexCount + 3) & ~3)) * 4;
        recU[r + 35] = size(h) * 4;
        let soaOff = 0;

        // Authoring and snapshots keep points/planes as their one source; derive Box3D's padded
        // streams only at upload. Tail vertices repeat element zero; tail normals are zero.
        const nv = (h.vertexCount + 3) & ~3;
        const nf = (h.faceCount + 3) & ~3;
        for (let p = 0; p < nv; ++p) {
            const pt = h.points[p < h.vertexCount ? p : 0];
            soa[soaOff + p] = pt.x;
            soa[soaOff + nv + p] = pt.y;
            soa[soaOff + 2 * nv + p] = pt.z;
        }
        soaOff += 3 * nv;
        for (let f = 0; f < nf; ++f) {
            const normal = f < h.faceCount ? h.planes[f].normal : undefined;
            soa[soaOff + f] = normal?.x ?? 0;
            soa[soaOff + nf + f] = normal?.y ?? 0;
            soa[soaOff + 2 * nf + f] = normal?.z ?? 0;
        }
        soaOff += 3 * nf;

        for (let p = 0; p < h.vertexCount; ++p) {
            const pt = h.points[p];
            const o = p * 3;
            points[o] = pt.x;
            points[o + 1] = pt.y;
            points[o + 2] = pt.z;
            vertices[p] = h.vertices[p].edge;
        }
        for (let e = 0; e < h.edgeCount; ++e) {
            const ed = h.edges[e];
            const o = e * 4;
            edgeCol[o] = ed.next;
            edgeCol[o + 1] = ed.twin;
            edgeCol[o + 2] = ed.origin;
            edgeCol[o + 3] = ed.face;
        }
        for (let f = 0; f < h.faceCount; ++f) {
            faceCol[f] = h.faces[f].edge;
            const pl = h.planes[f];
            const o = f * 4;
            planes[o] = pl.normal.x;
            planes[o + 1] = pl.normal.y;
            planes[o + 2] = pl.normal.z;
            planes[o + 3] = pl.offset;
        }

        base += size(h);
    }
}

/** Rebuild this World's geometry after its geometry set changes.
 * Hull references are record indices; non-convex references are word offsets within EXTRA. Mesh
 * records hold counts and offsets to 11-word nodes, xyz vertices, index triples, flags and materials.
 * Height records hold bounds, quantization, scale, dimensions, winding and array offsets. Compound
 * records hold the tree root, node/child counts and offsets to 12-word tree nodes and 19-word children
 * (kind, transform, four material indices, seven geometry words). Pool-relative offsets survive reallocation. */
export function rebuildGeometry(world: WorldState): void {
    const hullArray = Array.from(world.hullDatabase.values(), (entry) => entry.hull);
    for (let i = 0; i < hullArray.length; ++i) hullArray[i].geoIndex = i;

    // Pool-relative references need no patching when the geometry allocation moves.
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
                words.push(
                    world.hullDatabase.get(child.hull.hash | 0)!.hull.geoIndex,
                    bits(child.hull.innerRadius),
                );
            else if (child.mesh) {
                words.push(world.meshDatabase.get(child.mesh.data)!.geoIndex);
                vec(child.mesh.scale);
            }
            while (words.length < start + 7) words.push(0);
        }
    }
    kernel(world.ecsState).shapeSetActiveWorld(world.worldId);
    uploadGeometry(world.ecsState, hullArray, words);
    world.geometryUploadCount += 1;
    world.shapeStore.refreshViews();
    world.bodyStore.refreshViews();
    world.manifoldStore.refreshViews();
    for (let s = 0; s < world.shapeGeometry.length; ++s) {
        if (!kernel(world.ecsState).shapeAlive(world.worldId, s)) continue;
        world.shapeStore.writeGeometryReference(world, s);
    }
}
