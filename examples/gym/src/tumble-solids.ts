// Native solid rendering for the tumble sample host, derived entirely from the world's own shape data.
//
// Solid bodies render as shallot-native instanced Parts: the live `World` is walked through the same
// `world.draw` walk the samples use, one shallot mesh is generated per unique shape geometry (deduped by
// shape identity — 725 dominoes sharing one `makeBoxHull` register ONE mesh), and each frame a
// Part+Transform+Color instance per drawn shape syncs from the body pose + state color the walk resolves.
// The `world.draw` gizmo pass (joints, contact points) stays on top as lines (`tumble-sample.ts`).
//
// The walk is read-only and the visuals derive from post-step world state, so nothing here feeds the
// oracle: the bit-exact gold contract holds regardless of what renders.

import { Compute } from "@dylanebert/shallot";
import { Meshes, meshBounds, packMeshes, quantizeMeshes } from "@dylanebert/shallot/render/core";
import type {
    Capsule,
    DebugDraw,
    HullData,
    Mesh,
    Sphere,
    Vec3,
    World,
} from "@dylanebert/shallot/tumble/core";
import { defaultDebugDraw } from "@dylanebert/shallot/tumble/core";

const SPHERE_STACKS = 16;
const SPHERE_SLICES = 24;
const CAP_SEG = 20;
const CAP_RINGS = 6;

// The solid layer's derivation must be TOTAL — every world solid body is derived, always. `world.draw`
// only visits shapes whose fat AABB overlaps `drawingBounds`, and `defaultDebugDraw()` defaults that to a
// ±100 m debug box; a body driven past it (an aggressive mouse-grab drag) would be silently clipped from
// the derivation while still simulating. On-screen visibility is the Part pack's frustum cull's job, not
// the derivation's, so both the discovery and the per-frame walks (`tumble-sample.ts`) use these effectively
// unbounded bounds so no body is ever dropped for being far from the origin.
const H = 1e9;
export const TOTAL_DRAW_BOUNDS = {
    lowerBound: { x: -H, y: -H, z: -H },
    upperBound: { x: H, y: H, z: H },
};

/** one generated mesh: a unique shape geometry resolved to shallot's `posU + normalV` vertex layout. */
interface GeomDef {
    name: string;
    vertices: Float32Array;
    indices: Uint32Array;
}

/** the discovery result: the geometry→mesh-name map the per-frame reconcile keys on, the generated mesh
 *  defs to register once a device exists, and the total shape-instance count (capacity sizing). */
export interface Solids {
    /** stable shape-geometry object → registered mesh name. The reconcile resolves a drawn shape's mesh
     *  by the same key discovery used, so a mesh is shared across every instance of one geometry. */
    keyToName: Map<object, string>;
    defs: GeomDef[];
    instanceCount: number;
    /** shape kinds with no solid mesh mapping that fell back to wireframe (empty for the sample corpus). */
    fallbacks: Set<string>;
}

/** the stable object identity a shape's geometry keys on. A compound hands fresh `Mesh` wrappers each
 *  walk (`getCompoundChild`), so a triangle mesh keys on its immutable `data`; the other kinds key on the
 *  geometry object the shape owns for its life. */
function meshKey(mesh: Mesh): object {
    return mesh.data;
}

// --- geometry → shallot mesh vertices (posU + normalV, 8 floats/vertex) -----------------------------------

function pushVertex(
    out: number[],
    px: number,
    py: number,
    pz: number,
    nx: number,
    ny: number,
    nz: number,
    u: number,
    v: number,
): void {
    out.push(px, py, pz, u, nx, ny, nz, v);
}

function sphereMesh(sphere: Sphere): GeomDef {
    const { center, radius } = sphere;
    const verts: number[] = [];
    const idx: number[] = [];
    const cols = SPHERE_SLICES + 1;
    for (let i = 0; i <= SPHERE_STACKS; i++) {
        const phi = (Math.PI * i) / SPHERE_STACKS;
        const sp = Math.sin(phi);
        const cp = Math.cos(phi);
        for (let j = 0; j <= SPHERE_SLICES; j++) {
            const theta = (2 * Math.PI * j) / SPHERE_SLICES;
            const nx = sp * Math.cos(theta);
            const ny = cp;
            const nz = sp * Math.sin(theta);
            pushVertex(
                verts,
                center.x + radius * nx,
                center.y + radius * ny,
                center.z + radius * nz,
                nx,
                ny,
                nz,
                j / SPHERE_SLICES,
                i / SPHERE_STACKS,
            );
        }
    }
    for (let i = 0; i < SPHERE_STACKS; i++) {
        for (let j = 0; j < SPHERE_SLICES; j++) {
            const a = i * cols + j;
            const b = a + cols;
            // CCW-from-outside (sear culls back faces): +stack is +φ, +slice is +θ, and Tφ×Tθ points
            // inward, so the outward-facing triangle order is a → a+1 → b, not a → b → a+1.
            idx.push(a, a + 1, b, a + 1, b + 1, b);
        }
    }
    return { name: "", vertices: new Float32Array(verts), indices: new Uint32Array(idx) };
}

function normalize(x: number, y: number, z: number): [number, number, number] {
    const l = Math.hypot(x, y, z) || 1;
    return [x / l, y / l, z / l];
}

function capsuleMesh(cap: Capsule): GeomDef {
    const { center1: c1, center2: c2, radius: r } = cap;
    const [ax, ay, az] = normalize(c2.x - c1.x, c2.y - c1.y, c2.z - c1.z);
    // an axis-perpendicular basis (u, w) for the rings
    let [ux, uy, uz] = normalize(-ay, ax, 0);
    if (!Number.isFinite(ux) || ux * ux + uy * uy + uz * uz < 0.5) {
        [ux, uy, uz] = normalize(0, -az, ay);
    }
    const wx = ay * uz - az * uy;
    const wy = az * ux - ax * uz;
    const wz = ax * uy - ay * ux;

    const verts: number[] = [];
    const idx: number[] = [];
    const cols = CAP_SEG + 1;
    // rings: bottom hemisphere (cap around c1, axis −a) then top hemisphere (cap around c2, axis +a). The
    // bottom-equator and top-equator rings sit at c1 and c2 respectively, so the band between them is the tube.
    const ring = (cx: number, cy: number, cz: number, sign: number, beta: number): void => {
        const sb = Math.sin(beta);
        const cb = Math.cos(beta);
        for (let j = 0; j <= CAP_SEG; j++) {
            const theta = (2 * Math.PI * j) / CAP_SEG;
            const ct = Math.cos(theta);
            const st = Math.sin(theta);
            const nx = sign * sb * ax + cb * (ct * ux + st * wx);
            const ny = sign * sb * ay + cb * (ct * uy + st * wy);
            const nz = sign * sb * az + cb * (ct * uz + st * wz);
            pushVertex(verts, cx + r * nx, cy + r * ny, cz + r * nz, nx, ny, nz, j / CAP_SEG, 0);
        }
    };
    for (let k = 0; k <= CAP_RINGS; k++)
        ring(c1.x, c1.y, c1.z, -1, (Math.PI / 2) * (1 - k / CAP_RINGS));
    for (let k = 0; k <= CAP_RINGS; k++) ring(c2.x, c2.y, c2.z, 1, (Math.PI / 2) * (k / CAP_RINGS));
    const rings = 2 * (CAP_RINGS + 1);
    for (let i = 0; i < rings - 1; i++) {
        for (let j = 0; j < CAP_SEG; j++) {
            const a = i * cols + j;
            const b = a + cols;
            // outward-facing winding, same reasoning as the sphere (a → a+1 → b).
            idx.push(a, a + 1, b, a + 1, b + 1, b);
        }
    }
    return { name: "", vertices: new Float32Array(verts), indices: new Uint32Array(idx) };
}

function hullMesh(hull: HullData): GeomDef {
    const verts: number[] = [];
    const idx: number[] = [];
    for (let fi = 0; fi < hull.faceCount; fi++) {
        const n = hull.planes[fi].normal;
        const start = hull.faces[fi].edge;
        const loop: number[] = [];
        let e = start;
        do {
            loop.push(hull.edges[e].origin);
            e = hull.edges[e].next;
        } while (e !== start && loop.length <= hull.edgeCount);
        // fan-triangulate the face, flat-shaded with the face plane normal
        for (let k = 1; k < loop.length - 1; k++) {
            for (const vi of [loop[0], loop[k], loop[k + 1]]) {
                const p = hull.points[vi];
                idx.push(verts.length / 8);
                pushVertex(verts, p.x, p.y, p.z, n.x, n.y, n.z, 0, 0);
            }
        }
    }
    return { name: "", vertices: new Float32Array(verts), indices: new Uint32Array(idx) };
}

function triangleMesh(mesh: Mesh): GeomDef {
    const { vertices, triangles } = mesh.data;
    const s = mesh.scale;
    const verts: number[] = [];
    const idx: number[] = [];
    const scaled = (v: Vec3): [number, number, number] => [v.x * s.x, v.y * s.y, v.z * s.z];
    for (const t of triangles) {
        const [ax, ay, az] = scaled(vertices[t.index1]);
        const [bx, by, bz] = scaled(vertices[t.index2]);
        const [cx, cy, cz] = scaled(vertices[t.index3]);
        const [nx, ny, nz] = normalize(
            (by - ay) * (cz - az) - (bz - az) * (cy - ay),
            (bz - az) * (cx - ax) - (bx - ax) * (cz - az),
            (bx - ax) * (cy - ay) - (by - ay) * (cx - ax),
        );
        idx.push(verts.length / 8, verts.length / 8 + 1, verts.length / 8 + 2);
        pushVertex(verts, ax, ay, az, nx, ny, nz, 0, 0);
        pushVertex(verts, bx, by, bz, nx, ny, nz, 0, 0);
        pushVertex(verts, cx, cy, cz, nx, ny, nz, 0, 0);
    }
    return { name: "", vertices: new Float32Array(verts), indices: new Uint32Array(idx) };
}

// --- discovery: walk the world once, generate one mesh per unique geometry -------------------------------

/**
 * Walk `world` through its own `world.draw` walk, resolving each shape's geometry to a shallot mesh once
 * per unique geometry (compounds pre-decomposed into primitive callbacks by the walk). Pure — no device,
 * no entities: it produces the mesh defs + the key map the per-frame reconcile uses and the shape count
 * for capacity sizing. Register the defs with {@link registerSolids} once a device exists.
 */
export function collectSolids(world: World): Solids {
    const keyToName = new Map<object, string>();
    const defs: GeomDef[] = [];
    const fallbacks = new Set<string>();
    let instanceCount = 0;
    let n = 0;

    const record = (key: object, gen: () => GeomDef): void => {
        instanceCount++;
        if (keyToName.has(key)) return;
        const name = `tumble-solid-${n++}`;
        keyToName.set(key, name);
        const def = gen();
        def.name = name;
        defs.push(def);
    };

    const dd: DebugDraw = {
        ...defaultDebugDraw(),
        drawingBounds: TOTAL_DRAW_BOUNDS,
        drawShapes: true,
        drawSolidSphere: (_xf, sphere) => record(sphere, () => sphereMesh(sphere)),
        drawSolidCapsule: (_xf, cap) => record(cap, () => capsuleMesh(cap)),
        drawSolidHull: (_xf, hull) => record(hull, () => hullMesh(hull)),
        drawSolidMesh: (_xf, mesh) => record(meshKey(mesh), () => triangleMesh(mesh)),
        drawSolidHeightField: () => {
            fallbacks.add("heightField");
        },
    };
    world.draw(dd);
    return { keyToName, defs, instanceCount, fallbacks };
}

/**
 * Register the generated mesh defs with the engine, packing them into one shared buffer family (the
 * procedural-producer path, `render.md` — `mesh()`'s static staging only flushes at warm, so a
 * runtime-built set registers its own buffers). No-op without a device.
 * @returns each registered mesh's name → its {@link Meshes} id, for `Part.mesh` writes.
 */
export function registerSolids(defs: GeomDef[]): Map<string, number> {
    const nameToId = new Map<string, number>();
    const device = Compute.device;
    if (!device || defs.length === 0) return nameToId;
    const packed = packMeshes(defs);
    const q = quantizeMeshes(packed.vertices, packed.slices);
    const bounds = new Map(defs.map((d) => [d.name, meshBounds(d.vertices)]));
    const storage = (label: string, data: Uint32Array | Float32Array, index = false): GPUBuffer => {
        const buf = device.createBuffer({
            label,
            size: data.byteLength,
            usage:
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_DST |
                (index ? GPUBufferUsage.INDEX : 0),
        });
        device.queue.writeBuffer(buf, 0, data as Uint32Array<ArrayBuffer>);
        return buf;
    };
    const vertices = storage("tumble-solid-main", q.main);
    const position = storage("tumble-solid-pos", q.position);
    const quant = storage("tumble-solid-quant", q.quant);
    const indices = storage("tumble-solid-idx", packed.indices, true);
    for (const s of packed.slices) {
        const id = Meshes.register({
            name: s.name,
            vertices,
            position,
            quant,
            indices,
            indexBase: s.indexBase,
            indexCount: s.indexCount,
            bounds: bounds.get(s.name),
        });
        nameToId.set(s.name, id);
    }
    return nameToId;
}

/** the stable key a triangle-mesh shape resolves to (its immutable `data`) — the reconcile keys drawn
 *  shapes the same way {@link collectSolids} did, so every instance of one geometry shares its mesh. */
export { meshKey };
