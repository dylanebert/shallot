import type { IndexFlag, StorageFlag, TgpuBuffer, UniformFlag } from "typegpu";
import type { AnyData, AnyWgslData, WgslArray } from "typegpu/data";
import * as d from "typegpu/data";
import { Registry, type System, type World } from "../../engine";

import { MeshQuant, octEncode, packUnorm2 } from "../../engine/utils";

export type MeshStorage<T extends AnyWgslData> = TgpuBuffer<WgslArray<T>> & StorageFlag;
export type MeshIndex = MeshStorage<d.U32> & IndexFlag;
export type MeshBinding =
    | GPUTexture
    | GPUSampler
    | GPUBuffer
    | (TgpuBuffer<AnyData> & (StorageFlag | UniformFlag));

/**
 * registered vertex-pull geometry: a slice descriptor into the quantized vertex
 * streams + `indices` GPU storage. `vertices` is the 16 B/vertex main stream
 * (`vec4<u32>`: unorm16 pos + meshId, oct normal, unorm16 uv); `position` is the
 * 8 B/vertex depth/shadow stream (pos + meshId only); `quant` is the per-mesh
 * `MeshQuant` table the decode dequantizes against. `indices` is
 * `u32` absolute vertex positions. `indexBase`/`indexCount` slice the index
 * stream. All have `STORAGE` usage: consumer renderers pull indexed vertices in
 * WGSL, never via `setVertexBuffer` / `setIndexBuffer`.
 *
 * Producers stage typed arrays via {@link registerMesh}, which {@link flushMeshes} packs into
 * shared family buffer sets: every mesh is a slice (its own `indexBase` + meshId) of one set.
 * Registrations during `initialize` pack at warm; each later frame's registrations pack
 * before drawing. Each batch splits into families by attribute names and element schemas.
 *
 * `bounds` is the local-space bounding sphere `[cx, cy, cz, radius]` a producer's
 * frustum cull GlobalTransforms per instance. {@link registerMesh} derives it from the staged
 * vertices; procedural producers may supply their own or omit it (a culler then
 * treats the mesh as always-visible)
 */
export interface Mesh {
    name: string;
    vertices: MeshStorage<d.Vec4u>;
    /** the 8 B/vertex position-only stream the depth + shadow passes pull (standard binds this in the prepass group) */
    position?: MeshStorage<d.Vec2u>;
    /** the per-mesh `MeshQuant` dequant table (position + uv AABB), indexed by the meshId packed in the stream */
    quant?: MeshStorage<typeof MeshQuant>;
    indices: MeshIndex;
    indexBase: number;
    indexCount: number;
    bounds?: [number, number, number, number];
    /** `true` while {@link registerMesh}'s data waits for its pack, which replaces the entry before the next frame's draw */
    pending?: true;
    /**
     * per-mesh binding overrides: resources scoped to *this* mesh's draws, keyed by the surface's binding
     * name. A surface binding resolves to `mesh.bindings?.[name]` when present, else the published global
     * (`world.gpu.*`). Per-mesh resources are shared by that mesh's draws.
     */
    bindings?: Record<string, MeshBinding>;
    /** Named storage arrays, indexed by the same absolute vertex index as `vertices`.
     * Each batch splits by names and element schemas; buffers belong to that family and
     * clearMeshes destroys them. Standard resolves attribute bindings only here and refuses
     * a missing or incompatible stream rather than using published world resources. */
    attributes?: Record<string, MeshStorage<AnyWgslData>>;
}

/**
 * every registered mesh, keyed by name with a stable numeric ID. Each frame's draw reads the
 * entries as they stand. Standard draws an entry only with its `position` and `quant` streams,
 * which {@link registerMesh} builds; a direct `register` without them is skipped with a warning,
 * and a `pending` entry is skipped silently.
 */
export const Meshes: import("../../engine").Resource<Registry<Mesh>> = {
    create: (world) => world.resource(meshResourcesKey).meshes,
};

/** f32 lanes per vertex in the staging array: `px py pz u  nx ny nz v` (the `posU` + `normalV` authoring layout) */
export const VERTEX_FLOATS = 8;

// `registerMesh()` stages the typed arrays + a placeholder registry entry, so the mesh's id and
// `Meshes.size` are known at once; `flushMeshes()` packs staged meshes by stream signature.
interface PendingMesh {
    name: string;
    vertices: Float32Array;
    indices: Uint32Array;
    bounds: [number, number, number, number];
    attributes?: Record<string, { element: AnyWgslData; data: ArrayBufferView }>;
}
interface MeshResources {
    meshes: Registry<Mesh>;
    pending: PendingMesh[];
    initialized: boolean;
    placeholderVertices: MeshStorage<d.Vec4u> | null;
    placeholderIndices: MeshIndex | null;
    /** every buffer of every family flushMeshes packed since the last clear */
    families: { destroy(): void }[];
}

export const meshResourcesKey = { create: createMeshResources };

function createMeshResources(): MeshResources {
    return {
        meshes: new Registry<Mesh>(),
        pending: [],
        initialized: false,
        placeholderVertices: null,
        placeholderIndices: null,
        families: [],
    };
}

function meshResources(world: World): MeshResources {
    return world.resource(meshResourcesKey);
}

/** Create this world's mesh registry and staging during MeshPlugin initialization. */
export function initializeMeshState(world: World): void {
    world.resource(meshResourcesKey).initialized = true;
}

/**
 * local-space axis-aligned bounds `{ min, max }` of a vertex buffer (the shared
 * `posU + normalV` layout, position in the first three floats per record).
 * Pure: the one position-AABB scan both the cull sphere ({@link meshBounds})
 * and the unorm16 dequant range (the per-mesh `MeshQuant`) derive
 * from, so they share one source. An empty buffer returns a zero box.
 */
function meshAabb(vertices: Float32Array): {
    min: [number, number, number];
    max: [number, number, number];
} {
    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    for (let i = 0; i < vertices.length; i += VERTEX_FLOATS) {
        const x = vertices[i];
        const y = vertices[i + 1];
        const z = vertices[i + 2];
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (z < minZ) minZ = z;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
        if (z > maxZ) maxZ = z;
    }
    if (!Number.isFinite(minX)) return { min: [0, 0, 0], max: [0, 0, 0] }; // no vertices
    return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
}

/**
 * local-space bounding sphere `[cx, cy, cz, radius]` of a vertex buffer. Center
 * is the {@link meshAabb} midpoint; radius is the farthest vertex from it, so the
 * sphere is tight and a producer's cull can scale it by world scale. Pure:
 * derived once at registration, never per frame.
 */
export function meshBounds(vertices: Float32Array): [number, number, number, number] {
    const { min, max } = meshAabb(vertices);
    const cx = (min[0] + max[0]) * 0.5;
    const cy = (min[1] + max[1]) * 0.5;
    const cz = (min[2] + max[2]) * 0.5;
    let r2 = 0;
    for (let i = 0; i < vertices.length; i += VERTEX_FLOATS) {
        const dx = vertices[i] - cx;
        const dy = vertices[i + 1] - cy;
        const dz = vertices[i + 2] - cz;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > r2) r2 = d2;
    }
    return [cx, cy, cz, Math.sqrt(r2)];
}

/**
 * register a mesh from typed arrays, from `MeshPlugin.initialize` on. Its id is assigned now;
 * its data is packed by {@link flushMeshes} before the next frame's draw: at warm for
 * registrations during `initialize`, otherwise at the start of the next draw group, so a
 * registration in a draw-group system draws a frame later. Refuses before `MeshPlugin`
 * initializes (`AppConfig.setup`), whose initialize would drop it. Requires
 * `world.gpu.device`; no-ops otherwise. Optional named attributes contain raw storage bytes:
 * each byte length must equal vertex count times the element's storage-array stride
 * (including padding), otherwise registration refuses naming mesh and stream. A batch splits
 * into families by attribute names and schemas; each array uses absolute vertex indices.
 */
export function registerMesh(
    world: World,
    spec: {
        name: string;
        vertices: Float32Array;
        indices: Uint32Array;
        attributes?: PendingMesh["attributes"];
    },
): void {
    if (spec.vertices.length % VERTEX_FLOATS !== 0) {
        throw new Error(
            `mesh "${spec.name}": vertices length ${spec.vertices.length} is not a multiple of ${VERTEX_FLOATS} (one Vertex = posU + normalV)`,
        );
    }
    for (const [name, stream] of Object.entries(spec.attributes ?? {})) {
        const stride = d.sizeOf(d.arrayOf(stream.element, 1));
        const expected = (spec.vertices.length / VERTEX_FLOATS) * stride;
        if (stream.data.byteLength !== expected)
            throw new Error(
                `mesh "${spec.name}": attribute "${name}" has ${stream.data.byteLength} bytes; expected ${expected} (storage stride ${stride})`,
            );
    }
    const resources = meshResources(world);
    if (!resources.initialized) {
        throw new Error(
            `mesh "${spec.name}": registerMesh needs MeshPlugin initialized; register from a plugin's initialize or later, not AppConfig.setup`,
        );
    }
    const device = world.gpu.device;
    if (!device) return;
    // a placeholder reserves the registry entry now; flushMeshes swaps in the
    // real shared buffer + correct indexBase
    resources.placeholderVertices ??= world.gpu.root
        .createBuffer(d.arrayOf(d.vec4u, 1))
        .$usage("storage")
        .$name("shallot-mesh-pending-vertices");
    resources.placeholderIndices ??= world.gpu.root
        .createBuffer(d.arrayOf(d.u32, 1))
        .$usage("storage", "index")
        .$name("shallot-mesh-pending-indices");
    const bounds = meshBounds(spec.vertices);
    resources.pending.push({ ...spec, bounds });
    world.resource(Meshes).register({
        name: spec.name,
        vertices: resources.placeholderVertices,
        indices: resources.placeholderIndices,
        indexBase: 0,
        indexCount: spec.indices.length,
        bounds,
        pending: true,
    });
}

/**
 * concatenate staged meshes into one vertex + index pair, shifting each mesh's
 * indices by its vertex base so the index stream holds absolute positions.
 * Pure: {@link flushMeshes} uploads it
 */
function packMeshes(staged: { name: string; vertices: Float32Array; indices: Uint32Array }[]): {
    vertices: Float32Array;
    indices: Uint32Array;
    slices: {
        name: string;
        indexBase: number;
        indexCount: number;
        vertexBase: number;
        vertexCount: number;
    }[];
} {
    let totalVerts = 0;
    let totalIndices = 0;
    for (const m of staged) {
        totalVerts += m.vertices.length / VERTEX_FLOATS;
        totalIndices += m.indices.length;
    }
    const vertices = new Float32Array(totalVerts * VERTEX_FLOATS);
    const indices = new Uint32Array(totalIndices);
    const slices: {
        name: string;
        indexBase: number;
        indexCount: number;
        vertexBase: number;
        vertexCount: number;
    }[] = [];
    let vertexBase = 0;
    let indexBase = 0;
    for (const m of staged) {
        const vertexCount = m.vertices.length / VERTEX_FLOATS;
        vertices.set(m.vertices, vertexBase * VERTEX_FLOATS);
        for (let i = 0; i < m.indices.length; i++)
            indices[indexBase + i] = m.indices[i] + vertexBase;
        slices.push({
            name: m.name,
            indexBase,
            indexCount: m.indices.length,
            vertexBase,
            vertexCount,
        });
        vertexBase += vertexCount;
        indexBase += m.indices.length;
    }
    return { vertices, indices, slices };
}

// f32 lanes per mesh in a quant table — the `MeshQuant` record (3 × `vec4<f32>`)
const MESH_QUANT_FLOATS = 12;

/**
 * the GPU vertex streams a quantized family uploads, derived from the packed f32
 *. `main` is 16 B/vertex (`vec4<u32>`): w0 = unorm16 pos.xy,
 * w1 = unorm16 pos.z | (meshId << 16), w2 = oct normal, w3 = unorm16 uv.
 * `position` is the 8 B/vertex depth/shadow stream (w0, w1: pos + meshId only).
 * `quant` is `MeshQuant` per mesh (the position + uv AABB the decode dequantizes
 * against, selected by meshId). The f32 stays the lossless authoring form: only
 * the GPU mirror quantizes (the slab packed-mirror discipline).
 */
export interface QuantStreams {
    main: Uint32Array;
    position: Uint32Array;
    quant: Float32Array;
}

/**
 * quantize a packed f32 vertex stream ({@link packMeshes}'s output) into the GPU
 * formats. One AABB per mesh slice (its own position + uv range), so a small mesh
 * keeps full unorm16 precision; meshId is the slice index, packed into the stream
 * so the decode selects the right `MeshQuant` from a plain storage table: no
 * per-draw uniform, works unchanged in render bundles. Pure: the single emitter, paired with
 * the WGSL `decodePos` (`posQuantWgsl()`) so the lattice can't drift between writer and reader.
 */
function quantizeMeshes(
    vertices: Float32Array,
    slices: { vertexBase: number; vertexCount: number }[],
): QuantStreams {
    const vertexCount = vertices.length / VERTEX_FLOATS;
    const main = new Uint32Array(vertexCount * 4);
    const position = new Uint32Array(vertexCount * 2);
    const quant = new Float32Array(slices.length * MESH_QUANT_FLOATS);
    slices.forEach((s, meshId) => {
        if (meshId > 0xffff)
            throw new Error(
                `quantizeMeshes: ${slices.length} meshes exceeds the 16-bit meshId cap (65535) per family`,
            );
        // per-mesh position + uv AABB over the slice (a vertex belongs to one mesh)
        const pmin = [Infinity, Infinity, Infinity];
        const pmax = [-Infinity, -Infinity, -Infinity];
        const umin = [Infinity, Infinity];
        const umax = [-Infinity, -Infinity];
        for (let v = 0; v < s.vertexCount; v++) {
            const i = (s.vertexBase + v) * VERTEX_FLOATS;
            for (let a = 0; a < 3; a++) {
                pmin[a] = Math.min(pmin[a], vertices[i + a]);
                pmax[a] = Math.max(pmax[a], vertices[i + a]);
            }
            umin[0] = Math.min(umin[0], vertices[i + 3]);
            umax[0] = Math.max(umax[0], vertices[i + 3]);
            umin[1] = Math.min(umin[1], vertices[i + 7]);
            umax[1] = Math.max(umax[1], vertices[i + 7]);
        }
        if (s.vertexCount === 0) {
            pmin.fill(0);
            pmax.fill(0);
            umin.fill(0);
            umax.fill(0);
        }
        const pext = [pmax[0] - pmin[0], pmax[1] - pmin[1], pmax[2] - pmin[2]];
        const uext = [umax[0] - umin[0], umax[1] - umin[1]];
        // MeshQuant: posOffset(pmin.xyz, umin.x), posScale(pext.xyz, umin.y), uvScale(uext.xy, 0, 0)
        const q = meshId * MESH_QUANT_FLOATS;
        quant.set([pmin[0], pmin[1], pmin[2], umin[0]], q);
        quant.set([pext[0], pext[1], pext[2], umin[1]], q + 4);
        quant.set([uext[0], uext[1], 0, 0], q + 8);
        // a degenerate axis (extent 0 — a flat quad's z) writes 0 → decode returns the offset
        const norm = (val: number, lo: number, ext: number) => (ext === 0 ? 0 : (val - lo) / ext);
        for (let v = 0; v < s.vertexCount; v++) {
            const i = (s.vertexBase + v) * VERTEX_FLOATS;
            const vi = s.vertexBase + v;
            const w0 = packUnorm2(
                norm(vertices[i], pmin[0], pext[0]),
                norm(vertices[i + 1], pmin[1], pext[1]),
            );
            // the z lane rides the same unorm16 lattice as w0's x/y — pack it as a lane and keep the
            // low half, so all three position lanes round on one lattice (a bare f64 round here lands
            // on the other lattice point wherever the f32 product straddles a midpoint)
            const z16 = packUnorm2(norm(vertices[i + 2], pmin[2], pext[2]), 0) & 0xffff;
            const w1 = (z16 | (meshId << 16)) >>> 0;
            const w2 = octEncode(vertices[i + 4], vertices[i + 5], vertices[i + 6]);
            const w3 = packUnorm2(
                norm(vertices[i + 3], umin[0], uext[0]),
                norm(vertices[i + 7], umin[1], uext[1]),
            );
            main[vi * 4] = w0;
            main[vi * 4 + 1] = w1;
            main[vi * 4 + 2] = w2;
            main[vi * 4 + 3] = w3;
            position[vi * 2] = w0;
            position[vi * 2 + 1] = w1;
        }
    });
    return { main, position, quant };
}

// drop the staged-but-unflushed mesh data + the placeholder buffers. flushMeshes calls it after packing,
// clearMeshes after discarding — one source of truth for the staging state to reset. No bind group holds
// a placeholder: standard skips an entry without its quantized streams before binding it.
function resetStaging(world: World): void {
    const resources = meshResources(world);
    resources.pending.length = 0;
    resources.placeholderVertices?.destroy();
    resources.placeholderIndices?.destroy();
    resources.placeholderVertices = null;
    resources.placeholderIndices = null;
}

/**
 * pack staged meshes into families keyed by attribute names and element schemas, each with
 * quantized vertex streams and a shared index buffer. Re-register each as a slice;
 * earlier families and their entries are untouched.
 * `MeshPlugin.warm` runs it after every `initialize`, and {@link PrepareMeshesSystem} at the
 * start of each draw group; with nothing staged it does nothing.
 */
export function flushMeshes(world: World): void {
    const device = world.gpu.device;
    const resources = meshResources(world);
    if (!device || resources.pending.length === 0) return;
    const families: PendingMesh[][] = [];
    for (const mesh of resources.pending) {
        const names = Object.keys(mesh.attributes ?? {}).sort();
        const family = families.find((group) => {
            const other = group[0].attributes ?? {};
            const keys = Object.keys(other).sort();
            return (
                names.length === keys.length &&
                names.every(
                    (name, i) =>
                        name === keys[i] &&
                        d.deepEqual(mesh.attributes![name].element, other[name].element),
                )
            );
        });
        if (family) family.push(mesh);
        else families.push([mesh]);
    }
    for (const staged of families) packFamily(world, staged);
    resetStaging(world);
}

function packFamily(world: World, staged: PendingMesh[]): void {
    const resources = meshResources(world);
    const packed = packMeshes(staged);
    const q = quantizeMeshes(packed.vertices, packed.slices);
    const vertices = world.gpu.root
        .createBuffer(d.arrayOf(d.vec4u, q.main.length / 4))
        .$usage("storage")
        .$name("shallot-mesh-main");
    const position = world.gpu.root
        .createBuffer(d.arrayOf(d.vec2u, q.position.length / 2))
        .$usage("storage")
        .$name("shallot-mesh-pos");
    const quant = world.gpu.root
        .createBuffer(d.arrayOf(MeshQuant, q.quant.length / 12))
        .$usage("storage")
        .$name("shallot-mesh-quant");
    const indices = world.gpu.root
        .createBuffer(d.arrayOf(d.u32, packed.indices.length))
        .$usage("storage", "index")
        .$name("shallot-mesh-indices");
    vertices.write(q.main.buffer as ArrayBuffer);
    position.write(q.position.buffer as ArrayBuffer);
    quant.write(q.quant.buffer as ArrayBuffer);
    indices.write(packed.indices.buffer as ArrayBuffer);
    resources.families.push(vertices, position, quant, indices);
    const attributes: NonNullable<Mesh["attributes"]> = {};
    for (const [name, stream] of Object.entries(staged[0].attributes ?? {})) {
        const stride = d.sizeOf(d.arrayOf(stream.element, 1));
        const bytes = new Uint8Array((packed.vertices.length / VERTEX_FLOATS) * stride);
        let offset = 0;
        for (const mesh of staged) {
            const data = mesh.attributes![name].data;
            bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset);
            offset += data.byteLength;
        }
        const buffer = world.gpu.root
            .createBuffer(d.arrayOf(stream.element, bytes.length / stride))
            .$usage("storage")
            .$name(`shallot-mesh-attribute-${name}`);
        buffer.write(bytes.buffer);
        attributes[name] = buffer;
        resources.families.push(buffer);
    }
    const bounds = new Map(staged.map((m) => [m.name, m.bounds]));
    for (const s of packed.slices) {
        world.resource(Meshes).register({
            name: s.name,
            vertices,
            position,
            quant,
            indices,
            indexBase: s.indexBase,
            indexCount: s.indexCount,
            bounds: bounds.get(s.name),
            attributes,
        });
    }
}

/** packs the meshes registered since the last pack before this frame's draw, as Bevy prepares
 * render assets added during the frame */
export const PrepareMeshesSystem: System = {
    name: "prepareMeshes",
    group: "draw",
    first: true,
    update: flushMeshes,
};

/**
 * drop every registered mesh + any staged-but-unflushed data and destroy every family packed for them,
 * resetting the registry for a fresh build (`MeshPlugin.initialize`, clear then rebuild). Static producers re-stage via {@link registerMesh} in
 * their own initialize, so a producer toggled off leaves no stale slice to be paired against
 * a live surface (the pack registers a Draw per `(surface, mesh)` pair, including a dead one otherwise).
 */
export function clearMeshes(world: World): void {
    world.resource(Meshes).clear();
    resetStaging(world);
    // initialize runs between frames, so no open encoder holds a family, submitted work keeps its
    // storage, and a cleared registry names none of them for any later bind group
    const families = meshResources(world).families;
    for (const buffer of families) buffer.destroy();
    families.length = 0;
}
