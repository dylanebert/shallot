import type { IndexFlag, StorageFlag, TgpuBuffer, UniformFlag } from "typegpu";
import type { AnyData, AnyWgslData, WgslArray } from "typegpu/data";
import * as d from "typegpu/data";
import type { System, World } from "../../engine";

import { octEncode, packUnorm2 } from "../../engine/utils";
import { MeshQuant } from "./vertex";

export type MeshStorage<T extends AnyWgslData> = TgpuBuffer<WgslArray<T>> & StorageFlag;
export type MeshIndex = MeshStorage<d.U32> & IndexFlag;
export type MeshBinding =
    | GPUTexture
    | GPUSampler
    | GPUBuffer
    | (TgpuBuffer<AnyData> & (StorageFlag | UniformFlag));

declare const meshHandleBrand: unique symbol;

/** World-local identity returned by `Meshes.add`, valid until removal or clear. */
export type MeshHandle = number & { readonly [meshHandleBrand]: true };
const DEFAULT_MESH_HANDLE = 0 as MeshHandle;

/** Vertex-pull geometry in shared storage buffers, not vertex/index bindings.
 * `vertices` uses the packed layout in {@link QuantStreams}; `indices` holds absolute vertex indices,
 * sliced by `indexBase` and `indexCount`. Typed-array additions share family buffers built by
 * {@link flushMeshes}. */
export interface Mesh {
    /** Display label; mesh identity is its {@link MeshHandle}. */
    name: string;
    vertices: MeshStorage<d.Vec4u>;
    /** the 8 B/vertex position-only stream the depth + shadow passes pull (standard binds this in the prepass group) */
    position?: MeshStorage<d.Vec2u>;
    /** the per-mesh `MeshQuant` dequant table (position + uv AABB), indexed by the meshId packed in the stream */
    quant?: MeshStorage<typeof MeshQuant>;
    indices: MeshIndex;
    indexBase: number;
    /** Index count, or the upper bound when `indirectIndexCount` is present. */
    indexCount: number;
    /** Producer-owned STORAGE buffer holding a u32 index count at a four-byte-aligned byte offset
     * (default zero). Immutable while registered. Standard clamps to `indexCount` each frame without
     * readback, after passes ordered before MeshPreprocessSystem; applies to all views and shadows. */
    indirectIndexCount?: { buffer: GPUBuffer; offset?: number };
    /** Local-space culling sphere `[cx, cy, cz, radius]`, derived for typed-array additions.
     * GPU producers may supply it or omit it to remain always visible to standard's culler. */
    bounds?: [number, number, number, number];
    /** `true` while `Meshes.add`'s data waits for its pack, which replaces the entry before the next frame's draw */
    pending?: true;
    /**
     * per-mesh binding overrides: resources scoped to this mesh's draws, keyed by the surface's binding
     * name. Mesh attributes take precedence; otherwise a binding resolves to `mesh.bindings?.[name]`
     * when present, else the live resources owned by the material type through its `bindings` callback.
     * Built-ins resolve from their engine owners. Per-mesh resources are shared by that mesh's draws.
     */
    bindings?: Record<string, MeshBinding>;
    /** Named storage arrays, indexed by the same absolute vertex index as `vertices`.
     * Each batch splits by names and element schemas; buffers belong to that family and
     * clearMeshes destroys them. Standard resolves attribute bindings only here and refuses
     * a missing or incompatible stream rather than using published world resources. */
    attributes?: Record<string, MeshStorage<AnyWgslData>>;
}

/** Typed-array mesh data accepted by {@link MeshRegistry.add}, which stages and packs it. */
interface MeshData {
    name: string;
    vertices: Float32Array;
    indices: Uint32Array;
    attributes?: PendingMesh["attributes"];
}

/**
 * Every added mesh, keyed by its world-local handle. Standard draws an entry only with its
 * `position` and `quant` streams, which the typed-array form of {@link MeshRegistry.add} builds;
 * an entry without them is skipped with a warning, and a pending entry is skipped silently.
 */
interface MeshRegistry extends Iterable<Mesh> {
    /**
     * Add a mesh from typed arrays, from `MeshPlugin.initialize` on. Its handle is assigned now;
     * its data is packed by {@link flushMeshes} before the next frame's draw: at warm for
     * additions during `initialize`, otherwise at the start of the next draw group, so an
     * addition in a draw-group system draws a frame later. Refuses before `MeshPlugin`
     * initializes (`AppConfig.setup`), whose initialize would drop it. Requires
     * `world.gpu.device`; it refuses without one. `name` is a display label, so duplicate labels are valid.
     * Optional named attributes contain raw storage bytes:
     * each byte length must equal vertex count times the element's storage-array stride
     * (including padding), otherwise it refuses naming mesh and stream. A batch splits
     * into families by attribute names and schemas; each array uses absolute vertex indices.
     */
    add(mesh: MeshData): MeshHandle;
    /**
     * Add a GPU-resident mesh as-is: its buffers stay the producer's, so a compute pass may write
     * them and an indirect draw may size them each frame. Nothing is staged or packed.
     */
    add(mesh: Mesh): MeshHandle;
    /** Removes and returns the mesh, if present, without destroying its buffers. Typed-array
     * families retain their slices until `clearMeshes`. A later add reissues the lowest free
     * handle: remove the mesh's instances too, or they will reference its successor. */
    remove(handle: MeshHandle): Mesh | undefined;
    get(handle: MeshHandle): Mesh | undefined;
    has(handle: MeshHandle): boolean;
    entries(): IterableIterator<[MeshHandle, Mesh]>;
    values(): IterableIterator<Mesh>;
    readonly size: number;
    clear(): void;
}

class MeshRegistryImpl implements MeshRegistry {
    private readonly _world: World;
    private _nextHandle = 0;
    private _freeHandle = 0;
    version = 0;
    private readonly _entriesByHandle: (Mesh | undefined)[] = [];
    private readonly _order: MeshHandle[] = [];

    constructor(world: World) {
        this._world = world;
    }

    add(mesh: MeshData | Mesh): MeshHandle {
        // ArrayBuffer.isView also recognizes typed arrays from another realm.
        if (ArrayBuffer.isView(mesh.vertices)) return addMesh(this._world, mesh as MeshData);
        const source = (mesh as Mesh).indirectIndexCount;
        if (source) {
            const offset = source.offset ?? 0;
            if (
                !Number.isFinite(offset) ||
                offset < 0 ||
                offset % 4 !== 0 ||
                offset + 4 > source.buffer.size ||
                !(source.buffer.usage & GPUBufferUsage.STORAGE)
            )
                throw new Error(
                    `mesh ${mesh.name}: indirectIndexCount requires a STORAGE buffer and a finite, aligned in-range u32 byte offset`,
                );
        }
        const handle = this._freeHandle as MeshHandle;
        this._nextHandle = Math.max(this._nextHandle, handle + 1);
        this._entriesByHandle[handle] = mesh as Mesh;
        this._order.push(handle);
        while (this._entriesByHandle[this._freeHandle] !== undefined) this._freeHandle++;
        this.version++;
        return handle;
    }

    replace(handle: MeshHandle, mesh: Mesh): void {
        if (!this.has(handle)) throw new Error(`unknown mesh handle ${handle}`);
        this._entriesByHandle[handle] = mesh;
        this.version++;
    }

    remove(handle: MeshHandle): Mesh | undefined {
        const mesh = this._entriesByHandle[handle];
        if (!mesh) return undefined;
        this._entriesByHandle[handle] = undefined;
        const orderIndex = this._order.indexOf(handle);
        for (let i = orderIndex + 1; i < this._order.length; i++)
            this._order[i - 1] = this._order[i];
        this._order.pop();
        this._freeHandle = Math.min(this._freeHandle, handle);
        const pending = this._world.resource(meshResourcesKey).pending;
        let index = -1;
        for (let i = 0; i < pending.length; i++) {
            if (pending[i].handle === handle) {
                index = i;
                break;
            }
        }
        if (index !== -1) pending.splice(index, 1);
        this.version++;
        return mesh;
    }

    get(handle: MeshHandle): Mesh | undefined {
        return this._entriesByHandle[handle];
    }

    has(handle: MeshHandle): boolean {
        return this._entriesByHandle[handle] !== undefined;
    }

    *entries(): IterableIterator<[MeshHandle, Mesh]> {
        for (const handle of this._order) yield [handle, this._entriesByHandle[handle]!];
    }

    *values(): IterableIterator<Mesh> {
        for (const handle of this._order) yield this._entriesByHandle[handle]!;
    }

    [Symbol.iterator](): IterableIterator<Mesh> {
        return this.values();
    }

    get size(): number {
        return this._order.length;
    }

    get capacity(): number {
        return this._nextHandle;
    }

    clear(): void {
        this._entriesByHandle.fill(undefined);
        this._order.length = 0;
        // Handle-indexed consumers retain their allocation and addressing stride across clear.
        this._freeHandle = 0;
        this.version++;
    }
}

export const Meshes: import("../../engine").Resource<MeshRegistry> = {
    create: (world) => world.resource(meshResourcesKey).meshes,
};

/** f32 lanes per vertex in the staging array: `px py pz u  nx ny nz v` (the `posU` + `normalV` authoring layout) */
export const VERTEX_FLOATS = 8;

interface PendingMesh {
    handle: MeshHandle;
    name: string;
    vertices: Float32Array;
    indices: Uint32Array;
    bounds: [number, number, number, number];
    attributes?: Record<string, { element: AnyWgslData; data: ArrayBufferView }>;
}
interface MeshResources {
    meshes: MeshRegistryImpl;
    pending: PendingMesh[];
    defaultMesh: MeshHandle;
    initialized: boolean;
    placeholderVertices: MeshStorage<d.Vec4u> | null;
    placeholderIndices: MeshIndex | null;
    /** every buffer of every family flushMeshes packed since the last clear */
    families: { destroy(): void }[];
}

/** World-owned packing and handle-indexed consumer state. @internal */
export const meshResourcesKey = { create: createMeshResources };

function createMeshResources(world: World): MeshResources {
    return {
        meshes: new MeshRegistryImpl(world),
        pending: [],
        defaultMesh: DEFAULT_MESH_HANDLE,
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

/** Local-space bounds of the {@link VERTEX_FLOATS} staging layout; an empty buffer returns a zero box. */
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
    if (!Number.isFinite(minX)) return { min: [0, 0, 0], max: [0, 0, 0] };
    return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
}

/** Local-space sphere `[cx, cy, cz, radius]` for the {@link VERTEX_FLOATS} staging layout.
 * Center is the AABB midpoint; radius reaches the farthest vertex from that center.
 * Typed-array additions derive it once, not per frame. */
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

function addMesh(world: World, spec: MeshData): MeshHandle {
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
            `mesh "${spec.name}": Meshes.add needs MeshPlugin initialized; add from a plugin's initialize or later, not AppConfig.setup`,
        );
    }
    const device = world.gpu.device;
    if (!device) throw new Error(`mesh "${spec.name}": Meshes.add needs a GPU device`);
    // Reserve a handle before its family is packed.
    resources.placeholderVertices ??= world.gpu.root
        .createBuffer(d.arrayOf(d.vec4u, 1))
        .$usage("storage")
        .$name("shallot-mesh-pending-vertices");
    resources.placeholderIndices ??= world.gpu.root
        .createBuffer(d.arrayOf(d.u32, 1))
        .$usage("storage", "index")
        .$name("shallot-mesh-pending-indices");
    const bounds = meshBounds(spec.vertices);
    const handle = resources.meshes.add({
        name: spec.name,
        vertices: resources.placeholderVertices,
        indices: resources.placeholderIndices,
        indexBase: 0,
        indexCount: spec.indices.length,
        bounds,
        pending: true,
    });
    resources.pending.push({ ...spec, handle, bounds });
    return handle;
}

function packMeshes(staged: PendingMesh[]): {
    vertices: Float32Array;
    indices: Uint32Array;
    slices: {
        handle: MeshHandle;
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
        handle: MeshHandle;
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
            handle: m.handle,
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
 * GPU streams for one family. `main` is 16 B/vertex (`vec4<u32>`): w0 = unorm16 pos.xy,
 * w1 = unorm16 pos.z | (meshId << 16), w2 = oct normal, w3 = unorm16 uv.
 * `position` is the 8 B/vertex depth/shadow stream (w0, w1: pos + meshId only).
 * `quant` is `MeshQuant` per mesh (the position + uv AABB the decode dequantizes
 * against, selected by meshId). Only the GPU mirror quantizes; authoring data stays f32.
 */
export interface QuantStreams {
    main: Uint32Array;
    position: Uint32Array;
    quant: Float32Array;
}

// Per-mesh AABBs keep small meshes from losing precision to a family's larger meshes.
// Packed meshId selects MeshQuant without a per-draw uniform, including in render bundles.
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

// No bind group holds a placeholder: standard skips pending entries before binding them.
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
 * quantized vertex streams and a shared index buffer. Replace each entry with its slice;
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
    const bounds = new Map(staged.map((m) => [m.handle, m.bounds]));
    for (const s of packed.slices) {
        resources.meshes.replace(s.handle, {
            name: s.name,
            vertices,
            position,
            quant,
            indices,
            indexBase: s.indexBase,
            indexCount: s.indexCount,
            bounds: bounds.get(s.handle),
            attributes,
        });
    }
}

/** Packs staged additions at the start of the draw group. */
export const PrepareMeshesSystem: System = {
    name: "prepareMeshes",
    group: "draw",
    first: true,
    update: flushMeshes,
};

/** Clear the registry and staging, destroying packed family buffers but not producer-owned buffers.
 * Runs during MeshPlugin initialization; producers must re-stage geometry in their own initialize
 * so disabled producers leave no stale slices paired with live surfaces. */
export function clearMeshes(world: World): void {
    const resources = meshResources(world);
    resources.meshes.clear();
    resources.defaultMesh = DEFAULT_MESH_HANDLE;
    resetStaging(world);
    // initialize runs between frames, so no open encoder holds a family, submitted work keeps its
    // storage, and a cleared registry names none of them for any later bind group
    for (const buffer of resources.families) buffer.destroy();
    resources.families.length = 0;
}

/** Set MeshPlugin's built-in default after registering its primitives. */
export function setDefaultMeshHandle(world: World, handle: MeshHandle): void {
    meshResources(world).defaultMesh = handle;
}

/** Handle used when an entity adds MeshInstance without an explicit mesh. */
export function defaultMeshHandle(world: World): MeshHandle {
    return meshResources(world).defaultMesh;
}
