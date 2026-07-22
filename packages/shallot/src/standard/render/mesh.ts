import { Compute, Registry } from "../../engine";
import { octEncodeNormal, pack2x16unorm } from "../../engine/utils/core";

/**
 * registered vertex-pull geometry: a slice descriptor into the quantized vertex
 * streams + `indices` GPU storage. `vertices` is the 16 B/vertex main stream
 * (`vec4<u32>`: unorm16 pos + meshId, oct normal, unorm16 uv); `position` is the
 * 8 B/vertex depth/shadow stream (pos + meshId only); `quant` is the per-mesh
 * `MeshQuant` table the decode dequantizes against (gpu.md rule 6). `indices` is
 * `u32` absolute vertex positions. `indexBase`/`indexCount` slice the index
 * stream. All have `STORAGE` usage: consumer renderers pull indexed vertices in
 * WGSL, never via `setVertexBuffer` / `setIndexBuffer`.
 *
 * The registry is source-agnostic. Static producers stage typed arrays via
 * {@link mesh}, which {@link flushMeshes} packs into one shared family buffer
 * set — every static mesh is a slice (its own `indexBase` + meshId) of the same
 * buffers, so sear binds geometry once and the layout is `multi-draw-indirect`
 * ready. Procedural producers (compute-driven terrain, particle ribbons) allocate
 * their own `GPUBuffer`s (emitting the same quantized format via `POS_QUANT_PACK_WGSL`)
 * and register directly. Meshes sharing a buffer set share a bind group in sear
 *
 * `bounds` is the local-space bounding sphere `[cx, cy, cz, radius]` a producer's
 * frustum cull transforms per instance. {@link mesh} derives it from the staged
 * vertices; procedural producers may supply their own or omit it (a culler then
 * treats the mesh as always-visible)
 *
 * @expand
 */
export interface Mesh {
    name: string;
    vertices: GPUBuffer;
    /** the 8 B/vertex position-only stream the depth + shadow passes pull (sear binds this in the prepass group) */
    position?: GPUBuffer;
    /** the per-mesh `MeshQuant` dequant table (position + uv AABB), indexed by the meshId packed in the stream */
    quant?: GPUBuffer;
    indices: GPUBuffer;
    indexBase: number;
    indexCount: number;
    bounds?: [number, number, number, number];
    /**
     * whether this mesh's geometry changes after registration. `false` (default) =
     * static: a consumer that builds a per-mesh acceleration structure (the RT-shadow
     * BLAS) builds it once and reuses it. `true` = the `vertices`/`indices` are rewritten per
     * frame (a deforming or compute-emitted mesh), so the structure rebuilds every frame. A
     * mesh whose geometry changes but is left static casts stale shadows. Mark it dynamic.
     */
    dynamic?: boolean;
    /**
     * optional GPU buffer whose `[0]` is this mesh's *live* index count, ≤ `indexCount`. A
     * compute-emitting producer that materializes only its live elements supplies it so the
     * RT-shadow BLAS builds over the live triangle range each frame, not the registered cap:
     * the GPU-count contract, the count never crossing to the CPU. Omit for a fixed mesh: its
     * `indexCount` is the live count. Pair with `dynamic: true`.
     */
    count?: GPUBuffer;
    /**
     * whether the RT-shadow caster builds a BLAS for this mesh. `true` (default) = every mesh
     * casts (the caster auto-builds its BLAS; a producer contributes instances). Set `false` for
     * a **draw-only** mesh that another mesh already casts for: a producer that materializes a
     * world-space draw mesh but casts via a deduped object-space copy would otherwise reserve a
     * redundant slot in the shared caster budget for the draw mesh.
     */
    cast?: boolean;
    /**
     * a surface-specialization index (default 0): for a draw whose surface declares
     * {@link Surface.specialize}, it selects the compiled pipeline variant. The glTF importer sets it to
     * a primitive's material map-set bitmask so a textured draw samples only the maps its material
     * carries; a mesh whose surface doesn't specialize ignores it. Constant per mesh (a mesh is one glTF
     * primitive = one material), so it specializes the `(surface, mesh)` draw with no per-instance branch.
     */
    variant?: number;
    /**
     * per-mesh binding overrides: resources scoped to *this* mesh's draws, keyed by the surface's binding
     * name. A surface binding resolves to `mesh.bindings?.[name]` when present, else the published global
     * (`Compute.*`). The skinned-mesh VAT is the worked case: each skinned mesh owns its position/normal VAT
     * textures + params, so N skinned meshes coexist in one scene (the textured firehose shares its albedo
     * arrays globally; a VAT can't (different size per mesh), so it binds per-mesh). A mesh is already its
     * own bind group (own geometry buffers), so this adds no draw.
     */
    bindings?: Record<string, GPUTexture | GPUSampler | GPUBuffer>;
}

/** every registered mesh, keyed by name with a stable numeric ID */
export const Meshes: Registry<Mesh> = new Registry<Mesh>();

/** bytes per vertex in the **f32 staging array** producers fill (8 floats × 4 = 32 B). The lossless
 *  authoring layout. {@link quantizeMeshes} packs it to the 16 B GPU main stream + 8 B position stream
 *  (gpu.md rule 6); a GPU producer writes those directly via `POS_QUANT_PACK_WGSL`. */
export const VERTEX_STRIDE = 32;

/** f32 lanes per vertex in the staging array: `px py pz u  nx ny nz v` (the `posU` + `normalV` authoring layout) */
export const VERTEX_FLOATS = 8;

// static meshes pack into one shared family buffer pair. `mesh()` stages the
// typed arrays + a placeholder registry entry (so `Meshes.size` is final after
// `initialize`, before any warm); `flushMeshes()` concatenates them at warm.
interface PendingMesh {
    name: string;
    vertices: Float32Array;
    indices: Uint32Array;
    bounds: [number, number, number, number];
}
const _pending: PendingMesh[] = [];
let _placeholder: GPUBuffer | null = null;

/**
 * local-space axis-aligned bounds `{ min, max }` of a vertex buffer (the shared
 * `posU + normalV` layout, position in the first three floats per record).
 * Pure: the one position-AABB scan both the cull sphere ({@link meshBounds})
 * and the unorm16 dequant range (the per-mesh `MeshQuant`, gpu.md rule 6) derive
 * from, so they share one source. An empty buffer returns a zero box.
 */
export function meshAabb(vertices: Float32Array): {
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
 * register a static mesh from typed arrays. The data is staged now and packed
 * into the shared family buffer at warm by {@link flushMeshes}; the registry
 * entry is a slice of that shared buffer. Procedural producers skip this: they
 * own their `GPUBuffer`s and call `Meshes.register(...)` directly. Requires
 * `Compute.device`; no-ops otherwise
 *
 * @example
 * mesh({ name: "cube", vertices, indices })
 */
export function mesh(spec: { name: string; vertices: Float32Array; indices: Uint32Array }): void {
    if (spec.vertices.length % VERTEX_FLOATS !== 0) {
        throw new Error(
            `mesh "${spec.name}": vertices length ${spec.vertices.length} is not a multiple of ${VERTEX_FLOATS} (one Vertex = posU + normalV)`,
        );
    }
    const device = Compute.device;
    if (!device) return;
    // a placeholder reserves the registry entry now (fixing Meshes.size before
    // warm); flushMeshes swaps in the real shared buffer + correct indexBase
    _placeholder ??= device.createBuffer({
        label: "kitchen-mesh-pending",
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDEX,
    });
    const bounds = meshBounds(spec.vertices);
    _pending.push({ ...spec, bounds });
    Meshes.register({
        name: spec.name,
        vertices: _placeholder,
        indices: _placeholder,
        indexBase: 0,
        indexCount: spec.indices.length,
        bounds,
    });
}

/**
 * concatenate staged meshes into one vertex + index pair, shifting each mesh's
 * indices by its vertex base so the index stream holds absolute positions.
 * Pure: exported for the index-shift test; {@link flushMeshes} uploads it
 */
export function packMeshes(
    staged: { name: string; vertices: Float32Array; indices: Uint32Array }[],
): {
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
 * (gpu.md rule 6). `main` is 16 B/vertex (`vec4<u32>`): w0 = unorm16 pos.xy,
 * w1 = unorm16 pos.z | (meshId << 16), w2 = oct normal, w3 = unorm16 uv.
 * `position` is the 8 B/vertex depth/shadow stream (w0, w1: pos + meshId only).
 * `quant` is `MeshQuant` per mesh (the position + uv AABB the decode dequantizes
 * against, selected by meshId). The f32 stays the lossless authoring form: only
 * the GPU mirror quantizes (the slab packed-mirror discipline, ecs.md).
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
 * per-draw uniform, works unchanged in render bundles. Pure: the single emitter
 * both `flushMeshes` and the glTF importer call, paired with the WGSL `decodePos`
 * (`POS_QUANT_WGSL`) so the lattice can't drift between writer and reader.
 */
export function quantizeMeshes(
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
            const w0 = pack2x16unorm(
                norm(vertices[i], pmin[0], pext[0]),
                norm(vertices[i + 1], pmin[1], pext[1]),
            );
            const z16 = Math.round(
                Math.max(0, Math.min(1, norm(vertices[i + 2], pmin[2], pext[2]))) * 65535,
            );
            const w1 = ((z16 & 0xffff) | (meshId << 16)) >>> 0;
            const w2 = octEncodeNormal({
                x: vertices[i + 4],
                y: vertices[i + 5],
                z: vertices[i + 6],
            });
            const w3 = pack2x16unorm(
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

// drop the staged-but-unflushed mesh data + the placeholder buffer. flushMeshes calls it after packing,
// clearMeshes after discarding — one source of truth for the staging state to reset.
function resetStaging(): void {
    _pending.length = 0;
    _placeholder?.destroy();
    _placeholder = null;
}

/**
 * pack every staged static mesh into the quantized vertex streams + a shared
 * index buffer and re-register each as a slice. Called once from
 * `RenderPlugin.warm`, after all `initialize` hooks (so every `mesh(...)` has run)
 */
export function flushMeshes(): void {
    const device = Compute.device;
    if (!device || _pending.length === 0) return;
    const packed = packMeshes(_pending);
    const q = quantizeMeshes(packed.vertices, packed.slices);
    const storage = (label: string, data: Uint32Array | Float32Array) => {
        const buf = device.createBuffer({
            label,
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buf, 0, data as Uint32Array<ArrayBuffer>);
        return buf;
    };
    const vertices = storage("kitchen-mesh-main", q.main);
    const position = storage("kitchen-mesh-pos", q.position);
    const quant = storage("kitchen-mesh-quant", q.quant);
    const indices = device.createBuffer({
        label: "kitchen-mesh-indices",
        size: packed.indices.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(indices, 0, packed.indices as Uint32Array<ArrayBuffer>);
    const bounds = new Map(_pending.map((m) => [m.name, m.bounds]));
    for (const s of packed.slices) {
        Meshes.register({
            name: s.name,
            vertices,
            position,
            quant,
            indices,
            indexBase: s.indexBase,
            indexCount: s.indexCount,
            bounds: bounds.get(s.name),
        });
    }
    resetStaging();
}

/**
 * drop every registered mesh + any staged-but-unflushed data, resetting the registry for a fresh build
 * (`RenderPlugin.initialize`, ecs.md "clear then rebuild"). Static producers re-stage via {@link mesh} in
 * their own initialize, so a producer toggled off leaves no stale slice to be paired against
 * a live surface (the pack registers a Draw per `(surface, mesh)` pair, including a dead one otherwise).
 */
export function clearMeshes(): void {
    Meshes.clear();
    resetStaging();
}
