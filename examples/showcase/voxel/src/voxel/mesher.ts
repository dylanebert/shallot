// The GPU face-cull mesher (Phase 2): a compute pass walks the voxel grid and emits an exposed-face quad
// for every solid voxel whose neighbour across that face is air, atomically appending vertices / indices
// into producer-owned buffers and the index count into an indirect draw record. Sear rasterizes the
// result through one `drawIndirect` — the mesh never crosses back to the CPU. The 3D promotion of
// a 2D terrain emit: swap "cliff per non-ground
// neighbour" for "cube face per air neighbour", in all six directions including interior (the bored
// tunnel) and across chunk seams (the cross-chunk sphere). Single pass over a static grid — the two-pass
// count-then-`dispatchWorkgroupsIndirect` and per-chunk draws arrive with streaming (Phase 4).

import { Compute, type Plugin, RenderPlugin, type State, type System } from "@dylanebert/shallot";
import {
    BeginFrameSystem,
    type Draw,
    DrawIndexedIndirect,
    Draws,
    type Mesh,
    Meshes,
    Render,
} from "@dylanebert/shallot/render/core";
import { precompile, precompileScope } from "@dylanebert/shallot/runtime";
import {
    fsCtxSchema,
    lit,
    PrepassSystem,
    registerSurface,
    surfaceLayout,
} from "@dylanebert/shallot/sear/core";
import {
    encodePos,
    encodeUv,
    idiv,
    MeshQuant,
    octEncodeNormal,
} from "@dylanebert/shallot/utils/core";
import type { TgpuBindGroup } from "typegpu";
import tgpu, { writeToArrayBuffer } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { BINDING_FLOOR, BYTES, CHUNK_CELLS, DIM, GridData, ISO, VOXEL, voxelIndex } from "./grid";

const VERTS_PER_QUAD = 4;
const INDICES_PER_QUAD = 6;
const MAIN_STRIDE = 16; // the quantized main stream: vec4<u32> per vertex
const POS_STRIDE = 8; // the position-only depth stream: vec2<u32> per vertex
export const WG = 4; // workgroup edge → 4³ = 64 threads

// the grid's analytic MeshQuant (gpu.md rule 6): position spans [-HALF·VOXEL, +HALF·VOXEL] per axis
// (extent DIM·VOXEL); uv is constant (blockU=1, v=0), so its range is degenerate. The emit shader holds
// the same record as a const (GRID_QUANT) — one source, two emitters: this for sear's decode, that for
// the producer's encode. Layout: posOffset(posMin.xyz, uvMin.x), posScale(posExt.xyz, uvMin.y), uvScale
const GRID_QUANT = new Float32Array([
    -0.5 * DIM.x * VOXEL,
    -0.5 * DIM.y * VOXEL,
    -0.5 * DIM.z * VOXEL,
    1.0,
    DIM.x * VOXEL,
    DIM.y * VOXEL,
    DIM.z * VOXEL,
    0.0,
    0,
    0,
    0,
    0,
]);
export const DISPATCH = {
    x: Math.ceil(DIM.x / WG),
    y: Math.ceil(DIM.y / WG),
    z: Math.ceil(DIM.z / WG),
};

// the 16 B main stream must fit one storage binding (the largest of the three per-face buffers), so cap
// faces at ¾ of the portable 128 MiB floor. The canonical worst case (the full ground slab, ~134k faces)
// sits far under this; a grid that would exceed it overflows the guard in `emitQuad` and the count gate
// (atomic vs `faces()` oracle) catches the cap.
export const MAX_FACES = Math.floor((BINDING_FLOOR * 3) / 4 / (VERTS_PER_QUAD * MAIN_STRIDE));

/**
 * the one voxel world: the CPU grid (`data`, null when generated GPU-side), the GPU grid buffer (`grid` —
 * what the mesher reads and the generator writes), and the mesher's outputs. A scenario either authors
 * `data` before the first frame (the way a scene declares entities) and the mesher uploads + meshes it, or
 * leaves it null and fills `grid` directly on the GPU; either way it exposes `indirect` — the draw record
 * whose first word is the index count, mirrored to read the atomic face count back. {@link uploadVoxels}
 * rewrites the grid and re-meshes (the assert's per-pattern swap; Phase 5's carve will write through it too).
 */
export const Voxels = {
    data: null as Float32Array | null,
    grid: null as GPUBuffer | null,
    indirect: null as GPUBuffer | null,
    dirty: false,
};

/** read-only observability for the perf gate (`showcase-frame-floor` S2): the workgroup count issued by
 *  the most recent emit dispatch, set only by {@link VoxelEmitSystem} — no production code reads it, so it
 *  changes no behavior. `test/perf.spec.ts` (via `src/perf.ts`) gates dispatch scope against touched-chunk
 *  volume; today it's always the full-grid `DISPATCH` product, regardless of what fired the re-mesh. */
export const EmitTelemetry = { lastWorkgroups: 0 };

const gpu = {
    voxels: null as GPUBuffer | null,
    vertices: null as GPUBuffer | null,
    position: null as GPUBuffer | null,
    quant: null as GPUBuffer | null,
    indices: null as GPUBuffer | null,
    pipeline: null as ReturnType<typeof Compute.root.createComputePipeline> | null,
    bindGroup: null as TgpuBindGroup<typeof mesherLayout.entries> | null,
};

let warmEpoch = 0;

const VoxelVertices = d.arrayOf(d.vec4u, MAX_FACES * VERTS_PER_QUAD);
const VoxelPosition = d.arrayOf(d.vec2u, MAX_FACES * VERTS_PER_QUAD);
const VoxelIndices = d.arrayOf(d.u32, MAX_FACES * INDICES_PER_QUAD);
const VoxelAtomicIndirect = d.arrayOf(d.atomic(d.u32), 1);

type WarmOwner<Buffer, MeshEntry extends { name: string }, DrawEntry extends { name: string }> = {
    readonly buffers: Buffer[];
    readonly meshes: MeshEntry[];
    readonly draws: DrawEntry[];
};

type WarmCleanupOps<
    Buffer,
    MeshEntry extends { name: string },
    DrawEntry extends { name: string },
> = {
    mesh: (name: string) => MeshEntry | undefined;
    draw: (name: string) => DrawEntry | undefined;
    deleteMesh: (name: string) => void;
    deleteDraw: (name: string) => void;
    destroy: (buffer: Buffer) => void;
    active: (owner: WarmOwner<Buffer, MeshEntry, DrawEntry>) => boolean;
    clearActive: () => void;
};

const cleanedWarmOwners = new WeakSet<object>();

/** @internal */
export function cleanupWarmOwner<
    Buffer,
    MeshEntry extends { name: string },
    DrawEntry extends { name: string },
>(
    owner: WarmOwner<Buffer, MeshEntry, DrawEntry>,
    ops: WarmCleanupOps<Buffer, MeshEntry, DrawEntry>,
): void {
    if (cleanedWarmOwners.has(owner)) return;
    cleanedWarmOwners.add(owner);
    for (const mesh of owner.meshes) {
        if (ops.mesh(mesh.name) === mesh) ops.deleteMesh(mesh.name);
    }
    for (const draw of owner.draws) {
        if (ops.draw(draw.name) === draw) ops.deleteDraw(draw.name);
    }
    for (const buffer of owner.buffers) ops.destroy(buffer);
    if (ops.active(owner)) ops.clearActive();
}

type WarmState = { readonly disposed: boolean; onDispose: (cleanup: () => void) => void };

type WarmLifecycleOps<
    StateKey extends WarmState,
    Owner extends object,
    Prepared,
    Buffer,
    Binding,
> = {
    current: () => Owner | null;
    owned: (state: StateKey) => Owner | undefined;
    create: (state: StateKey) => Owner;
    activate: (owner: Owner) => void;
    cleanup: (owner: Owner) => void;
    prepare: (owner: Owner) => Prepared;
    precompile: (owner: Owner, prepared: Prepared, force: () => unknown) => Promise<void>;
    createWarmBinding: (own: (buffer: Buffer) => void) => Binding;
    force: (prepared: Prepared, binding: Binding) => unknown;
    destroyWarm: (buffer: Buffer) => void;
    publish: (owner: Owner, prepared: Prepared) => void;
};

/** @internal */
export function createWarmLifecycle<
    StateKey extends WarmState,
    Owner extends object,
    Prepared,
    Buffer,
    Binding,
>(ops: WarmLifecycleOps<StateKey, Owner, Prepared, Buffer, Binding>) {
    const warm = async (state: StateKey): Promise<void> => {
        if (state.disposed) return;
        const prior = ops.current();
        if (prior) ops.cleanup(prior);

        const owner = ops.create(state);
        ops.activate(owner);
        state.onDispose(() => ops.cleanup(owner));
        if (ops.current() !== owner) return;

        try {
            const prepared = ops.prepare(owner);
            const force = () => {
                const buffers: Buffer[] = [];
                try {
                    const binding = ops.createWarmBinding((buffer) => buffers.push(buffer));
                    return ops.force(prepared, binding);
                } finally {
                    for (const buffer of buffers) ops.destroyWarm(buffer);
                }
            };
            await ops.precompile(owner, prepared, force);
            if (ops.current() !== owner) return;
            ops.publish(owner, prepared);
        } catch (cause) {
            ops.cleanup(owner);
            throw cause;
        }
    };
    const dispose = (state: StateKey): void => {
        const owner = ops.owned(state);
        if (owner) ops.cleanup(owner);
    };
    return { warm, dispose };
}

type VoxelWarmOwner = WarmOwner<GPUBuffer, Mesh, Draw> & { readonly state: State };
let activeOwner: VoxelWarmOwner | null = null;
const stateOwners = new WeakMap<State, VoxelWarmOwner>();

/** rewrite the grid and mark it dirty so the next frame re-meshes. No-op before the mesher's buffers
 *  exist (first-frame setup uploads `Voxels.data` itself). */
export function uploadVoxels(data: Float32Array): void {
    Voxels.data = data;
    if (!gpu.voxels) return;
    Compute.device.queue.writeBuffer(gpu.voxels, 0, data as Float32Array<ArrayBuffer>);
    Voxels.dirty = true;
}

/** one-shot GPU→CPU readback of the whole grid buffer — the CPU twin the voxel gate runs the `faces()`
 *  oracle + density stats over a GPU-generated grid (the path with no CPU-authoritative `Voxels.data`).
 *  64 MiB per call, so an assert-only bridge, never a per-frame readback. */
export async function readGrid(): Promise<Float32Array> {
    if (!gpu.voxels) throw new Error("voxel: readGrid before the grid buffer exists");
    const { device } = Compute;
    const staging = device.createBuffer({
        label: "voxel-readback",
        size: BYTES,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = device.createCommandEncoder({ label: "voxel-readback" });
    enc.copyBufferToBuffer(gpu.voxels, 0, staging, 0, BYTES);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
}

/** pull the GPU grid back into the CPU mirror `Voxels.data`, so the GPU-generated terrain (no authoring
 *  copy) becomes carveable — the carve path DDA-marches + edits `Voxels.data`, then re-uploads its touched
 *  chunks. 64 MiB, so call after generate() (build / reseed), never per frame. A pattern path already holds
 *  the authored grid as the mirror, so it needs no sync. */
export async function syncGrid(): Promise<void> {
    Voxels.data = await readGrid();
}

/** push the edited chunk slices from the CPU mirror to the GPU grid and dirty so the next frame re-meshes.
 *  Chunk-major makes each chunk one contiguous range, so a carve uploads only the slots it touched — not the
 *  whole 64 MiB. {@link Voxels.data} is authoritative; this commits a {@link brush}'s changes. */
export function commitEdit(chunks: Iterable<number>): void {
    if (!gpu.voxels || !Voxels.data) return;
    const data = Voxels.data as Float32Array<ArrayBuffer>;
    let any = false;
    for (const slot of chunks) {
        const start = slot * CHUNK_CELLS;
        Compute.device.queue.writeBuffer(gpu.voxels, start * 4, data, start, CHUNK_CELLS);
        any = true;
    }
    if (any) Voxels.dirty = true;
}

// posU.w carries a per-vertex material slot (read as `uv.x` in the fs) for the Phase-3 palette — one block
// type today, so it's a constant 1.0 placeholder and the fs colours by face direction instead: grass-green
// tops, dirt-brown sides, to read the meshed structure (the six face directions, the tunnel's inward faces,
// the sphere's curvature). Linear base colours (sear's composite encodes sRGB), tuned to land near grass /
// dirt after `lit`'s ~1.66×.
const voxelLayout = surfaceLayout({});
const voxelFs = tgpu.fn(
    [fsCtxSchema()],
    d.vec4f,
)((ctx) => {
    "use gpu";
    const base = std.select(
        d.vec3f(0.1, 0.06, 0.03),
        d.vec3f(0.1, 0.2, 0.05),
        ctx.worldNormal.y > 0.5,
    );
    return d.vec4f(lit(base, ctx.worldNormal), 1);
});

const HALF_X = 0.5 * DIM.x;
const HALF_Y = 0.5 * DIM.y;
const HALF_Z = 0.5 * DIM.z;

const mesherLayout = tgpu.bindGroupLayout({
    voxels: { storage: GridData, access: "readonly" },
    vertices: { storage: VoxelVertices, access: "mutable" },
    indices: { storage: VoxelIndices, access: "mutable" },
    indirect: { storage: VoxelAtomicIndirect, access: "mutable" },
    position: { storage: VoxelPosition, access: "mutable" },
});

const gridQuant: d.Infer<typeof MeshQuant> = {
    posOffset: d.vec4f(-HALF_X * VOXEL, -HALF_Y * VOXEL, -HALF_Z * VOXEL, 1),
    posScale: d.vec4f(2 * HALF_X * VOXEL, 2 * HALF_Y * VOXEL, 2 * HALF_Z * VOXEL, 0),
    uvScale: d.vec4f(0, 0, 0, 0),
};

const emitKernel = tgpu
    .computeFn({
        workgroupSize: [WG, WG, WG],
        in: { gid: d.builtin.globalInvocationId },
    })((input) => {
        "use gpu";
        const gid = input.gid;
        if (gid.x >= d.u32(DIM.x) || gid.y >= d.u32(DIM.y) || gid.z >= d.u32(DIM.z)) return;
        if (mesherLayout.$.voxels[voxelIndex(gid.x, gid.y, gid.z)] < d.f32(ISO)) return;

        const ix = gid.x;
        const iy = gid.y;
        const iz = gid.z;
        const x0 = (d.f32(gid.x) - d.f32(HALF_X)) * d.f32(VOXEL);
        const y0 = (d.f32(gid.y) - d.f32(HALF_Y)) * d.f32(VOXEL);
        const z0 = (d.f32(gid.z) - d.f32(HALF_Z)) * d.f32(VOXEL);
        const x1 = x0 + d.f32(VOXEL);
        const y1 = y0 + d.f32(VOXEL);
        const z1 = z0 + d.f32(VOXEL);
        const blockU = d.f32(1);
        const uvw = encodeUv(d.vec2f(blockU, d.f32(0)), gridQuant);

        if (
            ix + d.u32(1) >= d.u32(DIM.x) ||
            mesherLayout.$.voxels[voxelIndex(ix + d.u32(1), iy, iz)] < d.f32(ISO)
        ) {
            const base = std.atomicAdd(mesherLayout.$.indirect[0], d.u32(INDICES_PER_QUAD));
            const fi = idiv(base, d.u32(INDICES_PER_QUAD));
            if (fi < d.u32(MAX_FACES)) {
                const v0 = fi * d.u32(VERTS_PER_QUAD);
                const octN = octEncodeNormal(d.vec3f(1, 0, 0));
                const m0 = encodePos(d.vec3f(x1, y0, z0), d.u32(0), gridQuant);
                const m1 = encodePos(d.vec3f(x1, y1, z0), d.u32(0), gridQuant);
                const m2 = encodePos(d.vec3f(x1, y1, z1), d.u32(0), gridQuant);
                const m3 = encodePos(d.vec3f(x1, y0, z1), d.u32(0), gridQuant);
                mesherLayout.$.vertices[v0 + d.u32(0)] = d.vec4u(m0.x, m0.y, octN, uvw);
                mesherLayout.$.vertices[v0 + d.u32(1)] = d.vec4u(m1.x, m1.y, octN, uvw);
                mesherLayout.$.vertices[v0 + d.u32(2)] = d.vec4u(m2.x, m2.y, octN, uvw);
                mesherLayout.$.vertices[v0 + d.u32(3)] = d.vec4u(m3.x, m3.y, octN, uvw);
                mesherLayout.$.position[v0 + d.u32(0)] = d.vec2u(m0.x, m0.y);
                mesherLayout.$.position[v0 + d.u32(1)] = d.vec2u(m1.x, m1.y);
                mesherLayout.$.position[v0 + d.u32(2)] = d.vec2u(m2.x, m2.y);
                mesherLayout.$.position[v0 + d.u32(3)] = d.vec2u(m3.x, m3.y);
                mesherLayout.$.indices[base + d.u32(0)] = v0 + d.u32(0);
                mesherLayout.$.indices[base + d.u32(1)] = v0 + d.u32(1);
                mesherLayout.$.indices[base + d.u32(2)] = v0 + d.u32(2);
                mesherLayout.$.indices[base + d.u32(3)] = v0 + d.u32(0);
                mesherLayout.$.indices[base + d.u32(4)] = v0 + d.u32(2);
                mesherLayout.$.indices[base + d.u32(5)] = v0 + d.u32(3);
            }
        }
        if (
            ix === d.u32(0) ||
            mesherLayout.$.voxels[voxelIndex(ix - d.u32(1), iy, iz)] < d.f32(ISO)
        ) {
            const base = std.atomicAdd(mesherLayout.$.indirect[0], d.u32(INDICES_PER_QUAD));
            const fi = idiv(base, d.u32(INDICES_PER_QUAD));
            if (fi < d.u32(MAX_FACES)) {
                const v0 = fi * d.u32(VERTS_PER_QUAD);
                const octN = octEncodeNormal(d.vec3f(-1, 0, 0));
                const m0 = encodePos(d.vec3f(x0, y0, z0), d.u32(0), gridQuant);
                const m1 = encodePos(d.vec3f(x0, y0, z1), d.u32(0), gridQuant);
                const m2 = encodePos(d.vec3f(x0, y1, z1), d.u32(0), gridQuant);
                const m3 = encodePos(d.vec3f(x0, y1, z0), d.u32(0), gridQuant);
                mesherLayout.$.vertices[v0 + d.u32(0)] = d.vec4u(m0.x, m0.y, octN, uvw);
                mesherLayout.$.vertices[v0 + d.u32(1)] = d.vec4u(m1.x, m1.y, octN, uvw);
                mesherLayout.$.vertices[v0 + d.u32(2)] = d.vec4u(m2.x, m2.y, octN, uvw);
                mesherLayout.$.vertices[v0 + d.u32(3)] = d.vec4u(m3.x, m3.y, octN, uvw);
                mesherLayout.$.position[v0 + d.u32(0)] = d.vec2u(m0.x, m0.y);
                mesherLayout.$.position[v0 + d.u32(1)] = d.vec2u(m1.x, m1.y);
                mesherLayout.$.position[v0 + d.u32(2)] = d.vec2u(m2.x, m2.y);
                mesherLayout.$.position[v0 + d.u32(3)] = d.vec2u(m3.x, m3.y);
                mesherLayout.$.indices[base + d.u32(0)] = v0 + d.u32(0);
                mesherLayout.$.indices[base + d.u32(1)] = v0 + d.u32(1);
                mesherLayout.$.indices[base + d.u32(2)] = v0 + d.u32(2);
                mesherLayout.$.indices[base + d.u32(3)] = v0 + d.u32(0);
                mesherLayout.$.indices[base + d.u32(4)] = v0 + d.u32(2);
                mesherLayout.$.indices[base + d.u32(5)] = v0 + d.u32(3);
            }
        }
        if (
            iy + d.u32(1) >= d.u32(DIM.y) ||
            mesherLayout.$.voxels[voxelIndex(ix, iy + d.u32(1), iz)] < d.f32(ISO)
        ) {
            const base = std.atomicAdd(mesherLayout.$.indirect[0], d.u32(INDICES_PER_QUAD));
            const fi = idiv(base, d.u32(INDICES_PER_QUAD));
            if (fi < d.u32(MAX_FACES)) {
                const v0 = fi * d.u32(VERTS_PER_QUAD);
                const octN = octEncodeNormal(d.vec3f(0, 1, 0));
                const m0 = encodePos(d.vec3f(x0, y1, z0), d.u32(0), gridQuant);
                const m1 = encodePos(d.vec3f(x0, y1, z1), d.u32(0), gridQuant);
                const m2 = encodePos(d.vec3f(x1, y1, z1), d.u32(0), gridQuant);
                const m3 = encodePos(d.vec3f(x1, y1, z0), d.u32(0), gridQuant);
                mesherLayout.$.vertices[v0 + d.u32(0)] = d.vec4u(m0.x, m0.y, octN, uvw);
                mesherLayout.$.vertices[v0 + d.u32(1)] = d.vec4u(m1.x, m1.y, octN, uvw);
                mesherLayout.$.vertices[v0 + d.u32(2)] = d.vec4u(m2.x, m2.y, octN, uvw);
                mesherLayout.$.vertices[v0 + d.u32(3)] = d.vec4u(m3.x, m3.y, octN, uvw);
                mesherLayout.$.position[v0 + d.u32(0)] = d.vec2u(m0.x, m0.y);
                mesherLayout.$.position[v0 + d.u32(1)] = d.vec2u(m1.x, m1.y);
                mesherLayout.$.position[v0 + d.u32(2)] = d.vec2u(m2.x, m2.y);
                mesherLayout.$.position[v0 + d.u32(3)] = d.vec2u(m3.x, m3.y);
                mesherLayout.$.indices[base + d.u32(0)] = v0 + d.u32(0);
                mesherLayout.$.indices[base + d.u32(1)] = v0 + d.u32(1);
                mesherLayout.$.indices[base + d.u32(2)] = v0 + d.u32(2);
                mesherLayout.$.indices[base + d.u32(3)] = v0 + d.u32(0);
                mesherLayout.$.indices[base + d.u32(4)] = v0 + d.u32(2);
                mesherLayout.$.indices[base + d.u32(5)] = v0 + d.u32(3);
            }
        }
        if (
            iy === d.u32(0) ||
            mesherLayout.$.voxels[voxelIndex(ix, iy - d.u32(1), iz)] < d.f32(ISO)
        ) {
            const base = std.atomicAdd(mesherLayout.$.indirect[0], d.u32(INDICES_PER_QUAD));
            const fi = idiv(base, d.u32(INDICES_PER_QUAD));
            if (fi < d.u32(MAX_FACES)) {
                const v0 = fi * d.u32(VERTS_PER_QUAD);
                const octN = octEncodeNormal(d.vec3f(0, -1, 0));
                const m0 = encodePos(d.vec3f(x0, y0, z0), d.u32(0), gridQuant);
                const m1 = encodePos(d.vec3f(x1, y0, z0), d.u32(0), gridQuant);
                const m2 = encodePos(d.vec3f(x1, y0, z1), d.u32(0), gridQuant);
                const m3 = encodePos(d.vec3f(x0, y0, z1), d.u32(0), gridQuant);
                mesherLayout.$.vertices[v0 + d.u32(0)] = d.vec4u(m0.x, m0.y, octN, uvw);
                mesherLayout.$.vertices[v0 + d.u32(1)] = d.vec4u(m1.x, m1.y, octN, uvw);
                mesherLayout.$.vertices[v0 + d.u32(2)] = d.vec4u(m2.x, m2.y, octN, uvw);
                mesherLayout.$.vertices[v0 + d.u32(3)] = d.vec4u(m3.x, m3.y, octN, uvw);
                mesherLayout.$.position[v0 + d.u32(0)] = d.vec2u(m0.x, m0.y);
                mesherLayout.$.position[v0 + d.u32(1)] = d.vec2u(m1.x, m1.y);
                mesherLayout.$.position[v0 + d.u32(2)] = d.vec2u(m2.x, m2.y);
                mesherLayout.$.position[v0 + d.u32(3)] = d.vec2u(m3.x, m3.y);
                mesherLayout.$.indices[base + d.u32(0)] = v0 + d.u32(0);
                mesherLayout.$.indices[base + d.u32(1)] = v0 + d.u32(1);
                mesherLayout.$.indices[base + d.u32(2)] = v0 + d.u32(2);
                mesherLayout.$.indices[base + d.u32(3)] = v0 + d.u32(0);
                mesherLayout.$.indices[base + d.u32(4)] = v0 + d.u32(2);
                mesherLayout.$.indices[base + d.u32(5)] = v0 + d.u32(3);
            }
        }
        if (
            iz + d.u32(1) >= d.u32(DIM.z) ||
            mesherLayout.$.voxels[voxelIndex(ix, iy, iz + d.u32(1))] < d.f32(ISO)
        ) {
            const base = std.atomicAdd(mesherLayout.$.indirect[0], d.u32(INDICES_PER_QUAD));
            const fi = idiv(base, d.u32(INDICES_PER_QUAD));
            if (fi < d.u32(MAX_FACES)) {
                const v0 = fi * d.u32(VERTS_PER_QUAD);
                const octN = octEncodeNormal(d.vec3f(0, 0, 1));
                const m0 = encodePos(d.vec3f(x0, y0, z1), d.u32(0), gridQuant);
                const m1 = encodePos(d.vec3f(x1, y0, z1), d.u32(0), gridQuant);
                const m2 = encodePos(d.vec3f(x1, y1, z1), d.u32(0), gridQuant);
                const m3 = encodePos(d.vec3f(x0, y1, z1), d.u32(0), gridQuant);
                mesherLayout.$.vertices[v0 + d.u32(0)] = d.vec4u(m0.x, m0.y, octN, uvw);
                mesherLayout.$.vertices[v0 + d.u32(1)] = d.vec4u(m1.x, m1.y, octN, uvw);
                mesherLayout.$.vertices[v0 + d.u32(2)] = d.vec4u(m2.x, m2.y, octN, uvw);
                mesherLayout.$.vertices[v0 + d.u32(3)] = d.vec4u(m3.x, m3.y, octN, uvw);
                mesherLayout.$.position[v0 + d.u32(0)] = d.vec2u(m0.x, m0.y);
                mesherLayout.$.position[v0 + d.u32(1)] = d.vec2u(m1.x, m1.y);
                mesherLayout.$.position[v0 + d.u32(2)] = d.vec2u(m2.x, m2.y);
                mesherLayout.$.position[v0 + d.u32(3)] = d.vec2u(m3.x, m3.y);
                mesherLayout.$.indices[base + d.u32(0)] = v0 + d.u32(0);
                mesherLayout.$.indices[base + d.u32(1)] = v0 + d.u32(1);
                mesherLayout.$.indices[base + d.u32(2)] = v0 + d.u32(2);
                mesherLayout.$.indices[base + d.u32(3)] = v0 + d.u32(0);
                mesherLayout.$.indices[base + d.u32(4)] = v0 + d.u32(2);
                mesherLayout.$.indices[base + d.u32(5)] = v0 + d.u32(3);
            }
        }
        if (
            iz === d.u32(0) ||
            mesherLayout.$.voxels[voxelIndex(ix, iy, iz - d.u32(1))] < d.f32(ISO)
        ) {
            const base = std.atomicAdd(mesherLayout.$.indirect[0], d.u32(INDICES_PER_QUAD));
            const fi = idiv(base, d.u32(INDICES_PER_QUAD));
            if (fi < d.u32(MAX_FACES)) {
                const v0 = fi * d.u32(VERTS_PER_QUAD);
                const octN = octEncodeNormal(d.vec3f(0, 0, -1));
                const m0 = encodePos(d.vec3f(x0, y0, z0), d.u32(0), gridQuant);
                const m1 = encodePos(d.vec3f(x0, y1, z0), d.u32(0), gridQuant);
                const m2 = encodePos(d.vec3f(x1, y1, z0), d.u32(0), gridQuant);
                const m3 = encodePos(d.vec3f(x1, y0, z0), d.u32(0), gridQuant);
                mesherLayout.$.vertices[v0 + d.u32(0)] = d.vec4u(m0.x, m0.y, octN, uvw);
                mesherLayout.$.vertices[v0 + d.u32(1)] = d.vec4u(m1.x, m1.y, octN, uvw);
                mesherLayout.$.vertices[v0 + d.u32(2)] = d.vec4u(m2.x, m2.y, octN, uvw);
                mesherLayout.$.vertices[v0 + d.u32(3)] = d.vec4u(m3.x, m3.y, octN, uvw);
                mesherLayout.$.position[v0 + d.u32(0)] = d.vec2u(m0.x, m0.y);
                mesherLayout.$.position[v0 + d.u32(1)] = d.vec2u(m1.x, m1.y);
                mesherLayout.$.position[v0 + d.u32(2)] = d.vec2u(m2.x, m2.y);
                mesherLayout.$.position[v0 + d.u32(3)] = d.vec2u(m3.x, m3.y);
                mesherLayout.$.indices[base + d.u32(0)] = v0 + d.u32(0);
                mesherLayout.$.indices[base + d.u32(1)] = v0 + d.u32(1);
                mesherLayout.$.indices[base + d.u32(2)] = v0 + d.u32(2);
                mesherLayout.$.indices[base + d.u32(3)] = v0 + d.u32(0);
                mesherLayout.$.indices[base + d.u32(4)] = v0 + d.u32(2);
                mesherLayout.$.indices[base + d.u32(5)] = v0 + d.u32(3);
            }
        }
    })
    .$name("voxel-emit-kernel");

export function emitWgsl(): string {
    return tgpu.resolve([octEncodeNormal, encodePos, encodeUv, voxelIndex, emitKernel], {
        names: "strict",
    });
}

const INDIRECT_INIT = new ArrayBuffer(d.sizeOf(DrawIndexedIndirect));
writeToArrayBuffer(INDIRECT_INIT, DrawIndexedIndirect, {
    indexCount: 0,
    instanceCount: 1,
    firstIndex: 0,
    baseVertex: 0,
    firstInstance: 0,
});

export const VoxelEmitSystem: System = {
    name: "voxel-emit",
    group: "draw",
    annotations: { mode: "always" },
    after: [BeginFrameSystem],
    // re-emit before sear reads geometry. Sear reads the same vertex/index buffers across its prepass,
    // shadow map, and color passes within a frame, so the emit must land ahead of all of them or a
    // re-mesh would tear new geometry against a stale read. `before: [PrepassSystem]` (the geometry-emit
    // anchor) pins it; `after: [BeginFrameSystem]` alone wouldn't.
    before: [PrepassSystem],
    setup() {},
    update() {
        if (!Voxels.dirty || !gpu.pipeline || !gpu.bindGroup || !Render.encoder) return;
        Compute.device.queue.writeBuffer(Voxels.indirect!, 0, INDIRECT_INIT);
        const pass = Render.encoder.beginComputePass({
            label: "voxel-emit",
            timestampWrites: Compute.span?.("voxel:emit"),
        });
        gpu.pipeline
            .with(gpu.bindGroup)
            .with(pass)
            .dispatchWorkgroups(DISPATCH.x, DISPATCH.y, DISPATCH.z);
        EmitTelemetry.lastWorkgroups = DISPATCH.x * DISPATCH.y * DISPATCH.z;
        pass.end();
        Voxels.dirty = false;
    },
};

type VoxelRawBuffers = {
    readonly voxels: GPUBuffer;
    readonly vertices: GPUBuffer;
    readonly position: GPUBuffer;
    readonly quant: GPUBuffer;
    readonly indices: GPUBuffer;
    readonly indirect: GPUBuffer;
};

function allocateVoxelBuffers(
    device: GPUDevice,
    own: (buffer: GPUBuffer) => void,
): VoxelRawBuffers {
    const allocate = (descriptor: GPUBufferDescriptor): GPUBuffer => {
        const buffer = device.createBuffer(descriptor);
        own(buffer);
        return buffer;
    };
    const maxVerts = MAX_FACES * VERTS_PER_QUAD;
    const maxIndices = MAX_FACES * INDICES_PER_QUAD;
    const voxels = allocate({
        label: "voxel-grid",
        size: BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const vertices = allocate({
        label: "voxel-main",
        size: maxVerts * MAIN_STRIDE,
        usage: GPUBufferUsage.STORAGE,
    });
    const position = allocate({
        label: "voxel-pos",
        size: maxVerts * POS_STRIDE,
        usage: GPUBufferUsage.STORAGE,
    });
    const quant = allocate({
        label: "voxel-quant",
        size: GRID_QUANT.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(quant, 0, GRID_QUANT as Float32Array<ArrayBuffer>);
    const indices = allocate({
        label: "voxel-indices",
        size: maxIndices * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDEX,
    });
    const indirect = allocate({
        label: "voxel-indirect",
        size: 20,
        usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.INDIRECT |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
    });
    return { voxels, vertices, position, quant, indices, indirect };
}

function wrapVoxelBuffers(raw: VoxelRawBuffers) {
    const typedVertices = Compute.root.createBuffer(VoxelVertices, raw.vertices).$usage("storage");
    const typedPosition = Compute.root.createBuffer(VoxelPosition, raw.position).$usage("storage");
    const typedQuant = Compute.root
        .createBuffer(d.arrayOf(MeshQuant, 1), raw.quant)
        .$usage("storage");
    const typedIndices = Compute.root
        .createBuffer(VoxelIndices, raw.indices)
        .$usage("storage", "index");
    const typedVoxels = Compute.root.createBuffer(GridData, raw.voxels).$usage("storage");
    const typedAtomicIndirect = Compute.root
        .createBuffer(VoxelAtomicIndirect, raw.indirect)
        .$usage("storage");
    const typedIndirect = Compute.root
        .createBuffer(DrawIndexedIndirect, raw.indirect)
        .$usage("storage", "indirect");
    const bindGroup = Compute.root.createBindGroup(mesherLayout, {
        voxels: typedVoxels,
        vertices: typedVertices,
        indices: typedIndices,
        indirect: typedAtomicIndirect,
        position: typedPosition,
    });
    return { typedVertices, typedPosition, typedQuant, typedIndices, typedIndirect, bindGroup };
}

function clearVoxelWarmOwner(owner: VoxelWarmOwner): void {
    cleanupWarmOwner(owner, {
        mesh: (name) => Meshes.get(name),
        draw: (name) => Draws.get(name),
        deleteMesh: (name) => Meshes.delete(name),
        deleteDraw: (name) => Draws.delete(name),
        destroy: (buffer) => buffer.destroy(),
        active: (candidate) => activeOwner === candidate,
        clearActive: () => {
            activeOwner = null;
            gpu.voxels = null;
            gpu.vertices = null;
            gpu.position = null;
            gpu.quant = null;
            gpu.indices = null;
            gpu.pipeline = null;
            gpu.bindGroup = null;
            Voxels.grid = null;
            Voxels.indirect = null;
            Voxels.dirty = false;
        },
    });
    if (stateOwners.get(owner.state) === owner) stateOwners.delete(owner.state);
}

type VoxelPrepared = {
    readonly raw: VoxelRawBuffers;
    readonly wrapped: ReturnType<typeof wrapVoxelBuffers>;
    readonly pipeline: ReturnType<typeof Compute.root.createComputePipeline>;
    readonly scope: string;
};

const voxelWarmLifecycle = createWarmLifecycle<
    State,
    VoxelWarmOwner,
    VoxelPrepared,
    GPUBuffer,
    TgpuBindGroup<typeof mesherLayout.entries>
>({
    current: () => activeOwner,
    owned: (state) => stateOwners.get(state),
    create: (state) =>
        Object.freeze({
            state,
            buffers: [] as GPUBuffer[],
            meshes: [] as Mesh[],
            draws: [] as Draw[],
        }),
    activate(owner) {
        activeOwner = owner;
        stateOwners.set(owner.state, owner);
    },
    cleanup: clearVoxelWarmOwner,
    prepare(owner) {
        const { device } = Compute;
        // COPY_SRC so readGrid() can mirror the grid back for the voxel gate; a fresh STORAGE buffer is
        // zero-initialised (all air), so the generated path (no CPU-authoritative data) needs no upload —
        // generate() fills it directly on the GPU.
        const raw = allocateVoxelBuffers(device, (buffer) => owner.buffers.push(buffer));
        if (Voxels.data)
            device.queue.writeBuffer(raw.voxels, 0, Voxels.data as Float32Array<ArrayBuffer>);

        const wrapped = wrapVoxelBuffers(raw);
        const mesh: Mesh = {
            name: "voxel",
            vertices: wrapped.typedVertices,
            position: wrapped.typedPosition,
            quant: wrapped.typedQuant,
            indices: wrapped.typedIndices,
            indexBase: 0,
            indexCount: MAX_FACES * INDICES_PER_QUAD,
        };
        owner.meshes.push(mesh);
        Meshes.register(mesh);
        const draw: Draw = {
            name: "voxel",
            surface: "voxel",
            mesh: "voxel",
            args: { indirect: wrapped.typedIndirect },
        };
        owner.draws.push(draw);
        Draws.register(draw);
        Object.freeze(owner.buffers);
        Object.freeze(owner.meshes);
        Object.freeze(owner.draws);

        return {
            raw,
            wrapped,
            pipeline: Compute.root
                .createComputePipeline({ compute: emitKernel })
                .$name("voxel-emit"),
            scope: precompileScope(`voxel-emit-${++warmEpoch}`),
        };
    },
    precompile: (_owner, prepared, force) => precompile(prepared.scope, force),
    createWarmBinding: (own) =>
        wrapVoxelBuffers(allocateVoxelBuffers(Compute.device, own)).bindGroup,
    // `initAsync()` on the returned pipeline (the precompile drain) pays Dawn's deferred compile
    // under the loading screen — no encoder/pass/submit needed to get there
    force(prepared) {
        return prepared.pipeline;
    },
    destroyWarm: (buffer) => buffer.destroy(),
    publish(_owner, prepared) {
        gpu.voxels = prepared.raw.voxels;
        gpu.vertices = prepared.raw.vertices;
        gpu.position = prepared.raw.position;
        gpu.quant = prepared.raw.quant;
        gpu.indices = prepared.raw.indices;
        gpu.pipeline = prepared.pipeline;
        gpu.bindGroup = prepared.wrapped.bindGroup;
        Voxels.grid = prepared.raw.voxels;
        Voxels.indirect = prepared.raw.indirect;
        Voxels.dirty = true;
    },
});

/** @internal */
export const warmVoxelEmit = voxelWarmLifecycle.warm;
/** @internal */
export const disposeVoxelWarm = voxelWarmLifecycle.dispose;

export const VoxelPlugin: Plugin = {
    name: "Voxel",
    dependencies: [RenderPlugin],
    systems: [VoxelEmitSystem],
    initialize(state) {
        registerSurface(state, { name: "voxel", layout: voxelLayout, fs: voxelFs });
    },
    warm: warmVoxelEmit,
    dispose: disposeVoxelWarm,
};

export default VoxelPlugin;
