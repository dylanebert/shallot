// The GPU face-cull mesher (Phase 2): a compute pass walks the voxel grid and emits an exposed-face quad
// for every solid voxel whose neighbour across that face is air, atomically appending vertices / indices
// into producer-owned buffers and the index count into an indirect draw record. Sear rasterizes the
// result through one `drawIndirect` — the mesh never crosses back to the CPU. The 3D promotion of
// orrstead's 2D terrain emit (`orrstead/package/src/render/terrain`): swap "cliff per non-ground
// neighbour" for "cube face per air neighbour", in all six directions including interior (the bored
// tunnel) and across chunk seams (the cross-chunk sphere). Single pass over a static grid — the two-pass
// count-then-`dispatchWorkgroupsIndirect` and per-chunk draws arrive with streaming (Phase 4).

import { Compute, type Plugin, RenderPlugin, type System } from "@dylanebert/shallot";
import { BeginFrameSystem, Draws, Meshes, Render, Surfaces } from "@dylanebert/shallot/render/core";
import { PrepassSystem } from "@dylanebert/shallot/sear/core";
import {
    OCT_ENCODE_WGSL,
    POS_QUANT_PACK_WGSL,
    POS_QUANT_WGSL,
} from "@dylanebert/shallot/utils/core";
import { addressingWgsl, BINDING_FLOOR, BYTES, CHUNK_CELLS, DIM, ISO, VOXEL } from "./grid";

const VERTS_PER_QUAD = 4;
const INDICES_PER_QUAD = 6;
const MAIN_STRIDE = 16; // the quantized main stream: vec4<u32> per vertex
const POS_STRIDE = 8; // the position-only depth stream: vec2<u32> per vertex
const WG = 4; // workgroup edge → 4³ = 64 threads

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
const DISPATCH = { x: Math.ceil(DIM.x / WG), y: Math.ceil(DIM.y / WG), z: Math.ceil(DIM.z / WG) };

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

const gpu = {
    voxels: null as GPUBuffer | null,
    vertices: null as GPUBuffer | null,
    position: null as GPUBuffer | null,
    quant: null as GPUBuffer | null,
    indices: null as GPUBuffer | null,
    pipeline: null as GPUComputePipeline | null,
    bindGroup: null as GPUBindGroup | null,
};

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
function surfaceFs(): string {
    return /* wgsl */ `
let isTop = worldNormal.y > 0.5;
let base = select(vec3<f32>(0.10, 0.06, 0.03), vec3<f32>(0.10, 0.20, 0.05), isTop);
col = vec4<f32>(lit(base, worldNormal), 1.0);
`;
}

function emitWgsl(): string {
    return /* wgsl */ `
${OCT_ENCODE_WGSL}
${POS_QUANT_WGSL}
${POS_QUANT_PACK_WGSL}
${addressingWgsl()}

@group(0) @binding(0) var<storage, read> voxels: array<f32>;
@group(0) @binding(1) var<storage, read_write> vertices: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> indirect: array<atomic<u32>, 5>;
@group(0) @binding(4) var<storage, read_write> position: array<vec2<u32>>;

const HALF_X: f32 = ${(0.5 * DIM.x).toExponential()};
const HALF_Y: f32 = ${(0.5 * DIM.y).toExponential()};
const HALF_Z: f32 = ${(0.5 * DIM.z).toExponential()};
const VOXEL: f32 = ${VOXEL.toExponential()};
const ISO: f32 = ${ISO.toExponential()};
const MAX_FACES: u32 = ${MAX_FACES}u;

// the grid's analytic dequant range — the CPU twin (GRID_QUANT) registers the identical record as the
// mesh's quant table, so the producer's encode and sear's decode share one lattice
const GRID_QUANT = MeshQuant(
    vec4<f32>(-HALF_X * VOXEL, -HALF_Y * VOXEL, -HALF_Z * VOXEL, 1.0),
    vec4<f32>(2.0 * HALF_X * VOXEL, 2.0 * HALF_Y * VOXEL, 2.0 * HALF_Z * VOXEL, 0.0),
    vec4<f32>(0.0, 0.0, 0.0, 0.0),
);

fn solidAt(x: i32, y: i32, z: i32) -> bool {
    if (x < 0 || y < 0 || z < 0 || x >= DIM_X || y >= DIM_Y || z >= DIM_Z) { return false; }
    return voxels[voxelIndex(u32(x), u32(y), u32(z))] >= ISO;
}

// append one quad, emitting the quantized main + position streams (gpu.md rule 6). Winding: (p1-p0)×(p2-p0)
// points along the outward normal n — sear's front face, so back-face culling keeps the outward-facing quad.
fn emitQuad(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, n: vec3<f32>, blockU: f32) {
    let base = atomicAdd(&indirect[0], ${INDICES_PER_QUAD}u);
    let fi = base / ${INDICES_PER_QUAD}u;
    if (fi >= MAX_FACES) { return; } // buffer full — drop (the count gate flags the over-budget grid)
    let v0 = fi * ${VERTS_PER_QUAD}u;
    let octN = octEncodeNormal(n); // one normal + uv per flat quad; only the position varies per corner
    let uvw = encodeUv(vec2<f32>(blockU, 0.0), GRID_QUANT);
    let m0 = encodePos(p0, 0u, GRID_QUANT);
    let m1 = encodePos(p1, 0u, GRID_QUANT);
    let m2 = encodePos(p2, 0u, GRID_QUANT);
    let m3 = encodePos(p3, 0u, GRID_QUANT);
    vertices[v0 + 0u] = vec4<u32>(m0, octN, uvw);
    vertices[v0 + 1u] = vec4<u32>(m1, octN, uvw);
    vertices[v0 + 2u] = vec4<u32>(m2, octN, uvw);
    vertices[v0 + 3u] = vec4<u32>(m3, octN, uvw);
    position[v0 + 0u] = m0;
    position[v0 + 1u] = m1;
    position[v0 + 2u] = m2;
    position[v0 + 3u] = m3;
    indices[base + 0u] = v0 + 0u;
    indices[base + 1u] = v0 + 1u;
    indices[base + 2u] = v0 + 2u;
    indices[base + 3u] = v0 + 0u;
    indices[base + 4u] = v0 + 2u;
    indices[base + 5u] = v0 + 3u;
}

@compute @workgroup_size(${WG}, ${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u32(DIM_X) || gid.y >= u32(DIM_Y) || gid.z >= u32(DIM_Z)) { return; }
    if (voxels[voxelIndex(gid.x, gid.y, gid.z)] < ISO) { return; }

    let ix = i32(gid.x);
    let iy = i32(gid.y);
    let iz = i32(gid.z);
    let x0 = (f32(gid.x) - HALF_X) * VOXEL;
    let y0 = (f32(gid.y) - HALF_Y) * VOXEL;
    let z0 = (f32(gid.z) - HALF_Z) * VOXEL;
    let x1 = x0 + VOXEL;
    let y1 = y0 + VOXEL;
    let z1 = z0 + VOXEL;
    let blockU = 1.0; // one block type; the fs colours by face direction (the Phase-3 palette reads this)

    // emit a face iff the neighbour across it is air (or out of bounds).
    if (!solidAt(ix + 1, iy, iz)) {
        emitQuad(vec3<f32>(x1, y0, z0), vec3<f32>(x1, y1, z0), vec3<f32>(x1, y1, z1), vec3<f32>(x1, y0, z1), vec3<f32>(1.0, 0.0, 0.0), blockU);
    }
    if (!solidAt(ix - 1, iy, iz)) {
        emitQuad(vec3<f32>(x0, y0, z0), vec3<f32>(x0, y0, z1), vec3<f32>(x0, y1, z1), vec3<f32>(x0, y1, z0), vec3<f32>(-1.0, 0.0, 0.0), blockU);
    }
    if (!solidAt(ix, iy + 1, iz)) {
        emitQuad(vec3<f32>(x0, y1, z0), vec3<f32>(x0, y1, z1), vec3<f32>(x1, y1, z1), vec3<f32>(x1, y1, z0), vec3<f32>(0.0, 1.0, 0.0), blockU);
    }
    if (!solidAt(ix, iy - 1, iz)) {
        emitQuad(vec3<f32>(x0, y0, z0), vec3<f32>(x1, y0, z0), vec3<f32>(x1, y0, z1), vec3<f32>(x0, y0, z1), vec3<f32>(0.0, -1.0, 0.0), blockU);
    }
    if (!solidAt(ix, iy, iz + 1)) {
        emitQuad(vec3<f32>(x0, y0, z1), vec3<f32>(x1, y0, z1), vec3<f32>(x1, y1, z1), vec3<f32>(x0, y1, z1), vec3<f32>(0.0, 0.0, 1.0), blockU);
    }
    if (!solidAt(ix, iy, iz - 1)) {
        emitQuad(vec3<f32>(x0, y0, z0), vec3<f32>(x0, y1, z0), vec3<f32>(x1, y1, z0), vec3<f32>(x1, y0, z0), vec3<f32>(0.0, 0.0, -1.0), blockU);
    }
}
`;
}

const INDIRECT_INIT = new Uint32Array([0, 1, 0, 0, 0]); // {indexCount=0, instanceCount=1, firstIndex=0, baseVertex=0, firstInstance=0}

const VoxelEmitSystem: System = {
    name: "voxel-emit",
    group: "draw",
    annotations: { mode: "always" },
    after: [BeginFrameSystem],
    // re-emit before sear reads geometry. Sear reads the same vertex/index buffers across its prepass,
    // shadow map, and color passes within a frame, so the emit must land ahead of all of them or a
    // re-mesh would tear new geometry against a stale read. `before: [PrepassSystem]` (the geometry-emit
    // anchor) pins it; `after: [BeginFrameSystem]` alone wouldn't.
    before: [PrepassSystem],
    async setup() {
        const { device } = Compute;

        // COPY_SRC so readGrid() can mirror the grid back for the voxel gate; a fresh STORAGE buffer is
        // zero-initialised (all air), so the generated path (no CPU-authoritative data) needs no upload —
        // generate() fills it directly on the GPU.
        gpu.voxels = device.createBuffer({
            label: "voxel-grid",
            size: BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        Voxels.grid = gpu.voxels;
        if (Voxels.data)
            device.queue.writeBuffer(gpu.voxels, 0, Voxels.data as Float32Array<ArrayBuffer>);

        const maxVerts = MAX_FACES * VERTS_PER_QUAD;
        const maxIndices = MAX_FACES * INDICES_PER_QUAD;
        gpu.vertices = device.createBuffer({
            label: "voxel-main",
            size: maxVerts * MAIN_STRIDE,
            usage: GPUBufferUsage.STORAGE,
        });
        gpu.position = device.createBuffer({
            label: "voxel-pos",
            size: maxVerts * POS_STRIDE,
            usage: GPUBufferUsage.STORAGE,
        });
        gpu.quant = device.createBuffer({
            label: "voxel-quant",
            size: GRID_QUANT.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(gpu.quant, 0, GRID_QUANT as Float32Array<ArrayBuffer>);
        gpu.indices = device.createBuffer({
            label: "voxel-indices",
            size: maxIndices * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDEX,
        });
        // STORAGE for the atomic emit, INDIRECT for the draw, COPY_DST to reset each emit, COPY_SRC so a
        // Mirror reads the face count back.
        Voxels.indirect = device.createBuffer({
            label: "voxel-indirect",
            size: 20,
            usage:
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.INDIRECT |
                GPUBufferUsage.COPY_DST |
                GPUBufferUsage.COPY_SRC,
        });

        Meshes.register({
            name: "voxel",
            vertices: gpu.vertices,
            position: gpu.position,
            quant: gpu.quant,
            indices: gpu.indices,
            indexBase: 0,
            indexCount: maxIndices,
        });
        Draws.register({
            name: "voxel",
            surface: "voxel",
            mesh: "voxel",
            args: { indirect: Voxels.indirect },
        });

        const module = device.createShaderModule({ label: "voxel-emit", code: emitWgsl() });
        gpu.pipeline = await device.createComputePipelineAsync({
            label: "voxel-emit",
            layout: "auto",
            compute: { module, entryPoint: "main" },
        });
        gpu.bindGroup = device.createBindGroup({
            label: "voxel-emit",
            layout: gpu.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: gpu.voxels } },
                { binding: 1, resource: { buffer: gpu.vertices } },
                { binding: 2, resource: { buffer: gpu.indices } },
                { binding: 3, resource: { buffer: Voxels.indirect } },
                { binding: 4, resource: { buffer: gpu.position } },
            ],
        });

        Voxels.dirty = true; // mesh once on the first frame
    },
    update() {
        if (!Voxels.dirty || !gpu.pipeline || !gpu.bindGroup || !Render.encoder) return;
        Compute.device.queue.writeBuffer(
            Voxels.indirect!,
            0,
            INDIRECT_INIT as Uint32Array<ArrayBuffer>,
        );
        const pass = Render.encoder.beginComputePass({
            label: "voxel-emit",
            timestampWrites: Compute.span?.("voxel:emit"),
        });
        pass.setPipeline(gpu.pipeline);
        pass.setBindGroup(0, gpu.bindGroup);
        pass.dispatchWorkgroups(DISPATCH.x, DISPATCH.y, DISPATCH.z);
        pass.end();
        Voxels.dirty = false;
    },
};

const VoxelPlugin: Plugin = {
    name: "Voxel",
    dependencies: [RenderPlugin],
    systems: [VoxelEmitSystem],
    initialize() {
        Surfaces.register({ name: "voxel", fs: surfaceFs() });
    },
};

export default VoxelPlugin;
