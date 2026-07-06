import { Compute } from "@dylanebert/shallot";
import { BVH_ROOT_WGSL, BVH_TRAVERSE_WGSL } from "@dylanebert/shallot/bvh/core";
// the BVH oracle + fixtures are test scaffolding (tests/, out of the published src/), so the
// scenarios reach them by relative path.
import { PRIM_F32, type Prims } from "../../../../packages/shallot/tests/bvh/fixtures";
import type { Bvh2, Ray } from "../../../../packages/shallot/tests/bvh/oracle";

// Shared BVH gym scaffolding — the pure pieces the `accel` scenario's build + traverse layers lean
// on. The GPU builder (bvh/core) and the CPU oracle (tests/bvh/oracle) are the validated
// modules; this only adapts their I/O to the gym contract: a Mirror's bytes become the oracle's
// Bvh2, a ray batch packs into the trace layout, and a software tracer splices the shipped
// ray-AABB traverser so the GPU traverse has a span-timed pass to assert against.

/** the gym builders are sized for thousands of prims — the scale the wireframe overlay dogfoods, not
 *  a 1M perf sweep. Covers every `allFixtures` scene + the coherence-guard scales below. */
export const MAX_PRIMS = 1 << 13;

/** view a {@link Mirror} of the builder's `nodes` buffer as the oracle's {@link Bvh2} — the same
 *  shared-ArrayBuffer bounds/child overlay the assert reads back, so the oracle gates it unchanged. */
export function readbackBvh(bytes: ArrayBufferLike, n: number): Bvh2 {
    return {
        bounds: new Float32Array(bytes),
        child: new Uint32Array(bytes),
        count: Math.max(1, 2 * n - 1),
        root: n === 1 ? 0 : 2 * n - 2,
        primCount: n,
    };
}

// a reordered copy of the same prims — same geometry, different leaf-index order. Mirrors a
// producer's non-deterministic materialize slot order, where each frame the build sees the same
// prims in a fresh permutation, so a correct build must be order-invariant (the coherence guard).
export function shuffledPrims(prims: Prims, seed: number): Prims {
    let s = seed >>> 0 || 1;
    const rand = (): number => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 0x100000000;
    };
    const idx = Array.from({ length: prims.count }, (_, i) => i);
    for (let i = prims.count - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    const data = new Float32Array(prims.count * PRIM_F32);
    for (let i = 0; i < prims.count; i++) {
        data.set(prims.data.subarray(idx[i] * PRIM_F32, (idx[i] + 1) * PRIM_F32), i * PRIM_F32);
    }
    return { count: prims.count, data };
}

/** pack a ray batch into the trace input layout (2 vec4/ray: origin.xyz, dir.xyz) */
export function packRays(batch: Ray[]): Float32Array<ArrayBuffer> {
    const data = new Float32Array(batch.length * 8);
    for (let i = 0; i < batch.length; i++) {
        const o = i * 8;
        data[o] = batch[i].origin[0];
        data[o + 1] = batch[i].origin[1];
        data[o + 2] = batch[i].origin[2];
        data[o + 4] = batch[i].dir[0];
        data[o + 5] = batch[i].dir[1];
        data[o + 6] = batch[i].dir[2];
    }
    return data;
}

const MAX_RAYS = 1 << 16;

// the root is computed on the GPU from the count buffer (bvhRoot, the GPU-count contract) — no CPU
// root crosses to the shader. closest-hit (mode 0) writes bitcast(t) + prim; any-hit (mode 1) writes
// 1/0 occlusion. The pass carries `bvh:trace` so ProfilePlugin times it alongside the build stages.
const TRACE_WGSL = /* wgsl */ `
struct Params { rayCount: u32, mode: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> nodes: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> rayData: array<vec4<f32>>; // 2 vec4/ray: origin.xyz, dir.xyz
@group(0) @binding(2) var<storage, read_write> hits: array<vec2<u32>>; // x = bitcast(t), y = prim
@group(0) @binding(3) var<storage, read> countBuf: array<u32>; // [0] = GPU-driven prim count
@group(1) @binding(0) var<uniform> P: Params;
${BVH_ROOT_WGSL}
${BVH_TRAVERSE_WGSL}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.rayCount) { return; }
    let root = bvhRoot(countBuf[0]);
    let ro = rayData[i * 2u].xyz;
    let rd = rayData[i * 2u + 1u].xyz;
    if (P.mode == 1u) {
        let occ = bvhAnyHit(root, ro, 1.0 / rd, 1.0e30);
        hits[i] = vec2<u32>(select(0u, 1u, occ), 0u);
        return;
    }
    let h = bvhClosestHit(root, ro, 1.0 / rd, 1.0e30);
    hits[i] = vec2<u32>(bitcast<u32>(h.t), h.prim);
}
`;

export interface Tracer {
    rays: GPUBuffer;
    hits: GPUBuffer;
    trace(encoder: GPUCommandEncoder, rayCount: number, mode: number): void;
    destroy(): void;
}

/** a ray-AABB tracer over a builder's `nodes` + GPU `count`, splicing the shipped traverser. The
 *  `hits` buffer is `COPY_SRC` so a {@link Mirror} reads it for the assert. */
export async function createTracer(
    device: GPUDevice,
    nodes: GPUBuffer,
    count: GPUBuffer,
): Promise<Tracer> {
    const rayBuf = device.createBuffer({
        label: "gym-trace-rays",
        size: MAX_RAYS * 32,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const hits = device.createBuffer({
        label: "gym-trace-hits",
        size: MAX_RAYS * 8,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const params = device.createBuffer({
        label: "gym-trace-params",
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const ioLayout = device.createBindGroupLayout({
        label: "gym-trace-io",
        entries: [0, 1, 2, 3].map((binding) => ({
            binding,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: binding === 2 ? "storage" : ("read-only-storage" as GPUBufferBindingType),
            },
        })),
    });
    const uniformLayout = device.createBindGroupLayout({
        label: "gym-trace-uniform",
        entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }],
    });
    const pipeline = await device.createComputePipelineAsync({
        label: "gym-trace",
        layout: device.createPipelineLayout({ bindGroupLayouts: [ioLayout, uniformLayout] }),
        compute: {
            module: device.createShaderModule({ label: "gym-trace", code: TRACE_WGSL }),
            entryPoint: "main",
        },
    });
    const ioBg = device.createBindGroup({
        layout: ioLayout,
        entries: [
            { binding: 0, resource: { buffer: nodes } },
            { binding: 1, resource: { buffer: rayBuf } },
            { binding: 2, resource: { buffer: hits } },
            { binding: 3, resource: { buffer: count } },
        ],
    });
    const paramsBg = device.createBindGroup({
        layout: uniformLayout,
        entries: [{ binding: 0, resource: { buffer: params } }],
    });
    return {
        rays: rayBuf,
        hits,
        trace(encoder, rayCount, mode): void {
            device.queue.writeBuffer(params, 0, new Uint32Array([rayCount, mode, 0, 0]));
            const pass = encoder.beginComputePass({ timestampWrites: Compute.span?.("bvh:trace") });
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, ioBg);
            pass.setBindGroup(1, paramsBg);
            pass.dispatchWorkgroups(Math.ceil(rayCount / 64));
            pass.end();
        },
        destroy(): void {
            for (const b of [rayBuf, hits, params]) b.destroy();
        },
    };
}
