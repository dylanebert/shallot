import { Compute, precompile } from "@dylanebert/shallot";
import { bvhAnyHit, bvhClosestHit, bvhRoot } from "@dylanebert/shallot/bvh/core";
import { precompileScope } from "@dylanebert/shallot/runtime";
import { bitcastF32toU32 } from "@dylanebert/shallot/utils/core";
import type {
    StorageFlag,
    TgpuBindGroup,
    TgpuBuffer,
    TgpuComputePipeline,
    UniformFlag,
} from "typegpu";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
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

const TraceParams = d
    .struct({ rayCount: d.u32, mode: d.u32, pad0: d.u32, pad1: d.u32 })
    .$name("TraceParams");
const TraceNodes = d.arrayOf(d.vec4f);
const TraceRays = d.arrayOf(d.vec4f, MAX_RAYS * 2);
const TraceHits = d.arrayOf(d.vec2u, MAX_RAYS);
const TraceCount = d.arrayOf(d.u32);
const traceLayout = tgpu.bindGroupLayout({
    nodes: { storage: TraceNodes, access: "readonly" },
    rayData: { storage: TraceRays, access: "readonly" },
    hits: { storage: TraceHits, access: "mutable" },
    countBuf: { storage: TraceCount, access: "readonly" },
    params: { uniform: TraceParams },
});

// `bvhClosestHit`/`bvhAnyHit` retain the certified traversal graph. Their WGSL-bodied node accessors
// read the consumer-owned `nodes` declaration by name, so the live zero fold makes that typed binding
// visible to TypeGPU's resolver and to the traversal without maintaining a second accessor graph here.
const traceKernel = tgpu
    .computeFn({
        in: { gid: d.builtin.globalInvocationId },
        workgroupSize: [64],
    })((input) => {
        "use gpu";
        const i = input.gid.x;
        if (i >= traceLayout.$.params.rayCount) return;
        const ray = i * d.u32(2);
        const ro = traceLayout.$.rayData[ray].xyz;
        const rd = traceLayout.$.rayData[ray + d.u32(1)].xyz;
        const inv = std.div(d.vec3f(1), rd);
        const root = bvhRoot(traceLayout.$.countBuf[d.u32(0)]);
        const tMax = d.f32(1e30) + traceLayout.$.nodes[d.u32(0)].x * d.f32(0);
        if (traceLayout.$.params.mode === d.u32(1)) {
            const occluded = bvhAnyHit(root, ro, inv, tMax);
            traceLayout.$.hits[i] = d.vec2u(std.select(d.u32(0), d.u32(1), occluded), d.u32(0));
            return;
        }
        const hit = bvhClosestHit(root, ro, inv, tMax);
        traceLayout.$.hits[i] = d.vec2u(bitcastF32toU32(hit.t), hit.prim);
    })
    .$name("traceBvh");

/** the emitted trace entry and certified traversal graph. @internal */
export function traceEntryWgsl(): string {
    return tgpu.resolve([traceKernel], { names: "strict" });
}

type TraceNodesBuffer = TgpuBuffer<ReturnType<typeof TraceNodes>> & StorageFlag;
type TraceRaysBuffer = TgpuBuffer<typeof TraceRays> & StorageFlag;
type TraceHitsBuffer = TgpuBuffer<typeof TraceHits> & StorageFlag;
type TraceCountBuffer = TgpuBuffer<ReturnType<typeof TraceCount>> & StorageFlag;
type TraceParamsBuffer = TgpuBuffer<typeof TraceParams> & UniformFlag;
type TraceGroup = TgpuBindGroup<typeof traceLayout.entries>;
type TracePipeline = TgpuComputePipeline;

interface TraceOwner {
    readonly raws: readonly GPUBuffer[];
    readonly raysRaw: GPUBuffer;
    readonly hitsRaw: GPUBuffer;
    readonly rays: TraceRaysBuffer;
    readonly hits: TraceHitsBuffer;
    readonly params: TraceParamsBuffer;
    readonly group: TraceGroup;
    readonly pipeline: TracePipeline;
}

const cleanedTraceOwners = new WeakSet<object>();

function cleanupTrace(owner: { readonly raws: readonly GPUBuffer[] }): void {
    if (cleanedTraceOwners.has(owner)) return;
    cleanedTraceOwners.add(owner);
    for (const raw of owner.raws) raw.destroy();
}

function forceTrace(
    root: typeof Compute.root,
    device: GPUDevice,
    pipeline: TracePipeline,
): TracePipeline {
    const raws: GPUBuffer[] = [];
    try {
        const storage = (label: string, size: number): GPUBuffer => {
            const raw = device.createBuffer({
                label,
                size,
                usage: GPUBufferUsage.STORAGE,
            });
            raws.push(raw);
            return raw;
        };
        const nodes = root
            .createBuffer(TraceNodes(1), storage("gym-trace-force-nodes", 16))
            .$usage("storage");
        const rayData = root
            .createBuffer(TraceRays, storage("gym-trace-force-rays", d.sizeOf(TraceRays)))
            .$usage("storage");
        const hits = root
            .createBuffer(TraceHits, storage("gym-trace-force-hits", d.sizeOf(TraceHits)))
            .$usage("storage");
        const countBuf = root
            .createBuffer(TraceCount(1), storage("gym-trace-force-count", 4))
            .$usage("storage");
        const paramsRaw = device.createBuffer({
            label: "gym-trace-force-params",
            size: d.sizeOf(TraceParams),
            usage: GPUBufferUsage.UNIFORM,
        });
        raws.push(paramsRaw);
        const params = root.createBuffer(TraceParams, paramsRaw).$usage("uniform");
        const group = root.createBindGroup(traceLayout, {
            nodes,
            rayData,
            hits,
            countBuf,
            params,
        });
        // the drain awaits the returned pipeline's own `initAsync()` — no dispatch needed to force the
        // compile, so the throwaway buffers above exist only to satisfy the bind group and are freed
        // the moment binding is done, same as glaze's dummy-texture forcer (standard/glaze/index.ts)
        return pipeline.with(group);
    } finally {
        for (const raw of raws) raw.destroy();
    }
}

/** one exact single-flight initialization record per root; a rejected record is retryable. @internal */
export function createPipelineCache<Root extends object, Context, Pipeline>(
    create: (root: Root, context: Context) => Pipeline,
    initialize: (root: Root, context: Context, pipeline: Pipeline) => Promise<void>,
): (root: Root, context: Context) => Promise<Pipeline> {
    const cache = new WeakMap<Root, { pipeline: Pipeline; ready: Promise<void> }>();
    return async (root, context) => {
        let entry = cache.get(root);
        if (!entry) {
            const pipeline = create(root, context);
            const draft = { pipeline, ready: Promise.resolve() };
            entry = draft;
            cache.set(root, entry);
            try {
                draft.ready = initialize(root, context, pipeline).catch((cause) => {
                    if (cache.get(root) === draft) cache.delete(root);
                    throw cause;
                });
            } catch (cause) {
                if (cache.get(root) === draft) cache.delete(root);
                throw cause;
            }
        }
        await entry.ready;
        return entry.pipeline;
    };
}

const tracePipeline = createPipelineCache(
    (root: typeof Compute.root) =>
        root.createComputePipeline({ compute: traceKernel }).$name("gym-trace"),
    async (root, device: GPUDevice, pipeline) => {
        await precompile(precompileScope("gym-trace"), () => forceTrace(root, device, pipeline));
    },
);

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
    const root = Compute.root;
    const raws: GPUBuffer[] = [];
    const draft = { raws };
    try {
        const raysRaw = device.createBuffer({
            label: "gym-trace-rays",
            size: d.sizeOf(TraceRays),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        raws.push(raysRaw);
        const rays: TraceRaysBuffer = root.createBuffer(TraceRays, raysRaw).$usage("storage");
        const hitsRaw = device.createBuffer({
            label: "gym-trace-hits",
            size: d.sizeOf(TraceHits),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        raws.push(hitsRaw);
        const hits: TraceHitsBuffer = root.createBuffer(TraceHits, hitsRaw).$usage("storage");
        const paramsRaw = device.createBuffer({
            label: "gym-trace-params",
            size: d.sizeOf(TraceParams),
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        raws.push(paramsRaw);
        const params: TraceParamsBuffer = root
            .createBuffer(TraceParams, paramsRaw)
            .$usage("uniform");
        const typedNodes: TraceNodesBuffer = root
            .createBuffer(TraceNodes(nodes.size / d.sizeOf(d.vec4f)), nodes)
            .$usage("storage");
        const typedCount: TraceCountBuffer = root
            .createBuffer(TraceCount(count.size / d.sizeOf(d.u32)), count)
            .$usage("storage");
        const pipeline = await tracePipeline(root, device);
        const group = root.createBindGroup(traceLayout, {
            nodes: typedNodes,
            rayData: rays,
            hits,
            countBuf: typedCount,
            params,
        });
        const owner: TraceOwner = Object.freeze({
            raws: Object.freeze([...raws]),
            raysRaw,
            hitsRaw,
            rays,
            hits,
            params,
            group,
            pipeline,
        });
        return {
            rays: owner.raysRaw,
            hits: owner.hitsRaw,
            trace(encoder, rayCount, mode): void {
                owner.params.write({ rayCount, mode, pad0: 0, pad1: 0 });
                const pass = encoder.beginComputePass({
                    timestampWrites: Compute.span?.("bvh:trace"),
                });
                owner.pipeline
                    .with(owner.group)
                    .with(pass)
                    .dispatchWorkgroups(Math.ceil(rayCount / 64));
                pass.end();
            },
            destroy(): void {
                cleanupTrace(owner);
            },
        };
    } catch (cause) {
        cleanupTrace(draft);
        throw cause;
    }
}
