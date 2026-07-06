// Scene-bounds reduction — primitive AABBs → one scene AABB, the first stage of the
// LBVH builder. The build's first pass: Morton
// normalization (Phase 3) needs the scene extent before any code is computed.
//
// Subgroup-first, the gpu.md "reduce in workgroup, then one atomic" rule in its
// subgroup form: each lane folds its grid-stride slice, `subgroupMin`/`subgroupMax`
// reduces across the subgroup, a second subgroup op folds the per-subgroup partials,
// and one lane does a single atomicMin/Max per axis to the global slot — no LDS tree
// where a subgroup op suffices, no per-thread global atomic.
//
// A subgroup-free LDS variant ships alongside it, selected by the `subgroups` flag
// (WebKit has no subgroups — Safari / WKWebView). It swaps the two subgroup reduces
// for one canonical workgroup-shared-memory tree reduce over the WG lanes; everything
// else (grid-stride fold, ordered-u32 atomics, finalize) is shared. Both produce the
// bit-identical scene AABB — min/max introduce no rounding — so they pass the same
// oracle gate, and the choice lands once at build, never per frame.
//
// WebGPU has integer atomics only, so the six axis extremes reduce as order-
// preserving u32 (`orderU32`): flip the sign bit on positives, all bits on negatives,
// which makes integer atomicMin/Max agree with float ordering across the whole range.
// A finalize pass decodes the six back to the f32 scene AABB. min/max introduce no
// rounding and the bitcast is bijective, so the result is bit-exact against the CPU
// oracle (tests/bvh/oracle.ts sceneBounds) — the Phase 2 gate, validated on the real
// GPU by the `accel` gym scenario's build gate, including degenerate (coplanar, single-prim).
//
// Designed for subgroup width 32, correct at 64 (the platform floor). The level-2
// reduce assumes the per-subgroup partials fit one subgroup (numSub <= sgsize, true
// for sgsize >= 16); software/sub-16 subgroups are out of scope.

import { Compute } from "../../engine";

const WG = 256; // workgroup size, both kernels
const MAX_SUB = 64; // max subgroup size on the floor — sizes the partials array so wg_min[sid] never reads OOB
const MAX_WG = 1024; // workgroup cap; the grid-stride loop folds any larger prim count
const SCRATCH_U32 = 6; // ordered-u32 axis extremes: [min.xyz, max.xyz]

// IEEE float → order-preserving u32 and back. Positive (sign 0): set the sign bit, so
// it sorts above every negative. Negative (sign 1): flip all bits, reversing magnitude
// order. Bijective, so the decode is exact.
const ORDER_WGSL = /* wgsl */ `
fn orderU32(f: f32) -> u32 {
    let u = bitcast<u32>(f);
    return select(~u, u | 0x80000000u, (u >> 31u) == 0u);
}
fn unorderU32(o: u32) -> f32 {
    return bitcast<f32>(select(~o, o ^ 0x80000000u, (o >> 31u) == 1u));
}
`;

// the workgroup reduce — subgroup-first by default, a canonical LDS tree reduce when
// subgroups are absent. Both fold the per-lane (lmin, lmax) into (tmin, tmax) before
// the single per-axis atomic, the only difference being how the WG lanes cooperate.
const subgroupReduce = /* wgsl */ `
    // level 1: reduce across the subgroup; lane 0 publishes the partial
    let sgid = tid / sgsize;
    let smin = subgroupMin(lmin);
    let smax = subgroupMax(lmax);
    if (sid == 0u) { wg_min[sgid] = smin; wg_max[sgid] = smax; }
    workgroupBarrier();

    // level 2: every subgroup redundantly folds the numSub partials (they fit one
    // subgroup since numSub = WG/sgsize <= sgsize for sgsize >= 16). No second barrier,
    // no LDS tree — the subgroup op is the whole reduce.
    let numSub = ${WG}u / sgsize;
    let vmin = select(vec3<f32>(FMAX), wg_min[sid], sid < numSub);
    let vmax = select(vec3<f32>(-FMAX), wg_max[sid], sid < numSub);
    let tmin = subgroupMin(vmin);
    let tmax = subgroupMax(vmax);
`;

// LDS tree reduce over the WG lanes: each lane seeds its slot, then a halving tree folds
// to slot 0. The canonical no-subgroup reduce (the gpu.md rule's pre-subgroup form).
const ldsReduce = /* wgsl */ `
    wg_min[tid] = lmin;
    wg_max[tid] = lmax;
    workgroupBarrier();
    for (var s = ${WG >> 1}u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_min[tid] = min(wg_min[tid], wg_min[tid + s]);
            wg_max[tid] = max(wg_max[tid], wg_max[tid + s]);
        }
        workgroupBarrier();
    }
    let tmin = wg_min[0];
    let tmax = wg_max[0];
`;

function reduceWgsl(subgroups: boolean): string {
    // the LDS path sizes its scratch to one slot per WG lane; the subgroup path needs
    // only one per subgroup (numSub <= MAX_SUB).
    const slots = subgroups ? MAX_SUB : WG;
    const sgParams = subgroups
        ? ", @builtin(subgroup_invocation_id) sid: u32, @builtin(subgroup_size) sgsize: u32"
        : "";
    return (
        (subgroups ? "enable subgroups;\n" : "") +
        ORDER_WGSL +
        /* wgsl */ `
@group(0) @binding(0) var<storage, read> prims: array<vec4<f32>>;   // 2 vec4/prim: min.xyz+pad, max.xyz+pad
@group(0) @binding(1) var<storage, read_write> scratch: array<atomic<u32>, ${SCRATCH_U32}>;
@group(0) @binding(2) var<storage, read> countBuf: array<u32>;       // [0] = GPU-driven prim count

var<workgroup> wg_min: array<vec3<f32>, ${slots}>;
var<workgroup> wg_max: array<vec3<f32>, ${slots}>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>${sgParams}) {
    let tid = lid.x;
    let stride = nwg.x * ${WG}u;
    let count = countBuf[0];
    let FMAX = bitcast<f32>(0x7f7fffffu);   // neutral element: empty lanes don't move the extreme

    // grid-stride fold of this lane's slice
    var lmin = vec3<f32>(FMAX);
    var lmax = vec3<f32>(-FMAX);
    for (var i = gid.x; i < count; i += stride) {
        lmin = min(lmin, prims[i * 2u].xyz);
        lmax = max(lmax, prims[i * 2u + 1u].xyz);
    }
${subgroups ? subgroupReduce : ldsReduce}
    // one global atomic per axis per workgroup
    if (tid == 0u) {
        atomicMin(&scratch[0], orderU32(tmin.x));
        atomicMin(&scratch[1], orderU32(tmin.y));
        atomicMin(&scratch[2], orderU32(tmin.z));
        atomicMax(&scratch[3], orderU32(tmax.x));
        atomicMax(&scratch[4], orderU32(tmax.y));
        atomicMax(&scratch[5], orderU32(tmax.z));
    }
}
`
    );
}

const FINALIZE_WGSL =
    ORDER_WGSL +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> scratch: array<u32>;
@group(0) @binding(1) var<storage, read_write> bounds: array<vec4<f32>, 2>;

// decode the six ordered-u32 extremes into the f32 scene AABB (2 vec4: min.xyz+pad, max.xyz+pad)
@compute @workgroup_size(1)
fn main() {
    bounds[0] = vec4<f32>(unorderU32(scratch[0]), unorderU32(scratch[1]), unorderU32(scratch[2]), 0.0);
    bounds[1] = vec4<f32>(unorderU32(scratch[3]), unorderU32(scratch[4]), unorderU32(scratch[5]), 0.0);
}
`;

function storageEntry(binding: number, readonly: boolean): GPUBindGroupLayoutEntry {
    return {
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: readonly ? "read-only-storage" : "storage" },
    };
}

/** buffers {@link createBvh} threads through so the stages share one set; an omitted field is allocated internally */
export interface SceneBoundsShared {
    /** input prim AABB buffer — 2 vec4<f32> per prim */
    prims?: GPUBuffer;
    /** scene AABB output (2 × vec4<f32>) */
    bounds?: GPUBuffer;
    /** GPU-driven prim count (one u32 at [0]); the reduce gates its loop on it */
    count?: GPUBuffer;
}

/**
 * a built scene-bounds reducer sized for `maxPrims`. Write primitive AABBs into {@link SceneBounds.prims}
 * (2 × vec4<f32> per prim), record {@link SceneBounds.reduce}, submit, then read the scene AABB back from
 * {@link SceneBounds.bounds} (2 × vec4<f32>: `min.xyz+pad`, `max.xyz+pad`).
 */
export interface SceneBounds {
    /** input prim AABB buffer — fill [0, count) prims, 2 vec4 each */
    readonly prims: GPUBuffer;
    /** output scene AABB — 2 vec4<f32> after the reduction (min then max) */
    readonly bounds: GPUBuffer;
    /** GPU-driven prim count (one u32 at [0]) — write it (≤ `maxPrims`) before recording */
    readonly count: GPUBuffer;
    /** capacity the input buffer is sized for */
    readonly maxPrims: number;
    /** record the bounds reduction over [0, count) prims onto `encoder`, count read from {@link SceneBounds.count} */
    reduce(encoder: GPUCommandEncoder): void;
    destroy(): void;
}

/**
 * build a scene-bounds reducer for up to `maxPrims` primitive AABBs. Compiles the
 * reduce + finalize kernels and allocates the input / scratch / output buffers up
 * front; {@link SceneBounds.reduce} then records with no further allocation.
 *
 * `subgroups` picks the reduce kernel — the subgroup reduce (default, when the device
 * has the feature) or the subgroup-free LDS tree reduce. Both yield the bit-identical
 * scene AABB; force `false` to exercise the LDS path on a subgroup-capable device.
 *
 * @example
 * const sb = await createSceneBounds(device, 1 << 20);
 * device.queue.writeBuffer(sb.prims, 0, primAabbs);
 * device.queue.writeBuffer(sb.count, 0, new Uint32Array([count]));
 * const enc = device.createCommandEncoder();
 * sb.reduce(enc);
 * device.queue.submit([enc.finish()]);
 */
export async function createSceneBounds(
    device: GPUDevice,
    maxPrims: number,
    shared: SceneBoundsShared = {},
    subgroups: boolean = device.features.has("subgroups"),
): Promise<SceneBounds> {
    const cap = Math.max(1, maxPrims);
    const owned: GPUBuffer[] = [];
    const own = (label: string, size: number, usage: number): GPUBuffer => {
        const b = device.createBuffer({ label, size, usage });
        owned.push(b);
        return b;
    };
    const StoreDst = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    const prims = shared.prims ?? own("bounds-prims", cap * 32, StoreDst);
    const scratch = own("bounds-scratch", SCRATCH_U32 * 4, StoreDst);
    const bounds =
        shared.bounds ?? own("bounds-out", 32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const count = shared.count ?? own("bounds-count", 4, StoreDst);

    const reduceLayout = device.createBindGroupLayout({
        label: "bounds-reduce",
        entries: [storageEntry(0, true), storageEntry(1, false), storageEntry(2, true)],
    });
    const finalizeLayout = device.createBindGroupLayout({
        label: "bounds-finalize",
        entries: [storageEntry(0, true), storageEntry(1, false)],
    });

    const pipe = (
        label: string,
        code: string,
        layout: GPUBindGroupLayout,
    ): Promise<GPUComputePipeline> =>
        device.createComputePipelineAsync({
            label,
            layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
            compute: { module: device.createShaderModule({ label, code }), entryPoint: "main" },
        });

    const [reducePipe, finalizePipe] = await Promise.all([
        pipe("bounds-reduce", reduceWgsl(subgroups), reduceLayout),
        pipe("bounds-finalize", FINALIZE_WGSL, finalizeLayout),
    ]);

    const reduceBg = device.createBindGroup({
        layout: reduceLayout,
        entries: [
            { binding: 0, resource: { buffer: prims } },
            { binding: 1, resource: { buffer: scratch } },
            { binding: 2, resource: { buffer: count } },
        ],
    });
    const finalizeBg = device.createBindGroup({
        layout: finalizeLayout,
        entries: [
            { binding: 0, resource: { buffer: scratch } },
            { binding: 1, resource: { buffer: bounds } },
        ],
    });

    // scratch reset before each reduce: min axes start above every value, max below.
    const init = new Uint32Array([0xffffffff, 0xffffffff, 0xffffffff, 0, 0, 0]);
    // dispatch is a CPU constant — the grid-stride loop folds [0, count) with count read
    // on the GPU, so the count buffer never crosses to the CPU
    const numWg = Math.min(Math.ceil(cap / WG), MAX_WG);

    return {
        prims,
        bounds,
        count,
        maxPrims,
        reduce(encoder: GPUCommandEncoder): void {
            device.queue.writeBuffer(scratch, 0, init);

            // profile under one accumulating row (no-op without ProfilePlugin → the gym measure unaffected)
            const reducePass = encoder.beginComputePass({
                timestampWrites: Compute.span?.("bvh:bounds"),
            });
            reducePass.setPipeline(reducePipe);
            reducePass.setBindGroup(0, reduceBg);
            reducePass.dispatchWorkgroups(numWg);
            reducePass.end();

            const finalizePass = encoder.beginComputePass({
                timestampWrites: Compute.span?.("bvh:bounds"),
            });
            finalizePass.setPipeline(finalizePipe);
            finalizePass.setBindGroup(0, finalizeBg);
            finalizePass.dispatchWorkgroups(1);
            finalizePass.end();
        },
        destroy(): void {
            for (const b of owned) b.destroy();
        },
    };
}
