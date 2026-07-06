// Morton-code assignment — primitive AABBs + scene AABB → one 30-bit spatial key
// per prim, plus an identity payload, a stage of the LBVH builder. The pass between
// scene-bounds (bounds.ts) and
// the sort (sort.ts): the key is what the radix sort orders, the payload rides along as
// the primitive index so the sorted order is a permutation the build reads back.
//
// Per prim: centroid = AABB midpoint, normalize by the scene extent into [0,1] per
// axis, quantize to 10 bits (×1023), interleave x | y<<1 | z<<2 into a 30-bit code.
// Bit-identical to NexusBVH's ComputeMortonCodesKernel + MortonCode<uint32_t>
// (reference/NexusBVH). A zero-extent axis (coplanar scene) normalizes to 0 rather
// than 0/0 = NaN — the defined-code requirement of Phase 3, matching the oracle's
// `if (ext <= 0) return 0` branch (tests/bvh/oracle.ts mortonCodes).
//
// Embarrassingly parallel — no subgroup ops, no shared memory, one grid-stride loop.
// The one float subtlety is the normalize divide. The oracle models it in f32
// (Math.fround) so the 10-bit quantization lands in the same bin as this GPU math,
// and the `accel` gym scenario's build gate's bit-identical check holds exactly on the
// fixture set. It is NOT exact in bulk: WGSL allows f32 divide up to 2.5 ULP and real
// hardware (NVIDIA Ada, observed) isn't correctly-rounded, so a centroid within ~1 ULP
// of a bin boundary lands ±1 bin off (~one prim per ~12k at 1M uniform-random). That
// reorders two near-coincident prims, which the topology-agnostic build doesn't care
// about — so it's not a correctness concern, only why the exact gate stays on the
// fixtures (no boundary centroids) and bulk scale is timed, not bit-compared.
//
// 64-bit codes (21 bits/axis, NexusBVH's huge-scene path) are deferred: they need the
// 8-pass sort variant that sort.ts (4×8-bit, u32-only) does not provide. Adding an
// unsortable key here would be a half-built feature, so the seam waits for that sort.

import { Compute } from "../../engine";

const WG = 256; // workgroup size; the grid-stride loop folds any count past one dispatch
const MAX_DISPATCH = 65535; // maxComputeWorkgroupsPerDimension floor

// Spread the low 10 bits of x to every third bit (two zeros between bits). Bit-
// identical to NexusBVH InterleaveBits32 and the oracle's interleaveBits32; every
// intermediate stays within 32 bits.
const INTERLEAVE_WGSL = /* wgsl */ `
fn interleaveBits32(x: u32) -> u32 {
    var v = x & 0x3ffu;
    v = (v | (v << 16u)) & 0x30000ffu;
    v = (v | (v << 8u)) & 0x300f00fu;
    v = (v | (v << 4u)) & 0x30c30c3u;
    v = (v | (v << 2u)) & 0x9249249u;
    return v;
}
`;

// `maxPrims` is baked as a const: the pass fills the whole [0, maxPrims) key range every
// run — a real 30-bit code for a live prim (`i < count`), a max-Morton sentinel for the
// padding tail (`i >= count`). The sentinel sorts above every real code, so the radix
// pushes padding to the end with no CPU tail-pad — the count stays GPU-resident.
const mortonWgsl = (maxPrims: number): string =>
    INTERLEAVE_WGSL +
    /* wgsl */ `
const MAX_PRIMS = ${Math.max(1, maxPrims)}u;
@group(0) @binding(0) var<storage, read> prims: array<vec4<f32>>;     // 2 vec4/prim: min.xyz+pad, max.xyz+pad
@group(0) @binding(1) var<storage, read> bounds: array<vec4<f32>, 2>; // scene AABB: min.xyz+pad, max.xyz+pad
@group(0) @binding(2) var<storage, read_write> keys: array<u32>;      // out: 30-bit Morton code / sentinel
@group(0) @binding(3) var<storage, read_write> payload: array<u32>;   // out: primIndex = identity
@group(0) @binding(4) var<storage, read> countBuf: array<u32>;        // [0] = GPU-driven prim count

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let bmin = bounds[0].xyz;
    let ext = bounds[1].xyz - bmin;
    let stride = nwg.x * ${WG}u;
    let count = countBuf[0];
    for (var i = gid.x; i < MAX_PRIMS; i += stride) {
        if (i < count) {
            let mn = prims[i * 2u].xyz;
            let mx = prims[i * 2u + 1u].xyz;
            let centroid = (mn + mx) * 0.5;
            // normalize into [0,1]; a zero-extent axis yields 0 (the divide there is
            // discarded), not NaN — defined codes on degenerate scenes
            let t = select((centroid - bmin) / ext, vec3<f32>(0.0), ext <= vec3<f32>(0.0));
            let c = clamp(t, vec3<f32>(0.0), vec3<f32>(1.0));
            let q = vec3<u32>(floor(c * 1023.0));
            keys[i] = interleaveBits32(q.x) | (interleaveBits32(q.y) << 1u) | (interleaveBits32(q.z) << 2u);
        } else {
            keys[i] = 0xffffffffu; // above any 30-bit code → sorts past the live prims
        }
        payload[i] = i;
    }
}
`;

function storageEntry(binding: number, readonly: boolean): GPUBindGroupLayoutEntry {
    return {
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: readonly ? "read-only-storage" : "storage" },
    };
}

/**
 * a built Morton-code pass sized for `maxPrims`. Write primitive AABBs into {@link
 * Morton.prims} (2 × vec4<f32> per prim) and the scene AABB into {@link Morton.bounds}
 * (2 × vec4<f32>: `min.xyz+pad`, `max.xyz+pad`, the {@link SceneBounds} output),
 * record {@link Morton.compute}, submit, then read {@link Morton.keys} (one 30-bit
 * code per prim) and {@link Morton.payload} (primitive index, identity before sort).
 * The two output buffers are the radix sort's key / payload input.
 */
/** buffers {@link createBvh} threads through so the stages share one set; an omitted field is allocated internally */
export interface MortonShared {
    /** input prim AABB buffer — 2 vec4<f32> per prim */
    prims?: GPUBuffer;
    /** input scene AABB (the bounds reduction's output) */
    bounds?: GPUBuffer;
    /** Morton code output (the sort's key input) */
    keys?: GPUBuffer;
    /** prim-index output (the sort's payload input) */
    payload?: GPUBuffer;
    /** GPU-driven prim count (one u32 at [0]); live prims are [0, count), the rest sentinel-padded */
    count?: GPUBuffer;
}

/**
 * a built Morton-code pass sized for `maxPrims`. Reads prim AABBs + the scene bounds ({@link
 * createSceneBounds}) and writes each prim's Morton code into {@link Morton.keys} with an identity
 * {@link Morton.payload}, sentinel-padding the [count, maxPrims) tail so the radix sort moves the live
 * range to the front. The sort's key/payload input.
 */
export interface Morton {
    /** input prim AABB buffer — fill [0, count) prims, 2 vec4 each */
    readonly prims: GPUBuffer;
    /** input scene AABB — 2 vec4<f32> (min then max), produced by the bounds reduction */
    readonly bounds: GPUBuffer;
    /** output Morton codes — real code for [0, count), max-Morton sentinel for [count, maxPrims) */
    readonly keys: GPUBuffer;
    /** output primitive indices — identity (payload[i] = i) over the full key range */
    readonly payload: GPUBuffer;
    /** GPU-driven prim count (one u32 at [0]) — write it (≤ `maxPrims`) before recording */
    readonly count: GPUBuffer;
    /** capacity the buffers are sized for */
    readonly maxPrims: number;
    /** record the Morton-code + sentinel-pad pass over [0, maxPrims) onto `encoder`, count read from {@link Morton.count} */
    compute(encoder: GPUCommandEncoder): void;
    destroy(): void;
}

/**
 * build a Morton-code pass for up to `maxPrims` primitive AABBs. Compiles the kernel
 * and allocates the input (prims, bounds) and output (keys, payload) buffers up front;
 * {@link Morton.compute} then records with no further allocation.
 *
 * @example
 * const mc = await createMorton(device, 1 << 20);
 * device.queue.writeBuffer(mc.prims, 0, primAabbs);
 * device.queue.writeBuffer(mc.bounds, 0, sceneAabb); // from createSceneBounds
 * device.queue.writeBuffer(mc.count, 0, new Uint32Array([count]));
 * const enc = device.createCommandEncoder();
 * mc.compute(enc);
 * device.queue.submit([enc.finish()]);
 */
export async function createMorton(
    device: GPUDevice,
    maxPrims: number,
    shared: MortonShared = {},
): Promise<Morton> {
    const cap = Math.max(1, maxPrims);
    const owned: GPUBuffer[] = [];
    const own = (label: string, size: number, usage: number): GPUBuffer => {
        const b = device.createBuffer({ label, size, usage });
        owned.push(b);
        return b;
    };
    const StoreSrc = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
    const StoreDst = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    const prims = shared.prims ?? own("morton-prims", cap * 32, StoreDst);
    const bounds = shared.bounds ?? own("morton-bounds", 32, StoreDst);
    const keys = shared.keys ?? own("morton-keys", cap * 4, StoreSrc);
    const payload = shared.payload ?? own("morton-payload", cap * 4, StoreSrc);
    const count = shared.count ?? own("morton-count", 4, StoreDst);

    const ioLayout = device.createBindGroupLayout({
        label: "morton-io",
        entries: [
            storageEntry(0, true),
            storageEntry(1, true),
            storageEntry(2, false),
            storageEntry(3, false),
            storageEntry(4, true),
        ],
    });

    const pipeline = await device.createComputePipelineAsync({
        label: "morton",
        layout: device.createPipelineLayout({ bindGroupLayouts: [ioLayout] }),
        compute: {
            module: device.createShaderModule({ label: "morton", code: mortonWgsl(cap) }),
            entryPoint: "main",
        },
    });

    const ioBg = device.createBindGroup({
        layout: ioLayout,
        entries: [
            { binding: 0, resource: { buffer: prims } },
            { binding: 1, resource: { buffer: bounds } },
            { binding: 2, resource: { buffer: keys } },
            { binding: 3, resource: { buffer: payload } },
            { binding: 4, resource: { buffer: count } },
        ],
    });

    // dispatch is a CPU constant — the kernel fills the whole [0, maxPrims) key range,
    // branching per-thread on the GPU-read count, so the count never crosses to the CPU
    const numWg = Math.min(Math.ceil(cap / WG), MAX_DISPATCH);

    return {
        prims,
        bounds,
        keys,
        payload,
        count,
        maxPrims,
        compute(encoder: GPUCommandEncoder): void {
            const pass = encoder.beginComputePass({
                timestampWrites: Compute.span?.("bvh:morton"),
            });
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, ioBg);
            pass.dispatchWorkgroups(numWg);
            pass.end();
        },
        destroy(): void {
            for (const b of owned) b.destroy();
        },
    };
}
