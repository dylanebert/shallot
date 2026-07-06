// LBVH — the public extension surface. One GPU BVH2 builder, rendering-unaware and
// unopinionated about its consumer: geometry in (primitive AABBs), BVH out — it names
// neither bodies nor draws, neither triangles nor instances. {@link createBvh} is the
// whole pipeline behind one prim-AABB input buffer and one BVH2 output buffer; the
// per-stage factories (re-exported below) stay available for isolated validation, and
// {@link BVH_TRAVERSE_WGSL} is the single-level ray-AABB traverser its consumers splice
// in (the physics broadphase / raycast; a native-RT lighting path when WebGPU ships RT).
// Consumer-specific layers — ray-triangle leaf tests, two-level BLAS/TLAS instancing —
// live with the consumer that needs them, not in the builder.
//
// The build is LBVH (Karras radix-tree topology + a coherence-safe bottom-up bounds
// relaxation), not the former single-kernel H-PLOC: the H-PLOC build (and its
// atomic-climb refit) relied on cross-workgroup memory ordering WGSL cannot express,
// which manifested as the forest shadow flicker (gpu.md "Cross-workgroup ordering").
// build + refit both live in {@link createBuild} (build.ts) — the bounds relaxation
// doubles as the refit, so there is no separate refit stage to derive parents for.

import { checkStorageBinding } from "../../engine";
import { createSceneBounds } from "./bounds";
import { createBuild } from "./build";
import { createMorton } from "./morton";
import { createRadixSort, KEYS_PER_BLOCK } from "./sort";

export type { SceneBounds } from "./bounds";
export { createSceneBounds } from "./bounds";
export type { Build } from "./build";
export { createBuild } from "./build";
export type { Morton } from "./morton";
export { createMorton } from "./morton";
export type { RadixSort } from "./sort";
export { createRadixSort, KEYS_PER_BLOCK } from "./sort";
export {
    BVH_INVALID,
    BVH_NODE_BYTES,
    BVH_ROOT_WGSL,
    BVH_TRAIL_LEVELS,
    BVH_TRAVERSE_WGSL,
    bvhRoot,
} from "./traverse";

// #doc:dev
// ### The BVH library
//
// `createBvh` builds a GPU bounding volume hierarchy — a spatial index that turns ray and overlap
// queries from an O(n) scan into a log-depth tree walk. It's rendering-unaware: primitive AABBs in, a
// BVH2 out, naming neither bodies nor triangles. Write AABBs into `prims`, record `build` into an
// encoder, submit, and read the tree from `nodes`. When the set is stable but the geometry moved, record
// `refit` instead — the same topology re-bounded, far cheaper than a rebuild.
//
// Query it from your own kernel: splice `BVH_TRAVERSE_WGSL` (a single-level ray-AABB traverser) into a
// compute shader and it walks the node buffer for you. Ray-triangle leaf tests and two-level TLAS/BLAS
// instancing belong with the consumer, not the builder — the physics broadphase is the worked case,
// building one BVH over body AABBs each step.
//
// The per-stage factories the builder composes are exposed for isolated validation — each stage runs and
// reads back on its own. The bounds reduce and the sort each carry a subgroup arm and an LDS fallback; a
// plugin that builds a BVH lists `BVH_FEATURES` in its `preferredFeatures` so a `subgroups`-less device
// (WebKit) still loads it, on the LDS arm.

/**
 * GPU features the builder's kernels run faster with — the bounds reduction and the radix sort have
 * a subgroup arm and an LDS fallback ({@link createBvh} reads `device.features` to pick). A plugin
 * that builds a BVH (the physics broadphase, an acceleration structure) lists these in its
 * `Plugin.preferredFeatures` so a `subgroups`-less device (WebKit) still loads it, on the LDS arm.
 */
export const BVH_FEATURES: readonly GPUFeatureName[] = ["subgroups"];

/**
 * an LBVH BVH2 builder sized for `maxPrims`. Write primitive AABBs into {@link
 * Bvh.prims} (2 × vec4<f32> per prim: `min.xyz+pad`, `max.xyz+pad`, leaf-index
 * order), record {@link Bvh.build}, submit, then read the BVH2 from {@link
 * Bvh.nodes} (`2N−1` nodes × 32 B; root is node `2N−2` for N≥2, else node 0 — see
 * {@link bvhRoot}). For stable topology under motion, write moved AABBs and record
 * {@link Bvh.refit} instead: the bounds relaxation alone, topology untouched. A {@link
 * Bvh.build} always leaves the tree refit-ready.
 */
export interface Bvh {
    /** input prim AABB buffer — fill [0, count) prims, 2 vec4 each */
    readonly prims: GPUBuffer;
    /** output BVH2 nodes — `2N−1` nodes, 32 B each (the traverser's input) */
    readonly nodes: GPUBuffer;
    /**
     * GPU-driven prim count (one u32 at [0]). Write it (≤ `maxPrims`) before {@link
     * Bvh.build} / {@link Bvh.refit} — a fixed-count producer via `writeBuffer`, a
     * GPU producer by writing it from its own compute. Read on the GPU into bounds,
     * Morton (gating), the build (indirect dispatch + the `2N−1` node range), and the
     * trace root (`bvhRoot`, {@link BVH_ROOT_WGSL}); it never crosses to the CPU.
     */
    readonly count: GPUBuffer;
    /** capacity the buffers are sized for */
    readonly maxPrims: number;
    /** total GPU bytes the builder allocated (owned buffers; injected shared nodes excluded) */
    readonly bytes: number;
    /** record a full rebuild (bounds → Morton → sort → build), count read from {@link Bvh.count} */
    build(encoder: GPUCommandEncoder): void;
    /** record a bounds-only refit over the existing topology, count read from {@link Bvh.count} */
    refit(encoder: GPUCommandEncoder): void;
    destroy(): void;
}

/**
 * build an LBVH BVH2 builder for up to `maxPrims` primitives. Allocates one shared
 * prim / node / working-buffer set and compiles every stage's kernels up front;
 * {@link Bvh.build} and {@link Bvh.refit} then record with no further allocation.
 *
 * Pass `sharedNodes` to build *in place* into a larger external buffer — concatenating
 * several BVHs into one node buffer (packing many small BLASes into one buffer): write the
 * slot's node base into `count[1]` before each build.
 * In-place builds share one node buffer across slots, so {@link Bvh.refit} is unavailable
 * — rebuild instead.
 *
 * `subgroups` selects the kernel family for the two subgroup-using stages (bounds reduce +
 * radix sort): the subgroup-accelerated path (default, when the device has the feature) or
 * the subgroup-free LDS path for WebKit (Safari / WKWebView, which ship no subgroups). Both
 * produce the identical BVH; force `false` to exercise the LDS path on a subgroup device.
 *
 * @example
 * const bvh = await createBvh(device, 1 << 16);
 * device.queue.writeBuffer(bvh.prims, 0, primAabbs);
 * device.queue.writeBuffer(bvh.count, 0, new Uint32Array([count]));
 * const enc = device.createCommandEncoder();
 * bvh.build(enc);
 * device.queue.submit([enc.finish()]);
 * // later, the prims moved but the set is unchanged:
 * device.queue.writeBuffer(bvh.prims, 0, movedAabbs);
 * const enc2 = device.createCommandEncoder();
 * bvh.refit(enc2);
 * device.queue.submit([enc2.finish()]);
 */
export async function createBvh(
    device: GPUDevice,
    maxPrims: number,
    sharedNodes?: GPUBuffer,
    subgroups: boolean = device.features.has("subgroups"),
): Promise<Bvh> {
    const cap = Math.max(1, maxPrims);
    const inPlace = sharedNodes !== undefined;
    // the sort pads keys to a block multiple, so the shared key/payload buffers must
    // hold that padded length, not just `cap`
    const paddedKeys = Math.ceil(cap / KEYS_PER_BLOCK) * KEYS_PER_BLOCK;

    // fail loud + clear before allocating: the prim + node buffers are the builder's largest storage
    // bindings, and a huge `maxPrims` blows the device's per-binding limit (the contact-store guard's
    // pattern). The node buffer (2·cap nodes) dominates the standalone case; an in-place build's nodes are
    // the caller's, so guard its largest owned buffer (prims) instead. Past the limit this throws here, not
    // at an opaque createBuffer validation error.
    const maxBinding = device.limits.maxStorageBufferBindingSize;
    checkStorageBinding(
        "[bvh] the prim buffer",
        cap * 32,
        maxBinding,
        "Lower maxPrims (the builder sizes its buffers to it).",
    );
    if (!inPlace)
        checkStorageBinding(
            "[bvh] the node buffer",
            2 * cap * 32,
            maxBinding,
            "Lower maxPrims (the node buffer holds 2·maxPrims nodes).",
        );

    const prims = device.createBuffer({
        label: "bvh-prims",
        size: cap * 32,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const bounds = device.createBuffer({
        label: "bvh-bounds",
        size: 32,
        usage: GPUBufferUsage.STORAGE,
    });
    const keys = device.createBuffer({
        label: "bvh-keys",
        size: paddedKeys * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const payload = device.createBuffer({
        label: "bvh-payload",
        size: paddedKeys * 4,
        usage: GPUBufferUsage.STORAGE,
    });
    const nodes =
        sharedNodes ??
        device.createBuffer({
            label: "bvh-nodes",
            size: 2 * cap * 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
    // the one control buffer, shared by every stage — written by the caller (CPU or GPU
    // producer): [0] = prim count, [1] = node-write base (folded here per gpu.md binding rule 3)
    const count = device.createBuffer({
        label: "bvh-count",
        size: 8,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const [sb, mc, rs, bd] = await Promise.all([
        createSceneBounds(device, cap, { prims, bounds, count }, subgroups),
        createMorton(device, cap, { prims, bounds, keys, payload, count }),
        createRadixSort(device, cap, { keys, payload, count }, subgroups),
        createBuild(device, cap, { prims, keys, payload, nodes, count }),
    ]);

    // owned-buffer total (the injected node buffer belongs to the caller; exclude it, matching destroy)
    const bytes =
        prims.size +
        bounds.size +
        keys.size +
        payload.size +
        count.size +
        (inPlace ? 0 : nodes.size);

    return {
        prims,
        nodes,
        count,
        maxPrims,
        bytes,
        build(encoder: GPUCommandEncoder): void {
            sb.reduce(encoder);
            mc.compute(encoder);
            // sort dispatched indirect off the count, so it scales with the actual prim count, not
            // the cap; Morton sentinel-padded the [count, cap) tail so the live range sorts first
            rs.sortIndirect(encoder);
            bd.build(encoder);
        },
        refit(encoder: GPUCommandEncoder): void {
            // an in-place build shares one node buffer across slots, so its persisted topology
            // isn't a single tree the refit can re-bound — rebuild those instead
            if (inPlace)
                throw new Error(
                    "createBvh: refit is unavailable for an in-place (shared-nodes) build",
                );
            bd.refit(encoder);
        },
        destroy(): void {
            sb.destroy();
            mc.destroy();
            rs.destroy();
            bd.destroy();
            // the caller owns an injected node buffer; destroy only what we allocated
            for (const b of [prims, bounds, keys, payload, count]) b.destroy();
            if (!inPlace) nodes.destroy();
        },
    };
}
