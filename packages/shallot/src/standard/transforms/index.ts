import type { StorageFlag, TgpuBuffer, TgpuComputePipeline } from "typegpu";
import * as d from "typegpu/data";
import { Compute, capacity, type Plugin, vec4 } from "../../engine";
import { precompile } from "../../engine/runtime";
import { eulerAlias } from "../../engine/utils";
import { Xform } from "../../engine/utils/core";
import { SlabPlugin, slab } from "../slab";
import { composeKernel, composeLayout } from "./compose";

// the transform firehose: one capacity-sized buffer of decomposed per-entity {pos, quat, scale} (`Xform`,
// 48 B) the compose pass gathers from the pos/rot/scale slabs; readers reconstruct the world transform on
// read (`xformWgsl()`), keeping the per-instance read one AoS cache line. Published to `Compute.buffers` as
// "transforms" (the access path — surfaces resolve it by name). A derived GPU buffer, not a per-entity
// field, so it lives here, not on the Transform component (mirrors `Lighting` vs `DirectionalLight` in
// render/). null until initialize (headless: stays null).
let _typed: (TgpuBuffer<d.WgslArray<typeof Xform>> & StorageFlag) | null = null;
let _composePipeline: TgpuComputePipeline | null = null;
// the compose pipeline with its bind group bound, built on first use — the slab mirrors and the
// membership buffer it reads are published by another plugin's `warm`, and `warm` hooks run
// concurrently (`Promise.all` in `build`), so nothing may read them from inside a sibling's warm
let _bound: TgpuComputePipeline | null = null;

// build once, on the first call that has every buffer: the forced precompile (drained after every
// plugin has warmed) or, failing that, the first frame's dispatch.
function bind(): TgpuComputePipeline | null {
    if (_bound) return _bound;
    if (!_composePipeline || !_typed) return null;
    // the firehose binds typed; the slab mirrors and `membership` are raw handles their owning modules
    // publish — a typed bind group takes either, which is what keeps the raw reach-in open. A missing
    // one is a wiring bug: let the bind group creation throw, never skip a frame
    const group = Compute.root.createBindGroup(composeLayout, {
        pos: Transform.pos.gpu!,
        rot: Transform.rot.gpu!,
        scale: Transform.scale.gpu!,
        transforms: _typed,
        membership: Compute.buffers.get("membership")!,
    });
    _bound = _composePipeline.with(group);
    return _bound;
}

/**
 * per-entity transform: pos, rot, scale as direct {@link Quad} fields. Lanes
 * are individually addressable as Singles (`Transform.pos.x.set(eid, v)`) and
 * writable in bulk (`Transform.pos.set(eid, x, y, z, 0)`). The CPU writes flow
 * through the slab plugin; SlabSystem flushes dirty slots once per frame and
 * the compose pass derives world matrices into the `"transforms"` firehose.
 *
 * @example
 * ```
 * <a transform="pos: 0 1 0; rot: 0 0 0 1; scale: 1 1 1" />
 * ```
 */
export const Transform = {
    pos: slab(vec4),
    rot: slab(vec4),
    scale: slab(vec4),
};

/**
 * record the per-frame world-matrix compose dispatch onto `encoder`. Reads
 * the slab canonical GPU buffers (populated by the prior frame's SlabSystem
 * submit), writes the `"transforms"` firehose. Headless (no device) leaves the
 * pipeline unbuilt and the call is a no-op
 */
export function composeTransforms(encoder: GPUCommandEncoder): void {
    const bound = bind();
    if (!bound) return;
    const pass = encoder.beginComputePass({
        label: "shallot-transforms-compose",
        timestampWrites: Compute.span?.("transforms:compose"),
    });
    bound.with(pass).dispatchWorkgroups(Math.ceil(capacity / 64));
    pass.end();
}

/**
 * compose one entity's world matrix on CPU from its slab fields. For camera
 * view derivation and other low-count CPU consumers. Per-frame entity loops
 * should read the `"transforms"` firehose on GPU instead
 *
 * @example
 * const world = composeTransform(eid, new Float32Array(16));
 */
export function composeTransform(eid: number, out: Float32Array): Float32Array {
    const { pos, rot, scale } = Transform;

    const px = pos.x.get(eid);
    const py = pos.y.get(eid);
    const pz = pos.z.get(eid);
    const qx = rot.x.get(eid);
    const qy = rot.y.get(eid);
    const qz = rot.z.get(eid);
    const qw = rot.w.get(eid);
    const sx = scale.x.get(eid);
    const sy = scale.y.get(eid);
    const sz = scale.z.get(eid);

    const x2 = qx + qx;
    const y2 = qy + qy;
    const z2 = qz + qz;
    const xx = qx * x2;
    const xy = qx * y2;
    const xz = qx * z2;
    const yy = qy * y2;
    const yz = qy * z2;
    const zz = qz * z2;
    const wx = qw * x2;
    const wy = qw * y2;
    const wz = qw * z2;

    out[0] = (1 - yy - zz) * sx;
    out[1] = (xy + wz) * sx;
    out[2] = (xz - wy) * sx;
    out[3] = 0;
    out[4] = (xy - wz) * sy;
    out[5] = (1 - xx - zz) * sy;
    out[6] = (yz + wx) * sy;
    out[7] = 0;
    out[8] = (xz + wy) * sz;
    out[9] = (yz - wx) * sz;
    out[10] = (1 - xx - yy) * sz;
    out[11] = 0;
    out[12] = px;
    out[13] = py;
    out[14] = pz;
    out[15] = 1;
    return out;
}

/**
 * the transform substrate: registers the {@link Transform} component and runs the per-frame compose pass
 * that gathers the pos/rot/scale slabs into the `"transforms"` GPU firehose surfaces read by name. In
 * `DEFAULT_PLUGINS`; every rendered entity needs a Transform.
 */
export const TransformsPlugin: Plugin = {
    name: "Transforms",
    components: { Transform },
    dependencies: [SlabPlugin],
    traits: {
        Transform: {
            defaults: () => ({
                pos: [0, 0, 0, 0],
                rot: [0, 0, 0, 1],
                scale: [1, 1, 1, 1],
            }),
            // rot is stored as a quaternion but authored as euler degrees
            aliases: { rot: eulerAlias("rot") },
        },
    },

    initialize(state) {
        _typed = null;
        _composePipeline = null;
        _bound = null;

        if (!Compute.device) return;

        // COPY_DST (typegpu grants it, with COPY_SRC, on every buffer it creates): a CPU physics backend
        // (tumble) writes a mover's interpolated pose straight in via `queue.writeBuffer` (standard/tumble
        // ComposeSystem). The GPU-only AVBD backend's compose is a compute pass and needs no CPU write,
        // but the buffer is shared, so the usage covers both.
        _typed = Compute.root
            .createBuffer(d.arrayOf(Xform, capacity))
            .$usage("storage")
            .$name("shallot-transforms");
        Compute.buffers.set("transforms", Compute.root.unwrap(_typed));
        Compute.typed.set("transforms", _typed);

        const t = state.membership.bit(Transform);
        _composePipeline = Compute.root
            .createComputePipeline({ compute: composeKernel(t.gen * capacity, t.mask, capacity) })
            .$name("shallot-transforms-compose");
    },

    warm() {
        if (!_composePipeline) return;
        // the bind, not just the dispatch, is deferred into the forcer: the drain runs after every
        // plugin's warm has resolved, which is the first moment the buffers this reads are all up
        precompile("shallot-transforms-compose", () => {
            const bound = bind();
            bound?.dispatchWorkgroups(0);
            return bound;
        });
    },
};
