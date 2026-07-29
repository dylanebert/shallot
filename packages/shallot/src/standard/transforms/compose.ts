import tgpu from "typegpu";
import * as d from "typegpu/data";
import { Xform } from "../../engine/utils/core";

// The compose kernel: gather the pos/rot/scale slabs into the decomposed `Xform` firehose. No matrix
// math — readers reconstruct on demand (`xformWgsl()`); the gather just lays the three SoA slabs into one
// AoS record so a reader's per-instance read is a single cache line. Gated on Transform MEMBERSHIP (the
// part-pack shape): the firehose has a second writer — a physics backend composes `Body` poses into the
// same buffer (`Body` excludes `Transform`, so the two writers partition by slot) — and a CPU backend's
// `queue.writeBuffer` executes in queue order BEFORE this frame's encoder, so an ungated scatter would
// stomp every physics record with the unset Transform slab (zero scale — an invisible Body+Part).

/** @internal */
export const composeLayout = tgpu.bindGroupLayout({
    pos: { storage: d.arrayOf(d.vec4f), access: "readonly" },
    rot: { storage: d.arrayOf(d.vec4f), access: "readonly" },
    scale: { storage: d.arrayOf(d.vec4f), access: "readonly" },
    transforms: { storage: d.arrayOf(Xform), access: "mutable" },
    membership: { storage: d.arrayOf(d.u32), access: "readonly" },
});

// `base` (the component's generation row), `mask` (its bit) and `cap` are captured numbers, so they
// resolve to literals — the gate stays constant-folded, with no uniform to bind or read.
/** @internal */
export function composeKernel(base: number, mask: number, cap: number) {
    return tgpu.computeFn({
        workgroupSize: [64],
        in: { gid: d.builtin.globalInvocationId },
    })((input) => {
        "use gpu";
        const i = input.gid.x;
        if (i >= cap) return;
        if ((composeLayout.$.membership[base + i] & mask) === 0) return;
        const p = composeLayout.$.pos[i];
        const q = composeLayout.$.rot[i];
        const s = composeLayout.$.scale[i];
        composeLayout.$.transforms[i] = Xform({
            pos: d.vec3f(p.x, p.y, p.z),
            quat: d.vec4f(q.x, q.y, q.z, q.w),
            scale: d.vec3f(s.x, s.y, s.z),
        });
    });
}

/** the emitted compose WGSL for a membership gate — the device-free structural seam its test resolves.
 *  @internal */
export function composeWgsl(base: number, mask: number, cap: number): string {
    return tgpu.resolve([composeKernel(base, mask, cap)], { names: "strict" });
}
