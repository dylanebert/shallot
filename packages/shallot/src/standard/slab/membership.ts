import { Compute, capacity, type State, type System } from "../../engine";

// GPU mirror of the ECS component-membership bitset. The CPU bitset is the
// source of truth; this is a write-only synced mirror published as the
// "membership" buffer. A GPU producer that scans a buffer by index gates on it
// — `(membership[gen * capacity + eid] & mask) != 0` is the authoritative "does
// eid carry this component" test — so a destroyed or detached slot stops
// satisfying the gate the frame after the change, with no per-field reset and
// no sentinel value reserved out of the data domain.

let _gpu: GPUBuffer | null = null;
let _mirror: Uint32Array<ArrayBuffer> | null = null;

/**
 * allocate the mirror + CPU staging. Sized from the generation count, which
 * `build` fixes (it assigns every registered component its bit up front), so
 * the size never changes after this. COPY_SRC is for readback in tests/debug;
 * production only reads it in shaders
 */
function alloc(state: State): void {
    _gpu?.destroy();
    _mirror = new Uint32Array(state.membership.generations * capacity);
    _gpu = Compute.device.createBuffer({
        label: "membership",
        size: _mirror.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    Compute.buffers.set("membership", _gpu);
}

/**
 * copy changed membership words up. A whole-buffer write on frames where
 * membership changed (spawn / despawn / add / remove), a no-op otherwise.
 * Churn is bounded by the structural-change rate — far below the per-frame
 * transform firehose — so the full re-upload on dirty frames stays cheap
 */
function flush(state: State): void {
    if (!_gpu || !_mirror) return;
    const mirror = _mirror;
    const changed = state.membership.drain((eid, gen, word) => {
        const i = gen * capacity + eid;
        if (i < mirror.length) mirror[i] = word;
    });
    if (changed) Compute.device.queue.writeBuffer(_gpu, 0, mirror);
}

function release(): void {
    if (_gpu && Compute.buffers?.get("membership") === _gpu) Compute.buffers.delete("membership");
    _gpu?.destroy();
    _gpu = null;
    _mirror = null;
}

/**
 * flushes the component-membership bitset to the `"membership"` GPU buffer.
 * Draw-group head, before any index-scan consumer (the Part pack) reads it.
 * `mode: "always"` so it runs in edit mode too — the pack it feeds does.
 * Owns the buffer: allocates at setup (generation count is fixed by `build`),
 * releases on dispose
 */
export const MembershipSystem: System = {
    group: "draw",
    first: true,
    annotations: { mode: "always" },
    setup(state) {
        alloc(state);
    },
    update(state) {
        flush(state);
    },
    dispose() {
        release();
    },
};
