import { Compute, type Plugin, type State, type System } from "../../engine";

/**
 * buffer-level GPU→CPU readback. Construct with a source buffer;
 * {@link MirrorSystem} encodes one `copyBufferToBuffer` + `mapAsync` per
 * frame into a staging ring slot, stamps the encode-time tick, and writes
 * {@link Mirror.snapshot} once the map resolves.
 *
 * Mirror operates at buffer granularity, not field granularity — the
 * snapshot is opaque bytes. It has no opinion about what they mean.
 * Compaction stays a consumer concern: write a smaller GPU-only buffer in
 * your compute graph and point Mirror at that.
 *
 * `snapshot.bytes` is a buffer reused across readbacks (the latest readback
 * overwrites it), so it allocates nothing per frame. Read it in the frame you
 * observe it — it's the current readback, not a retained per-frame copy; don't
 * hold it across frames expecting it to stay frozen.
 *
 * @example
 * const m = mirror(physics.compactBuffer);
 * // each frame: MirrorSystem copies + maps, eventually populating m.snapshot
 * if (m.snapshot) {
 *     const view = new Float32Array(m.snapshot.bytes);
 *     const age = state.time.fixedTick - m.snapshot.fixedTick;
 * }
 */
export class Mirror {
    private static _all: Mirror[] = [];

    readonly source: GPUBuffer;
    /** byte size of each staging slot and each {@link snapshot} */
    readonly size: number;

    /** latest map-resolved snapshot. `null` until the first map completes. `bytes` is reused across
     *  readbacks (see the class doc) — read it in-frame, don't retain it. */
    snapshot: { fixedTick: number; frame: number; bytes: ArrayBuffer } | null = null;

    private readonly _ringSize: number;
    private readonly _free: GPUBuffer[] = [];
    private readonly _slots: GPUBuffer[] = [];
    // the persistent CPU-side destination the mapped range is copied into, reused across readbacks so a
    // large/frequent mirror doesn't allocate its full size every frame (a major-GC source). Lazily sized.
    private _owned: ArrayBuffer | null = null;
    private _disposed: boolean = false;

    constructor(source: GPUBuffer, opts?: { ring?: number }) {
        this.source = source;
        this.size = source.size;
        this._ringSize = opts?.ring ?? 2;
        Mirror._all.push(this);
    }

    /** number of staging buffers currently allocated. capped at the ring depth. */
    get allocated(): number {
        return this._slots.length;
    }

    /** stop reading back; releases staging buffers. Pending map callbacks become no-ops. */
    dispose(): void {
        if (this._disposed) return;
        this._disposed = true;
        const i = Mirror._all.indexOf(this);
        if (i !== -1) Mirror._all.splice(i, 1);
        this._release();
    }

    private _release(): void {
        for (const b of this._slots) b.destroy();
        this._slots.length = 0;
        this._free.length = 0;
        this._owned = null;
    }

    static reset(): void {
        for (const m of Mirror._all) {
            m._disposed = true;
            m._release();
        }
        Mirror._all.length = 0;
    }

    static flush(state: State): void {
        if (Mirror._all.length === 0) return;
        const device = Compute.device;
        const fixedTick = state.time.fixedTick;
        const frame = Compute.frame;

        const encoder = device.createCommandEncoder({ label: "mirror-flush" });
        const pending: { m: Mirror; slot: GPUBuffer }[] = [];

        for (const m of Mirror._all) {
            let slot = m._free.pop();
            if (!slot) {
                // Ring saturated — every staging slot still mapping. Skip this tick.
                if (m._slots.length >= m._ringSize) continue;
                slot = device.createBuffer({
                    label: "mirror-staging",
                    size: m.size,
                    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
                });
                m._slots.push(slot);
            }
            encoder.copyBufferToBuffer(m.source, 0, slot, 0, m.size);
            pending.push({ m, slot });
        }

        if (pending.length === 0) return;
        // bare copyBufferToBuffer — WebGPU has no timestampWrites for a copy, so this submit is untimed by
        // design; its GPU cost surfaces as fence wait, not a pass span (gpu.md "GPU profiling"). Don't try
        // to wrap it in a span — measure it via fence wait instead.
        device.queue.submit([encoder.finish()]);

        for (const { m, slot } of pending) {
            slot.mapAsync(GPUMapMode.READ, 0, m.size).then(
                () => {
                    if (m._disposed) return;
                    // A stale (out-of-order) map resolution must not clobber a newer snapshot — without
                    // the per-readback fresh buffer, last-resolved-wins would otherwise overwrite the
                    // reused buffer with older data + an older frame stamp.
                    if (m.snapshot && frame < m.snapshot.frame) {
                        slot.unmap();
                        m._free.push(slot);
                        return;
                    }
                    const mapped = slot.getMappedRange(0, m.size);
                    if (!m._owned) m._owned = new ArrayBuffer(m.size);
                    new Uint8Array(m._owned).set(new Uint8Array(mapped));
                    slot.unmap();
                    m._free.push(slot);
                    if (m.snapshot) {
                        m.snapshot.fixedTick = fixedTick;
                        m.snapshot.frame = frame;
                    } else {
                        m.snapshot = { fixedTick, frame, bytes: m._owned };
                    }
                },
                () => {
                    if (!m._disposed) m._free.push(slot);
                },
            );
        }
    }
}

/** construct a buffer-level mirror; registers with {@link MirrorSystem} */
export function mirror(source: GPUBuffer, opts?: { ring?: number }): Mirror {
    return new Mirror(source, opts);
}

/**
 * per-frame readback for every registered mirror. Runs at the tail of the
 * draw group so any compute that wrote a mirror's source this frame has
 * already encoded.
 */
export const MirrorSystem: System = {
    group: "draw",
    // `mode: "always"` — a live authoring host builds with `mode: "edit"`, and a readback in edit mode
    // (a viewport pick samples `view.tag` through a Mirror) needs the flush to run there too. The flush is
    // non-destructive (buffer copies + a `snapshot` field, no component add/remove), so it's edit-safe.
    annotations: { mode: "always" },
    last: true,
    update(state) {
        Mirror.flush(state);
    },
};

/**
 * owns the per-frame mirror flush. Plugins that allocate mirrors in
 * `initialize` should declare `dependencies: [MirrorPlugin]` so the
 * registry is cleared before allocation.
 */
export const MirrorPlugin: Plugin = {
    name: "Mirror",
    systems: [MirrorSystem],

    initialize() {
        Mirror.reset();
    },

    dispose() {
        Mirror.reset();
    },
};
