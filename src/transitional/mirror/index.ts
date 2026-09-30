// Destination: engine; owner: engine-gpu-core.md.
import { isBuffer, type TgpuBuffer } from "typegpu";
import type { AnyData } from "typegpu/data";
import type { Plugin, State, System } from "../../engine";
import { deviceLost, type LazyAlloc } from "../../engine/runtime";

/** what {@link mirror} reads back: a raw `GPUBuffer` or its typed twin. Mirror is byte-granular either
 *  way — a typed source is unwrapped at construction and the snapshot stays opaque bytes. */
export type MirrorSource = GPUBuffer | TgpuBuffer<AnyData>;

function unwrap(state: State, source: MirrorSource): GPUBuffer {
    return isBuffer(source) ? state.gpu.root.unwrap(source) : source;
}

/**
 * buffer-level GPU→CPU readback. Construct with a source buffer;
 * {@link MirrorSystem} encodes one `copyBufferToBuffer` + `mapAsync` per
 * frame into a staging ring slot, stamps the encode-time tick, and writes
 * {@link Mirror.snapshot} once the map resolves.
 *
 * Mirror operates at buffer granularity, not field granularity — the
 * snapshot is opaque bytes. It has no opinion about what they mean.
 * Snapshot availability depends on device timing; derived state stays out of the determinism hash.
 * Compaction stays a consumer concern: write a smaller GPU-only buffer in
 * your compute graph and point Mirror at that.
 *
 * `snapshot.bytes` is a buffer reused across readbacks (the latest readback
 * overwrites it), so it allocates nothing per frame. Read it in the frame you
 * observe it — it's the current readback, not a retained per-frame copy; don't
 * hold it across frames expecting it to stay frozen.
 *
 * @example
 * const m = mirror(state, physics.compactBuffer);
 * // each frame: MirrorSystem copies + maps, eventually populating m.snapshot
 * if (m.snapshot) {
 *     const view = new Float32Array(m.snapshot.bytes);
 *     const age = state.time.fixedTick - m.snapshot.fixedTick;
 * }
 */
const mirrorsKey = Symbol("shallot.mirrors");

function mirrorsFor(state: State): Set<Mirror> {
    return state.resource(mirrorsKey, () => new Set());
}

export class Mirror<T extends MirrorSource = MirrorSource> {
    /** the State that owns this mirror's staging ring and snapshot */
    readonly state: State;
    /** schema-carrying source passed at construction; raw sources remain raw for the WebGPU escape */
    readonly source: T;
    /** byte size of each staging slot and each {@link snapshot} */
    get size(): number {
        return this._raw.size;
    }

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

    private _raw: GPUBuffer;
    private readonly _device: GPUDevice;
    private _generation = 0;
    private _unsubscribe: (() => void) | undefined;
    private readonly _timers = new Map<GPUBuffer, ReturnType<typeof setTimeout>>();

    constructor(state: State, source: T, opts?: { ring?: number }) {
        this.state = state;
        this.source = source;
        this._device = state.gpu.device;
        this._raw = unwrap(state, source);
        this._ringSize = opts?.ring ?? 2;
        const table = state.tableForBuffer(this._raw);
        if (table) {
            const rebind = (buffer: GPUBuffer) => {
                if (buffer === this._raw) return;
                this._raw = buffer;
                this._generation++;
                this._release();
            };
            this._unsubscribe =
                this._raw === table.activeRowsBuffer
                    ? table.subscribeActiveRows(rebind)
                    : this._raw === table.eidToRowBuffer
                      ? table.subscribeMap(rebind)
                      : table.subscribe(rebind);
        }
        mirrorsFor(state).add(this);
    }

    /** number of staging buffers currently allocated. capped at the ring depth. */
    get allocated(): number {
        return this._slots.length;
    }

    /** stop reading back; releases staging buffers. Pending map callbacks become no-ops. */
    dispose(): void {
        if (this._disposed) return;
        this._disposed = true;
        this._unsubscribe?.();
        mirrorsFor(this.state).delete(this);
        this._release();
    }

    private _release(): void {
        for (const timer of this._timers.values()) clearTimeout(timer);
        this._timers.clear();
        for (const b of this._slots) b.destroy();
        this._slots.length = 0;
        this._free.length = 0;
        this._owned = null;
        this.snapshot = null;
    }

    static reset(state: State): void {
        const mirrors = mirrorsFor(state);
        for (const m of mirrors) {
            m._disposed = true;
            m._unsubscribe?.();
            m._release();
        }
        mirrors.clear();
    }

    static flush(state: State): void {
        const mirrors = mirrorsFor(state);
        if (mirrors.size === 0) return;
        const device = state.gpu.device;
        if (deviceLost(device)) return;
        const fixedTick = state.time.fixedTick;
        const frame = state.gpu.frame;

        const encoder = device.createCommandEncoder({ label: "mirror-flush" });
        const pending: { m: Mirror; slot: GPUBuffer }[] = [];

        for (const m of mirrors) {
            if (m._device !== device) continue;
            let slot = m._free.pop();
            if (!slot) {
                // Ring saturated — every staging slot still mapping. Skip this tick.
                if (m._slots.length >= m._ringSize) continue;
                // `lazy: true` declares this ring to the profiler (`LazyAlloc`, engine/runtime): a slot
                // grows on real GPU backpressure — how many have grown by the moment `Profile` samples
                // depends on device readback timing, not scenario code or params — so a byte-budget gate
                // excludes these bytes from its exact total.
                const desc: GPUBufferDescriptor & LazyAlloc = {
                    label: "mirror-staging",
                    size: m.size,
                    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
                    lazy: true,
                };
                slot = device.createBuffer(desc);
                m._slots.push(slot);
            }
            encoder.copyBufferToBuffer(m._raw, 0, slot, 0, m.size);
            pending.push({ m, slot });
        }

        if (pending.length === 0) return;
        // bare copyBufferToBuffer — WebGPU has no timestampWrites for a copy, so this submit is untimed by
        // design; its GPU cost surfaces as fence wait, not a pass span. Don't try
        // to wrap it in a span — measure it via fence wait instead.
        device.queue.submit([encoder.finish()]);

        for (const { m, slot } of pending) {
            const generation = m._generation;
            const label = `Mirror ${m._raw.label} frame ${frame} readback`;
            const timer = setTimeout(() => {
                m._timers.delete(slot);
                if (m._disposed || generation !== m._generation) return;
                slot.destroy();
                const index = m._slots.indexOf(slot);
                if (index >= 0) m._slots.splice(index, 1);
                console.error(`${label} timed out after 750 ms`);
            }, 750);
            m._timers.set(slot, timer);
            slot.mapAsync(GPUMapMode.READ, 0, m.size).then(
                () => {
                    clearTimeout(timer);
                    m._timers.delete(slot);
                    if (m._disposed || generation !== m._generation || !m._slots.includes(slot))
                        return;
                    if (m.state.disposed || deviceLost(m._device)) {
                        m.dispose();
                        return;
                    }
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
                (error: unknown) => {
                    clearTimeout(timer);
                    m._timers.delete(slot);
                    if (m._disposed || generation !== m._generation || !m._slots.includes(slot))
                        return;
                    if (m.state.disposed || deviceLost(m._device)) {
                        m.dispose();
                        return;
                    }
                    m._free.push(slot);
                    console.error(`${label} map failed; the slot was recycled: ${String(error)}`);
                },
            );
        }
    }
}

/** construct a buffer-level mirror over a raw or typed buffer; registers with {@link MirrorSystem} */
export function mirror<T extends MirrorSource>(
    state: State,
    source: T,
    opts?: { ring?: number },
): Mirror<T> {
    return new Mirror(state, source, opts);
}

/**
 * per-frame readback for every registered mirror. Runs at the tail of the
 * draw group so any compute that wrote a mirror's source this frame has
 * already encoded.
 */
export const MirrorSystem: System = {
    group: "draw",
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

    initialize(state: State) {
        Mirror.reset(state);
    },

    dispose(state: State) {
        Mirror.reset(state);
    },
};
