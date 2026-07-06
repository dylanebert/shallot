import type { Plugin, State, System } from "../../engine";
import { Compute, mountOverlay } from "../../engine";
import { createMeasure, foldIndirect, INDIRECT_FLOOR_US } from "./benchmark";

export type {
    BenchmarkAPI,
    BenchmarkCompileStats,
    BenchmarkCpuStats,
    BenchmarkFrameStats,
    BenchmarkGpuStats,
    BenchmarkMeasurement,
} from "./benchmark";

/**
 * live read-only view of the profiler: per-frame CPU and GPU pass timings, memory, and one-shot compile timings. Populated by {@link ProfilePlugin}; empty without it.
 * @expand
 */
export interface Profile {
    /** per-system CPU timings for the current frame, in milliseconds */
    readonly cpu: ReadonlyMap<string, number>;
    /** per-pass GPU timings (most recent fully-resolved frame), in milliseconds. Greedy-held for
     *  display: a fixed-group pass keeps its last value across frames it doesn't fire (see the
     *  drain hold). For exact per-occurrence accounting, use {@link gpuTime} / {@link gpuFires}. */
    readonly gpu: ReadonlyMap<string, number>;
    /** cumulative GPU time per pass since attach, in milliseconds — summed over every actual
     *  occurrence (a fixed-group pass once per fixed step, a draw pass once per frame). Pair with
     *  {@link gpuFires} to derive the per-occurrence cost: `gpuTime / gpuFires`. Immune to the
     *  greedy hold the display {@link gpu} map applies — the source of truth for the benchmark. */
    readonly gpuTime: ReadonlyMap<string, number>;
    /** cumulative occurrence count per pass since attach — how many times the pass actually fired.
     *  Divided into a window's frame count it gives the pass's fire cadence (≈1 for a per-frame
     *  render pass, ≈fixed-steps-per-frame for a per-step sim pass). */
    readonly gpuFires: ReadonlyMap<string, number>;
    /** cumulative indirect-draw count per pass since attach — summed over every frame the pass issued
     *  draws. Pair with {@link indirectFires} for the per-frame count (`indirectCount / indirectFires`)
     *  and derive Dawn's injected-validation floor via `INDIRECT_FLOOR_US` (gpu.md). The benchmark
     *  window-diffs it like {@link gpuTime} / {@link gpuFires}, untimed by `timestampWrites`. */
    readonly indirectCount: ReadonlyMap<string, number>;
    /** cumulative frame count per pass — how many frames the pass reported indirect draws. */
    readonly indirectFires: ReadonlyMap<string, number>;
    /** per-pipeline compile durations from app startup, in milliseconds */
    readonly compile: ReadonlyMap<string, number>;
    /** wall-clock span from the first pipeline build start to the last build end */
    readonly compileMs: number;
    /** ms spent awaiting the prior frame's GPU fence before this frame began */
    readonly fenceWaitMs: number;
    /** cumulative `device.queue.submit` calls since attach. Each submit is a renderer→GPU-process IPC
     *  round-trip + a GPU serialization point, untimed by `timestampWrites` (the cost surfaces in fence
     *  wait, not a pass). A frame issues several — render, the slab flush, a mirror readback, the
     *  profiler's own resolve — so the benchmark window-diffs this into submits/frame, the lever for
     *  collapsing them into one encoder (gpu.md "Single queue"). */
    readonly submitCount: number;
}

interface ResourceAlloc {
    label: string;
    bytes: number;
    kind: "buffer" | "texture";
}

// one readback ring slot: a MAP_READ buffer plus the pass names + count for the queries copied into
// it this frame (snapshotted at resolve, so the shared `_passes` scratch can be reused next frame).
interface ReadSlot {
    buffer: GPUBuffer;
    passes: string[];
    count: number;
}

// A fixed-group GPU pass (physics solve) fires 0..N times per draw frame — the accumulator batches fixed
// steps unevenly (render commonly outruns the 60Hz fixed rate, so most draw frames take 0 steps).
// Reporting the display `gpu` map per frame then oscillates: 0 on a step-less frame, N× on a multi-step
// one. So drain HOLDS a pass's last value across frames it doesn't fire (shown as if it ran every frame),
// stacking when it fires multiple times — a steady readout. A pass absent this many drained frames is
// evicted, so a genuinely one-shot pass (an initial slab:flush) decays out fast — well inside a bench
// warmup — while a fixed-group pass (fires within the window) stays held. The hold is a DISPLAY smoothing
// only; the benchmark reads the exact per-occurrence counters (gpuTime / gpuFires), never the held map.
const GPU_HOLD_DRAINS = 8;

// the timestamp read-buffer ring depth. resolve copies each frame's query timestamps into a free slot;
// the slot maps async (1–2 frames) then drain reads + recycles it. Sized over the 2-frame fence pipeline
// + map latency so a slot is always free — a SINGLE buffer's copy-when-unmapped cadence (~3 frames) beats
// against the fixed-step cadence (~4 frames at headless 240Hz) and silently drops the physics passes:
// they age out of the held `gpu` map and the reported GPU total craters mid-run.
const READ_RING = 4;

// timestamp queries + pipeline-compile timing + live allocation tracking. Owns
// the GPU query set + staging buffers (singleton-lifetime — live with the
// device, never destroyed per-state) and patches `device.createBuffer` /
// `createTexture` / `createComputePipelineAsync` / `createRenderPipelineAsync`
// on attach.
class ProfileImpl implements Profile {
    readonly cpu = new Map<string, number>();
    readonly gpu = new Map<string, number>();
    readonly gpuTime = new Map<string, number>();
    readonly gpuFires = new Map<string, number>();
    // this frame's per-pass indirect-draw tally (summed across a pass's occurrences — sear:color reports
    // once per camera), cleared at frame begin like `cpu`. The overlay reads it for the live floor; `reset`
    // folds it into the cumulative counters first (one fire per pass per frame), the benchmark's window-diff
    // unit. Synchronous per-frame data — no readback delay, so no greedy hold like the timed `gpu` map
    readonly indirect = new Map<string, number>();
    readonly indirectCount = new Map<string, number>();
    readonly indirectFires = new Map<string, number>();
    readonly compile = new Map<string, number>();
    compileMs = 0;
    fenceWaitMs = 0;
    submitCount = 0;
    // bumps each time drain repopulates `gpu` so the overlay can integrate
    // new gpu samples once per readback (the map is sticky between drains;
    // sampling it every display frame would multiply-count).
    drainCount = 0;

    readonly buffers = new Set<GPUBuffer>();
    readonly textures = new Set<GPUTexture>();
    readonly sizes = new Map<object, ResourceAlloc>();
    bufferBytes = 0;
    textureBytes = 0;

    private _compileEarliest = Infinity;
    private _compileLatest = -Infinity;

    private _querySet: GPUQuerySet | null = null;
    private _resolveBuffer: GPUBuffer | null = null;
    private _capacity = 0;
    private _nextSlot = 0;
    private readonly _passes: string[] = [];
    private readonly _slotCache: GPUComputePassTimestampWrites[] = [];
    // drains since each pass last fired, for the greedy hold + eviction (see GPU_HOLD_DRAINS)
    private readonly _gpuMiss = new Map<string, number>();
    // reused scratch: one drained frame's per-pass summed time (cleared per slot, no per-frame alloc)
    private readonly _fired = new Map<string, number>();
    // the readback ring (see READ_RING): a slot carries its frame's resolved buffer + the pass names
    // for the queries it holds. `_free` slots are unmapped + ready to copy into; `_mapped` slots have
    // resolved their async map and await drain.
    private readonly _free: ReadSlot[] = [];
    private readonly _mapped: ReadSlot[] = [];

    attach(device: GPUDevice, capacity = 2048): void {
        if (this._querySet) return;
        this._capacity = capacity;
        this._querySet = device.createQuerySet({ type: "timestamp", count: capacity * 2 });
        const bytes = capacity * 2 * 8;
        this._resolveBuffer = device.createBuffer({
            label: "profile-resolve",
            size: bytes,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        for (let i = 0; i < READ_RING; i++) {
            const buffer = device.createBuffer({
                label: "profile-read",
                size: bytes,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            });
            this._free.push({ buffer, passes: [], count: 0 });
        }

        const origCompute = device.createComputePipelineAsync.bind(device);
        device.createComputePipelineAsync = async (desc: GPUComputePipelineDescriptor) => {
            const start = performance.now();
            const pipeline = await origCompute(desc);
            this.recordCompile(desc.label ?? "", start, performance.now());
            return pipeline;
        };
        const origRender = device.createRenderPipelineAsync.bind(device);
        device.createRenderPipelineAsync = async (desc: GPURenderPipelineDescriptor) => {
            const start = performance.now();
            const pipeline = await origRender(desc);
            this.recordCompile(desc.label ?? "", start, performance.now());
            return pipeline;
        };

        const origCreateBuffer = device.createBuffer.bind(device);
        device.createBuffer = (desc) =>
            this.trackAlloc(
                origCreateBuffer(desc),
                desc.label ?? "",
                desc.size,
                this.buffers,
                "buffer",
            );

        const origCreateTexture = device.createTexture.bind(device);
        device.createTexture = (desc) =>
            this.trackAlloc(
                origCreateTexture(desc),
                desc.label ?? "",
                textureByteSize(desc),
                this.textures,
                "texture",
            );

        // count every submit (the patch lives on the queue object, so a caller caching `device.queue`
        // still sees it). The profiler's own resolve submit counts too — it's a real per-frame submit.
        const queue = device.queue;
        const origSubmit = queue.submit.bind(queue);
        queue.submit = (buffers) => {
            this.submitCount++;
            origSubmit(buffers);
        };
    }

    private trackAlloc<T extends GPUBuffer | GPUTexture>(
        obj: T,
        label: string,
        bytes: number,
        set: Set<T>,
        kind: "buffer" | "texture",
    ): T {
        const totals = kind === "buffer" ? "bufferBytes" : "textureBytes";
        set.add(obj);
        this[totals] += bytes;
        this.sizes.set(obj, { label, bytes, kind });
        const origDestroy = obj.destroy.bind(obj);
        (obj as { destroy: () => void }).destroy = () => {
            set.delete(obj);
            this[totals] -= bytes;
            this.sizes.delete(obj);
            origDestroy();
        };
        return obj;
    }

    record(name: string, ms: number): void {
        this.cpu.set(name, (this.cpu.get(name) ?? 0) + ms);
    }

    recordIndirect(name: string, count: number): void {
        this.indirect.set(name, (this.indirect.get(name) ?? 0) + count);
    }

    reset(): void {
        // fold the just-completed frame's indirect tally into the cumulative counters before clearing (the
        // overlay read it at the prior frame's draw-last; the benchmark window-diffs the cumulative maps)
        foldIndirect(this.indirect, this.indirectCount, this.indirectFires);
        this.indirect.clear();
        this.cpu.clear();
        this._nextSlot = 0;
    }

    span(name: string): GPUComputePassTimestampWrites | undefined {
        if (!this._querySet) return undefined;
        const slot = this._nextSlot;
        if (slot >= this._capacity) return undefined;
        this._passes[slot] = name;
        this._nextSlot = slot + 1;
        let cached = this._slotCache[slot];
        if (!cached) {
            cached = {
                querySet: this._querySet,
                beginningOfPassWriteIndex: slot * 2,
                endOfPassWriteIndex: slot * 2 + 1,
            };
            this._slotCache[slot] = cached;
        }
        return cached;
    }

    // resolve the just-completed frame's queries into a free ring slot + kick its async map. Called at
    // the START of the next frame (ProfileFrameBeginSystem), so every one of the prior frame's submits —
    // the render encoder, the separate slab flush, any producer's pass — has settled into one coherent
    // capture. Resolving at the tail of the same frame raced the render submit (a `last: true` tie with
    // EndFrameSystem, which the render contract warns against) and could capture the render passes a frame
    // stale while the separately-submitted slab flush was current. destination offset is always 0 so
    // resolveQuerySet's 256-byte alignment holds trivially. A slot is taken every frame (the ring is
    // sized so one is free; see READ_RING).
    resolve(device: GPUDevice): void {
        if (!this._querySet || !this._resolveBuffer) return;
        const queryCount = this._nextSlot * 2;
        if (queryCount === 0) return;
        const slot = this._free.pop();
        if (!slot) return; // ring momentarily saturated; the held `gpu` values cover this frame
        const encoder = device.createCommandEncoder({ label: "profile-resolve" });
        encoder.resolveQuerySet(this._querySet, 0, queryCount, this._resolveBuffer, 0);
        encoder.copyBufferToBuffer(this._resolveBuffer, 0, slot.buffer, 0, queryCount * 8);
        slot.count = this._nextSlot;
        for (let i = 0; i < this._nextSlot; i++) slot.passes[i] = this._passes[i];
        device.queue.submit([encoder.finish()]);
        slot.buffer.mapAsync(GPUMapMode.READ).then(
            () => this._mapped.push(slot),
            () => this._free.push(slot),
        );
    }

    drain(): void {
        while (this._mapped.length > 0) {
            const slot = this._mapped.shift()!;
            if (slot.buffer.mapState !== "mapped") {
                this._free.push(slot);
                continue;
            }
            const data = new BigUint64Array(slot.buffer.getMappedRange());
            // sum this frame's occurrences per pass — a fixed-group pass fires once per fixed step, so
            // multiple steps in one draw frame stack here (their real combined per-frame GPU cost).
            const fired = this._fired;
            fired.clear();
            for (let i = 0; i < slot.count; i++) {
                const name = slot.passes[i];
                fired.set(
                    name,
                    (fired.get(name) ?? 0) + Number(data[i * 2 + 1] - data[i * 2]) / 1e6,
                );
            }
            slot.buffer.unmap();
            this._free.push(slot);
            // exact per-occurrence accounting — cumulative time + fire count, one fire per drained
            // frame the pass appears in. Untouched by the display hold below.
            for (const [name, ms] of fired) {
                this.gpuTime.set(name, (this.gpuTime.get(name) ?? 0) + ms);
                this.gpuFires.set(name, (this.gpuFires.get(name) ?? 0) + 1);
            }
            // greedy hold (see GPU_HOLD_DRAINS): a fired pass takes this frame's value; an absent pass
            // keeps its last value (held, as if it ran) until absent past the hold window, then evicts.
            for (const [name, ms] of fired) {
                this.gpu.set(name, ms);
                this._gpuMiss.set(name, 0);
            }
            for (const name of this.gpu.keys()) {
                if (fired.has(name)) continue;
                const miss = (this._gpuMiss.get(name) ?? 0) + 1;
                if (miss > GPU_HOLD_DRAINS) {
                    this.gpu.delete(name);
                    this._gpuMiss.delete(name);
                } else {
                    this._gpuMiss.set(name, miss);
                }
            }
            this.drainCount++;
        }
    }

    private recordCompile(label: string, start: number, end: number): void {
        this.compile.set(label, end - start);
        if (start < this._compileEarliest) this._compileEarliest = start;
        if (end > this._compileLatest) this._compileLatest = end;
        this.compileMs = this._compileLatest - this._compileEarliest;
    }
}

function texelBytes(format: string): number {
    if (format.startsWith("rgba32")) return 16;
    if (format.startsWith("rgba16")) return 8;
    if (format.startsWith("rgba8") || format.startsWith("bgra8")) return 4;
    if (format.startsWith("rg32")) return 8;
    if (format.startsWith("rg16")) return 4;
    if (format.startsWith("rg8")) return 2;
    if (format.startsWith("r32")) return 4;
    if (format.startsWith("r16")) return 2;
    if (format.startsWith("r8")) return 1;
    if (format === "depth24plus" || format === "depth32float" || format === "depth24plus-stencil8")
        return 4;
    return 4;
}

function textureByteSize(desc: GPUTextureDescriptor): number {
    const [w, h, d] = normalizeSize(desc.size);
    const bpp = texelBytes(desc.format);
    const mips = desc.mipLevelCount ?? 1;
    let total = 0;
    for (let i = 0; i < mips; i++) {
        total += Math.max(1, w >> i) * Math.max(1, h >> i) * d * bpp;
    }
    return total;
}

function normalizeSize(size: GPUExtent3DStrict): [number, number, number] {
    if (Array.isArray(size)) return [size[0], size[1] ?? 1, size[2] ?? 1];
    const dict = size as GPUExtent3DDictStrict;
    return [dict.width, dict.height ?? 1, dict.depthOrArrayLayers ?? 1];
}

interface ViewportData {
    canvasWidth: number;
    canvasHeight: number;
    cssWidth: number;
    cssHeight: number;
    dpr: number;
    fullscreen: boolean;
}

// the heavy display snapshot, built only on a throttled render (not per frame). The cpu/gpu
// rows aren't here — they're ticked into the pools from the live Maps each frame (tickPool)
interface OverlayData {
    fps: number;
    frameTime: number;
    fenceWaitMs: number;
    gapMs: number;
    fixedSteps: number;
    throttled: boolean;
    pending: number;
    memBuffers: number;
    memTextures: number;
    memTotal: number;
    memCount: number;
    memDetails: { kind: string; label: string; bytes: number }[];
    compileEntries: [string, number][];
    compileTotalMs: number;
    viewport: ViewportData | null;
}

interface Overlay {
    update(state: State, profile: ProfileImpl): void;
    destroy(): void;
}

interface OverlayOptions {
    position?: "top-left" | "top-right" | "bottom-left" | "bottom-right";
}

const BG = "rgba(14,13,12,0.88)";
const FG = "#cdc5bc";
const FG_BRIGHT = "#f0ece8";
const DIM = "#706860";
const ACCENT = "#d49560";
const WARN = "#e05050";
const BORDER = "rgba(255,255,255,0.06)";
const MB = 1024 * 1024;
const VALUE_WIDTH = "72px";
const ROW_HEIGHT = "15px";

function el(tag: string, styles?: Partial<CSSStyleDeclaration>): HTMLElement {
    const e = document.createElement(tag);
    if (styles) Object.assign(e.style, styles);
    return e;
}

function makeRow(
    parent: HTMLElement,
    labelColor = DIM,
    valueColor = FG,
): { label: HTMLElement; value: HTMLElement; row: HTMLElement } {
    const r = el("div", {
        display: "flex",
        alignItems: "baseline",
        height: ROW_HEIGHT,
        lineHeight: ROW_HEIGHT,
    });
    const label = el("span", {
        color: labelColor,
        flex: "1",
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
        minWidth: "0",
    });
    const value = el("span", {
        color: valueColor,
        width: VALUE_WIDTH,
        flexShrink: "0",
        textAlign: "right",
        fontVariantNumeric: "tabular-nums",
        whiteSpace: "nowrap",
    });
    r.append(label, value);
    parent.append(r);
    return { label, value, row: r };
}

interface RowPool {
    container: HTMLElement;
    rows: { label: HTMLElement; value: HTMLElement; row: HTMLElement }[];
    order: string[];
    // per-name sum of values seen in the current refresh window.
    accum: Map<string, number>;
    // number of frames sampled into accum since last flush.
    frames: number;
    // EMA across refresh windows of (window_sum / window_frames). This is
    // what gets displayed — gives a stable time-average for sparse signals
    // (fixed-tick passes that fire every Nth frame).
    smoothed: Map<string, number>;
    // refresh windows since each name last fired, used to evict dead entries.
    misses: Map<string, number>;
}

function createRowPool(parent: HTMLElement, initial: number): RowPool {
    const container = el("div");
    parent.append(container);
    const rows: RowPool["rows"] = [];
    for (let i = 0; i < initial; i++) {
        const r = makeRow(container, DIM, ACCENT);
        r.row.style.display = "none";
        rows.push(r);
    }
    return {
        container,
        rows,
        order: [],
        accum: new Map(),
        frames: 0,
        smoothed: new Map(),
        misses: new Map(),
    };
}

// Cross-window EMA — smooths flush-to-flush variation for sparse signals
// (e.g. a pass that fires 1× per refresh window). For high-frequency signals
// the per-window mean is already stable, so this layer is mostly cosmetic.
const WINDOW_ALPHA = 0.4;
// max consecutive refresh windows with no fire before the row is dropped.
const MAX_MISS_WINDOWS = 12;
// rounds to "0.00 ms" at this magnitude — safe to evict without visible flicker.
const EVICT_BELOW = 0.001;
// per-frame EMA for the FPS / frame-time hero readouts — both are scalars
// sampled fresh every frame, so the windowing layer isn't needed.
const HERO_ALPHA = 0.15;

// accumulate this frame's samples into the pool. Called every frame — takes the live
// Map directly (it's an Iterable<[name, ms]>) so the per-frame path allocates nothing.
function tickPool(pool: RowPool, entries: Iterable<[string, number]>): void {
    for (const [name, raw] of entries) {
        pool.accum.set(name, (pool.accum.get(name) ?? 0) + raw);
    }
    pool.frames++;
}

// fold the current window's accumulator into the cross-window EMA, evict
// long-dead rows, sort with hysteresis, and return sorted display rows + the
// sum of all live smoothed values (== visible row sum). Called per refresh.
function flushPool(
    pool: RowPool,
    fmt: (v: number) => string,
): { rows: [string, string][]; total: number } {
    const frames = pool.frames || 1;
    const seen = new Set<string>();
    for (const [name, sum] of pool.accum) {
        const mean = sum / frames;
        seen.add(name);
        const prev = pool.smoothed.get(name) ?? 0;
        pool.smoothed.set(name, prev + WINDOW_ALPHA * (mean - prev));
        pool.misses.set(name, 0);
    }
    for (const [name, value] of pool.smoothed) {
        if (seen.has(name)) continue;
        const next = value * (1 - WINDOW_ALPHA);
        const misses = (pool.misses.get(name) ?? 0) + 1;
        if (next < EVICT_BELOW || misses > MAX_MISS_WINDOWS) {
            pool.smoothed.delete(name);
            pool.misses.delete(name);
            const idx = pool.order.indexOf(name);
            if (idx !== -1) pool.order.splice(idx, 1);
        } else {
            pool.smoothed.set(name, next);
            pool.misses.set(name, misses);
        }
    }
    pool.accum.clear();
    pool.frames = 0;

    let total = 0;
    const live: [string, string, number][] = [];
    for (const [name, value] of pool.smoothed) {
        total += value;
        live.push([name, fmt(value), value]);
    }

    const sorted = live.slice().sort((a, b) => b[2] - a[2]);
    const rank = new Map<string, number>();
    for (let i = 0; i < sorted.length; i++) rank.set(sorted[i][0], i);

    const prev = pool.order;
    if (prev.length !== sorted.length || sorted.some(([name]) => !prev.includes(name))) {
        pool.order = sorted.map(([name]) => name);
    } else {
        const next = prev.slice();
        for (let i = 0; i < next.length; i++) {
            const desired = rank.get(next[i])!;
            if (Math.abs(desired - i) >= 2) {
                next.splice(i, 1);
                next.splice(desired, 0, prev[i]);
            }
        }
        pool.order = next;
    }

    const byName = new Map<string, [string, string]>();
    for (const [name, formatted] of sorted) byName.set(name, [name, formatted]);
    return { rows: pool.order.map((name) => byName.get(name)!), total };
}

function renderPool(pool: RowPool, entries: [string, string][]): void {
    while (pool.rows.length < entries.length) {
        const r = makeRow(pool.container, DIM, ACCENT);
        r.row.style.display = "none";
        pool.rows.push(r);
    }
    for (let i = 0; i < entries.length; i++) {
        const r = pool.rows[i];
        r.label.textContent = entries[i][0];
        r.value.textContent = entries[i][1];
        r.row.style.display = "flex";
    }
    for (let i = entries.length; i < pool.rows.length; i++) {
        pool.rows[i].row.style.display = "none";
    }
}

function section(parent: HTMLElement, title: string): { totalEl: HTMLElement; body: HTMLElement } {
    const wrapper = el("div", {
        marginTop: "2px",
        paddingTop: "4px",
        borderTop: `1px solid ${BORDER}`,
    });

    const headerRow = el("div", {
        display: "flex",
        alignItems: "baseline",
        height: ROW_HEIGHT,
        lineHeight: ROW_HEIGHT,
        cursor: "pointer",
        userSelect: "none",
    });
    const titleEl = el("span", {
        color: DIM,
        flex: "1",
        fontSize: "10px",
        letterSpacing: "0.05em",
        textTransform: "uppercase",
    });
    titleEl.textContent = "▸ " + title;
    const totalEl = el("span", {
        color: FG_BRIGHT,
        width: VALUE_WIDTH,
        flexShrink: "0",
        textAlign: "right",
        fontVariantNumeric: "tabular-nums",
        whiteSpace: "nowrap",
    });
    headerRow.append(titleEl, totalEl);

    const body = el("div", { display: "none", paddingLeft: "8px" });
    let open = false;
    headerRow.addEventListener("click", () => {
        open = !open;
        body.style.display = open ? "block" : "none";
        titleEl.textContent = (open ? "▾ " : "▸ ") + title;
    });

    wrapper.append(headerRow, body);
    parent.append(wrapper);
    return { totalEl, body };
}

function positionStyles(
    pos: "top-left" | "top-right" | "bottom-left" | "bottom-right",
): Partial<CSSStyleDeclaration> {
    const s: Partial<CSSStyleDeclaration> = { margin: "6px" };
    if (pos.includes("top")) s.top = "0";
    else s.bottom = "0";
    if (pos.includes("left")) s.left = "0";
    else s.right = "0";
    return s;
}

function formatBytes(bytes: number): string {
    if (bytes >= MB) return (bytes / MB).toFixed(1) + " MB";
    if (bytes >= 1024) return (bytes / 1024).toFixed(0) + " KB";
    return bytes + " B";
}

const msFmt = (v: number): string => v.toFixed(2) + " ms";

function pad(v: string, len: number): string {
    return v.length >= len ? v : " ".repeat(len - v.length) + v;
}

function createOverlay(opts?: OverlayOptions): Overlay {
    const pos = opts?.position ?? "top-left";
    // the stats HUD lives in the engine's sandboxed overlay (canvas-bounded, can't spill into an
    // embedding host like the editor viewport), the same surface `config.ui` hands an app
    const parent = mountOverlay(document.querySelector("canvas"));

    const root = el("div", {
        position: "absolute",
        zIndex: "10000",
        pointerEvents: "auto",
        background: BG,
        color: FG,
        fontFamily: "'JetBrains Mono', ui-monospace, 'Cascadia Code', 'Fira Code', monospace",
        fontSize: "10px",
        lineHeight: ROW_HEIGHT,
        padding: "8px 12px",
        minWidth: "220px",
        maxHeight: "90vh",
        overflowY: "auto",
        scrollbarGutter: "stable",
        borderRadius: "4px",
        border: `1px solid ${BORDER}`,
        boxShadow: "0 2px 12px rgba(0,0,0,0.4)",
        ...positionStyles(pos),
    });
    root.setAttribute("data-shallot-profile", "");

    const style = document.createElement("style");
    style.textContent = [
        "[data-shallot-profile]::-webkit-scrollbar { width: 6px }",
        "[data-shallot-profile]::-webkit-scrollbar-track { background: transparent }",
        "[data-shallot-profile]::-webkit-scrollbar-thumb { background: transparent; border-radius: 4px }",
        `[data-shallot-profile]:hover::-webkit-scrollbar-thumb { background: ${BORDER} }`,
    ].join("\n");
    root.append(style);

    const hero = el("div", {
        display: "flex",
        alignItems: "baseline",
        gap: "8px",
        paddingBottom: "4px",
        borderBottom: `1px solid ${BORDER}`,
        marginBottom: "2px",
    });
    const fpsEl = el("span", {
        color: FG_BRIGHT,
        fontSize: "14px",
        fontWeight: "600",
        fontVariantNumeric: "tabular-nums",
        letterSpacing: "-0.02em",
    });
    const ftEl = el("span", {
        color: DIM,
        fontSize: "10px",
        fontVariantNumeric: "tabular-nums",
    });
    const warnEl = el("span", {
        color: WARN,
        fontSize: "9px",
        marginLeft: "auto",
    });
    hero.append(fpsEl, ftEl, warnEl);
    root.append(hero);

    const frameGroup = el("div", { paddingTop: "2px" });
    const fenceRow = makeRow(frameGroup);
    // GPU-bound signal, EMA-smoothed. Per-frame fence is a pipelined residual (GPU frame time minus the
    // CPU/GPU overlap, beating against vsync/rAF pacing), so the raw value swings 0→budget frame to frame —
    // the trend is what reads, not the instantaneous sample. p95 (the worst-frame stall) lives in the bench
    fenceRow.label.textContent = "fence wait";
    const gapRow = makeRow(frameGroup);
    gapRow.label.textContent = "gap (vsync/throttle)";
    const fixedRow = makeRow(frameGroup);
    fixedRow.label.textContent = "fixed steps";
    const pendingRow = makeRow(frameGroup);
    pendingRow.label.textContent = "pending";
    root.append(frameGroup);

    const view = section(root, "viewport");
    const canvasRow = makeRow(view.body, DIM, ACCENT);
    canvasRow.label.textContent = "canvas";
    const cssRow = makeRow(view.body, DIM, ACCENT);
    cssRow.label.textContent = "css";
    const dprRow = makeRow(view.body, DIM, ACCENT);
    dprRow.label.textContent = "dpr";
    const fsRow = makeRow(view.body, DIM, ACCENT);
    fsRow.label.textContent = "fullscreen";

    const gpu = section(root, "gpu");
    const gpuPool = createRowPool(gpu.body, 16);
    // the untimed GPU cost the profiler can predict: Dawn's injected indirect-draw validation
    // (#drawIndexedIndirect × INDIRECT_FLOOR_US, gpu.md). Per-pass below + a Σ on the caption. It lives in
    // the gpu section because it's a GPU cost — NOT under fence wait, which is a pipelined residual it
    // doesn't sum into (fence can read below the floor when the frame isn't GPU-bound)
    const indirectCaption = el("div", {
        display: "flex",
        alignItems: "baseline",
        height: ROW_HEIGHT,
        lineHeight: ROW_HEIGHT,
        marginTop: "2px",
    });
    const indirectLabel = el("span", {
        color: DIM,
        flex: "1",
        fontSize: "9px",
        letterSpacing: "0.05em",
        textTransform: "uppercase",
    });
    indirectLabel.textContent = "indirect floor (µs)";
    const indirectTotalEl = el("span", {
        color: DIM,
        width: VALUE_WIDTH,
        flexShrink: "0",
        textAlign: "right",
        fontVariantNumeric: "tabular-nums",
    });
    indirectCaption.append(indirectLabel, indirectTotalEl);
    gpu.body.append(indirectCaption);
    const indirectPool = createRowPool(gpu.body, 4);

    const cpu = section(root, "cpu");
    const cpuPool = createRowPool(cpu.body, 16);

    const mem = section(root, "memory");
    const memBufRow = makeRow(mem.body, DIM, ACCENT);
    memBufRow.label.textContent = "buffers";
    const memTexRow = makeRow(mem.body, DIM, ACCENT);
    memTexRow.label.textContent = "textures";
    const memPool = createRowPool(mem.body, 16);

    const startup = section(root, "startup");
    const startupPool = createRowPool(startup.body, 16);
    let startupFrozen = false;

    parent.append(root);

    let emaFps = -1;
    let emaFt = -1;
    let emaFence = -1;
    let fenceSum = 0;
    let fenceCount = 0;
    let lastDrainCount = -1;
    let lastRender = 0;

    return {
        update(state: State, profile: ProfileImpl) {
            // per-frame: tick the pools straight from the live Maps (no allocation, no layout).
            // The heavy snapshot (getBoundingClientRect, map spreads, mem sort) is deferred to
            // the throttled render below, so the profiler's own per-frame CPU cost stays minimal
            tickPool(cpuPool, profile.cpu);
            // indirect counts are synchronous per-frame data (CPU-counted at draw time, no readback), so
            // tick every frame straight from the live map like cpu — not gated on the GPU drain
            tickPool(indirectPool, profile.indirect);
            if (profile.drainCount !== lastDrainCount) {
                lastDrainCount = profile.drainCount;
                tickPool(gpuPool, profile.gpu);
            }
            // fence is sampled every frame and averaged over the render window (then EMA'd) — a single
            // throttled sample of the swinging residual is noise; the windowed mean is the signal
            fenceSum += profile.fenceWaitMs;
            fenceCount++;

            const now = performance.now();
            if (now - lastRender < 250) return;
            lastRender = now;

            const data = collectStats(state, profile);
            emaFps = emaFps < 0 ? data.fps : emaFps + HERO_ALPHA * (data.fps - emaFps);
            emaFt = emaFt < 0 ? data.frameTime : emaFt + HERO_ALPHA * (data.frameTime - emaFt);
            fpsEl.textContent = pad(emaFps > 0 ? emaFps.toFixed(0) : "--", 3) + " fps";
            ftEl.textContent = (emaFt > 0 ? emaFt.toFixed(1) : "--") + " ms";
            warnEl.textContent = data.throttled ? "throttled" : "";

            const winFence = fenceCount > 0 ? fenceSum / fenceCount : data.fenceWaitMs;
            fenceSum = 0;
            fenceCount = 0;
            emaFence = emaFence < 0 ? winFence : emaFence + HERO_ALPHA * (winFence - emaFence);
            fenceRow.value.textContent = emaFence.toFixed(2) + " ms";
            gapRow.value.textContent = data.gapMs.toFixed(2) + " ms";
            fixedRow.value.textContent = String(data.fixedSteps);
            pendingRow.value.textContent = String(data.pending);

            if (data.viewport) {
                const vp = data.viewport;
                const mp = (vp.canvasWidth * vp.canvasHeight) / 1e6;
                view.totalEl.textContent = mp.toFixed(2) + " MP";
                canvasRow.value.textContent = `${vp.canvasWidth}×${vp.canvasHeight}`;
                cssRow.value.textContent = `${vp.cssWidth}×${vp.cssHeight}`;
                dprRow.value.textContent = vp.dpr.toFixed(2);
                fsRow.value.textContent = vp.fullscreen ? "yes" : "no";
            }

            const gpuFlush = flushPool(gpuPool, msFmt);
            gpu.totalEl.textContent = gpuFlush.total.toFixed(2) + " ms";
            renderPool(gpuPool, gpuFlush.rows);

            const indirectFlush = flushPool(indirectPool, (v) =>
                (v * INDIRECT_FLOOR_US).toFixed(0),
            );
            indirectCaption.style.display = indirectFlush.rows.length > 0 ? "flex" : "none";
            indirectTotalEl.textContent =
                (indirectFlush.total * INDIRECT_FLOOR_US).toFixed(0) + " µs";
            renderPool(indirectPool, indirectFlush.rows);

            const cpuFlush = flushPool(cpuPool, msFmt);
            cpu.totalEl.textContent = cpuFlush.total.toFixed(1) + " ms";
            renderPool(cpuPool, cpuFlush.rows);

            mem.totalEl.textContent =
                data.memTotal.toFixed(1) +
                " MB" +
                (data.memCount > 0 ? " (" + data.memCount + ")" : "");
            memBufRow.value.textContent = data.memBuffers.toFixed(1) + " MB";
            memTexRow.value.textContent = data.memTextures.toFixed(1) + " MB";
            renderPool(
                memPool,
                data.memDetails.map((a) => [
                    (a.kind === "buffer" ? "column " : "tex ") + (a.label || "(unlabeled)"),
                    formatBytes(a.bytes),
                ]),
            );

            if (!startupFrozen && data.compileTotalMs > 0) {
                startup.totalEl.textContent = data.compileTotalMs.toFixed(0) + " ms";
                renderPool(
                    startupPool,
                    data.compileEntries
                        .sort((a, b) => b[1] - a[1])
                        .map(([label, ms]) => [label || "(unnamed)", ms.toFixed(1) + " ms"]),
                );
                startupFrozen = true;
            }
        },
        destroy() {
            parent.remove(); // removes the sandboxed host (root lives inside it)
        },
    };
}

function collectViewport(): ViewportData | null {
    if (typeof document === "undefined") return null;
    const canvas = document.querySelector("canvas");
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    return {
        canvasWidth: canvas.width,
        canvasHeight: canvas.height,
        cssWidth: Math.round(rect.width),
        cssHeight: Math.round(rect.height),
        dpr: window.devicePixelRatio || 1,
        fullscreen: !!document.fullscreenElement,
    };
}

function collectStats(s: State, profile: ProfileImpl): OverlayData {
    const t = s.time;
    const rawDt = t.rawDeltaTime;
    const fenceWaitMs = profile.fenceWaitMs;
    const rawMs = rawDt * 1000;

    let cpuTotal = 0;
    for (const ms of profile.cpu.values()) cpuTotal += ms;

    return {
        fps: rawDt > 0 ? 1 / rawDt : 0,
        frameTime: rawMs,
        fenceWaitMs,
        gapMs: Math.max(0, rawMs - cpuTotal - fenceWaitMs),
        fixedSteps: t.fixedSteps,
        throttled: t.throttled,
        pending: Compute?.pending?.() ?? 0,
        memBuffers: profile.bufferBytes / MB,
        memTextures: profile.textureBytes / MB,
        memTotal: (profile.bufferBytes + profile.textureBytes) / MB,
        memCount: profile.buffers.size + profile.textures.size,
        memDetails: [...profile.sizes.values()].sort((a, b) => b.bytes - a.bytes),
        compileEntries: [...profile.compile.entries()],
        compileTotalMs: profile.compileMs,
        viewport: collectViewport(),
    };
}

// Profile singleton lives for the module's lifetime. The plugin attaches the
// GPU device on init and tears down the DOM overlay + hook wiring on dispose;
// GPU resources persist with the device.
const _profile = new ProfileImpl();
export const Profile: Profile = _profile;
let _overlay: Overlay | null = null;
let _benchmarkReady = false;
// the overlay is a convenience HUD, off by default and toggled with F3 (owned here, not per-consumer).
// it lives inside the canvas's container so it sits within the view, not over the whole window — for a
// fullscreen example that reads the same; for the editor it stays inside the viewport. persists across
// rebuilds (module-scoped), so an editor scene edit doesn't re-hide it.
let _visible = false;
let _keyHandler: ((e: KeyboardEvent) => void) | null = null;

// runs FIRST in setup group, before any of the new frame's GPU work: drain every ring slot whose async
// map has resolved (each carries one past frame's timestamps), resolve the just-completed prior frame's
// queries (every submit settled — coherent timeline; see resolve()), then zero the per-frame query
// counter for the new frame. resolve reads `_nextSlot` (the prior frame's pass count) so it must run
// before reset zeroes it. The map is kicked here, so draining trails by 1–2 frames.
const ProfileFrameBeginSystem: System = {
    group: "setup",
    annotations: { mode: "always" },
    first: true,
    update() {
        _profile.drain();
        const compute = Compute;
        if (compute) _profile.resolve(compute.device);
        _profile.reset();
    },
};

// runs in the draw group's "last" bucket: signals benchmark readiness and refreshes the overlay from the
// drained per-frame state. Resolve lives at the frame's start, not here — capturing queries at the tail
// races the render submit (a `last: true` tie with EndFrameSystem, which the render contract warns off).
const ProfileRenderSystem: System = {
    group: "draw",
    annotations: { mode: "always" },
    last: true,
    update(state: State) {
        _benchmarkReady = true;
        if (typeof document === "undefined" || !_visible) return;
        if (!_overlay) _overlay = createOverlay();
        _overlay.update(state, _profile);
    },
};

/**
 * performance profiler: an F3-toggled stats overlay (FPS, per-pass GPU/CPU timings, memory, shader
 * compile) plus the {@link Profile} singleton and the `window.__benchmark` measurement API. Off by
 * default — add it and press F3 to show the overlay; the data is on `Profile` whether it's shown or not.
 * Register it first so its `createBuffer` / pipeline patches catch every allocation.
 * @example
 * const config = { plugins: [ProfilePlugin] };
 */
export const ProfilePlugin: Plugin = {
    name: "Profile",
    systems: [ProfileFrameBeginSystem, ProfileRenderSystem],
    dependencies: [],

    initialize(state: State) {
        const compute = Compute;
        if (!compute) return;

        _profile.attach(compute.device);

        compute.span = (name) => _profile.span(name);
        compute.indirect = (name, count) => _profile.recordIndirect(name, count);
        state.recordSink = (name, ms) => _profile.record(name, ms);
        state.fenceWaitSink = (ms) => {
            _profile.fenceWaitMs = ms;
        };

        if (typeof window !== "undefined") {
            _benchmarkReady = false;
            const measure = createMeasure(state, _profile);
            window.__benchmark = {
                get ready() {
                    return _benchmarkReady;
                },
                measure,
            };

            if (_keyHandler) window.removeEventListener("keydown", _keyHandler);
            _keyHandler = (e: KeyboardEvent) => {
                if (e.key !== "F3") return;
                e.preventDefault();
                _visible = !_visible;
                if (!_visible && _overlay) {
                    _overlay.destroy();
                    _overlay = null;
                }
            };
            window.addEventListener("keydown", _keyHandler);
        }
    },

    dispose(state: State) {
        const compute = Compute;
        if (compute) {
            compute.span = undefined;
            compute.indirect = undefined;
        }
        state.recordSink = undefined;
        state.fenceWaitSink = undefined;
        _overlay?.destroy();
        _overlay = null;
        if (typeof window !== "undefined") {
            if (_keyHandler) {
                window.removeEventListener("keydown", _keyHandler);
                _keyHandler = null;
            }
            delete window.__benchmark;
        }
        _benchmarkReady = false;
    },
};
