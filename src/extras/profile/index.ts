import type { Plugin, System, World } from "../../engine";
import { mountOverlay } from "../../engine";
import { cpuTotal } from "./cpu";
import { reorderRows } from "./reorder";

const INDIRECT_FLOOR_US = 1;

/**
 * live read-only view of the profiler: per-frame CPU and GPU pass timings, memory, and one-shot compile timings. Populated by {@link ProfilePlugin}; empty without it.
 * @expand
 */
export interface Profile {
    /** Whether this device grants the optional GPU timing capability. */
    readonly gpuTiming: "available" | "requires timestamp-query";
    /** Per-system CPU timings for the current frame, in milliseconds. A recorded name extending
     * another recorded name past a `/` is part of that timing; totals count only outermost timings.
     */
    readonly cpu: ReadonlyMap<string, number>;
    /** Per-pass GPU milliseconds summed within a drained frame, held across absent frames for display
     * until eviction. {@link gpuTime} accumulates durations; {@link gpuFires} counts frames containing a pass. */
    readonly gpu: ReadonlyMap<string, number>;
    /** cumulative GPU time per pass since attach, in milliseconds, summed over every occurrence. */
    readonly gpuTime: ReadonlyMap<string, number>;
    /** Number of drained frames containing each pass since attach; repeated occurrences in a frame count once. */
    readonly gpuFires: ReadonlyMap<string, number>;
    /** Per-pipeline compile durations from app startup, in milliseconds, keyed by descriptor label.
     *  A repeated named label overwrites. TypeGPU's synchronous constructors return before the driver
     *  finishes compiling, so their duration is near-zero; a `precompile` scope can measure the later
     *  completion under the pipeline's own label. */
    readonly compile: ReadonlyMap<string, number>;
    /** wall-clock span from the first pipeline build start to the last build end */
    readonly compileMs: number;
    /** ms spent awaiting the prior frame's GPU fence before this frame began */
    readonly fenceWaitMs: number;
    /** live GPU buffer bytes, summed over every buffer allocated through this World's device from acquisition and decremented on
     *  `destroy()`. */
    readonly bufferBytes: number;
    /** live GPU texture bytes, summed over every texture allocated through this World's device from acquisition and decremented on
     *  `destroy()`. */
    readonly textureBytes: number;
}

interface ResourceAlloc {
    label: string;
    bytes: number;
    kind: "buffer" | "texture";
    destroy: () => void;
}

// One timestamp result plus the pass names and count captured at its request. Staging belongs to the
// world's readback pool; this metadata lets the shared `_passes` scratch be reused next frame.
interface ReadSlot {
    bytes: ArrayBuffer;
    passes: string[];
    count: number;
}

// Hold absent passes' last durations to smooth uneven fixed-step scheduling.
// Eviction counts drained frames, not display frames; cumulative totals are unaffected.
const GPU_HOLD_DRAINS = 8;

// Bound the number of timestamp requests in flight. Metadata returns after the result is drained;
// the world's pool owns mapping and staging reuse. Saturation skips a profiling sample, not gameplay.
const READ_RING = 4;

// GPU timings are optional; CPU timings and resource tracking work without them.
const TIMESTAMP: readonly GPUFeatureName[] = ["timestamp-query"];
/** The profiler value owned by this World, populated by ProfilePlugin. */
export const Profile: import("../../engine").Resource<Profile> = {
    create: () => new ProfileImpl(),
};

class ProfileImpl implements Profile {
    gpuTiming: Profile["gpuTiming"] = "requires timestamp-query";
    readonly cpu = new Map<string, number>();
    readonly gpu = new Map<string, number>();
    readonly gpuTime = new Map<string, number>();
    readonly gpuFires = new Map<string, number>();
    // Per-frame CPU-counted indirect draws for the overlay's validation-floor estimate.
    readonly indirect = new Map<string, number>();
    readonly compile = new Map<string, number>();
    compileMs = 0;
    fenceWaitMs = 0;
    // The gpu map persists between drains; this prevents the overlay from counting a sample again.
    drainCount = 0;

    readonly sizes = new Map<object, ResourceAlloc>();
    bufferBytes = 0;
    textureBytes = 0;

    private _compileEarliest = Infinity;
    private _compileLatest = -Infinity;
    // Unlabelled compile records need unique map keys.
    private _unlabeledCompileCount = 0;

    private _querySet: GPUQuerySet | null = null;
    private _resolveBuffer: GPUBuffer | null = null;
    private _capacity = 0;
    private _nextSlot = 0;
    private readonly _passes: string[] = [];
    private readonly _slotCache: GPUComputePassTimestampWrites[] = [];
    // drains since each pass last fired, for the greedy hold + eviction (see GPU_HOLD_DRAINS)
    private readonly _gpuMiss = new Map<string, number>();
    private _readbackError: unknown;
    private _readbackFailed = false;
    // reused scratch: one drained frame's per-pass summed time (cleared per slot, no per-frame alloc)
    private readonly _fired = new Map<string, number>();
    // Bounded timestamp metadata: `_free` can accept a request; `_mapped` holds owned result bytes
    // waiting for drain. Neither list owns staging buffers.
    private readonly _free: ReadSlot[] = [];
    private readonly _mapped: ReadSlot[] = [];
    private readonly _restorePatches: (() => void)[] = [];
    private _disposed = false;

    attach(device: GPUDevice, capacity = 2048): void {
        if (this._querySet) return;
        this.gpuTiming = device.features.has("timestamp-query")
            ? "available"
            : "requires timestamp-query";
        const origCompute = device.createComputePipelineAsync.bind(device);
        device.createComputePipelineAsync = async (desc: GPUComputePipelineDescriptor) => {
            const start = performance.now();
            const pipeline = await origCompute(desc);
            this.recordCompile(desc.label ?? "", start, performance.now());
            return pipeline;
        };
        this._restorePatches.push(() => {
            device.createComputePipelineAsync = origCompute;
        });
        const origRender = device.createRenderPipelineAsync.bind(device);
        device.createRenderPipelineAsync = async (desc: GPURenderPipelineDescriptor) => {
            const start = performance.now();
            const pipeline = await origRender(desc);
            this.recordCompile(desc.label ?? "", start, performance.now());
            return pipeline;
        };
        this._restorePatches.push(() => {
            device.createRenderPipelineAsync = origRender;
        });

        // The synchronous constructor returns before the driver finishes compiling. A precompile scope
        // can record the later completion under the pipeline's own label.
        const origComputeSync = device.createComputePipeline.bind(device);
        device.createComputePipeline = (desc: GPUComputePipelineDescriptor) => {
            const start = performance.now();
            const pipeline = origComputeSync(desc);
            this.recordCompile(desc.label ?? "", start, performance.now());
            return pipeline;
        };
        this._restorePatches.push(() => {
            device.createComputePipeline = origComputeSync;
        });
        const origRenderSync = device.createRenderPipeline.bind(device);
        device.createRenderPipeline = (desc: GPURenderPipelineDescriptor) => {
            const start = performance.now();
            const pipeline = origRenderSync(desc);
            this.recordCompile(desc.label ?? "", start, performance.now());
            return pipeline;
        };
        this._restorePatches.push(() => {
            device.createRenderPipeline = origRenderSync;
        });

        const origCreateBuffer = device.createBuffer.bind(device);
        device.createBuffer = (desc) =>
            this.trackAlloc(origCreateBuffer(desc), desc.label ?? "", desc.size, "buffer");
        this._restorePatches.push(() => {
            device.createBuffer = origCreateBuffer;
        });

        const origCreateTexture = device.createTexture.bind(device);
        device.createTexture = (desc) =>
            this.trackAlloc(
                origCreateTexture(desc),
                desc.label ?? "",
                textureByteSize(desc),
                "texture",
            );
        this._restorePatches.push(() => {
            device.createTexture = origCreateTexture;
        });

        if (this.gpuTiming === "available") {
            this._capacity = capacity;
            this._querySet = device.createQuerySet({ type: "timestamp", count: capacity * 2 });
            const bytes = capacity * 2 * 8;
            this._resolveBuffer = device.createBuffer({
                label: "profile-resolve",
                size: bytes,
                usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
            });
            for (let i = 0; i < READ_RING; i++) {
                this._free.push({ bytes: new ArrayBuffer(0), passes: [], count: 0 });
            }
        }
    }

    private trackAlloc<T extends GPUBuffer | GPUTexture>(
        obj: T,
        label: string,
        bytes: number,
        kind: "buffer" | "texture",
    ): T {
        const totals = kind === "buffer" ? "bufferBytes" : "textureBytes";
        const origDestroy = obj.destroy.bind(obj);
        const allocation: ResourceAlloc = { label, bytes, kind, destroy: origDestroy };
        this[totals] += bytes;
        this.sizes.set(obj, allocation);
        (obj as { destroy: () => void }).destroy = () => {
            if (this.sizes.get(obj) !== allocation) return;
            this[totals] -= bytes;
            this.sizes.delete(obj);
            origDestroy();
        };
        return obj;
    }

    dispose(): void {
        this._disposed = true;
        this._readbackFailed = false;
        this._readbackError = undefined;
        for (const restore of this._restorePatches.reverse()) restore();
        this._restorePatches.length = 0;
        this._querySet?.destroy();
        this._querySet = null;
        this._resolveBuffer?.destroy();
        this._resolveBuffer = null;
        this._free.length = 0;
        this._mapped.length = 0;
        for (const [resource, allocation] of this.sizes) {
            (resource as { destroy: () => void }).destroy = allocation.destroy;
        }
        this.sizes.clear();
        this.cpu.clear();
        this.gpu.clear();
        this.gpuTime.clear();
        this.gpuFires.clear();
        this.indirect.clear();
        this.compile.clear();
        this._slotCache.length = 0;
        this._passes.length = 0;
        this._capacity = 0;
        this.bufferBytes = 0;
        this.textureBytes = 0;
    }

    record(name: string, ms: number): void {
        this.cpu.set(name, (this.cpu.get(name) ?? 0) + ms);
    }

    recordIndirect(name: string, count: number): void {
        this.indirect.set(name, (this.indirect.get(name) ?? 0) + count);
    }

    reset(): void {
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

    // Request prior-frame query timestamps at frame begin, before resetting query indices. Queue
    // order puts the resolve and copy after every prior submission and before the next frame's writes.
    // Capture pass names with the request so later frames cannot relabel an arriving result. The world
    // pool owns staging; READ_RING bounds only the profiling metadata in flight.
    resolve(world: World): void {
        const querySet = this._querySet;
        const resolveBuffer = this._resolveBuffer;
        if (!querySet || !resolveBuffer) return;
        const queryCount = this._nextSlot * 2;
        if (queryCount === 0) return;
        const slot = this._free.pop();
        if (!slot) return;
        slot.count = this._nextSlot;
        for (let i = 0; i < this._nextSlot; i++) slot.passes[i] = this._passes[i];
        world.readback
            .request(queryCount * 8, "profile-timestamps", (encoder, staging) => {
                encoder.resolveQuerySet(querySet, 0, queryCount, resolveBuffer, 0);
                encoder.copyBufferToBuffer(resolveBuffer, 0, staging, 0, queryCount * 8);
            })
            .then(
                (result) => {
                    if (this._disposed) return;
                    slot.bytes = result.bytes;
                    this._mapped.push(slot);
                },
                (error: unknown) => {
                    if (this._disposed) return;
                    this._free.push(slot);
                    this._readbackError = error;
                    this._readbackFailed = true;
                },
            );
    }

    drain(): void {
        if (this._readbackFailed)
            throw new Error("profile timestamp readback failed", { cause: this._readbackError });
        while (this._mapped.length > 0) {
            const slot = this._mapped.shift()!;
            const data = new BigUint64Array(slot.bytes);
            const fired = this._fired;
            fired.clear();
            for (let i = 0; i < slot.count; i++) {
                const name = slot.passes[i];
                fired.set(
                    name,
                    (fired.get(name) ?? 0) + Number(data[i * 2 + 1] - data[i * 2]) / 1e6,
                );
            }
            this._free.push(slot);
            // Count each pass once per drained frame, independently of the display hold.
            for (const [name, ms] of fired) {
                this.gpuTime.set(name, (this.gpuTime.get(name) ?? 0) + ms);
                this.gpuFires.set(name, (this.gpuFires.get(name) ?? 0) + 1);
            }
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

    /** @internal — also the `world.gpu.precompiled` sink for typed pipelines. */
    recordCompile(label: string, start: number, end: number): void {
        // Repeated named labels share an entry; unlabelled compile records get distinct keys.
        let key: string;
        if (label) {
            key = label;
        } else {
            const seen = ++this._unlabeledCompileCount;
            key = seen === 1 ? "(unlabeled)" : `(unlabeled)#${seen}`;
        }
        this.compile.set(key, end - start);
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
    update(world: World, profile: ProfileImpl): void;
    destroy(): void;
}

interface OverlayOptions {
    position?: "top-left" | "top-right" | "bottom-left" | "bottom-right";
}

const BG = "rgba(20,18,16,0.88)";
const FG = "#cdc5bc";
const FG_BRIGHT = "#f0e6d6";
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
    // EMA of window means, so sparse timings retain their time-average.
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

// Smooth refresh-window means so sparse signals do not flicker.
const WINDOW_ALPHA = 0.4;
// max consecutive refresh windows with no fire before the row is dropped.
const MAX_MISS_WINDOWS = 12;
// rounds to "0.00 ms" at this magnitude — safe to evict without visible flicker.
const EVICT_BELOW = 0.001;
const HERO_ALPHA = 0.15;

// Consume live maps without copying on the frame path.
function tickPool(pool: RowPool, entries: Iterable<[string, number]>): void {
    for (const [name, raw] of entries) {
        pool.accum.set(name, (pool.accum.get(name) ?? 0) + raw);
    }
    pool.frames++;
}

function flushPool(
    pool: RowPool,
    fmt: (v: number) => string,
    cpu = false,
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

    let total = cpu ? cpuTotal(pool.smoothed) : 0;
    const live: [string, string, number][] = [];
    for (const [name, value] of pool.smoothed) {
        if (!cpu) total += value;
        live.push([name, fmt(value), value]);
    }

    const sorted = live.slice().sort((a, b) => b[2] - a[2]);
    const rank = new Map<string, number>();
    for (let i = 0; i < sorted.length; i++) rank.set(sorted[i][0], i);

    const prev = pool.order;
    if (prev.length !== sorted.length || sorted.some(([name]) => !prev.includes(name))) {
        pool.order = sorted.map(([name]) => name);
    } else {
        pool.order = reorderRows(prev, rank);
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
    // Bound the profiler to the canvas so it cannot spill into an embedding host page.
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
    // Fence wait is a pipelined residual affected by CPU/GPU overlap and vsync/rAF pacing.
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
    // Dawn's indirect-draw validation estimate is GPU work, not additive to the pipelined fence wait.
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
    let lastRender = -Infinity;

    return {
        update(world: World, profile: ProfileImpl) {
            // Defer layout reads and allocating snapshots to the throttled render.
            tickPool(cpuPool, profile.cpu);
            // CPU-counted indirect draws are current-frame data, independent of GPU readback.
            tickPool(indirectPool, profile.indirect);
            if (profile.drainCount !== lastDrainCount) {
                lastDrainCount = profile.drainCount;
                tickPool(gpuPool, profile.gpu);
            }
            // Average fence wait across the window to avoid aliasing a single throttled sample.
            fenceSum += profile.fenceWaitMs;
            fenceCount++;

            const now = performance.now();
            if (now - lastRender < 250) return;
            lastRender = now;

            const data = collectStats(world, profile);
            emaFps = emaFps < 0 ? data.fps : emaFps + HERO_ALPHA * (data.fps - emaFps);
            emaFt = emaFt < 0 ? data.frameTime : emaFt + HERO_ALPHA * (data.frameTime - emaFt);
            fpsEl.textContent = pad(emaFps > 0 ? emaFps.toFixed(0) : "--", 3) + " fps";
            ftEl.textContent = (emaFt > 0 ? emaFt.toFixed(1) : "--") + " ms";
            warnEl.textContent = data.throttled ? "throttled" : "";

            const winFence = fenceSum / fenceCount;
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
            root.dataset.gpuTiming = profile.gpuTiming;
            gpu.totalEl.textContent =
                profile.gpuTiming === "available"
                    ? gpuFlush.total.toFixed(2) + " ms"
                    : profile.gpuTiming;
            renderPool(gpuPool, gpuFlush.rows);

            const indirectFlush = flushPool(indirectPool, (v) =>
                (v * INDIRECT_FLOOR_US).toFixed(0),
            );
            indirectCaption.style.display = indirectFlush.rows.length > 0 ? "flex" : "none";
            indirectTotalEl.textContent =
                (indirectFlush.total * INDIRECT_FLOOR_US).toFixed(0) + " µs";
            renderPool(indirectPool, indirectFlush.rows);

            const cpuFlush = flushPool(cpuPool, msFmt, true);
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
                        .map(([label, ms]) => [label, ms.toFixed(1) + " ms"]),
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

function collectStats(s: World, profile: ProfileImpl): OverlayData {
    const t = s.time;
    const rawDt = t.rawDeltaTime;
    const fenceWaitMs = profile.fenceWaitMs;
    const rawMs = rawDt * 1000;

    const total = cpuTotal(profile.cpu);

    return {
        fps: rawDt > 0 ? 1 / rawDt : 0,
        frameTime: rawMs,
        fenceWaitMs,
        gapMs: Math.max(0, rawMs - total - fenceWaitMs),
        fixedSteps: t.fixedSteps,
        throttled: t.throttled,
        pending: s.gpu.pending(),
        memBuffers: profile.bufferBytes / MB,
        memTextures: profile.textureBytes / MB,
        memTotal: (profile.bufferBytes + profile.textureBytes) / MB,
        memCount: profile.sizes.size,
        memDetails: [...profile.sizes.values()].sort((a, b) => b.bytes - a.bytes),
        compileEntries: [...profile.compile.entries()],
        compileTotalMs: profile.compileMs,
        viewport: collectViewport(),
    };
}

const profileUi = {
    create: () => ({
        overlay: null as Overlay | null,
        visible: false,
    }),
};

/**
 * show or hide the profiler overlay — the same HUD F3 toggles, driven from code. Off by default; call
 * `showProfiler(world)` in a plugin `warm` to surface the numbers on open without a keypress. F3 still toggles
 * it alongside. No-op without {@link ProfilePlugin} (nothing populates the overlay).
 */
export function showProfiler(world: World, show = true): void {
    const _profileUi = world.resource(profileUi);

    _profileUi.visible = show;
    if (!_profileUi.visible && _profileUi.overlay) {
        _profileUi.overlay.destroy();
        _profileUi.overlay = null;
    }
}

// Resolve prior-frame queries before reset clears their count, and before new GPU work overwrites them.
// Drain only completed readbacks; arrival may trail the sampled frame.
const ProfileFrameBeginSystem: System = {
    group: "setup",
    first: true,
    update(world: World) {
        const profiler = world.resource(Profile) as ProfileImpl;
        profiler.drain();
        profiler.resolve(world);
        profiler.reset();
    },
};

// runs in the draw group's "last" bucket and refreshes the overlay from the drained per-frame state.
// Resolve lives at the frame's start, not here — capturing queries at the tail races the render submit.
const ProfileRenderSystem: System = {
    group: "draw",
    last: true,
    update(world: World) {
        const _profileUi = world.resource(profileUi);

        if (typeof document === "undefined" || !_profileUi.visible) return;
        if (!_profileUi.overlay) _profileUi.overlay = createOverlay();
        _profileUi.overlay.update(world, world.resource(Profile) as ProfileImpl);
    },
};

/**
 * performance profiler: an F3-toggled stats overlay (FPS, per-pass GPU/CPU timings, memory, shader
 * compile) plus the World-owned {@link Profile} view. Off by default: add it and press F3 to show the
 * overlay; the data is on `Profile` whether it's shown or not.
 * Tracking starts at device acquisition, independent of plugin order. Composed systems
 * may record phase timings through the world's record sink; this plugin adds no simulation.
 */
export const ProfilePlugin: Plugin = {
    name: "Profile",
    gpu: { preferredFeatures: TIMESTAMP },
    systems: [ProfileFrameBeginSystem, ProfileRenderSystem],
    dependencies: [],

    prepareGpu(world, compute) {
        const profiler = world.resource(Profile) as ProfileImpl;
        world.onDispose(() => profiler.dispose());
        profiler.attach(compute.device);

        compute.span = (name) => profiler.span(name);
        compute.indirect = (name, count) => profiler.recordIndirect(name, count);
        compute.precompiled = (label, start, end) => profiler.recordCompile(label, start, end);
        world.recordSink = (name, ms) => profiler.record(name, ms);
        world.fenceWaitSink = (ms) => {
            profiler.fenceWaitMs = ms;
        };
    },

    initialize(world: World) {
        const ui = world.resource(profileUi);
        // World disposal must remove the DOM mount even without App.dispose().
        world.onDispose(() => {
            ui.overlay?.destroy();
            ui.overlay = null;
        });

        if (typeof window !== "undefined") {
            window.addEventListener(
                "keydown",
                (e: KeyboardEvent) => {
                    if (e.key !== "F3") return;
                    e.preventDefault();
                    showProfiler(world, !ui.visible);
                },
                { signal: world.signal },
            );
        }
    },

    dispose(world: World) {
        const compute = world.gpu;
        compute.span = undefined;
        compute.indirect = undefined;
        compute.precompiled = undefined;
        world.recordSink = undefined;
        world.fenceWaitSink = undefined;
    },
};
