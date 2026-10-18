import type { Plugin, System, World } from "../../engine";
import { Diagnostic, DiagnosticsStore, mountOverlay, sampleDiagnosticTime } from "../../engine";
import { cpuTotal } from "./cpu";
import { reorderRows } from "./reorder";

const INDIRECT_FLOOR_US = 1;

// Bevy joins span names as path components. WebGPU labels need not be valid components:
// escape separators and percent signs, reserving a lone percent sign for the empty label.
function encodeLabel(label: string): string {
    return label.replaceAll("%", "%25").replaceAll("/", "%2F") || "%";
}

function decodeLabel(label: string): string {
    if (label === "%") return "(unlabeled)";
    return label.replace(/%25|%2F/g, (code) => (code === "%25" ? "%" : "/"));
}

interface LabelPaths {
    cpu: string;
    gpu: string;
    indirect: string;
    compile: string;
}

interface ResourceAlloc {
    destroy: () => void;
}

// One timestamp result plus the pass names and count captured at its request. Staging belongs to the
// world's readback pool; this metadata lets the shared `_passes` scratch be reused next frame.
interface ReadSlot {
    bytes: ArrayBuffer;
    passes: string[];
    count: number;
}

// Bound the number of timestamp requests in flight. Metadata returns after the result is drained;
// the world's pool owns mapping and staging reuse. Saturation skips a profiling sample, not gameplay.
const READ_RING = 4;

// GPU timings are optional; CPU timings and resource tracking work without them.
const TIMESTAMP: readonly GPUFeatureName[] = ["timestamp-query"];
const recorder = {
    create: () => new DiagnosticsRecorder(),
};

class DiagnosticsRecorder {
    store!: DiagnosticsStore;
    gpuTiming: "available" | "requires timestamp-query" = "requires timestamp-query";
    // Per-frame CPU-counted indirect draws for the overlay's validation-floor estimate.
    readonly indirect = new Map<string, number>();
    readonly sizes = new Map<object, ResourceAlloc>();
    readonly measurement = new Float64Array(2);

    private readonly _paths = new Map<string, LabelPaths>();
    private _allocationId = 0;
    private _compileEarliest = Infinity;
    private _compileLatest = -Infinity;

    private _querySet: GPUQuerySet | null = null;
    private _resolveBuffer: GPUBuffer | null = null;
    private _capacity = 0;
    private _nextSlot = 0;
    private readonly _passes: string[] = [];
    private readonly _slotCache: GPUComputePassTimestampWrites[] = [];
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

    private paths(label: string): LabelPaths {
        return this._paths.get(label) ?? this.createPaths(label);
    }

    private createPaths(label: string): LabelPaths {
        const component = encodeLabel(label);
        const paths = {
            cpu: `render/${component}/elapsed_cpu`,
            gpu: `render/${component}/elapsed_gpu`,
            indirect: `render/${component}/indirect`,
            compile: `compile/${component}/elapsed`,
        };
        this._paths.set(label, paths);
        return paths;
    }

    attach(device: GPUDevice, capacity = 2048): void {
        if (this._querySet) return;
        this.gpuTiming = device.features.has("timestamp-query")
            ? "available"
            : "requires timestamp-query";
        this.store.record("render/timestamp_query", this.gpuTiming === "available" ? 1 : 0);
        for (const path of ["fps", "frame_time", "fixed_steps", "pending", "throttled"]) {
            if (!this.store.get(path))
                this.store.add(new Diagnostic(path, { suffix: path === "frame_time" ? "ms" : "" }));
        }
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
        const path = `memory/allocations/${kind}/${encodeLabel(label)}#${this._allocationId++}`;
        const allocation: ResourceAlloc = { destroy: origDestroy };
        this.store.record(path, bytes, undefined, "B");
        const totalPath = `memory/${totals}`;
        this.store.record(
            totalPath,
            (this.store.get(totalPath)?.value() ?? 0) + bytes,
            undefined,
            "B",
        );
        this.sizes.set(obj, allocation);
        (obj as { destroy: () => void }).destroy = () => {
            if (this.sizes.get(obj) !== allocation) return;
            this.store.record(
                totalPath,
                this.store.get(totalPath)!.value()! - bytes,
                undefined,
                "B",
            );
            this.sizes.delete(obj);
            this.store.remove(path);
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
        for (const diagnostic of this.store.iter()) {
            if (diagnostic.path.path.startsWith("memory/allocations/"))
                this.store.remove(diagnostic.path);
        }
        this.indirect.clear();
        this._paths.clear();
        this._slotCache.length = 0;
        this._passes.length = 0;
        this._capacity = 0;
        this.store.record("memory/bufferBytes", 0, undefined, "B");
        this.store.record("memory/textureBytes", 0, undefined, "B");
    }

    record(name: string, measurement: Readonly<Float64Array>): void {
        const path = this.paths(name).cpu;
        const diagnostic = this.store.get(path) ?? this.registerCpu(path);
        diagnostic.addMeasurementBuffer(measurement);
    }

    private registerCpu(path: string): Diagnostic {
        const diagnostic = new Diagnostic(path, { suffix: "ms" });
        this.store.add(diagnostic);
        return diagnostic;
    }

    recordIndirect(name: string, count: number): void {
        this.indirect.set(name, (this.indirect.get(name) ?? 0) + count);
        this.store.record(this.paths(name).indirect, this.indirect.get(name)!);
    }

    reset(): void {
        if (this.indirect.size) this.indirect.clear();
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
            const time = performance.now() / 1000;
            for (const [name, ms] of fired) {
                this.store.record(this.paths(name).gpu, ms, time, "ms");
            }
        }
    }

    /** @internal — also the `world.gpu.precompiled` sink for typed pipelines. */
    recordCompile(label: string, start: number, end: number): void {
        this.store.record(this.paths(label).compile, end - start, end / 1000, "ms");
        if (start < this._compileEarliest) this._compileEarliest = start;
        if (end > this._compileLatest) this._compileLatest = end;
        this.store.record(
            "compile/elapsed",
            this._compileLatest - this._compileEarliest,
            end / 1000,
            "ms",
        );
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
    update(store: DiagnosticsStore): void;
    destroy(): void;
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
    values: Map<string, number>;
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
        values: new Map(),
    };
}

function tickPool(pool: RowPool, entries: Iterable<[string, number]>): void {
    pool.values.clear();
    for (const [name, value] of entries) pool.values.set(name, value);
}

function flushPool(
    pool: RowPool,
    fmt: (v: number) => string,
    cpu = false,
): { rows: [string, string][]; total: number } {
    let total = cpu ? cpuTotal(pool.values) : 0;
    const live: [string, string, number][] = [];
    for (const [name, value] of pool.values) {
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

function formatBytes(bytes: number): string {
    if (bytes >= MB) return (bytes / MB).toFixed(1) + " MB";
    if (bytes >= 1024) return (bytes / 1024).toFixed(0) + " KB";
    return bytes + " B";
}

const msFmt = (v: number): string => v.toFixed(2) + " ms";

function pad(v: string, len: number): string {
    return v.length >= len ? v : " ".repeat(len - v.length) + v;
}

function createOverlay(): Overlay {
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
        top: "0",
        left: "0",
        margin: "6px",
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

    let lastRender = -Infinity;

    return {
        update(store: DiagnosticsStore) {
            const now = performance.now();
            if (now - lastRender < 250) return;
            lastRender = now;
            tickPool(cpuPool, diagnosticEntries(store, "render/", "/elapsed_cpu"));
            tickPool(gpuPool, diagnosticEntries(store, "render/", "/elapsed_gpu"));
            tickPool(indirectPool, diagnosticEntries(store, "render/", "/indirect"));
            const gpuTiming =
                store.get("render/timestamp_query")?.value() === 1
                    ? "available"
                    : "requires timestamp-query";
            const data = collectStats(store);
            fpsEl.textContent = pad(data.fps > 0 ? data.fps.toFixed(0) : "--", 3) + " fps";
            ftEl.textContent = (data.frameTime > 0 ? data.frameTime.toFixed(1) : "--") + " ms";
            warnEl.textContent = data.throttled ? "throttled" : "";

            fenceRow.value.textContent = data.fenceWaitMs.toFixed(2) + " ms";
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
            root.dataset.gpuTiming = gpuTiming;
            gpu.totalEl.textContent =
                gpuTiming === "available" ? gpuFlush.total.toFixed(2) + " ms" : gpuTiming;
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

function* diagnosticEntries(
    store: DiagnosticsStore,
    prefix: string,
    field: string,
): IterableIterator<[string, number]> {
    for (const diagnostic of store.iter()) {
        const path = diagnostic.path.path;
        if (path.startsWith(prefix) && path.endsWith(field)) {
            const value = diagnostic.smoothed();
            if (value !== undefined)
                yield [decodeLabel(path.slice(prefix.length, -field.length)), value];
        }
    }
}

function collectStats(store: DiagnosticsStore): OverlayData {
    const value = (path: string) => store.get(path)?.value() ?? 0;
    const rawMs = store.get("frame_time")?.smoothed() ?? 0;
    const fenceWaitMs = store.get("frame/fence_wait")?.smoothed() ?? 0;
    const memDetails: OverlayData["memDetails"] = [];
    for (const diagnostic of store.iter()) {
        const path = diagnostic.path.path;
        const bytes = diagnostic.value() ?? 0;
        if (!path.startsWith("memory/allocations/") || bytes === 0) continue;
        const parts = path.slice(19).split("/");
        memDetails.push({
            kind: parts[0],
            label: decodeLabel(parts[1].replace(/#\d+$/, "")),
            bytes,
        });
    }
    return {
        fps: store.get("fps")?.smoothed() ?? 0,
        frameTime: rawMs,
        fenceWaitMs,
        gapMs: Math.max(
            0,
            rawMs -
                cpuTotal(new Map(diagnosticEntries(store, "render/", "/elapsed_cpu"))) -
                fenceWaitMs,
        ),
        fixedSteps: value("fixed_steps"),
        throttled: value("throttled") !== 0,
        pending: value("pending"),
        memBuffers: value("memory/bufferBytes") / MB,
        memTextures: value("memory/textureBytes") / MB,
        memTotal: (value("memory/bufferBytes") + value("memory/textureBytes")) / MB,
        memCount: memDetails.length,
        memDetails: memDetails.sort((a, b) => b.bytes - a.bytes),
        compileEntries: [...diagnosticEntries(store, "compile/", "/elapsed")],
        compileTotalMs: value("compile/elapsed"),
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
 * it alongside. Requires {@link DiagnosticsOverlayPlugin}.
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
        const profiler = world.resource(recorder);
        profiler.drain();
        profiler.resolve(world);
        profiler.reset();
    },
};

const ProfileFrameStatsSystem: System = {
    group: "simulation",
    first: true,
    update(world) {
        const t = world.time;
        const store = world.resource(DiagnosticsStore);
        const measurement = world.resource(recorder).measurement;
        sampleDiagnosticTime(measurement);
        if (t.rawDeltaTime > 0) {
            measurement[1] = 1 / t.rawDeltaTime;
            store.get("fps")!.addMeasurementBuffer(measurement);
            measurement[1] = t.rawDeltaTime * 1000;
            store.get("frame_time")!.addMeasurementBuffer(measurement);
        }
        measurement[1] = t.fixedSteps;
        store.get("fixed_steps")!.addMeasurementBuffer(measurement);
        measurement[1] = world.gpu.pending();
        store.get("pending")!.addMeasurementBuffer(measurement);
        measurement[1] = Number(t.throttled);
        store.get("throttled")!.addMeasurementBuffer(measurement);
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
        _profileUi.overlay.update(world.resource(DiagnosticsStore));
    },
};

/**
 * Records CPU/GPU spans, memory and pipeline compilation into the World's DiagnosticsStore.
 * Add DiagnosticsOverlayPlugin separately for the F3 HUD.
 * Tracking starts at device acquisition, independent of plugin order. Composed systems
 * may record phase timings through the world's record sink; this plugin adds no simulation.
 */
export const ProfilePlugin: Plugin = {
    name: "Profile",
    gpu: { preferredFeatures: TIMESTAMP },
    systems: [ProfileFrameBeginSystem, ProfileFrameStatsSystem],
    dependencies: [],

    prepareGpu(world, compute) {
        const profiler = world.resource(recorder);
        profiler.store = world.resource(DiagnosticsStore);
        world.onDispose(() => profiler.dispose());
        profiler.attach(compute.device);

        compute.span = (name) => profiler.span(name);
        compute.indirect = (name, count) => profiler.recordIndirect(name, count);
        compute.precompiled = (label, start, end) => profiler.recordCompile(label, start, end);
        world.recordSink = (name, measurement) => profiler.record(name, measurement);
        world.fenceWaitSink = (ms) => {
            profiler.store.record("frame/fence_wait", ms, undefined, "ms");
        };
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

/** Optional F3 overlay reading the diagnostics collected by ProfilePlugin. */
export const DiagnosticsOverlayPlugin: Plugin = {
    name: "DiagnosticsOverlay",
    dependencies: [ProfilePlugin],
    systems: [ProfileRenderSystem],
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
};
