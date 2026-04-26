import type { Plugin, State, System } from "../../engine";
import { Compute, ComputePlugin } from "../../standard/compute";
import { GpuProfile, GpuRegistryResource, trackDevice } from "../../standard/compute/core";
import { Views } from "../../standard/viewport";

interface ViewportData {
    canvasWidth: number;
    canvasHeight: number;
    cssWidth: number;
    cssHeight: number;
    dpr: number;
    fullscreen: boolean;
}

interface StatsData {
    fps: number;
    frameTime: number;
    fenceWaitMs: number;
    gapMs: number;
    fixedSteps: number;
    throttled: boolean;
    pending: number;
    gpuEntries: [string, number][];
    gpuTotal: number;
    cpuEntries: [string, number][];
    cpuTotal: number;
    memBuffers: number;
    memTextures: number;
    memTotal: number;
    memCount: number;
    memDetails: { kind: string; label: string; bytes: number }[];
    compileEntries: [string, number][];
    compileTotalMs: number;
    viewport: ViewportData | null;
}

interface StatsOverlay {
    update(data: StatsData): void;
    destroy(): void;
    element: HTMLDivElement;
}

interface StatsOverlayOptions {
    position?: "top-left" | "top-right" | "bottom-left" | "bottom-right";
    parent?: HTMLElement;
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
    ema?: Map<string, number>;
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
    return { container, rows, order: [] };
}

const EMA_ALPHA = 0.15;

function smoothEntries(
    pool: RowPool,
    entries: [string, number][],
    fmt: (v: number) => string,
): [string, string, number][] {
    if (!pool.ema) pool.ema = new Map();
    const ema = pool.ema;
    const seen = new Set<string>();
    const result: [string, string, number][] = [];
    for (const [name, raw] of entries) {
        seen.add(name);
        const prev = ema.get(name);
        const smoothed = prev !== undefined ? prev + EMA_ALPHA * (raw - prev) : raw;
        ema.set(name, smoothed);
        result.push([name, fmt(smoothed), smoothed]);
    }
    for (const key of ema.keys()) {
        if (!seen.has(key)) ema.delete(key);
    }
    return result;
}

function stableSort(pool: RowPool, entries: [string, string, number][]): [string, string][] {
    const sorted = [...entries].sort((a, b) => b[2] - a[2]);
    const newRank = new Map<string, number>();
    for (let i = 0; i < sorted.length; i++) newRank.set(sorted[i][0], i);

    const prev = pool.order;
    if (prev.length !== sorted.length || sorted.some(([name]) => !prev.includes(name))) {
        pool.order = sorted.map(([name]) => name);
    } else {
        const next: string[] = [...prev];
        for (let i = 0; i < next.length; i++) {
            const desired = newRank.get(next[i])!;
            if (Math.abs(desired - i) >= 2) {
                next.splice(i, 1);
                next.splice(desired, 0, prev[i]);
            }
        }
        pool.order = next;
    }

    const byName = new Map<string, [string, string]>();
    for (const [name, formatted] of sorted) byName.set(name, [name, formatted]);
    return pool.order.map((name) => byName.get(name)!);
}

function poolUpdate(pool: RowPool, entries: [string, string][]): void {
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

function sectionBlock(
    parent: HTMLElement,
    title: string,
): {
    titleEl: HTMLElement;
    totalEl: HTMLElement;
    body: HTMLElement;
    wrapper: HTMLElement;
} {
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
    titleEl.textContent = "\u25b8 " + title;
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
        titleEl.textContent = (open ? "\u25be " : "\u25b8 ") + title;
    });

    wrapper.append(headerRow, body);
    parent.append(wrapper);
    return { titleEl, totalEl, body, wrapper };
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

function pad(v: string, len: number): string {
    return v.length >= len ? v : " ".repeat(len - v.length) + v;
}

function createStatsOverlay(opts?: StatsOverlayOptions): StatsOverlay {
    const pos = opts?.position ?? "top-left";
    const parent = opts?.parent ?? document.body;

    const root = el("div", {
        position: "fixed",
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
    }) as HTMLDivElement;
    root.setAttribute("data-shallot-stats", "");

    const style = document.createElement("style");
    style.textContent = [
        "[data-shallot-stats]::-webkit-scrollbar { width: 6px }",
        "[data-shallot-stats]::-webkit-scrollbar-track { background: transparent }",
        "[data-shallot-stats]::-webkit-scrollbar-thumb { background: transparent; border-radius: 4px }",
        `[data-shallot-stats]:hover::-webkit-scrollbar-thumb { background: ${BORDER} }`,
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
    fenceRow.label.textContent = "fence wait";
    const gapRow = makeRow(frameGroup);
    gapRow.label.textContent = "gap (vsync/throttle)";
    const fixedRow = makeRow(frameGroup);
    fixedRow.label.textContent = "fixed steps";
    const pendingRow = makeRow(frameGroup);
    pendingRow.label.textContent = "pending";
    root.append(frameGroup);

    const view = sectionBlock(root, "viewport");
    const canvasRow = makeRow(view.body, DIM, ACCENT);
    canvasRow.label.textContent = "canvas";
    const cssRow = makeRow(view.body, DIM, ACCENT);
    cssRow.label.textContent = "css";
    const dprRow = makeRow(view.body, DIM, ACCENT);
    dprRow.label.textContent = "dpr";
    const fsRow = makeRow(view.body, DIM, ACCENT);
    fsRow.label.textContent = "fullscreen";

    const gpu = sectionBlock(root, "gpu");
    const gpuPool = createRowPool(gpu.body, 16);

    const cpu = sectionBlock(root, "cpu");
    const cpuPool = createRowPool(cpu.body, 16);

    const mem = sectionBlock(root, "memory");
    const memBufRow = makeRow(mem.body, DIM, ACCENT);
    memBufRow.label.textContent = "buffers";
    const memTexRow = makeRow(mem.body, DIM, ACCENT);
    memTexRow.label.textContent = "textures";
    const memPool = createRowPool(mem.body, 16);

    const startup = sectionBlock(root, "startup");
    const startupPool = createRowPool(startup.body, 16);
    let startupFrozen = false;

    parent.append(root);

    let emaFps = -1;
    let emaFt = -1;
    let emaGpuTotal = -1;
    let emaCpuTotal = -1;

    return {
        element: root,
        update(data: StatsData) {
            emaFps = emaFps < 0 ? data.fps : emaFps + EMA_ALPHA * (data.fps - emaFps);
            emaFt = emaFt < 0 ? data.frameTime : emaFt + EMA_ALPHA * (data.frameTime - emaFt);
            fpsEl.textContent = pad(emaFps > 0 ? emaFps.toFixed(0) : "--", 3) + " fps";
            ftEl.textContent = (emaFt > 0 ? emaFt.toFixed(1) : "--") + " ms";
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

            emaGpuTotal =
                emaGpuTotal < 0
                    ? data.gpuTotal
                    : emaGpuTotal + EMA_ALPHA * (data.gpuTotal - emaGpuTotal);
            gpu.totalEl.textContent = emaGpuTotal.toFixed(2) + " ms";
            poolUpdate(
                gpuPool,
                stableSort(
                    gpuPool,
                    smoothEntries(gpuPool, data.gpuEntries, (v) => v.toFixed(2) + " ms"),
                ),
            );

            emaCpuTotal =
                emaCpuTotal < 0
                    ? data.cpuTotal
                    : emaCpuTotal + EMA_ALPHA * (data.cpuTotal - emaCpuTotal);
            cpu.totalEl.textContent = emaCpuTotal.toFixed(1) + " ms";
            poolUpdate(
                cpuPool,
                stableSort(
                    cpuPool,
                    smoothEntries(cpuPool, data.cpuEntries, (v) => v.toFixed(2) + " ms"),
                ),
            );

            mem.totalEl.textContent =
                data.memTotal.toFixed(1) +
                " MB" +
                (data.memCount > 0 ? " (" + data.memCount + ")" : "");
            memBufRow.value.textContent = data.memBuffers.toFixed(1) + " MB";
            memTexRow.value.textContent = data.memTextures.toFixed(1) + " MB";
            poolUpdate(
                memPool,
                data.memDetails.map((a) => [
                    (a.kind === "buffer" ? "buf " : "tex ") + (a.label || "(unlabeled)"),
                    formatBytes(a.bytes),
                ]),
            );

            if (!startupFrozen && data.compileTotalMs > 0) {
                startup.totalEl.textContent = data.compileTotalMs.toFixed(0) + " ms";
                poolUpdate(
                    startupPool,
                    data.compileEntries
                        .sort((a, b) => b[1] - a[1])
                        .map(([label, ms]) => [label || "(unnamed)", ms.toFixed(1) + " ms"]),
                );
                startupFrozen = true;
            }
        },
        destroy() {
            root.remove();
        },
    };
}

function collectViewport(s: State): ViewportData | null {
    const views = Views.from(s);
    let canvas: HTMLCanvasElement | null = null;
    if (views) {
        for (const view of views.values()) {
            if (view.element) {
                canvas = view.element;
                break;
            }
        }
    }
    if (!canvas && typeof document !== "undefined") {
        canvas = document.querySelector("canvas");
    }
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

function collectStats(s: State): StatsData {
    const t = s.time;
    const rawDt = t.rawDeltaTime;

    const gpuAccum = new Map<string, number>();
    let gpuTotal = 0;
    const profiles = GpuProfile.from(s);
    if (profiles && profiles.length > 0) {
        for (const p of profiles) {
            for (const [name, ms] of p) {
                gpuTotal += ms;
                gpuAccum.set(name, (gpuAccum.get(name) ?? 0) + ms);
            }
        }
    }

    const cpuTimings = s.scheduler.cpu;
    let cpuTotal = 0;
    const cpuEntries: [string, number][] = [];
    for (const [name, ms] of cpuTimings) {
        cpuTotal += ms;
        cpuEntries.push([name, ms]);
    }

    const fenceWaitMs = t.fenceWaitMs;
    const rawMs = rawDt * 1000;
    const gapMs = Math.max(0, rawMs - cpuTotal - fenceWaitMs);

    const registry = GpuRegistryResource.from(s);
    if (registry && registry.compileSpans.length > 0) registry.finalizeCompile();
    return {
        fps: rawDt > 0 ? 1 / rawDt : 0,
        frameTime: rawMs,
        fenceWaitMs,
        gapMs,
        fixedSteps: t.fixedSteps,
        throttled: t.throttled,
        pending: Compute.from(s)?.pending ?? 0,
        gpuEntries: [...gpuAccum.entries()],
        gpuTotal,
        cpuEntries,
        cpuTotal,
        memBuffers: registry ? registry.bufferBytes / MB : 0,
        memTextures: registry ? registry.textureBytes / MB : 0,
        memTotal: registry ? registry.totalBytes() / MB : 0,
        memCount: registry ? registry.count() : 0,
        memDetails: registry
            ? [...registry.sizes.values()]
                  .sort((a, b) => b.bytes - a.bytes)
                  .map((a) => ({ kind: a.kind, label: a.label, bytes: a.bytes }))
            : [],
        compileEntries: registry
            ? registry.compileTimings.map((e) => [e.label, e.ms] as [string, number])
            : [],
        compileTotalMs: registry ? registry.compileTotalMs : 0,
        viewport: collectViewport(s),
    };
}

let _overlay: StatsOverlay | null = null;
let _lastUpdate = 0;

const StatsSystem: System = {
    group: "draw",
    update(s: State) {
        if (!_overlay) {
            _overlay = createStatsOverlay();
            s.onDispose(() => {
                _overlay?.destroy();
                _overlay = null;
            });
        }
        const now = performance.now();
        if (now - _lastUpdate < 250) return;
        _lastUpdate = now;
        _overlay.update(collectStats(s));
    },
};

export const StatsPlugin: Plugin = {
    name: "StatsPlugin",
    systems: [StatsSystem],
    dependencies: [ComputePlugin],

    initialize(state: State) {
        const compute = Compute.from(state);
        if (!compute) return;
        const registry = trackDevice(compute.device);
        state.setResource(GpuRegistryResource, registry);
    },
};
