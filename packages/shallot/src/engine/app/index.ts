import { type Component, State, type System } from "../ecs";
import { entries, fields, register, type Traits } from "../ecs/core";
import { Compute, now, Runtime, readFile, requestFrame, requestGPU } from "../runtime";
import { diagnose, load, parse } from "../scene";
import { preload } from "../scene/core";
import { coalesce, median } from "./coalesce";

/**
 * bundle of components, systems, and lifecycle hooks — the unit of behavior a project enables.
 * @expand
 */
export interface Plugin {
    /** unique name; the manifest enables the plugin by this name, and `swap` pairs reloads by it */
    readonly name: string;
    /** systems this plugin adds to the scheduler */
    readonly systems?: readonly System[];
    /** components this plugin registers, keyed by scene-attribute name */
    readonly components?: Record<string, Component>;
    /** per-component traits (requires/excludes/singleton/defaults), keyed like `components` */
    readonly traits?: Record<string, Traits>;
    /** other plugins that must load first; a missing one skips this plugin with a warning */
    readonly dependencies?: readonly Plugin[];
    /**
     * GPU features this plugin's shaders require beyond the engine's base floor. The active
     * plugins' features are unioned and requested at device acquisition; a device missing one
     * fails with {@link UnsupportedError} before any plugin loads.
     */
    readonly features?: readonly GPUFeatureName[];
    /**
     * GPU features this plugin runs faster with but does not require. Unioned across the active
     * plugins and requested at device acquisition only where the adapter has them — a device
     * missing one loads normally, and the plugin takes its fallback path (it reads
     * `device.features.has(...)` to pick the arm). e.g. the BVH builder prefers `subgroups` for
     * its bounds reduction + radix sort, falling back to an LDS arm on a device without it.
     */
    readonly preferredFeatures?: readonly GPUFeatureName[];
    /** registration-only setup, run before scene parse (no entities exist yet); idempotent, may report progress */
    readonly initialize?: (
        state: State,
        onProgress?: (progress: number) => void,
    ) => void | Promise<void>;
    /** post-scene GPU setup and derived spawns, run once entities exist; idempotent, may report progress */
    readonly warm?: (state: State, onProgress?: (progress: number) => void) => void | Promise<void>;
    /** teardown, run in reverse dependency order on `App.dispose` */
    readonly dispose?: (state: State) => void;
}

/**
 * a startup/error screen driven by the build's progress. the engine calls `show` before loading,
 * `update` across every lifecycle step, and `error` if the build throws. see the built-in
 * {@link shallotDark} family.
 * @expand
 */
export interface Loading {
    /** display the screen; return a cleanup called once the build finishes, or nothing to leave it up */
    show(): (() => void) | void;
    /** report build progress, `0`–`1` across initialize, scene load, and warm */
    update(progress: number): void;
    /** render the thrown value; the default screen branches on `UnsupportedError` */
    error?(error: unknown): void;
}

/**
 * the {@link build} / {@link run} configuration — plugins, scene, and startup behavior.
 * @expand
 */
export interface Config {
    /** plugins to load, unioned with the built-in defaults unless `defaults` is `false` */
    plugins: Plugin[];
    /** `.scene` file path(s), or an inline XML string (any value starting with `<`) */
    scene?: string | string[];
    /** startup screen; defaults to the one set by `setDefaultLoading` */
    loading?: Loading;
    /** `false` skips the built-in default plugins entirely */
    defaults?: boolean;
    /** specific default plugins to drop while keeping the rest */
    exclude?: Plugin[];
    /** hook run after registration, before any plugin `initialize` */
    setup?: (state: State) => void;
    /** mount app UI into the canvas-bounded overlay; return a cleanup. see `mountOverlay` */
    ui?: (container: HTMLElement, state: State) => () => void;
    /** externally-acquired GPU device; if omitted, the engine acquires one */
    device?: GPUDevice;
    /** entity capacity; fixed at app construction. defaults to 65536. */
    capacity?: number;
    /**
     * render device-pixel ratio for canvas views; fixed at app construction. `"auto"` (default)
     * clamps `devicePixelRatio` to `[1, 2]`. A number forces a fixed ratio (`1` = CSS resolution /
     * cheapest; below 1 = pixel-art downscale). See {@link pixelRatio}.
     */
    pixelRatio?: number | "auto";
    /** filter systems by `annotations.mode`; fixed at construction. omit to run every system. */
    mode?: "edit" | "play";
}

/** result of {@link build} / {@link run}. owns the plugin teardown order. */
export interface App {
    readonly state: State;
    /**
     * names of plugins skipped at build (missing dependency). The built reality {@link swap}
     * validates against — a skipped plugin never initialized, so swapping it would half-apply.
     */
    readonly skipped: readonly string[];
    dispose(): void;
}

// runaway backstop: the most frames the loop may run ahead of the GPU before skipping a step, so the CPU
// can't queue unboundedly past a saturated GPU. Sized well above a present-throttled pipeline's depth —
// `Compute.sync` counts frames by `onSubmittedWorkDone`, which is present-gated, so a 60Hz fullscreen
// throttle reads ~3 frames in flight with the GPU otherwise idle; a tighter cap would drop frames Chrome
// is ready to present (the fullscreen-throttle judder). Per-present pacing is the double-fire `coalesce`,
// not this; the bound engages only under sustained genuine GPU saturation.
const MAX_FRAMES_IN_FLIGHT = 6;

let _defaultPlugins: readonly Plugin[] = [];
let _defaultLoading: (() => Loading) | null = null;

/**
 * set the global default plugin set every {@link build} unions in (unless `defaults: false`). `standard`
 * calls this at import with `DEFAULT_PLUGINS`; override it to define your own zero-config baseline.
 */
export function setDefaultPlugins(plugins: readonly Plugin[]): void {
    _defaultPlugins = plugins;
}

/**
 * set the global default {@link Loading} screen every {@link build} uses when `config.loading` is omitted.
 * `standard` calls this at import with {@link shallotDark}.
 */
export function setDefaultLoading(factory: () => Loading): void {
    _defaultLoading = factory;
}

/**
 * build the app: collect plugins, acquire the GPU device, register, run `initialize`, load scenes, and
 * `warm` — returning a live {@link State} without starting a frame loop. drive `state.step(dt)` yourself,
 * or use {@link run} for the managed loop.
 * @example
 * const app = await build({ plugins: [MyPlugin], scene: "/scenes/demo.scene" });
 * app.state.step(1 / 60);
 */
export async function build(config: Config): Promise<App> {
    const state = new State({
        capacity: config.capacity,
        pixelRatio: config.pixelRatio,
        mode: config.mode,
    });
    const useDefaults = config.defaults !== false;
    const loading = config.loading ?? _defaultLoading?.();
    const cleanup = loading?.show();
    const initialized: Plugin[] = [];

    try {
        const excluded = new Set(config.exclude ?? []);
        const pluginSet = new Set<Plugin>();
        if (useDefaults) {
            for (const p of _defaultPlugins) if (!excluded.has(p)) pluginSet.add(p);
        }
        for (const p of config.plugins) pluginSet.add(p);

        const skipped = new Set<Plugin>();
        const edges: [Plugin, Plugin][] = [];
        for (const plugin of pluginSet) {
            for (const dep of plugin.dependencies ?? []) {
                if (pluginSet.has(dep)) continue;
                console.warn(
                    `Missing plugin dependency: ${plugin.name ?? "?"} requires ${dep.name ?? "?"}`,
                );
                skipped.add(plugin);
                break;
            }
            for (const dep of plugin.dependencies ?? []) {
                edges.push([dep, plugin]);
            }
        }
        const sorted = sortPlugins([...pluginSet], edges).filter((p) => !skipped.has(p));

        // acquire the device for exactly the loaded plugins' feature needs — the union of every
        // active plugin's required `features` and best-effort `preferredFeatures`. A scene without
        // physics (no BVH) requests neither, so it never asks for `subgroups`.
        const features = [...new Set(sorted.flatMap((p) => p.features ?? []))];
        const preferred = [...new Set(sorted.flatMap((p) => p.preferredFeatures ?? []))];
        await requestGPU(config.device, features, preferred);

        for (const plugin of sorted) {
            const components = plugin.components ?? {};
            const traits = plugin.traits ?? {};
            for (const [name, component] of Object.entries(components)) {
                register(name, component, traits[name]);
            }
            for (const name of Object.keys(traits)) {
                if (!components[name]) {
                    console.warn(
                        `Plugin "${plugin.name}": traits["${name}"] has no matching component`,
                    );
                }
            }
            for (const system of plugin.systems ?? []) {
                state.addSystem(system, plugin.name);
            }
        }

        // assign every registered component its membership bit now, so the GPU
        // membership mirror's generation count is fixed before any warm sizes it
        for (const { component } of entries()) state.membership.bit(component);

        const scenes = config.scene
            ? Array.isArray(config.scene)
                ? config.scene
                : [config.scene]
            : [];

        const total = sorted.length * 2 + scenes.length;

        config.setup?.(state);

        for (let i = 0; i < sorted.length; i++) {
            const onProgress = loading
                ? (progress: number) => loading.update((i + progress) / total)
                : undefined;
            await sorted[i].initialize?.(state, onProgress);
            initialized.push(sorted[i]);
            loading?.update((i + 1) / total);
        }

        for (let i = 0; i < scenes.length; i++) {
            const scene = scenes[i];
            const xml = scene.startsWith("<") ? scene : await readFile(scene);
            const nodes = parse(xml);
            for (const d of diagnose(nodes)) console.warn(`[shallot] ${d.message}`);
            // the pre-load resolve pass: a plugin whose assets the scene references by name (glTF)
            // imports them here, so every mesh name resolves when `load` applies the attrs
            await preload(nodes, state);
            load(nodes, state);
            loading?.update((sorted.length + i + 1) / total);
        }

        const warmable = sorted.filter((p) => p.warm);
        const warmBase = sorted.length + scenes.length;
        let warmDone = 0;
        await Promise.all(
            warmable.map(async (plugin) => {
                await plugin.warm!(state, (progress: number) => {
                    loading?.update((warmBase + warmDone + progress) / total);
                });
                warmDone++;
                loading?.update((warmBase + warmDone) / total);
            }),
        );

        if (cleanup) {
            loading?.update(1);
            await new Promise<void>((r) => requestFrame(() => r()));
            cleanup();
        }

        return {
            state,
            skipped: [...skipped].map((p) => p.name),
            dispose() {
                for (let i = sorted.length - 1; i >= 0; i--) {
                    sorted[i].dispose?.(state);
                }
                state.dispose();
            },
        };
    } catch (e) {
        // a failed build leaves nothing live: plugins that completed initialize dispose in
        // reverse, then the State, so a retry builds against clean module singletons. A dispose
        // throw here is reported, never allowed to mask the build error.
        for (let i = initialized.length - 1; i >= 0; i--) {
            try {
                initialized[i].dispose?.(state);
            } catch (err) {
                console.error(`Plugin "${initialized[i].name}" threw during cleanup:`, err);
            }
        }
        try {
            state.dispose();
        } catch (err) {
            console.error("State dispose threw during cleanup:", err);
        }
        loading?.error?.(e);
        throw e;
    }
}

function sortPlugins(nodes: Plugin[], edges: [Plugin, Plugin][]): Plugin[] {
    const adj = new Map<Plugin, Plugin[]>();
    const inDegree = new Map<Plugin, number>();
    for (const node of nodes) {
        adj.set(node, []);
        inDegree.set(node, 0);
    }
    for (const [from, to] of edges) {
        if (!adj.has(from) || !inDegree.has(to)) continue;
        adj.get(from)!.push(to);
        inDegree.set(to, inDegree.get(to)! + 1);
    }
    const sorted: Plugin[] = [];
    const queue: Plugin[] = [];
    for (const node of nodes) if (inDegree.get(node) === 0) queue.push(node);
    while (queue.length) {
        const node = queue.shift()!;
        sorted.push(node);
        for (const next of adj.get(node)!) {
            const d = inDegree.get(next)! - 1;
            inDegree.set(next, d);
            if (d === 0) queue.push(next);
        }
    }
    if (sorted.length !== nodes.length) {
        const remaining = nodes.filter((n) => (inDegree.get(n) ?? 0) > 0).map((n) => n.name);
        throw new Error(`Circular plugin dependency: ${remaining.join(", ")}`);
    }
    return sorted;
}

/**
 * create the sandboxed UI overlay over a canvas — the single DOM surface a shallot app's UI mounts
 * into ({@link Config.ui}). It fills the canvas's parent and is a true sandbox: `contain: layout
 * paint` makes it the containing block for absolute *and* fixed descendants and clips paint to its
 * box, so app UI is bounded to the canvas region and can never spill into an embedding host (an
 * editor viewport, a host page) — even a stray `position: fixed`. `pointer-events: none` lets input
 * reach the canvas; UI panels re-enable it. Returns the overlay; the caller removes it on dispose.
 */
export function mountOverlay(canvas: HTMLElement | null): HTMLDivElement {
    const parent = canvas?.parentElement ?? document.body;
    parent.style.position = "relative";
    const overlay = document.createElement("div");
    overlay.style.cssText =
        "position:absolute;inset:0;pointer-events:none;z-index:1;contain:layout paint;overflow:hidden";
    parent.appendChild(overlay);
    return overlay;
}

/**
 * build the app and start the `requestAnimationFrame` frame loop, mounting `config.ui` (web only). the
 * loop drives `state.step(dt)` each frame, GPU-fence backpressured so it never runs far ahead of the GPU.
 * @example
 * const app = await run({ plugins: [MyPlugin], scene: "/scenes/demo.scene" });
 * // later: app.dispose();
 */
export async function run(config: Config): Promise<App> {
    const app = await build(config);
    const state = app.state;
    let uiCleanup: (() => void) | undefined;
    let overlay: HTMLDivElement | undefined;
    if (config.ui && Runtime === "web") {
        overlay = mountOverlay(document.querySelector("canvas"));
        uiCleanup = config.ui(overlay, state);
    }

    let disposed = false;
    let lastTime = now();
    let pendingFenceWaitMs = 0;
    // recent raw-callback intervals + a reused sort scratch, feeding the double-fire coalescer's median
    let lastCallback = -1;
    const intervals: number[] = [];
    const scratch: number[] = [];

    function frame(timestamp?: number): void {
        if (disposed) return;
        // rAF clocks the loop and reschedules first, before any GPU work: the next frame is registered
        // while the browser's paint deadline is still open, so frame delivery stays vsync-aligned. The
        // alternative — scheduling the next rAF off the completion fence — slips a paint whenever the
        // fence resolves late, and under a throttled present (fullscreen vsync) its phase drifts against
        // the deadline, turning a steady rate into visible judder.
        requestFrame(frame);
        // drive dt from the rAF presentation timestamp (the frame's vsync-aligned start time the browser
        // assigns), not now() at callback time: the callback runs after a variable event-loop delay, so
        // now() carries that jitter into the sim timebase and misaligns the fixed-step interpolation from
        // the actual present (Raph Levien, "Swapchains and frame pacing"). The headless setTimeout path,
        // with no timestamp, falls back to now().
        const t = timestamp ?? now();
        if (lastCallback >= 0) {
            const raw = t - lastCallback;
            if (raw > 0) {
                intervals.push(raw);
                if (intervals.length > 20) intervals.shift();
            }
        }
        lastCallback = t;
        // coalesce a Chrome rAF double-fire so the loop submits once per present (else the extra frame fills
        // the swapchain queue → input latency). This is the present-pacing mechanism; MAX_FRAMES_IN_FLIGHT is
        // only the runaway backstop below.
        if (coalesce(t, lastTime, median(intervals, scratch))) return;
        // backstop only: under genuine GPU saturation the CPU would queue unboundedly past the GPU, so cap
        // the in-flight depth. The bound sits well above a present-throttled pipeline's depth (~3 frames),
        // since `onSubmittedWorkDone` is present-gated and a tighter cap would drop frames Chrome is ready to
        // present (a 60Hz fullscreen throttle reads ~3 in flight with the GPU idle).
        if ((Compute.pending?.() ?? 0) >= MAX_FRAMES_IN_FLIGHT) return;
        const dt = (t - lastTime) / 1000;
        lastTime = t;
        state.fenceWait(pendingFenceWaitMs);
        pendingFenceWaitMs = 0;
        state.step(dt);
        const fence = Compute.sync?.();
        if (fence) {
            const waitStart = now();
            fence.then(() => {
                pendingFenceWaitMs = now() - waitStart;
            });
        }
    }

    requestFrame(frame);

    return {
        state,
        skipped: app.skipped,
        dispose() {
            disposed = true;
            uiCleanup?.();
            overlay?.remove();
            app.dispose();
        },
    };
}

/** outcome of a {@link swap}: `ok` when the in-place swap applied, else `reason` says why a rebuild is needed */
export interface SwapResult {
    ok: boolean;
    reason?: string;
}

/**
 * hot-swap a live `State`'s plugins in place, preserving runtime state. For each
 * plugin (paired by name) it re-registers the components — the stable-id layer
 * reuses their storage and id, so membership, queries, and the GPU firehose
 * (slab buffers, bind groups, pipelines) survive untouched — swaps each system's
 * behavior onto the live scheduler object (identity + ordering + setup state
 * preserved), and re-runs `initialize` to repopulate module singletons with the
 * reloaded code. A schema / system-set / ordering / feature change it can't carry
 * safely returns `{ ok: false, reason }`; the caller then rebuilds from the
 * document. A `warm`- or `setup`-body edit is undetectable (a closure body can't
 * be diffed) and lands on the next rebuild rather than this swap. The editor
 * drives this from its HMR seam
 * — `prev`/`next` are the project's own plugins before and after the reload.
 *
 * `skipped` is the build's skip set ({@link App.skipped}): a plugin skipped at
 * build never initialized, so swapping it would half-apply — it's rejected here.
 * A user `initialize` that throws mid-swap also returns `{ ok: false }`: systems
 * are already swapped at that point, so the State is half-updated and the
 * rebuild the caller falls back to is the recovery.
 */
export async function swap(
    state: State,
    prev: readonly Plugin[],
    next: readonly Plugin[],
    skipped: readonly string[] = [],
): Promise<SwapResult> {
    const prevByName = new Map(prev.map((p) => [p.name, p]));
    const nextByName = new Map(next.map((p) => [p.name, p]));
    if (prevByName.size !== nextByName.size) return { ok: false, reason: "plugin set changed" };

    // ordering refs into the swapped set resolve by slot — a reload recreates the objects, so
    // identity can't pair them. Index every system across the whole set in one shared order.
    const prevIndex = new Map<System, number>();
    const nextIndex = new Map<System, number>();
    let slot = 0;
    for (const [name, nextPlugin] of nextByName) {
        const prevPlugin = prevByName.get(name);
        if (!prevPlugin) continue;
        const ps = prevPlugin.systems ?? [];
        const ns = nextPlugin.systems ?? [];
        for (let i = 0; i < Math.min(ps.length, ns.length); i++) {
            prevIndex.set(ps[i], slot);
            nextIndex.set(ns[i], slot);
            slot++;
        }
    }

    // validate every pair before mutating anything — any shape change falls back to a rebuild
    const skippedSet = new Set(skipped);
    for (const [name, nextPlugin] of nextByName) {
        const prevPlugin = prevByName.get(name);
        if (!prevPlugin) return { ok: false, reason: `plugin "${name}" added or renamed` };
        if (skippedSet.has(name)) return { ok: false, reason: `${name}: skipped at build` };
        const diff = shapeDiff(prevPlugin, nextPlugin, prevIndex, nextIndex);
        if (diff) return { ok: false, reason: `${name}: ${diff}` };
        // a plugin skipped at build (missing dependency) never reached the scheduler — swapping
        // it would half-apply (no-op system swap, fresh component registration), so rebuild
        for (const system of prevPlugin.systems ?? []) {
            if (!state.hasSystem(system))
                return { ok: false, reason: `${name}: system not live (skipped at build?)` };
        }
    }

    for (const [name, nextPlugin] of nextByName) {
        const prevPlugin = prevByName.get(name)!;
        const components = nextPlugin.components ?? {};
        const traits = nextPlugin.traits ?? {};
        for (const [cname, component] of Object.entries(components)) {
            register(cname, component, traits[cname]);
        }
        const prevSystems = prevPlugin.systems ?? [];
        const nextSystems = nextPlugin.systems ?? [];
        for (let i = 0; i < nextSystems.length; i++) state.swap(prevSystems[i], nextSystems[i]);
    }

    // initialize is registration-only and idempotent (the lifecycle contract), so re-running it
    // repopulates singletons with the new code without touching entities or warm GPU state.
    // A throw here lands after the system swap, so the State is half-updated — report ok:false
    // and let the caller's rebuild fallback recover, never wedge on an unhandled throw
    for (const nextPlugin of nextByName.values()) {
        try {
            await nextPlugin.initialize?.(state);
        } catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            return { ok: false, reason: `${nextPlugin.name}: initialize threw — ${msg}` };
        }
    }

    return { ok: true };
}

// a plugin's shape signature for the swap-vs-rebuild decision: a component-schema, system-set,
// ordering, or device-feature change can't be carried in place, so it forces a rebuild instead.
function shapeDiff(
    prev: Plugin,
    next: Plugin,
    prevIndex: Map<System, number>,
    nextIndex: Map<System, number>,
): string | null {
    const pc = prev.components ?? {};
    const nc = next.components ?? {};
    const pcKeys = Object.keys(pc).sort();
    const ncKeys = Object.keys(nc).sort();
    if (pcKeys.join(",") !== ncKeys.join(",")) return "component set changed";
    for (const key of ncKeys) {
        if (fieldSig(pc[key]) !== fieldSig(nc[key])) return `component "${key}" schema changed`;
    }
    const ps = prev.systems ?? [];
    const ns = next.systems ?? [];
    if (ps.length !== ns.length) return "system set changed";
    for (let i = 0; i < ns.length; i++) {
        if (systemSig(ps[i]) !== systemSig(ns[i])) return `system ${i} scheduling changed`;
        if (
            !sameRefs(ps[i].before, ns[i].before, prevIndex, nextIndex) ||
            !sameRefs(ps[i].after, ns[i].after, prevIndex, nextIndex)
        )
            return `system ${i} scheduling changed`;
    }
    if ((prev.features ?? []).join(",") !== (next.features ?? []).join(","))
        return "features changed";
    if ((prev.preferredFeatures ?? []).join(",") !== (next.preferredFeatures ?? []).join(","))
        return "preferred features changed";
    return null;
}

function fieldSig(component: Component): string {
    return fields(component)
        .map((f) => `${f.name}:${f.store.type.name}`)
        .join(",");
}

function systemSig(s: System): string {
    return [
        s.group ?? "simulation",
        s.first ? 1 : 0,
        s.last ? 1 : 0,
        (s.annotations?.mode as string) ?? "",
        (s.annotations?.layer as string) ?? "",
    ].join("|");
}

// an ordering ref into the swapped set compares by slot; one outside it (an engine
// anchor like BeginFrameSystem, never reloaded) compares by identity
function sameRefs(
    prev: readonly System[] | undefined,
    next: readonly System[] | undefined,
    prevIndex: Map<System, number>,
    nextIndex: Map<System, number>,
): boolean {
    const a = prev ?? [];
    const b = next ?? [];
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
        const pi = prevIndex.get(a[i]) ?? -1;
        const ni = nextIndex.get(b[i]) ?? -1;
        if (pi !== ni) return false;
        if (pi === -1 && a[i] !== b[i]) return false;
    }
    return true;
}
