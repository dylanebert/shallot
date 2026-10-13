/// <reference types="@webgpu/types" />

import { type Component, type System, World } from "../ecs";
import {
    type AdapterVerdict,
    deviceLost,
    type GpuRequirements,
    now,
    precompileAll,
    Runtime,
    requestFrame,
    requestGPU,
    validateGpu,
    withPipelineOwner,
} from "../runtime";
import { coalesce, frameDelta, median } from "./coalesce";
import { resolvePlugins } from "./compose";

export * from "./compose";

/**
 * bundle of components, systems, and lifecycle hooks: the unit of behavior a project enables.
 */
export interface Plugin {
    /** unique name */
    readonly name: string;
    /** Declares whether fixed simulation replays deterministically (default true).
     * Fixed consumers of GPU readback declare false. Byte access is not guarded. */
    readonly deterministic?: boolean;
    /** Binds a snapshot/restore pair to each world during plugin registration, before initialize.
     * A plugin recovers state it owns that any fixed system reads, including read-only authoring
     * resources. Fixed plugins must declare a participant or `"stateless"` when components or
     * per-call reconstruction hold it all; registration refusal is not a completeness proof. Images are independent
     * and reusable; restore sees recovered ECS identity, fields and fixedTick.
     * Host handles and presentation are outside capture; live device reads are not replay inputs. */
    readonly recovery?: ((world: World) => import("../ecs").Recovery) | "stateless";
    /** systems this plugin adds to the scheduler */
    readonly systems?: readonly System[];
    /** components registered under exact stable keys, with their defaults and requirements */
    readonly components?: readonly Component[];
    /** other plugins that must load first; every dependency must be present in the composition */
    readonly dependencies?: readonly Plugin[];
    /** Required and preferred features and required limits for this plugin's GPU use. Presence, including `{}`, declares a device need; an unmet requirement refuses before initialization with this plugin and the cause named. */
    readonly gpu?: GpuRequirements;
    /** initial setup, run before warm-up; idempotent, may report progress */
    readonly initialize?: (
        world: World,
        onProgress?: (progress: number) => void,
    ) => void | Promise<void>;
    /**
     * GPU setup and derived spawns after initialization; may report progress.
     * idempotent: entity spawns need nothing (they live in the `World`). Tie any external effect
     * (a DOM mount, listener, or rAF loop) to the World via `world.onDispose` / `world.signal` so it
     * unwinds at dispose.
     */
    readonly warm?: (world: World, onProgress?: (progress: number) => void) => void | Promise<void>;
    /**
     * teardown, run in reverse dependency order on `App.dispose`. reserve it for process/module-lifetime
     * teardown — engine singletons, globals — not per-build external effects. tie a per-build mount,
     * listener, or rAF loop to the World via `world.onDispose` / `world.signal` at its creation site
     * instead: a `dispose` hook fires only on the `App.dispose` path, never on a direct `world.dispose()`
     * (a host driving the World without an App), so a World-registered effect is the only one covering both.
     */
    readonly dispose?: (world: World) => void;
}

/**
 * a startup/error screen driven by the build's progress. the engine calls `show` before loading,
 * `update` across every lifecycle step, `complete` once progress reaches `1`, and `error` instead
 * if the build throws. see the built-in `shallotDark` family in `standard`.
 */
export interface Loading {
    /** display the screen; return a cleanup called once the build finishes, or nothing to leave it up */
    show(): (() => void) | void;
    /** report build progress, `0`–`1` across initialize and warm */
    update(progress: number): void;
    /**
     * progress reached `1`; return a promise to hold the screen (an outro, a minimum dwell) until
     * it resolves. the engine awaits it, then one frame, then the cleanup. omit to dismiss at once
     */
    complete?(): Promise<void> | void;
    /** show a non-blocking adapter notice while the app continues on a fallback or unidentified adapter */
    notice?(verdict: AdapterVerdict): void;
    /** render the thrown value; `runApp` reports every non-destroyed device loss here. */
    error?(error: unknown): void;
}

/**
 * the {@link createApp} / {@link runApp} configuration: plugins and startup behavior.
 */
export interface AppConfig {
    /** plugins to load, unioned with the built-in defaults unless `defaults` is `false` */
    plugins: Plugin[];
    /** startup screen; defaults to the one set by `setDefaultLoading` */
    loading?: Loading;
    /** `false` skips the built-in default plugins entirely */
    defaults?: boolean;
    /** specific default plugins to drop while keeping the rest */
    exclude?: Plugin[];
    /** hook run after registration, before any plugin `initialize` */
    setup?: (world: World) => void;
    /** mount app UI into the canvas-bounded overlay; return a cleanup. see `mountOverlay` */
    ui?: (container: HTMLElement, world: World) => () => void;
    /** externally acquired GPU device, checked against enabled declarations; if omitted, the engine acquires one for declared GPU needs */
    device?: GPUDevice;
    /** adapter that supplied an externally-acquired {@link device}; omitted devices are stamped unidentified */
    adapter?: GPUAdapter;
    /**
     * render device-pixel ratio for canvas views; fixed at app construction. `"auto"` (default)
     * clamps `devicePixelRatio` to `[1, 2]`. A number forces a fixed ratio (`1` = CSS resolution /
     * cheapest; `2` = twice CSS resolution; below 1 = pixel-art downscale with nearest-neighbor
     * upscaling). Canvas resize reads the ratio again, including after moving between monitors.
     */
    pixelRatio?: number | "auto";
}

/** result of {@link createApp} / {@link runApp}. owns the plugin teardown order. */
export interface App {
    readonly world: World;
    dispose(): void;
}

/** Settle every started warm; validate and precompile only when a device is present. @internal */
export async function warmPlugins(
    device: GPUDevice | undefined,
    world: World,
    plugins: readonly Plugin[],
    onProgress?: (progress: number) => void,
): Promise<void> {
    const warm = async (): Promise<void> => {
        const perPlugin = new Array(plugins.length).fill(0);
        const report = () => onProgress?.(perPlugin.reduce((a, b) => a + b, 0));
        const results = await Promise.allSettled(
            plugins.map(async (plugin, i) => {
                try {
                    await withPipelineOwner(device, plugin.name, () =>
                        plugin.warm!(world, (p) => {
                            perPlugin[i] = Math.max(perPlugin[i], p);
                            report();
                        }),
                    );
                } catch (error) {
                    throw pluginHookError(plugin, "warm", error);
                } finally {
                    perPlugin[i] = 1;
                    report();
                }
            }),
        );
        const rejected = results.filter(
            (result): result is PromiseRejectedResult => result.status === "rejected",
        );
        if (rejected.length > 0) {
            const reasons = rejected.map((r) => r.reason);
            throw new AggregateError(
                reasons,
                reasons.map((r) => (r instanceof Error ? r.message : String(r))).join(", "),
            );
        }

        if (device) {
            // typegpu creates pipelines synchronously and Dawn defers that compile to the first dispatch,
            // so every registered pipeline is forced here — under the loading screen, not on frame one.
            await precompileAll(world);
        }
    };

    if (device) await validateGpu(device, "pipeline warm", warm);
    else await warm();
}

function pluginHookError(plugin: Plugin, hook: "initialize" | "warm", error: unknown): Error {
    const detail = error instanceof Error ? error.message : String(error);
    return new Error(`Plugin "${plugin.name}" ${hook} threw: ${detail}`);
}

// runaway backstop: the most frames the loop may run ahead of the GPU before skipping a step, so the CPU
// can't queue unboundedly past a saturated GPU. Sized well above a present-throttled pipeline's depth —
// `world.gpu.sync` counts frames by `onSubmittedWorkDone`, which is present-gated, so a 60Hz fullscreen
// throttle reads ~3 frames in flight with the GPU otherwise idle; a tighter cap would drop frames Chrome
// is ready to present (the fullscreen-throttle judder). Per-present pacing is the double-fire `coalesce`,
// not this; the bound engages only under sustained genuine GPU saturation.
const MAX_FRAMES_IN_FLIGHT = 6;
// the recent rAF-callback intervals the double-fire coalescer's median reads
const MEDIAN_WINDOW = 20;

export const ClearChangeMarksSystem: System = {
    group: "draw",
    first: true,
    name: "clear-component-changes",
    update(world) {
        world.uploadTables();
        world.clearChangesIfNeeded();
    },
};

let _defaultPlugins: readonly Plugin[] = [];
let _defaultLoading: (() => Loading) | null = null;

/**
 * set the global default plugin set every {@link createApp} unions in (unless `defaults: false`). `standard`
 * calls this at import with `DEFAULT_PLUGINS`; override it to define your own zero-config baseline.
 */
export function setDefaultPlugins(plugins: readonly Plugin[]): void {
    _defaultPlugins = plugins;
}

/**
 * set the global default {@link Loading} screen every {@link createApp} uses when `config.loading` is omitted.
 * `standard` calls this at import with its `shallotDark` screen.
 */
export function setDefaultLoading(factory: () => Loading): void {
    _defaultLoading = factory;
}

let buildTail: Promise<void> = Promise.resolve();

/**
 * build the app: collect plugins, acquire a GPU device only for declared needs or a supplied device,
 * register, run `initialize` and `warm`, returning a live {@link World} without starting a frame loop.
 * Build setup is serialized, and
 * completed Apps may coexist with separate World-owned storage and GPU registries. Plugin resources retained
 * in module globals are not isolated by this guarantee. Drive `world.step(dt)` yourself, or use {@link runApp}
 * for the managed loop.
 */
export function createApp(config: AppConfig): Promise<App> {
    const pending = buildTail.then(() => buildNow(config));
    buildTail = pending.then(
        () => undefined,
        () => undefined,
    );
    return pending;
}

async function buildNow(config: AppConfig): Promise<App> {
    let world!: World;
    let loading: Loading | undefined;
    let cleanup: (() => void) | undefined;
    const initialized: Plugin[] = [];

    try {
        const useDefaults = config.defaults !== false;
        const excluded = new Set(config.exclude ?? []);
        const pluginSet = new Set<Plugin>();
        if (useDefaults) {
            for (const plugin of _defaultPlugins) if (!excluded.has(plugin)) pluginSet.add(plugin);
        }
        for (const plugin of config.plugins) pluginSet.add(plugin);

        // A public plugin selection pulls in its declared substrates. The resolver remains pure and strict;
        // this boundary makes `createApp({ defaults: false, plugins: [StandardPhysicsPlugin] })` a complete composition.
        const requested = [...pluginSet];
        for (let i = 0; i < requested.length; i++) {
            for (const dependency of requested[i].dependencies ?? []) {
                if (!pluginSet.has(dependency)) {
                    pluginSet.add(dependency);
                    requested.push(dependency);
                }
            }
        }

        const composition = resolvePlugins([...pluginSet]);
        if (composition.missing.length > 0) {
            const edges = composition.missing.map(
                ({ plugin, dependency }) => `${plugin.name} requires ${dependency.name}`,
            );
            throw new Error(`Missing plugin dependencies:\n${edges.join("\n")}`);
        }

        const sorted = composition.plugins;
        world = new World({ pixelRatio: config.pixelRatio });
        loading = config.loading ?? _defaultLoading?.();
        cleanup = loading?.show() ?? undefined;

        const requirements = sorted.flatMap((plugin) =>
            plugin.gpu ? [{ plugin: plugin.name, ...plugin.gpu }] : [],
        );
        if (config.device !== undefined || requirements.length > 0) {
            const compute = await requestGPU(config.device, requirements, config.adapter, {
                own: world.own.bind(world),
                resource: world.resource.bind(world),
            });
            world.attachGpu(compute);
            if (world.gpu.adapter.class !== "real") loading?.notice?.(world.gpu.adapter);
        }

        for (const plugin of sorted) {
            if (
                plugin.recovery !== undefined ||
                plugin.systems?.some((system) => system.group === "fixed")
            )
                world.registerRecovery(
                    plugin.name,
                    typeof plugin.recovery === "function"
                        ? plugin.recovery(world)
                        : plugin.recovery,
                );
            for (const entry of plugin.components ?? []) {
                world.registry.register(entry, plugin.name);
                world.storage(entry);
            }
            for (const system of plugin.systems ?? []) {
                world.addSystem(system, plugin.name);
            }
        }

        world.addSystem(ClearChangeMarksSystem, "Engine");

        // Assign registered components their world-owned columns before loading.
        for (const { component } of world.registry.entries()) {
            world.storage(component);
        }

        const warmable = sorted.filter((p) => p.warm);
        const total = sorted.length + warmable.length;

        config.setup?.(world);

        for (let i = 0; i < sorted.length; i++) {
            const currentLoading = loading;
            const onProgress = currentLoading
                ? (progress: number) => currentLoading.update((i + progress) / total)
                : undefined;
            initialized.push(sorted[i]);
            try {
                await withPipelineOwner(world.gpuIfAvailable?.device, sorted[i].name, () =>
                    sorted[i].initialize?.(world, onProgress),
                );
            } catch (error) {
                throw pluginHookError(sorted[i], "initialize", error);
            }
            loading?.update((i + 1) / total);
        }

        const warmBase = sorted.length;
        await warmPlugins(world.gpuIfAvailable?.device, world, warmable, (progress) => {
            loading?.update((warmBase + progress) / total);
        });

        loading?.update(1);
        await loading?.complete?.();
        if (cleanup) {
            await new Promise<void>((r) => requestFrame(() => r()));
            cleanup();
            cleanup = undefined;
        }

        let disposed = false;
        return {
            world,
            dispose() {
                if (disposed) return;
                disposed = true;
                for (let i = sorted.length - 1; i >= 0; i--) {
                    try {
                        sorted[i].dispose?.(world);
                    } catch (err) {
                        console.error(`Plugin "${sorted[i].name}" threw during dispose:`, err);
                    }
                }
                try {
                    world.dispose();
                } catch (err) {
                    console.error("World dispose threw:", err);
                }
            },
        };
    } catch (e) {
        // a failed build leaves nothing live: plugins that started initialize dispose in
        // reverse, then the World, so a retry builds against clean module singletons. A dispose
        // throw here is reported, never allowed to mask the build error.
        for (let i = initialized.length - 1; i >= 0; i--) {
            try {
                initialized[i].dispose?.(world);
            } catch (err) {
                console.error(`Plugin "${initialized[i].name}" threw during cleanup:`, err);
            }
        }
        try {
            world?.dispose();
        } catch (err) {
            console.error("World dispose threw during cleanup:", err);
        }
        try {
            if (loading?.error) loading.error(e);
            else cleanup?.();
        } catch (err) {
            console.error("Loading cleanup threw during build failure:", err);
        }
        throw e;
    }
}

/**
 * create the sandboxed UI overlay over a canvas: the single DOM surface a shallot app's UI mounts
 * into ({@link AppConfig.ui}). It fills the canvas's parent and is a true sandbox: `contain: layout
 * paint` makes it the containing block for absolute *and* fixed descendants and clips paint to its
 * box, so app UI is bounded to the canvas region and can never spill into an embedding host (a
 * host page), even a stray `position: fixed`. `pointer-events: none` lets input
 * reach the canvas; UI panels re-enable it. Returns the overlay. Pass `world` to tie the overlay's
 * removal to the World's lifetime — it auto-registers `overlay.remove()` via {@link World.onDispose}
 * and restores the host element's prior inline `position`, so a build that disposes cleans it up; omit
 * `world` to remove the overlay yourself, in which case the `position` is not restored.
 */
export function mountOverlay(canvas: HTMLElement | null, world?: World): HTMLDivElement {
    const parent = canvas?.parentElement ?? document.body;
    const prior = parent.style.position;
    parent.style.position = "relative";
    const overlay = document.createElement("div");
    overlay.style.cssText =
        "position:absolute;inset:0;pointer-events:none;z-index:1;contain:layout paint;overflow:hidden";
    parent.appendChild(overlay);
    world?.onDispose(() => {
        overlay.remove();
        parent.style.position = prior;
    });
    return overlay;
}

/**
 * build the app and start the `requestAnimationFrame` frame loop, mounting `config.ui` (web only). the
 * loop drives `world.step(dt)` each frame, GPU-fence backpressured so it never runs far ahead of the GPU.
 * system setup/update errors are logged and the system pauses until swapped or rebuilt; the frame finishes. A
 * non-destroyed device loss stops the loop and reaches the loading screen, without recovery.
 */
export async function runApp(config: AppConfig): Promise<App> {
    const loading = config.loading ?? _defaultLoading?.();
    const app = await createApp({ ...config, loading });
    try {
        const world = app.world;
        world.logAndPauseSystemErrors();
        const gpu = world.gpuIfAvailable;
        const device = gpu?.device;
        const pending = gpu?.pending;
        const sync = gpu?.sync;
        // UI teardown is World-owned: the overlay auto-registers its removal (mountOverlay above), and the
        // ui cleanup registers beside it. Both run at world.dispose() — after the plugin dispose hooks on the
        // App.dispose path (UI cleanup is DOM/unmount work with no dependency on plugin GPU state), and it also
        // covers a host that calls world.dispose() directly.
        if (config.ui && Runtime === "web") {
            const overlay = mountOverlay(document.querySelector("canvas"), world);
            const uiCleanup = config.ui(overlay, world);
            if (uiCleanup) world.onDispose(uiCleanup);
        }

        let disposed = false;
        let stopped = false;
        if (device) {
            void device.lost.then((info) => {
                if (disposed || info.reason === "destroyed") return;
                stopped = true;
                loading?.error?.(
                    new Error(
                        `GPU device lost ${info.reason}: ${info.message}\nReloading the page may resolve this issue.`,
                    ),
                );
            });
        }
        // stop the rAF loop when the World tears down, so a host that calls world.dispose() directly
        // halts the loop too — not only the returned App.dispose(). Without this the loop keeps
        // stepping a torn-down World every frame (the stacked-rAF leak). App.dispose sets it first; this is
        // idempotent with that.
        world.onDispose(() => {
            disposed = true;
        });
        // seeded by the first frame's own timestamp (`frameDelta` steps 0 there), never by `now()` here: a
        // rAF timestamp is the frame's vsync-aligned start and can precede a wall-clock seed, which stepped
        // a negative dt into the scheduler's throw
        let lastTime = -1;
        let pendingFenceWaitMs = 0;
        // the last MEDIAN_WINDOW raw-callback intervals in a ring + an insertion scratch, feeding the
        // double-fire coalescer's median
        let lastCallback = -1;
        const intervals = new Float64Array(MEDIAN_WINDOW);
        const scratch = new Float64Array(MEDIAN_WINDOW);
        let intervalCount = 0;
        let intervalNext = 0;
        // the issue time of each fence still in flight, oldest at `fenceHead`: the queue settles fences in
        // submission order, and the in-flight backstop below keeps at most MAX_FRAMES_IN_FLIGHT of them
        const fenceIssued = new Float64Array(MAX_FRAMES_IN_FLIGHT);
        let fenceHead = 0;
        let fenceTail = 0;
        const fenceSettled = (): void => {
            pendingFenceWaitMs = now() - fenceIssued[fenceHead];
            fenceHead = (fenceHead + 1) % MAX_FRAMES_IN_FLIGHT;
        };
        // A rejected fence has no timing sample. The loss observer stops the loop and reports non-destroyed losses.
        const fenceRejected = (): void => {
            fenceHead = (fenceHead + 1) % MAX_FRAMES_IN_FLIGHT;
        };

        // A method rather than a function declaration: minification keeps property names, so a production
        // build's profile still names the frame loop.
        const loop = {
            frame(timestamp?: number): void {
                if (
                    disposed ||
                    stopped ||
                    (device !== undefined && deviceLost(device)) ||
                    world.gpuIfAvailable?.sync !== sync
                )
                    return;
                // rAF clocks the loop and reschedules first, before any GPU work: the next frame is registered
                // while the browser's paint deadline is still open, so frame delivery stays vsync-aligned. The
                // alternative — scheduling the next rAF off the completion fence — slips a paint whenever the
                // fence resolves late, and under a throttled present (fullscreen vsync) its phase drifts against
                // the deadline, turning a steady rate into visible judder.
                requestFrame(loop.frame);
                // drive dt from the rAF presentation timestamp (the frame's vsync-aligned start time the browser
                // assigns), not now() at callback time: the callback runs after a variable event-loop delay, so
                // now() carries that jitter into the sim timebase and misaligns the fixed-step interpolation from
                // the actual present (Raph Levien, "Swapchains and frame pacing"). The headless setTimeout path,
                // with no timestamp, falls back to now().
                const t = timestamp ?? now();
                if (lastCallback >= 0) {
                    const raw = t - lastCallback;
                    if (raw > 0) {
                        intervals[intervalNext] = raw;
                        intervalNext = (intervalNext + 1) % MEDIAN_WINDOW;
                        if (intervalCount < MEDIAN_WINDOW) intervalCount++;
                    }
                }
                lastCallback = t;
                // coalesce a Chrome rAF double-fire so the loop submits once per present (else the extra frame fills
                // the swapchain queue → input latency). This is the present-pacing mechanism; MAX_FRAMES_IN_FLIGHT is
                // only the runaway backstop below.
                if (coalesce(t, lastTime, median(intervals, intervalCount, scratch))) return;
                // Runaway backstop; MAX_FRAMES_IN_FLIGHT says why it sits above a throttled present's depth.
                if ((pending?.() ?? 0) >= MAX_FRAMES_IN_FLIGHT) return;
                const dt = frameDelta(t, lastTime);
                lastTime = t;
                world.fenceWaitSink?.(pendingFenceWaitMs);
                pendingFenceWaitMs = 0;
                world.step(dt);
                // Fence timing has a reader only when requested; time the World's submission fence, not a second one.
                const fence = world.frameFence;
                if (fence && world.fenceWaitSink) {
                    fenceIssued[fenceTail] = now();
                    fenceTail = (fenceTail + 1) % MAX_FRAMES_IN_FLIGHT;
                    fence.then(fenceSettled, fenceRejected);
                }
            },
        };

        requestFrame(loop.frame);

        return {
            world,
            dispose() {
                disposed = true;
                app.dispose();
            },
        };
    } catch (error) {
        try {
            app.dispose();
        } catch (disposeError) {
            console.error("run setup cleanup threw:", disposeError);
        }
        throw error;
    }
}
