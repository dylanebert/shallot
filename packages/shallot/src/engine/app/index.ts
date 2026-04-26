import { State, resource, type Plugin } from "../ecs";
import { toposort, registerComponent, registerRelation } from "../ecs/core";
import { parse, load, diagnose } from "../scene";
import { Runtime, readFile, requestFrame, now } from "../runtime";

export const FrameSync = resource<() => Promise<void> | null>("frame-sync");

export interface Loading {
    show(): (() => void) | void;
    update(progress: number): void;
    error?(message: string): void;
}

export const NoLoading: Loading = { show: () => {}, update: () => {} };

export interface Config {
    plugins: Plugin[];
    scene?: string | string[];
    loading?: Loading;
    defaults?: boolean;
    exclude?: Plugin[];
    setup?: (state: State) => void;
    ui?: (container: HTMLElement, state: State) => () => void;
}

let _defaultPlugins: readonly Plugin[] = [];
let _defaultLoading: (() => Loading) | null = null;

export function setDefaultPlugins(plugins: readonly Plugin[]): void {
    _defaultPlugins = plugins;
}

export function setDefaultLoading(factory: () => Loading): void {
    _defaultLoading = factory;
}

export async function build(config: Config): Promise<State> {
    const state = new State();

    const useDefaults = config.defaults !== false;
    const loading = config.loading ?? _defaultLoading?.();
    const cleanup = loading?.show();

    try {
        const excludeSet = config.exclude ? new Set(config.exclude) : null;

        const pluginSet = new Set<Plugin>();

        if (useDefaults) {
            for (const plugin of _defaultPlugins) {
                if (!excludeSet?.has(plugin)) {
                    pluginSet.add(plugin);
                }
            }
        }

        for (const plugin of config.plugins) {
            pluginSet.add(plugin);
        }

        const allPlugins = [...pluginSet];

        const skipped = new Set<Plugin>();
        for (const plugin of allPlugins) {
            for (const dep of plugin.dependencies ?? []) {
                if (!pluginSet.has(dep)) {
                    console.warn(
                        `Missing plugin dependency: ${plugin.name ?? "?"} requires ${dep.name ?? "?"}`,
                    );
                    skipped.add(plugin);
                    break;
                }
            }
        }

        for (const plugin of allPlugins) {
            if (skipped.has(plugin)) continue;
            if (plugin.components) {
                for (const [name, component] of Object.entries(plugin.components)) {
                    registerComponent(name, component);
                }
            }
            if (plugin.relations) {
                for (const relation of plugin.relations) {
                    registerRelation(relation);
                }
            }
            if (plugin.systems) {
                for (const system of plugin.systems) {
                    state.scheduler.register(system, plugin.name);
                }
            }
        }

        const edges: [Plugin, Plugin][] = [];
        for (const plugin of allPlugins) {
            for (const dep of plugin.dependencies ?? []) {
                edges.push([dep, plugin]);
            }
        }
        const sorted = toposort(allPlugins, edges);

        const scenes = config.scene
            ? Array.isArray(config.scene)
                ? config.scene
                : [config.scene]
            : [];

        const total = sorted.length * 2 + scenes.length;

        config.setup?.(state);

        for (let i = 0; i < sorted.length; i++) {
            const plugin = sorted[i];
            const onProgress = loading
                ? (progress: number) => loading.update((i + progress) / total)
                : undefined;
            if (!skipped.has(plugin)) {
                await plugin.initialize?.(state, onProgress);
            }
            loading?.update((i + 1) / total);
        }

        if (scenes.length > 0) {
            for (let i = 0; i < scenes.length; i++) {
                const scene = scenes[i];
                const xml = scene.startsWith("<") ? scene : await readFile(scene);
                const nodes = parse(xml);
                for (const d of diagnose(nodes)) console.warn(`[shallot] ${d.message}`);
                load(nodes, state);
                loading?.update((sorted.length + i + 1) / total);
            }
        }

        const warmable = sorted.filter((p) => p.warm && !skipped.has(p));
        let warmDone = 0;
        const warmBase = sorted.length + scenes.length;

        const warmPromises = warmable.map(async (plugin) => {
            await plugin.warm!(state, (progress: number) => {
                if (loading) {
                    loading.update((warmBase + warmDone + progress) / total);
                }
            });
            warmDone++;
            loading?.update((warmBase + warmDone) / total);
        });

        await Promise.all(warmPromises);

        if (cleanup) {
            loading?.update(1);
            await new Promise<void>((r) => requestFrame(r));
            cleanup();
        }

        return state;
    } catch (e) {
        const message = e instanceof Error ? e.message : String(e);
        loading?.error?.(message);
        throw e;
    }
}

export async function run(config: Config): Promise<State> {
    const state = await build(config);
    if (config.ui && Runtime === "web") {
        const canvas = document.querySelector("canvas");
        const parent = canvas?.parentElement ?? document.body;
        parent.style.position = "relative";
        const overlay = document.createElement("div");
        overlay.style.cssText = "position:absolute;inset:0;pointer-events:none;z-index:1";
        parent.appendChild(overlay);
        const cleanup = config.ui(overlay, state);
        state.onDispose(() => {
            cleanup();
            overlay.remove();
        });
    }

    let disposed = false;
    let lastTime = now();
    let pendingFenceWaitMs = 0;

    function scheduleFrame(): void {
        if (disposed) return;
        requestFrame(tick);
    }

    function tick(): void {
        if (disposed) return;
        const t = now();
        const dt = (t - lastTime) / 1000;
        lastTime = t;
        state.scheduler.reportFenceWait(pendingFenceWaitMs);
        pendingFenceWaitMs = 0;
        state.step(dt);
        const wait = FrameSync.from(state)?.();
        if (wait) {
            const waitStart = now();
            wait.then(() => {
                pendingFenceWaitMs = now() - waitStart;
                scheduleFrame();
            });
        } else {
            scheduleFrame();
        }
    }

    state.onDispose(() => {
        disposed = true;
    });
    scheduleFrame();
    return state;
}
