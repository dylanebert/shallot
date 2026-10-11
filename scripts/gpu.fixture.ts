import { afterAll, beforeAll } from "bun:test";
import { relative } from "node:path";
import { createApp, DEFAULT_PLUGINS, type Plugin } from "@dylanebert/shallot";
import { rawDevice } from "../src/engine/runtime";
import { expectWarning } from "./test-console";
import { CEILING } from "./test-tiers";

export function compileGpuFile<T>(path: string, compile: () => Promise<T>): () => T {
    let subject: T;
    beforeAll(async () => {
        const start = performance.now();
        const { setupGlobals } = await import("@dylanebert/shallot/webgpu");
        await setupGlobals();
        subject = await compile();
        if (process.env.SHALLOT_GPU_COMPILE_ORACLE === "1") {
            console.log(
                JSON.stringify({
                    gpuCompile: relative(process.cwd(), path),
                    ms: performance.now() - start,
                }),
            );
        }
    }, CEILING.startup);
    return () => subject;
}

/** Union declared GPU needs for one test device shared by independently built worlds. */
export function gpuRequirements(plugins: readonly Plugin[]): NonNullable<Plugin["gpu"]> {
    const required = new Set<GPUFeatureName>();
    const preferred = new Set<GPUFeatureName>();
    const limits: Record<string, number> = {};
    const seen = new Set<Plugin>();
    const visit = (plugin: Plugin): void => {
        if (seen.has(plugin)) return;
        seen.add(plugin);
        for (const dependency of plugin.dependencies ?? []) visit(dependency);
        for (const feature of plugin.gpu?.features ?? []) required.add(feature);
        for (const feature of plugin.gpu?.preferredFeatures ?? []) preferred.add(feature);
        for (const [name, value] of Object.entries(plugin.gpu?.limits ?? {})) {
            if (typeof value !== "number") continue;
            limits[name] = name.startsWith("min")
                ? Math.min(limits[name] ?? value, value)
                : Math.max(limits[name] ?? value, value);
        }
    };
    for (const plugin of plugins) visit(plugin);
    return {
        features: [...required],
        preferredFeatures: [...preferred],
        limits: limits as Partial<GPUSupportedLimits>,
    };
}

type GpuApp = Awaited<ReturnType<typeof createApp>>;

/** Build one test app while declaring an adapter notice for GPU-backed compositions. */
export function createTestGpuApp(
    config: Parameters<typeof createApp>[0],
): ReturnType<typeof createApp> {
    const visited = new Set<Plugin>();
    const usesGpu = (plugin: Plugin): boolean => {
        if (visited.has(plugin)) return false;
        visited.add(plugin);
        return plugin.gpu !== undefined || (plugin.dependencies ?? []).some(usesGpu);
    };
    if (
        config.device !== undefined ||
        (config.defaults !== false && DEFAULT_PLUGINS.some(usesGpu)) ||
        (config.plugins ?? []).some(usesGpu)
    ) {
        expectWarning(
            /^\[shallot\] WebGPU reports (?:an adapter with no identity|a fallback adapter)/,
            { optional: true },
        );
    }
    return createApp(config);
}

/** Wait for all fixture submissions before releasing the fixture's worlds. */
export async function disposeGpuApps(apps: GpuApp[]): Promise<void> {
    const devices = new Set<GPUDevice>();
    for (const app of apps) {
        try {
            devices.add(rawDevice(app.world.gpu.device));
        } catch {
            // Continue teardown if a world was already disposed or lost its device.
        }
    }
    try {
        await Promise.all(
            [...devices].map(async (device) => {
                try {
                    await device.queue.onSubmittedWorkDone();
                } catch {
                    // Device loss must not keep the fixture's owned worlds alive.
                }
            }),
        );
    } finally {
        for (const app of apps.reverse()) app.dispose();
    }
}

/** Prebuild independent worlds on one file device; never share pipelines between worlds. */
export function gpuApps(
    path: string,
    configs: Parameters<typeof createApp>[0][],
): () => Awaited<ReturnType<typeof createApp>>[] {
    const apps: Awaited<ReturnType<typeof createApp>>[] = [];
    const subject = compileGpuFile(path, async () => {
        const owner = await createTestGpuApp({
            defaults: false,
            plugins: [
                {
                    name: "GpuTestDevice",
                    gpu: gpuRequirements(configs.flatMap((config) => config.plugins)),
                },
            ],
        });
        apps.push(owner);
        const device = rawDevice(owner.world.gpu.device);
        const worlds = [];
        for (const config of configs) {
            const app = await createTestGpuApp({ ...config, device });
            apps.push(app);
            worlds.push(app);
        }
        return worlds;
    });
    afterAll(() => disposeGpuApps(apps));
    return subject;
}
