import { afterAll, beforeAll } from "bun:test";
import { relative } from "node:path";
import { build } from "@dylanebert/shallot";
import { rawDevice } from "../src/engine/runtime";
import { CEILING } from "./test-tiers";

export function compileGpuFile<T>(path: string, compile: () => Promise<T>): () => T {
    let subject: T;
    beforeAll(async () => {
        const start = performance.now();
        const peer = "bun-webgpu";
        await (await import(peer)).setupGlobals();
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

/** Prebuild independent worlds on one file device; never share pipelines between worlds. */
export function gpuApps(
    path: string,
    configs: Parameters<typeof build>[0][],
): () => Awaited<ReturnType<typeof build>>[] {
    const apps: Awaited<ReturnType<typeof build>>[] = [];
    const subject = compileGpuFile(path, async () => {
        const owner = await build({ defaults: false, plugins: [] });
        apps.push(owner);
        const device = rawDevice(owner.state.gpu.device);
        const worlds = [];
        for (const config of configs) {
            const app = await build({ ...config, device });
            apps.push(app);
            worlds.push(app);
        }
        return worlds;
    });
    afterAll(() => {
        for (const app of apps.reverse()) app.dispose();
    });
    return subject;
}
