import { loadavg } from "node:os";
import { setTimeout } from "node:timers/promises";
import { Camera, DiagnosticsStore, Meshes } from "@dylanebert/shallot";
import { DiagnosticsOverlayPlugin, ProfilePlugin, showProfiler } from "@dylanebert/shallot/extras";
import { attachTexture } from "@dylanebert/shallot/rendering";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { AppPlugin, Viewer } from "../src/app";
import { sceneSwitcher } from "../src/sample";
import { categories } from "../src/scenes";

// bun diagnostics/benchmark-profile/run.ts; hardware timings are observations, not a gate.
const app = await openGpuApp({
    plugins: [
        {
            ...AppPlugin,
            dependencies: AppPlugin.dependencies!.filter(
                (plugin) => plugin !== DiagnosticsOverlayPlugin,
            ),
        },
        ProfilePlugin,
    ],
});
try {
    const world = app.world;
    console.log(
        JSON.stringify({
            adapter: world.gpu.adapter,
            width: 1280,
            height: 720,
            warmup: 10,
            frames: 150,
            solverThreads: 1,
        }),
    );
    for (const camera of world.query([Camera]))
        attachTexture(world, camera, { width: 1280, height: 720 });
    const select = sceneSwitcher(world, world.resource(Viewer));
    const scenes = categories.find((category) => category.name === "Benchmark")!.scenes;
    for (let index = 0; index < scenes.length; index++) {
        while (loadavg()[0]! > 8) await setTimeout(10000);
        const { id } = scenes[index]!;
        select(id);
        showProfiler(world);
        const store = world.resource(DiagnosticsStore);
        for (const diagnostic of store.iter())
            if (String(diagnostic.path).startsWith("render/")) store.remove(diagnostic.path);
        for (let frame = 0; frame < 10; frame++) {
            world.step(1 / 60);
            await world.frameFence;
        }
        const rows: Record<string, number[]> = {};
        const frames: number[] = [];
        for (let frame = 0; frame < 150; frame++) {
            const start = performance.now();
            world.step(1 / 60);
            await world.frameFence;
            frames.push(performance.now() - start);
            for (const diagnostic of world.resource(DiagnosticsStore).iter()) {
                const path = String(diagnostic.path);
                if (path.startsWith("render/") && path.endsWith("/elapsed_cpu"))
                    (rows[path] ??= []).push(diagnostic.value() ?? 0);
            }
        }
        if (loadavg()[0]! > 8) {
            index--;
            continue;
        }
        const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;
        console.log(
            JSON.stringify({
                scene: id,
                load: loadavg()[0],
                frame: mean(frames),
                meshes: world.resource(Meshes).size,
                rows: Object.fromEntries(
                    Object.entries(rows).map(([path, values]) => [path, mean(values)]),
                ),
            }),
        );
    }
} finally {
    app.dispose();
}
