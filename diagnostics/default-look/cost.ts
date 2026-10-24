import {
    createApp,
    DiagnosticsStore,
    type Plugin,
    ScreenSpaceAmbientOcclusionPlugin,
} from "@dylanebert/shallot";
import { ProfilePlugin } from "@dylanebert/shallot/extras";
import { type CostRow, costCells } from "./cost-report";
import { PALETTES } from "./palette";
import { DIAGNOSTIC_PLUGINS, SCENES, type SceneDefinition } from "./scenes";
import { AO_LEVELS, addAmbientOcclusion } from "./ssao";

const status = document.querySelector("#status")!;
const output = document.querySelector("#results")!;
const median = (values: number[]) => values.sort((a, b) => a - b)[Math.floor(values.length / 2)]!;
const rows: CostRow[] = [];

function sceneApp(scene: SceneDefinition, quality: string) {
    const plugin: Plugin = {
        name: "DefaultLookCost",
        dependencies: [
            ...DIAGNOSTIC_PLUGINS,
            ...(scene.plugins ?? []),
            ProfilePlugin,
            ScreenSpaceAmbientOcclusionPlugin,
        ],
        initialize(world) {
            scene.build(world, PALETTES["aperture-carbon"]);
            addAmbientOcclusion(world, quality);
        },
    };
    return createApp({ plugins: [plugin], pixelRatio: devicePixelRatio });
}

try {
    status.textContent = "Warming device: 120 seconds at Ultra";
    console.log(status.textContent);
    const soak = await sceneApp(SCENES[0]!, "Ultra");
    try {
        const until = performance.now() + 120_000;
        while (performance.now() < until) {
            soak.world.step(1 / 60);
            await soak.world.frameFence;
        }
    } finally {
        soak.dispose();
    }
    for (const scene of SCENES) {
        for (const quality of AO_LEVELS) {
            status.textContent = `Running ${scene.id}/${quality}`;
            console.log(status.textContent);
            const app = await sceneApp(scene, quality);
            try {
                const world = app.world;
                const frames: number[] = [];
                const spans = new Map<string, number[]>();
                const lastSpanTime = new Map<string, number>();
                const store = world.resource(DiagnosticsStore);
                const warmUntil = performance.now() + 10_000;
                const measureUntil = warmUntil + 20_000;
                while (performance.now() < measureUntil) {
                    const start = performance.now();
                    world.step(1 / 60);
                    await world.frameFence;
                    if (performance.now() < warmUntil) continue;
                    frames.push(performance.now() - start);
                    for (const diagnostic of store.iter()) {
                        const path = diagnostic.path.path;
                        if (
                            !path.startsWith("render/standard:ssao-") ||
                            !path.endsWith("/elapsed_gpu")
                        )
                            continue;
                        const values = spans.get(path) ?? [];
                        const after = lastSpanTime.get(path) ?? warmUntil / 1000;
                        for (const measurement of diagnostic.measurements()) {
                            if (measurement.time <= after) continue;
                            values.push(measurement.value);
                            lastSpanTime.set(path, measurement.time);
                        }
                        if (values.length) spans.set(path, values);
                    }
                }
                if (
                    quality !== "Off" &&
                    world.gpu.device.features.has("timestamp-query") &&
                    spans.size === 0
                ) {
                    throw new Error(`${scene.id}/${quality}: no SSAO GPU spans recorded`);
                }
                const canvas = document.querySelector("canvas")!;
                const row = {
                    scene: scene.id,
                    quality,
                    adapter: {
                        vendor: world.gpu.device.adapterInfo.vendor,
                        architecture: world.gpu.device.adapterInfo.architecture,
                        device: world.gpu.device.adapterInfo.device,
                        description: world.gpu.device.adapterInfo.description,
                    },
                    browser: navigator.userAgent,
                    resolution: `${canvas.width}×${canvas.height}`,
                    timestampQuery: world.gpu.device.features.has("timestamp-query")
                        ? "available"
                        : "unavailable: frame time alone",
                    samples: frames.length,
                    frameMs: median(frames),
                    spanSamples: Object.fromEntries(
                        [...spans].map(([path, values]) => [path, values.length]),
                    ),
                    spansMs: Object.fromEntries(
                        [...spans].map(([path, values]) => [path, median(values)]),
                    ),
                };
                rows.push(row);
                const tr = document.createElement("tr");
                for (const value of costCells(row)) {
                    const cell = document.createElement("td");
                    cell.textContent = value;
                    tr.append(cell);
                }
                document.querySelector("tbody")!.append(tr);
                output.textContent = JSON.stringify(rows, null, 2);
                console.log(JSON.stringify(rows.at(-1)));
            } finally {
                app.dispose();
            }
        }
    }
    status.textContent = "Complete";
    document.body.dataset.complete = "true";
} catch (error) {
    status.textContent = String(error);
    document.body.dataset.error = String(error);
    throw error;
}
