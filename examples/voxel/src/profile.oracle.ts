import { setDefaultTimeout, test } from "bun:test";
import { DiagnosticsStore, probeBuffer } from "@dylanebert/shallot";
import { ProfilePlugin } from "@dylanebert/shallot/extras";
import { attachTexture } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { Terrain, TerrainPlugin } from "./terrain";

setDefaultTimeout(TIMEOUT.node);
const median = (values: number[]) => values.sort((a, b) => a - b)[Math.floor(values.length / 2)];
test("report shipped voxel CPU/GPU spans and bundle recordings while scrolling and held", async () => {
    const app = await openGpuApp({ plugins: [TerrainPlugin, ProfilePlugin] });
    const { world } = app;
    const device = world.gpu.device;
    const createBundle = device.createRenderBundleEncoder.bind(device);
    let bundles = 0;
    device.createRenderBundleEncoder = (descriptor) => {
        bundles++;
        return createBundle(descriptor);
    };
    try {
        if (!device.features.has("timestamp-query"))
            throw new Error("voxel pass attribution requires timestamp-query");
        attachTexture(world, world.resource(Terrain).camera, { width: 1280, height: 720 });
        for (const mode of ["scrolling", "held"] as const) {
            const delta = mode === "scrolling" ? 1 / 60 : 0;
            for (let i = 0; i < 30; i++) {
                world.step(delta);
                await world.gpu.sync();
                await new Promise((resolve) => setTimeout(resolve, 0));
            }
            const stats = world.resource(DiagnosticsStore);
            for (const diagnostic of stats.iter()) diagnostic.clearHistory();
            const cpu: number[] = [],
                fenced: number[] = [];
            const initialBundles = bundles;
            let geometryChangeFrames = 0;
            const slots = world.resource(Terrain).slots;
            const geometry = slots.map((slot) => slot.geometry);
            for (let i = 0; i < 120; i++) {
                for (let index = 0; index < slots.length; index++)
                    geometry[index] = slots[index].geometry;
                const start = performance.now();
                world.step(delta);
                const recorded = performance.now();
                await world.gpu.sync();
                const completed = performance.now();
                cpu.push(recorded - start);
                fenced.push(completed - start);
                let changed = false;
                for (let index = 0; index < slots.length; index++)
                    changed ||= geometry[index] !== slots[index].geometry;
                geometryChangeFrames += Number(changed);
                // Yield outside timing to let asynchronous timestamp readbacks settle.
                await new Promise((resolve) => setTimeout(resolve, 0));
            }
            const spans = [...stats.iter()]
                .filter(
                    (diagnostic) =>
                        diagnostic.historyLength &&
                        (diagnostic.path.path.endsWith("/elapsed_gpu") ||
                            diagnostic.path.path.endsWith("/elapsed_cpu")),
                )
                .map((diagnostic) => ({
                    span: diagnostic.path.path,
                    medianMs: median([...diagnostic.measurements()].map((sample) => sample.value)),
                    samples: diagnostic.historyLength,
                }));
            console.log({
                adapter: world.gpu.adapter,
                mode,
                resolution: "1280x720 offscreen",
                frames: cpu.length,
                detailRows: new Uint32Array(
                    (
                        await probeBuffer(
                            world,
                            world.gpu.root.unwrap(world.resource(Terrain).placement!.table.count),
                        )
                    ).bytes,
                )[0],
                medianCpuStepMs: median(cpu),
                medianFencedStepMs: median(fenced),
                recordedBundles: bundles - initialBundles,
                geometryChangeFrames,
                spans,
            });
        }
    } finally {
        device.createRenderBundleEncoder = createBundle;
        app.dispose();
    }
});
