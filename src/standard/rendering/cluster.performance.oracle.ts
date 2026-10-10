import { test } from "bun:test";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { attachCanvas, Camera, PointLight } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { createApp } from "../../engine/app";
import { CanvasContext } from "../../engine/app/canvas.fixture";
import { StandardRenderer, StandardRenderingPlugin } from "./index";

await setupGlobals();
if (typeof ResizeObserver === "undefined") {
    Object.assign(globalThis, {
        ResizeObserver: class {
            observe() {}
            unobserve() {}
            disconnect() {}
        },
    });
}

const TIMED_PASSES = 10;
const TRIALS = 5;
const WARM_FRAMES = 9;

type TimedPass = { pass: string; microseconds: number };
type FrameTenSample = {
    adapter: string;
    adapterClass: string;
    passes: TimedPass[];
};

async function measureFrameTen(lightCount: number): Promise<FrameTenSample> {
    const app = await createApp({
        defaults: false,
        plugins: [
            { name: "ClusterTimestampQueries", gpu: { features: ["timestamp-query"] } },
            StandardRenderingPlugin,
        ],
    });
    const { world } = app;
    const device = world.gpu.device;
    try {
        let context: CanvasContext;
        const canvas = {
            width: 32,
            height: 24,
            style: { imageRendering: "auto" },
            getContext: () => context,
            getBoundingClientRect: () => ({ width: 32, height: 24 }),
        } as unknown as HTMLCanvasElement;
        context = new CanvasContext(canvas, 32, 24);
        const camera = world.create();
        world.add(camera, Transform);
        world.add(camera, Camera);
        world.add(camera, StandardRenderer);
        world.storage(Transform).translation.set(camera, 0, 0, 5, 0);
        attachCanvas(camera, canvas, world);

        for (let i = 0; i < lightCount; i++) {
            const eid = world.create();
            world.add(eid, Transform);
            world.add(eid, PointLight);
            const x = i % 4 === 3 ? 12 : ((i % 13) - 6) * 0.25;
            const y = ((Math.floor(i / 13) % 9) - 4) * 0.2;
            const depth = 4 + ((i * 7) % 28);
            world.storage(Transform).translation.set(eid, x, y, 5 - depth, 0);
            world.storage(PointLight).range.set(eid, 1 + (i % 3));
        }

        const queries = device.createQuerySet({ type: "timestamp", count: TIMED_PASSES * 2 });
        const resolved = device.createBuffer({
            label: "cluster-frame-ten-resolved",
            size: TIMED_PASSES * 16,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        const readback = device.createBuffer({
            label: "cluster-frame-ten-readback",
            size: TIMED_PASSES * 16,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        world.own(queries);
        world.own(resolved);
        world.own(readback);
        const labels: string[] = [];
        let capture = false;
        world.gpu.span = (name) => {
            if (!capture) return;
            const slot = labels.length;
            if (slot >= TIMED_PASSES) throw new Error(`too many timestamped passes: ${name}`);
            labels.push(name);
            return {
                querySet: queries,
                beginningOfPassWriteIndex: slot * 2,
                endOfPassWriteIndex: slot * 2 + 1,
            };
        };

        for (let frame = 0; frame < WARM_FRAMES; frame++) world.step(1 / 60);
        await device.queue.onSubmittedWorkDone();
        capture = true;
        world.step(1 / 60);
        capture = false;
        await device.queue.onSubmittedWorkDone();
        if (world.gpu.frame !== 10) throw new Error(`expected frame 10, got ${world.gpu.frame}`);
        if (labels.length === 0) throw new Error("no light clustering passes were timestamped");

        const encoder = device.createCommandEncoder({ label: "cluster-frame-ten-readback" });
        encoder.resolveQuerySet(queries, 0, labels.length * 2, resolved, 0);
        encoder.copyBufferToBuffer(resolved, 0, readback, 0, labels.length * 16);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const timestamps = new BigUint64Array(readback.getMappedRange());
        const passes = labels.map((label, slot) => ({
            pass: label,
            microseconds: Number(timestamps[slot * 2 + 1]! - timestamps[slot * 2]!) / 1000,
        }));
        readback.unmap();
        return {
            adapter: world.gpu.adapter.identity,
            adapterClass: world.gpu.adapter.class,
            passes,
        };
    } finally {
        app.dispose();
    }
}

test("timestamp the warmed light passes at frame 10", async () => {
    for (const pointLights of [0, 256]) {
        const samples: FrameTenSample[] = [];
        for (let trial = 0; trial < TRIALS; trial++)
            samples.push(await measureFrameTen(pointLights));
        const passNames = [
            ...new Set(samples.flatMap((sample) => sample.passes.map(({ pass }) => pass))),
        ];
        const medianPasses = passNames.map((pass) => {
            const times = samples
                .map((sample) => sample.passes.find((entry) => entry.pass === pass)?.microseconds)
                .filter((time): time is number => time !== undefined)
                .sort((a, b) => a - b);
            return { pass, microseconds: times[Math.floor(times.length / 2)] };
        });
        console.info(
            JSON.stringify({
                oracle: "light-clustering-frame-ten",
                host: `${process.platform}/${process.arch}`,
                adapter: samples[0]!.adapter,
                adapterClass: samples[0]!.adapterClass,
                frame: 10,
                pointLights,
                trials: samples.map(({ passes }) => passes),
                medianPasses,
            }),
        );
    }
}, 0);
