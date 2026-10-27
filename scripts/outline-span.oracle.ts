import { test } from "bun:test";
import {
    Body,
    BodyType,
    Camera,
    GlobalTransform,
    MeshInstance,
    Outline,
    pointerMove,
} from "@dylanebert/shallot";
import { DiagnosticsOverlayPlugin } from "@dylanebert/shallot/extras";
import { attachTexture, worldToViewport } from "@dylanebert/shallot/rendering";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { AppPlugin, Viewer } from "../examples/physics/src/app";
import { Grab } from "../examples/physics/src/grab";
import { sceneSwitcher } from "../examples/physics/src/sample";

test("physics grab outline: summed GPU outline:jfa span on a fixed view", async () => {
    const app = await openGpuApp({
        plugins: [
            {
                ...AppPlugin,
                dependencies: AppPlugin.dependencies!.filter((p) => p !== DiagnosticsOverlayPlugin),
            },
            { name: "TimestampRequirement", gpu: { features: ["timestamp-query"] } },
        ],
    });
    try {
        const world = app.world,
            device = world.gpu.device;
        sceneSwitcher(world, world.resource(Viewer))("boxStack");
        const camera = [...world.query([Camera])][0]!;
        attachTexture(world, camera, { width: 2560, height: 1440 });
        world.tick();
        world.tick();
        world.step(0);
        await world.frameFence;
        let target = 0;
        for (const eid of world.query([Body, MeshInstance])) {
            if (world.storage(Body).type.get(eid) !== BodyType.Dynamic) continue;
            const p = world.get(eid, GlobalTransform)!.translation;
            const xy = worldToViewport(world, camera, [p[0]!, p[1]!, p[2]!]);
            if (!xy || xy[0] < 0 || xy[0] >= 2560 || xy[1] < 0 || xy[1] >= 1440) continue;
            pointerMove(world, { x: xy[0], y: xy[1], hover: true });
            world.step(0);
            await world.frameFence;
            target = world.resource(Grab).outlined;
            if (target) break;
        }
        if (!target || !world.has(target, Outline))
            throw new Error("physics grab outline not showing");
        console.log(
            JSON.stringify({
                adapter: {
                    vendor: device.adapterInfo.vendor,
                    architecture: device.adapterInfo.architecture,
                    device: device.adapterInfo.device,
                    description: device.adapterInfo.description,
                    isFallbackAdapter: device.adapterInfo.isFallbackAdapter,
                },
                scene: "boxStack",
                resolution: [2560, 1440],
                target,
                outline: world.get(target, Outline),
                warmup: 40,
                frames: 200,
            }),
        );
        const querySet = device.createQuerySet({ type: "timestamp", count: 32 });
        const resolve = device.createBuffer({
            size: 256,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        const read = device.createBuffer({
            size: 256,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        let count = 0;
        world.gpu.span = (label) => {
            if (label !== "outline:jfa") return undefined;
            const first = count;
            count += 2;
            return { querySet, beginningOfPassWriteIndex: first, endOfPassWriteIndex: first + 1 };
        };
        const samples: number[] = [];
        for (let frame = 0; frame < 240; frame++) {
            count = 0;
            world.step(0);
            await world.frameFence;
            if (!count) throw new Error("missing outline:jfa");
            const encoder = device.createCommandEncoder();
            encoder.resolveQuerySet(querySet, 0, count, resolve, 0);
            encoder.copyBufferToBuffer(resolve, 0, read, 0, count * 8);
            device.queue.submit([encoder.finish()]);
            await read.mapAsync(GPUMapMode.READ);
            const values = new BigUint64Array(read.getMappedRange());
            let ns = 0;
            for (let i = 0; i < count; i += 2) ns += Number(values[i + 1]! - values[i]!);
            read.unmap();
            if (frame >= 40) samples.push(ns / 1e6);
        }
        const sorted = [...samples].sort((a, b) => a - b);
        console.log(
            JSON.stringify({
                span: "outline:jfa",
                passes: count / 2,
                n: samples.length,
                meanMs: samples.reduce((a, b) => a + b, 0) / samples.length,
                medianMs: sorted[Math.floor(sorted.length / 2)],
                p95Ms: sorted[Math.floor(sorted.length * 0.95)],
            }),
        );
        world.gpu.span = undefined;
        querySet.destroy();
        resolve.destroy();
        read.destroy();
    } finally {
        app.dispose();
    }
});
