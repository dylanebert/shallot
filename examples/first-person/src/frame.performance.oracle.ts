import { expect, test } from "bun:test";
import {
    AmbientLight,
    Camera,
    CharacterPlugin,
    createApp,
    Materials,
    MeshInstance,
    MeshMaterial,
    PlayerPlugin,
    StandardMaterial,
    StandardPhysicsPlugin,
    StandardRenderer,
    Time,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { attachTexture } from "@dylanebert/shallot/rendering";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { Demo } from "./demo";

await bounded("frame probe WebGPU setup", setupGlobals());
function bounded<T>(label: string, promise: PromiseLike<T>): Promise<T> {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`${label} timed out after 5000 ms`)), 5000);
        promise.then(
            (value) => {
                clearTimeout(timer);
                resolve(value);
            },
            (error) => {
                clearTimeout(timer);
                reject(error);
            },
        );
    });
}
function median(values: number[]): number {
    return values.toSorted((a, b) => a - b)[Math.floor(values.length / 2)]!;
}

test("report production frame GPU time for first-person and a 10k-instance scene", async () => {
    const adapter = await bounded("frame probe adapter", navigator.gpu.requestAdapter());
    if (!adapter) throw new Error("frame probe requires a real adapter");
    const identity = `${adapter.info.vendor} ${adapter.info.architecture} ${adapter.info.device} ${adapter.info.description}`;
    console.info(`[frame-perf] adapter=${identity} resolution=1280x720`);
    expect(identity.toLowerCase()).not.toContain("swiftshader");
    const requiredLimits: Record<string, number> = { maxStorageBuffersPerShaderStage: 10 };
    for (const limit of [
        "maxStorageBuffersInVertexStage",
        "maxStorageBuffersInFragmentStage",
        "maxStorageTexturesInVertexStage",
        "maxStorageTexturesInFragmentStage",
    ] as const) {
        if (adapter.limits[limit] === 0) requiredLimits[limit] = 0;
    }
    const device = await bounded(
        "frame probe device",
        adapter.requestDevice({
            requiredFeatures: ["timestamp-query", "rg11b10ufloat-renderable"],
            requiredLimits,
        }),
    );
    device.addEventListener("uncapturederror", (event) => {
        throw new Error(event.error.message);
    });
    const querySet = device.createQuerySet({ type: "timestamp", count: 512 });
    const resolve = device.createBuffer({
        size: 4096,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const readback = device.createBuffer({
        size: 4096,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    let active = false;
    let slots = 0;
    const labels: string[] = [];
    const createEncoder = device.createCommandEncoder.bind(device);
    Object.defineProperty(device, "createCommandEncoder", {
        configurable: true,
        value: (descriptor?: GPUCommandEncoderDescriptor) => {
            const encoder = createEncoder(descriptor);
            const render = encoder.beginRenderPass.bind(encoder);
            const compute = encoder.beginComputePass.bind(encoder);
            function writes(): GPUComputePassTimestampWrites {
                if (slots + 2 > 512) throw new Error("frame probe exhausted timestamp slots");
                const start = slots;
                slots += 2;
                return {
                    querySet,
                    beginningOfPassWriteIndex: start,
                    endOfPassWriteIndex: start + 1,
                };
            }
            Object.defineProperty(encoder, "beginRenderPass", {
                value: (d: GPURenderPassDescriptor) => {
                    if (active) labels.push(d.label ?? "render");
                    return render(active ? { ...d, timestampWrites: writes() } : d);
                },
            });
            Object.defineProperty(encoder, "beginComputePass", {
                value: (d?: GPUComputePassDescriptor) => {
                    if (active) labels.push(d?.label ?? "compute");
                    return compute(active ? { ...d, timestampWrites: writes() } : d);
                },
            });
            return encoder;
        },
    });
    const stress = (world: World) => {
        const camera = world.create();
        world.add(camera, Camera);
        world.add(camera, AmbientLight, { brightness: 798.4766 });
        world.add(camera, StandardRenderer);
        world.add(camera, Transform, { translation: [0, 0, 90, 0] });
        for (let i = 0; i < 10000; i++) {
            const eid = world.create();
            world.add(eid, MeshInstance);
            world.add(eid, Transform, {
                translation: [((i % 100) - 50) * 0.4, Math.floor(i / 100 - 50) * 0.4, 0, 0],
                scale: [0.15, 0.15, 0.15, 1],
            });
            const material = world
                .resource(Materials)
                .add(StandardMaterial({ baseColor: [0.3, 0.6, 0.8, 1] }));
            world.add(eid, MeshMaterial, material);
        }
    };
    try {
        for (const name of ["first-person", "stress-10k"] as const) {
            const app = await createApp({
                device,
                plugins:
                    name === "first-person"
                        ? [StandardPhysicsPlugin, CharacterPlugin, PlayerPlugin, Demo]
                        : [],
                setup: name === "stress-10k" ? stress : undefined,
            });
            try {
                const camera = [...app.world.query([Camera])][0];
                if (camera === undefined) throw new Error(`${name} has no camera`);
                attachTexture(app.world, camera, { width: 1280, height: 720 });
                for (let frame = 0; frame < 30; frame++) app.world.step(Time.FIXED_DT);
                await bounded(`${name} warmup completion`, device.queue.onSubmittedWorkDone());
                for (let run = 0; run < 3; run++) {
                    const elapsed: number[] = [];
                    const passSums: number[] = [];
                    for (let frame = 0; frame < 60; frame++) {
                        device.pushErrorScope("validation");
                        slots = 0;
                        labels.length = 0;
                        active = true;
                        try {
                            app.world.step(Time.FIXED_DT);
                        } finally {
                            active = false;
                        }
                        if (!slots) throw new Error(`${name} produced no GPU passes`);
                        const encoder = createEncoder({ label: `${name}-frame-timestamps` });
                        encoder.resolveQuerySet(querySet, 0, slots, resolve, 0);
                        encoder.copyBufferToBuffer(resolve, 0, readback, 0, slots * 8);
                        device.queue.submit([encoder.finish()]);
                        const error = await bounded(
                            `${name} frame validation`,
                            device.popErrorScope(),
                        );
                        if (error) throw new Error(error.message);
                        await bounded(
                            `${name} frame timestamp readback`,
                            readback.mapAsync(GPUMapMode.READ),
                        );
                        const times = new BigUint64Array(readback.getMappedRange());
                        let start = times[0]!;
                        let end = times[1]!;
                        let sum = 0;
                        for (let i = 0; i < slots; i += 2) {
                            start = times[i]! < start ? times[i]! : start;
                            end = times[i + 1]! > end ? times[i + 1]! : end;
                            sum += Number(times[i + 1]! - times[i]!);
                        }
                        elapsed.push(Number(end - start) / 1e6);
                        passSums.push(sum / 1e6);
                        readback.unmap();
                    }
                    console.info(
                        `[frame-perf] scene=${name} run=${run} frame-span-ms=${median(elapsed)} pass-sum-ms=${median(passSums)} min/max-ms=${Math.min(...elapsed)}/${Math.max(...elapsed)} passes=${slots / 2} labels=${labels.join(",")}`,
                    );
                }
            } finally {
                app.dispose();
            }
        }
    } finally {
        querySet.destroy();
        resolve.destroy();
        readback.destroy();
        device.destroy();
    }
}, 0);
