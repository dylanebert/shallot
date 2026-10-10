import { expect, setDefaultTimeout, test } from "bun:test";
import { MeshInstance } from "../../core/mesh";
import { AmbientLight, attachTexture, Camera, captureTexture, Views } from "../../core/rendering";
import {
    Materials,
    MeshMaterial,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "../../standard/rendering";
import "../../standard";

import { CEILING } from "../../../scripts/test-tiers";
import { Transform } from "../../core/transform";
import { stampAdapter, Time, type World } from "../index";
import { createApp } from "./index";

setDefaultTimeout(CEILING.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();
function cameraPlugin(
    label: string,
    x: number,
    clearColor: number,
    color: readonly [number, number, number, number],
) {
    return {
        name: label,
        dependencies: [StandardRenderingPlugin],
        initialize(world: World) {
            const camera = world.create();
            world.add(camera, Transform, { translation: [x, 0, 5, 0] });
            world.add(camera, Camera, { clearColor });
            world.add(camera, AmbientLight, { brightness: 499.04787 });
            world.add(camera, StandardRenderer);
            attachTexture(world, camera, { width: 16, height: 16 });

            const mesh = world.create();
            world.add(mesh, Transform, { translation: [x, 0, 0, 0] });
            world.add(mesh, MeshInstance);
            const material = world
                .resource(Materials)
                .add(StandardMaterial({ baseColor: color, perceptualRoughness: 1 }));
            world.add(mesh, MeshMaterial, material);
        },
    };
}

for (const sharedDevice of [true, false]) {
    test(`default renderer worlds step independently on ${sharedDevice ? "a shared device" : "separate devices"}`, async () => {
        const makeTrackedDevice = async () => {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) throw new Error("Dawn adapter unavailable");
            console.info("default renderer adapter:", stampAdapter(adapter));
            const requiredLimits: Record<string, number> = { maxStorageBuffersPerShaderStage: 10 };
            for (const limit of [
                "maxStorageBuffersInVertexStage",
                "maxStorageBuffersInFragmentStage",
                "maxStorageTexturesInVertexStage",
                "maxStorageTexturesInFragmentStage",
            ] as const) {
                if (adapter.limits[limit] === 0) requiredLimits[limit] = 0;
            }
            const device = await adapter.requestDevice({
                requiredFeatures: ["indirect-first-instance", "rg11b10ufloat-renderable"],
                requiredLimits,
            });
            const live = new Set<GPUBuffer | GPUTexture>();
            const createBuffer = device.createBuffer.bind(device);
            const createTexture = device.createTexture.bind(device);
            Object.defineProperties(device, {
                createBuffer: {
                    configurable: true,
                    value: (descriptor: GPUBufferDescriptor) => {
                        const buffer = createBuffer(descriptor);
                        live.add(buffer);
                        const destroy = buffer.destroy.bind(buffer);
                        buffer.destroy = () => {
                            if (live.delete(buffer)) destroy();
                        };
                        return buffer;
                    },
                },
                createTexture: {
                    configurable: true,
                    value: (descriptor: GPUTextureDescriptor) => {
                        const texture = createTexture(descriptor);
                        live.add(texture);
                        const destroy = texture.destroy.bind(texture);
                        texture.destroy = () => {
                            if (live.delete(texture)) destroy();
                        };
                        return texture;
                    },
                },
            });
            return { device, live };
        };

        const exercise = async (
            firstDevice: GPUDevice,
            secondDevice: GPUDevice,
            live: Set<GPUBuffer | GPUTexture>,
        ) => {
            let first: Awaited<ReturnType<typeof createApp>> | undefined;
            let second: Awaited<ReturnType<typeof createApp>> | undefined;
            const queues = new Map<
                GPUQueue,
                {
                    descriptor: PropertyDescriptor | undefined;
                    submit: GPUQueue["submit"];
                    count: number;
                }
            >();
            try {
                first = await createApp({
                    plugins: [cameraPlugin("DefaultCameraA", 0, 0x204060, [0.8, 0.2, 0.1, 1])],
                    device: firstDevice,
                });
                second = await createApp({
                    plugins: [cameraPlugin("DefaultCameraB", 1, 0x603020, [0.1, 0.2, 0.8, 1])],
                    device: secondDevice,
                });
                expect(first.world.gpu.root).not.toBe(second.world.gpu.root);
                expect(first.world.resource(Views)).not.toBe(second.world.resource(Views));
                for (const world of [first.world, second.world]) {
                    const queue = world.gpu.device.queue;
                    if (queues.has(queue)) continue;
                    const tracked = {
                        descriptor: Object.getOwnPropertyDescriptor(queue, "submit"),
                        submit: queue.submit.bind(queue),
                        count: 0,
                    };
                    Object.defineProperty(queue, "submit", {
                        configurable: true,
                        value: (...args: Parameters<GPUQueue["submit"]>) => {
                            tracked.count++;
                            return tracked.submit(...args);
                        },
                    });
                    queues.set(queue, tracked);
                }
                const step = (world: World) => {
                    const tracked = queues.get(world.gpu.device.queue)!;
                    const before = tracked.count;
                    world.step(Time.FIXED_DT);
                    expect(tracked.count - before).toBe(1);
                    expect(world.frameFence).toBeDefined();
                };
                step(first.world);
                step(second.world);
                step(first.world);
                await Promise.all([first.world.frameFence!, second.world.frameFence!]);

                if (sharedDevice) {
                    const firstCamera = [...first.world.query([Camera])][0]!;
                    const secondCamera = [...second.world.query([Camera])][0]!;
                    const firstMesh = [...first.world.query([MeshInstance])][0]!;
                    expect(first.world.resource(Views).get(firstCamera)).not.toBe(
                        second.world.resource(Views).get(secondCamera),
                    );
                    const firstTexture = first.world.resource(Views).get(firstCamera)!.texture!;
                    expect(firstTexture).not.toBe(
                        second.world.resource(Views).get(secondCamera)!.texture,
                    );
                    const firstFrame = (await captureTexture(first.world, firstCamera)).rgba;
                    const secondFrame = (await captureTexture(second.world, secondCamera)).rgba;
                    expect(firstFrame).not.toEqual(secondFrame);

                    first.world.remove(firstMesh, MeshInstance);
                    step(first.world);
                    const sceneChanged = (await captureTexture(first.world, firstCamera)).rgba;
                    expect(sceneChanged).not.toEqual(firstFrame);
                    step(second.world);
                    expect((await captureTexture(second.world, secondCamera)).rgba).toEqual(
                        secondFrame,
                    );

                    first.world.storage(Camera).clearColor.set(firstCamera, 0x20c040);
                    step(first.world);
                    expect((await captureTexture(first.world, firstCamera)).rgba).not.toEqual(
                        sceneChanged,
                    );
                    step(second.world);
                    expect((await captureTexture(second.world, secondCamera)).rgba).toEqual(
                        secondFrame,
                    );

                    first.dispose();
                    first = undefined;
                    expect(live.has(firstTexture)).toBe(false);
                    step(second.world);
                    expect((await captureTexture(second.world, secondCamera)).rgba).toEqual(
                        secondFrame,
                    );
                }
            } finally {
                for (const [queue, tracked] of queues) {
                    if (tracked.descriptor)
                        Object.defineProperty(queue, "submit", tracked.descriptor);
                    else Reflect.deleteProperty(queue, "submit");
                }
                second?.dispose();
                first?.dispose();
            }
            expect(live.size).toBe(0);
        };

        const first = await makeTrackedDevice();
        const second = sharedDevice ? first : await makeTrackedDevice();
        try {
            await exercise(first.device, second.device, first.live);
            expect(second.live.size).toBe(0);
        } finally {
            first.device.destroy();
            if (!sharedDevice) second.device.destroy();
        }
    });
}
