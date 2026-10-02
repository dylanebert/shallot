import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { attachCanvas, Camera, RenderingPlugin } from "../../core/rendering";
import { CanvasContext } from "../app/canvas.fixture";
import { Time, Transform, type World } from "../index";

setDefaultTimeout(CEILING.gpu);
if (typeof ResizeObserver === "undefined") {
    Object.assign(globalThis, {
        ResizeObserver: class {
            observe() {}
            unobserve() {}
            disconnect() {}
        },
    });
}

import { gpuApps } from "../../../scripts/gpu.fixture";

const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [] },
    { defaults: false, plugins: [RenderingPlugin] },
]);

function attachTestCamera(world: World): void {
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
    world.storage(Transform).translation.set(camera, 0, 0, 5, 0);
    attachCanvas(camera, canvas, world);
}

for (const renderer of [false, true]) {
    test(
        renderer
            ? "GlobalTransform history and interpolation share the frame submission across catch-up ticks"
            : "a world with no interpolated GlobalTransform reader runs no GlobalTransform GPU work",
        async () => {
            const app = subjects()[renderer ? 1 : 0];
            const world = app.world;
            const eid = world.create();
            world.add(eid, Transform);
            world.storage(Transform).translation.set(eid, 3, 2, 1, 0);
            if (renderer) attachTestCamera(world);
            // Warm allocation/growth is not the stepped submission under observation.
            world.step(0);
            const device = world.gpu.device;
            const queue = device.queue;
            const encoderDescriptor = Object.getOwnPropertyDescriptor(
                device,
                "createCommandEncoder",
            );
            const submitDescriptor = Object.getOwnPropertyDescriptor(queue, "submit");
            const writeDescriptor = Object.getOwnPropertyDescriptor(queue, "writeBuffer");
            const create = device.createCommandEncoder.bind(device);
            const submit = queue.submit.bind(queue);
            const write = queue.writeBuffer.bind(queue);
            let encoders = 0,
                submissions = 0,
                globalTransformWrites = 0,
                copies = 0;
            // Only this world steps while the shared-device counters are installed.
            Object.defineProperty(device, "createCommandEncoder", {
                configurable: true,
                value: (...args: Parameters<GPUDevice["createCommandEncoder"]>) => {
                    encoders++;
                    const encoder = create(...args);
                    const copy = encoder.copyBufferToBuffer.bind(encoder);
                    Object.defineProperty(encoder, "copyBufferToBuffer", {
                        configurable: true,
                        value: (
                            ...copyArgs: Parameters<GPUCommandEncoder["copyBufferToBuffer"]>
                        ) => {
                            if (
                                copyArgs[0] === world.globalTransformRuntime!.current!.buffer &&
                                copyArgs[2] === world.globalTransformRuntime!.previous!.buffer
                            )
                                copies++;
                            return copy(...copyArgs);
                        },
                    });
                    return encoder;
                },
            });
            Object.defineProperty(queue, "submit", {
                configurable: true,
                value: (...args: Parameters<GPUQueue["submit"]>) => {
                    submissions++;
                    return submit(...args);
                },
            });
            Object.defineProperty(queue, "writeBuffer", {
                configurable: true,
                value: (...args: Parameters<GPUQueue["writeBuffer"]>) => {
                    if (
                        args[0] === world.globalTransformRuntime!.current!.buffer ||
                        args[0] === world.globalTransformRuntime!.params!
                    )
                        globalTransformWrites++;
                    return write(...args);
                },
            });
            try {
                world.storage(Transform).translation.set(eid, 9, 8, 7, 0);
                world.step(Time.FIXED_DT * 2.5);
                expect(world.time.fixedSteps).toBe(2);
                expect(encoders).toBe(renderer ? 1 : 0);
                expect(submissions).toBe(renderer ? 1 : 0);
                expect(copies).toBe(renderer ? 1 : 0);
                if (!renderer) {
                    expect(globalTransformWrites).toBe(0);
                    expect(world.globalTransformRuntime!.enabled).toBe(false);
                    expect(world.globalTransformRuntime!.current).toBeUndefined();
                }
            } finally {
                for (const [object, key, descriptor] of [
                    [device, "createCommandEncoder", encoderDescriptor],
                    [queue, "submit", submitDescriptor],
                    [queue, "writeBuffer", writeDescriptor],
                ] as const) {
                    if (descriptor) Object.defineProperty(object, key, descriptor);
                    else Reflect.deleteProperty(object, key);
                }
                app.dispose();
            }
        },
    );
}
