import { expect, setDefaultTimeout, test } from "bun:test";
import { attachCanvas, Camera, RenderPlugin } from "../../core/rendering";
import { CanvasContext } from "../app/canvas.fixture";
import { build, type State, Time, Transform } from "../index";

setDefaultTimeout(20_000);
if (typeof ResizeObserver === "undefined") {
    Object.assign(globalThis, {
        ResizeObserver: class {
            observe() {}
            unobserve() {}
            disconnect() {}
        },
    });
}
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();

function attachTestCamera(state: State): void {
    let context: CanvasContext;
    const canvas = {
        width: 32,
        height: 24,
        style: { imageRendering: "auto" },
        getContext: () => context,
        getBoundingClientRect: () => ({ width: 32, height: 24 }),
    } as unknown as HTMLCanvasElement;
    context = new CanvasContext(canvas, 32, 24);
    const camera = state.create();
    state.add(camera, Transform);
    state.add(camera, Camera);
    state.of(Transform).pos.set(camera, 0, 0, 5, 0);
    attachCanvas(camera, canvas, state);
}

for (const renderer of [false, true]) {
    test(
        renderer
            ? "GlobalTransform history and interpolation share the frame submission across catch-up ticks"
            : "a world with no interpolated GlobalTransform reader runs no GlobalTransform GPU work",
        async () => {
            const app = await build({ defaults: false, plugins: renderer ? [RenderPlugin] : [] });
            const state = app.state;
            const eid = state.create();
            state.add(eid, Transform);
            state.of(Transform).pos.set(eid, 3, 2, 1, 0);
            if (renderer) attachTestCamera(state);
            // Warm allocation/growth is not the stepped submission under observation.
            state.step(0);
            const device = state.gpu.device;
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
                                copyArgs[0] === state.globalTransformRuntime!.current!.buffer &&
                                copyArgs[2] === state.globalTransformRuntime!.previous!.buffer
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
                        args[0] === state.globalTransformRuntime!.current!.buffer ||
                        args[0] === state.globalTransformRuntime!.params!
                    )
                        globalTransformWrites++;
                    return write(...args);
                },
            });
            try {
                state.of(Transform).pos.set(eid, 9, 8, 7, 0);
                state.step(Time.FIXED_DT * 2.5);
                expect(state.time.fixedSteps).toBe(2);
                expect(encoders).toBe(renderer ? 1 : 0);
                expect(submissions).toBe(renderer ? 1 : 0);
                expect(copies).toBe(renderer ? 1 : 0);
                if (!renderer) {
                    expect(globalTransformWrites).toBe(0);
                    expect(state.globalTransformRuntime!.enabled).toBe(false);
                    expect(state.globalTransformRuntime!.current).toBeUndefined();
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
