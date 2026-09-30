import { expect, setDefaultTimeout, test } from "bun:test";
import { attachCanvas, Camera } from "../../core/rendering";
import { SearPlugin } from "../../standard/rendering";
import "../../standard";
import { type State, Time, Transform } from "../index";
import { CanvasContext } from "./canvas.fixture";
import { build } from "./index";

setDefaultTimeout(20_000);
const peerModule = "bun-webgpu";
await (await import(peerModule)).setupGlobals();
if (typeof ResizeObserver === "undefined") {
    globalThis.ResizeObserver = class {
        observe() {}
        unobserve() {}
        disconnect() {}
    };
}

function cameraPlugin(label: string) {
    const canvas = {
        width: 16,
        height: 16,
        style: { imageRendering: "auto" },
        getBoundingClientRect: () => ({ width: 16, height: 16 }),
    } as unknown as HTMLCanvasElement;
    const context = new CanvasContext(canvas, 16, 16);
    canvas.getContext = ((kind: string) =>
        kind === "webgpu" ? context : null) as typeof canvas.getContext;
    return {
        name: label,
        dependencies: [SearPlugin],
        initialize(state: State) {
            const eid = state.create();
            state.add(eid, Transform);
            state.add(eid, Camera);
            attachCanvas(eid, canvas, state);
        },
    };
}

for (const sharedDevice of [true, false]) {
    test(`default renderer worlds step independently on ${sharedDevice ? "a shared device" : "separate devices"}`, async () => {
        const makeTrackedDevice = async () => {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) throw new Error("Dawn adapter unavailable");
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
                requiredFeatures: ["bgra8unorm-storage", "rg11b10ufloat-renderable"],
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
            let first: Awaited<ReturnType<typeof build>> | undefined;
            let second: Awaited<ReturnType<typeof build>> | undefined;
            try {
                first = await build({
                    plugins: [cameraPlugin("DefaultCameraA")],
                    device: firstDevice,
                });
                second = await build({
                    plugins: [cameraPlugin("DefaultCameraB")],
                    device: secondDevice,
                });
                expect(first.state.gpu.root).not.toBe(second.state.gpu.root);
                first.state.step(Time.FIXED_DT);
                second.state.step(Time.FIXED_DT);
                first.state.step(Time.FIXED_DT);
            } finally {
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
