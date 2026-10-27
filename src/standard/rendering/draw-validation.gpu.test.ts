import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { gpuScene } from "../../../scripts/gpu.fixture";
import { expectError } from "../../../scripts/test-console";
import { MeshInstance } from "../../core/mesh";
import {
    attachCanvas,
    attachTexture,
    Camera,
    captureFrame,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import type { World } from "../../engine";
import { CanvasContext } from "../../engine/app/canvas.fixture";
import { observeDevice } from "../../engine/runtime";
import {
    MaterialPlugin,
    MeshMaterial,
    MeshRenderPlugin,
    materialLayout,
    materialType,
    StandardRenderer,
    StandardRenderingPlugin,
} from "./index";

setDefaultTimeout(TIMEOUT.gpu);
// A filtering binding handed a comparison sampler fails bind group validation in the frame's draw.
const Owned = {
    create: (world: World) => {
        const device = world.gpu.device;
        const image = device.createTexture({
            size: [1, 1],
            format: "rgba8unorm",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        world.own(image);
        device.queue.writeTexture(
            { texture: image },
            new Uint8Array([0, 255, 0, 255]),
            { bytesPerRow: 4 },
            [1, 1],
        );
        return { image, linear: device.createSampler({ magFilter: "linear" }) };
    },
};
const layout = materialLayout(d.struct({ unused: d.f32 }), {
    image: { type: "texture" },
    linear: { type: "sampler" },
});
const Sampled = materialType({
    name: "Sampled",
    layout,
    defaults: { unused: 0 },
    bindings: (world) => world.resource(Owned),
    fragment: () => {
        "use gpu";
        return std.textureSample(layout.$.image, layout.$.linear, d.vec2f(0.5));
    },
});
const subject = gpuScene(
    import.meta.path,
    {
        defaults: false,
        plugins: [StandardRenderingPlugin, MeshRenderPlugin, MaterialPlugin(Sampled)],
    },
    (world) => {
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 5, 0] });
        world.add(camera, Camera, { antialias: 0 });
        world.add(camera, StandardRenderer);
        world.add(camera, Tonemapping, { method: TonemappingMethod.None });
        attachTexture(world, camera, { width: 4, height: 4 });
        const canvas = {
            width: 4,
            height: 4,
            style: {},
            getBoundingClientRect: () => ({ width: 4, height: 4 }),
        } as unknown as HTMLCanvasElement;
        const context = new CanvasContext(canvas, 4, 4);
        canvas.getContext = (() => context) as unknown as typeof canvas.getContext;
        context.configure({
            device: world.gpu.device,
            format: "bgra8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });
        world.onDispose(() => context.unconfigure());
        const canvasCamera = world.create();
        world.add(canvasCamera, Transform, { translation: [0, 0, 5, 0] });
        world.add(canvasCamera, Camera, { antialias: 0 });
        world.add(canvasCamera, StandardRenderer);
        attachCanvas(world, canvasCamera, canvas);
        const mesh = world.create();
        world.add(mesh, Transform, { scale: [20, 20, 1, 0] });
        world.add(mesh, MeshInstance);
        world.add(mesh, MeshMaterial, world.resource(Sampled).add({ unused: 0 }));
        return { camera, canvas };
    },
);

const settle = (promise: Promise<unknown>) =>
    promise.then(
        (value) => value,
        (error) => error,
    );

let failing: Promise<{ valid: unknown; queued: Promise<unknown>; invalid: unknown }> | undefined;

/** Draw one valid frame, then one whose draw fails validation, once per file; the device stays failed. */
function failedFrame() {
    failing ??= (async () => {
        const {
            world,
            scene: { camera, canvas },
        } = subject();
        world.step(0);
        const valid = await settle(captureTexture(world, camera));
        expectError(
            /GpuValidationError\]: Comparison sampler .* is incompatible with non-comparison sampler binding/,
        );
        await world.frameFence;
        canvas.getContext("webgpu")!.configure({
            device: world.gpu.device,
            format: "bgra8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });
        const queued = settle(captureFrame(canvas));
        world.resource(Owned).linear = world.gpu.device.createSampler({ compare: "less" });
        world.step(0);
        const invalid = await settle(captureTexture(world, camera));
        // the device's one classified report, raised after its selection window
        await observeDevice(world.gpu.device).error;
        return { valid, queued, invalid };
    })();
    return failing;
}

test("a capture after a frame whose draw failed validation rejects instead of reading the stale target", async () => {
    const { valid, invalid } = await failedFrame();
    if (valid instanceof Error) throw valid;
    expect(Array.from((valid as { rgba: Uint8ClampedArray }).rgba.subarray(0, 4))).toEqual([
        0, 255, 0, 255,
    ]);
    expect(invalid).toBeInstanceOf(Error);
    expect((invalid as Error).message).toMatch(
        /readback failed: Comparison sampler .* is incompatible with non-comparison sampler binding/,
    );
});

test("a capture encoded into an invalid frame rejects instead of returning zero bytes", async () => {
    const result = await (await failedFrame()).queued;
    expect(result).toBeInstanceOf(Error);
    expect((result as Error).message).toMatch(
        /canvas frame capture: .*Comparison sampler .* is incompatible/,
    );
});

test("a bound canvas capture after a failed frame rejects without another step", async () => {
    await failedFrame();
    const {
        scene: { canvas },
    } = subject();
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
        await expect(
            Promise.race([
                captureFrame(canvas),
                new Promise<never>((_, reject) => {
                    timer = setTimeout(() => reject(new Error("captureFrame did not settle")), 200);
                }),
            ]),
        ).rejects.toThrow(/canvas frame capture: .*Comparison sampler .* is incompatible/);
    } finally {
        clearTimeout(timer);
    }
});

test("a capture after the failed device is lost still names the error that stopped it", async () => {
    await failedFrame();
    const {
        world,
        scene: { camera },
    } = subject();
    world.gpu.device.destroy();
    await world.gpu.device.lost;
    await expect(captureTexture(world, camera)).rejects.toThrow(
        /readback failed: Comparison sampler .* is incompatible with non-comparison sampler binding/,
    );
});
