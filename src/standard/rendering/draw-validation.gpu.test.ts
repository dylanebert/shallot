import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { gpuScene } from "../../../scripts/gpu.fixture";
import { expectError } from "../../../scripts/test-console";
import { MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import type { World } from "../../engine";
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
    image: { type: "texture-2d" },
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
        const mesh = world.create();
        world.add(mesh, Transform, { scale: [20, 20, 1, 0] });
        world.add(mesh, MeshInstance);
        world.add(mesh, MeshMaterial, world.resource(Sampled).add({ unused: 0 }));
        return camera;
    },
);

test("a capture after a frame whose draw failed validation rejects instead of reading the stale target", async () => {
    const { world, scene: camera } = subject();
    world.step(0);
    expect(Array.from((await captureTexture(world, camera)).rgba.subarray(0, 4))).toEqual([
        0, 255, 0, 255,
    ]);
    expectError(
        /GpuValidationError\]: Comparison sampler .* is incompatible with non-comparison sampler binding/,
    );
    world.resource(Owned).linear = world.gpu.device.createSampler({ compare: "less" });
    world.step(0);
    await expect(captureTexture(world, camera)).rejects.toThrow(
        /readback failed: Comparison sampler .* is incompatible with non-comparison sampler binding/,
    );
    // the device's one classified report, raised after its selection window
    await observeDevice(world.gpu.device).error;
});

test("a capture after the failed device is lost still names the error that stopped it", async () => {
    // the device the previous test stopped
    const { world, scene: camera } = subject();
    world.gpu.device.destroy();
    await world.gpu.device.lost;
    await expect(captureTexture(world, camera)).rejects.toThrow(
        /readback failed: Comparison sampler .* is incompatible with non-comparison sampler binding/,
    );
});
