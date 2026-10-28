import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { gpuApps } from "../../../scripts/gpu.fixture";
import type { World } from "../../engine";
import {
    MaterialPlugin,
    MeshMaterial,
    MeshRenderPlugin,
    materialLayout,
    materialType,
    StandardRenderer,
    StandardRenderingPlugin,
} from "../../standard/rendering";
import { MeshInstance } from "../mesh";
import { Transform } from "../transform";
import {
    attachTexture,
    Camera,
    captureTexture,
    RenderLayers,
    Tonemapping,
    TonemappingMethod,
} from "./index";

setDefaultTimeout(TIMEOUT.node);
await setupGlobals();

const Source = {
    create(world: World) {
        const camera = world.create();
        world.add(camera, Transform);
        world.add(camera, Camera, { clearColor: 0xff00ff, order: -1 });
        world.add(camera, StandardRenderer);
        world.add(camera, RenderLayers, { mask: 2 });
        world.add(camera, Tonemapping, { method: TonemappingMethod.None });
        return { image: attachTexture(world, camera, { width: 4, height: 4 }) };
    },
};
const layout = materialLayout(d.struct({ unused: d.f32 }), { image: { type: "texture" } });
const Sampled = materialType({
    name: "CameraTextureSample",
    layout,
    defaults: { unused: 0 },
    bindings: (world) => world.resource(Source),
    fragment: () => {
        "use gpu";
        return std.textureLoad(layout.$.image, d.vec2i(2, 2), 0);
    },
});
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [StandardRenderingPlugin, MeshRenderPlugin, MaterialPlugin(Sampled)],
    },
]);

test("a material samples a camera's texture target after that camera draws", async () => {
    const { world } = subjects()[0];
    world.resource(Source);
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera, { clearColor: 0x000000 });
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 8, height: 8 });
    const mesh = world.create();
    world.add(mesh, Transform, { scale: [20, 20, 1, 0] });
    world.add(mesh, MeshInstance);
    world.add(mesh, MeshMaterial, world.resource(Sampled).add({ unused: 0 }));
    world.gpu.device.pushErrorScope("validation");
    world.step(0);
    world.step(0);
    const error = await world.gpu.device.popErrorScope();
    expect(error?.message ?? null).toBeNull();
    const { rgba } = await captureTexture(world, camera);
    const center = (4 * 8 + 4) * 4;
    expect(Array.from(rgba.subarray(center, center + 4))).toEqual([255, 0, 255, 255]);
});
