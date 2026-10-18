import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import {
    attachTexture,
    Camera,
    CorePipelinePlugin,
    captureTexture,
    fullscreenVertex,
    RenderPhases,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { fogReadBindGroup, fogReadLayout, readFog, readFogInScatter } from ".";

setDefaultTimeout(TIMEOUT.node);
const subjects = gpuApps(import.meta.path, [{ defaults: false, plugins: [CorePipelinePlugin] }]);
const fragment = tgpu
    .fragmentFn({ out: d.vec4f })(() => {
        "use gpu";
        const fog = readFog(d.vec3f(0, 0, -2));
        const light = readFogInScatter(d.vec3f(0, 0, -2));
        if (
            fog.r === 0 &&
            fog.g === 0 &&
            fog.b === 0 &&
            fog.a === 1 &&
            light.x === 0 &&
            light.y === 0 &&
            light.z === 0
        )
            return d.vec4f(0, 1, 0, 1);
        return d.vec4f(1, 0, 0, 1);
    })
    .$name("identityFragment");

test("a core-only camera can bind and read identity fog without FogPlugin or standard lighting", async () => {
    const { world } = subjects()[0]!;
    const camera = world.create();
    world.add(camera, Transform);
    world.add(camera, Camera, { antialias: 0 });
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 8, height: 8 });
    world.step(0);
    const group = fogReadBindGroup(world, camera);
    expect(fogReadBindGroup(world, camera)).toBe(group);
    const device = world.gpu.device;
    const module = device.createShaderModule({
        code: tgpu.resolve([fullscreenVertex, fragment], { names: "strict" }),
    });
    const empty = device.createBindGroupLayout({ entries: [] });
    const pipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [empty, empty, empty, world.gpu.root.unwrap(fogReadLayout)],
        }),
        vertex: { module, entryPoint: "fullscreenVertex" },
        fragment: { module, entryPoint: "identityFragment", targets: [{ format: "rgba16float" }] },
        depthStencil: { format: "depth32float", depthWriteEnabled: false, depthCompare: "always" },
    });
    world.resource(RenderPhases).push({
        afterOpaque(_world, _eid, _view, pass) {
            pass.setPipeline(pipeline);
            pass.setBindGroup(3, group);
            pass.draw(3);
        },
    });
    world.step(0);
    const frame = await captureTexture(world, camera);
    expect(Array.from(frame.rgba.subarray((4 * 8 + 4) * 4, (4 * 8 + 4) * 4 + 4))).toEqual([
        0, 255, 0, 255,
    ]);
});
