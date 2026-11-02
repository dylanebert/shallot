import { expect, setDefaultTimeout, test } from "bun:test";
import type { World } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuScene } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import { attachTexture, Camera, DepthPrepass } from "../../core/rendering";
import { Transform } from "../../core/transform";
import {
    AlphaMode,
    AlphaPipelineKey,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "./index";

setDefaultTimeout(TIMEOUT.node);
const Shaders = { create: () => new Map<string, string>() };
const Observer = {
    name: "DiscardObservations",
    gpu: {},
    prepareGpu(world: World, gpu: World["gpu"]) {
        const modules = new Map<GPUShaderModule, string>();
        const device = gpu.device;
        const shader = device.createShaderModule.bind(device);
        device.createShaderModule = (desc) => {
            const module = shader(desc);
            modules.set(module, desc.code);
            return module;
        };
        const observe = (desc: GPURenderPipelineDescriptor) => {
            if (desc.fragment)
                world.resource(Shaders).set(desc.label!, modules.get(desc.fragment.module)!);
        };
        const pipeline = device.createRenderPipeline.bind(device);
        device.createRenderPipeline = (desc) => {
            observe(desc);
            return pipeline(desc);
        };
        const asyncPipeline = device.createRenderPipelineAsync.bind(device);
        device.createRenderPipelineAsync = (desc) => {
            observe(desc);
            return asyncPipeline(desc);
        };
    },
};
const setup = gpuScene(
    import.meta.path,
    {
        defaults: false,
        plugins: [Observer, StandardRenderingPlugin, MeshRenderPlugin],
    },
    (world) => {
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 5, 0] });
        world.add(camera, Camera);
        world.add(camera, StandardRenderer);
        world.add(camera, DepthPrepass);
        attachTexture(world, camera, { width: 32, height: 32 });
        const singleCamera = world.create();
        world.add(singleCamera, Transform, { translation: [0, 0, 5, 0] });
        world.add(singleCamera, Camera, { antialias: 0 });
        world.add(singleCamera, StandardRenderer);
        world.add(singleCamera, DepthPrepass);
        attachTexture(world, singleCamera, { width: 32, height: 32 });
        for (const alphaMode of [
            AlphaMode.Opaque,
            AlphaMode.Mask(0.5),
            AlphaMode.AlphaToCoverage,
            AlphaMode.Blend,
            AlphaMode.Premultiplied,
            AlphaMode.Multiply,
        ]) {
            const eid = world.create();
            world.add(eid, Transform);
            world.add(eid, MeshInstance);
            world.add(
                eid,
                MeshMaterial,
                world.resource(Materials).add(StandardMaterial(), { alphaMode }),
            );
        }
        return world.resource(Shaders);
    },
);
test("only masked fragments compile alpha-cutoff discards; opaque atlas shadows retain seam clipping", () => {
    const { world, scene: shaders } = setup();
    world.step(0);
    for (const key of Object.values(AlphaPipelineKey)) {
        for (const pass of [
            "standard-StandardMaterial",
            "standard-prepass-StandardMaterial",
            "standard-point-StandardMaterial",
            "standard-cascade-StandardMaterial",
        ]) {
            const blended =
                key === AlphaPipelineKey.Blend ||
                key === AlphaPipelineKey.Premultiplied ||
                key === AlphaPipelineKey.Multiply;
            if (blended && pass !== "standard-StandardMaterial") continue;
            const entries = [...shaders].filter(([name]) => name.startsWith(`${pass}-${key}`));
            expect(entries.length, `${pass}-${key}`).toBeGreaterThan(0);
            for (const [name, code] of entries) {
                const shadow = pass.includes("point") || pass.includes("cascade");
                const cutoff =
                    key === AlphaPipelineKey.Mask ||
                    (key === AlphaPipelineKey.AlphaToCoverage &&
                        (shadow || pass.includes("prepass") || name.includes("1x")));
                expect((code.match(/\bdiscard\b/g) ?? []).length, name).toBe(
                    Number(shadow) + Number(cutoff),
                );
                expect(/\.a < cutoff/.test(code), name).toBe(cutoff);
            }
        }
    }
});
