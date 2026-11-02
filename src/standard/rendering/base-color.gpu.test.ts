import { beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    captureTexture,
    DepthPrepass,
    DirectionalLight,
    Views,
} from "../../core/rendering";
import { imageTextures } from "../../core/rendering/image";
import { Transform } from "../../core/transform";
import { lookAtRotation, precompile, type World } from "../../engine";
import { probeBuffer } from "../../engine/runtime";
import { loadGltfScene, poseGltfScene, spawnGltfScene } from "../../extras/gltf/scene";
import { GltfPlugin, SkinnedMaterial } from "../../extras/gltf/skinning";
import { skinningFixtureUrl } from "../../extras/gltf/skinning.fixture";
import { sunShadowView } from "./atlas";
import {
    AlphaMode,
    applyPbrLighting,
    BaseColor,
    extendedMaterial,
    MaterialPlugin,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    pbrInputFromStandardMaterial,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "./index";
import { MeshDrawBuffers } from "./preprocess";

setDefaultTimeout(TIMEOUT.gpu);
const Extended = extendedMaterial(Materials, {
    name: "BaseColorExtended",
    fragment: (ctx) => {
        "use gpu";
        return applyPbrLighting(pbrInputFromStandardMaterial(ctx));
    },
});
const ExtendedPlugin = MaterialPlugin(Extended);
const cases = ["skinned", "lit", "unlit", "textured", "masked", "blended", "extended"] as const;
const SIDE = 256;
const depthLayout = tgpu.bindGroupLayout({
    view: { texture: d.textureDepth2d() },
    sun: { texture: d.textureDepth2d() },
    out: { storage: d.arrayOf(d.f32), access: "mutable" },
});
const sampleDepth = tgpu
    .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [8, 8] })(`{
    let i = in.gid.y * ${SIDE}u + in.gid.x;
    bound.out[i] = textureLoad(bound.view, vec2i(in.gid.xy * textureDimensions(bound.view) / ${SIDE}u), 0);
    bound.out[${SIDE * SIDE}u + i] = textureLoad(bound.sun, vec2i(in.gid.xy * textureDimensions(bound.sun) / ${SIDE}u), 0);
}`)
    .$uses({ bound: depthLayout.$ });
const DepthProbe = {
    create(world: World) {
        const pipeline = world.gpu.root.createComputePipeline({ compute: sampleDepth });
        precompile(world, "base-color-depth-probe", () => [pipeline]);
        const out = world.gpu.root
            .createBuffer(d.arrayOf(d.f32, 2 * SIDE * SIDE))
            .$usage("storage");
        return { pipeline, out };
    },
};
async function depths(world: World, camera: number) {
    const { pipeline, out } = world.resource(DepthProbe);
    const group = world.gpu.root.createBindGroup(depthLayout, {
        view: world.resource(Views).get(camera)!.depth!,
        sun: sunShadowView(world)!,
        out,
    });
    return (
        await probeBuffer(world, world.gpu.root.unwrap(out), {
            encode(encoder) {
                const pass = encoder.beginComputePass();
                pass.setPipeline(world.gpu.root.unwrap(pipeline));
                pass.setBindGroup(0, world.gpu.root.unwrap(group));
                pass.dispatchWorkgroups(SIDE / 8, SIDE / 8);
                pass.end();
            },
        })
    ).bytes;
}
const color = [0.2, 0.7, 0.4, 0.3] as const;
const scenes: { camera: number; entity: number; reference: import("./index").MaterialHandle }[] =
    [];
const apps = gpuApps(
    import.meta.path,
    cases.map((mode, index) => ({
        defaults: false,
        plugins: [
            StandardRenderingPlugin,
            MeshRenderPlugin,
            ExtendedPlugin,
            GltfPlugin,
            {
                name: `BaseColorScene-${mode}`,
                dependencies: [
                    StandardRenderingPlugin,
                    MeshRenderPlugin,
                    ExtendedPlugin,
                    GltfPlugin,
                ],
                async initialize(world: World) {
                    const camera = world.create();
                    world.add(camera, Transform, { translation: [0, 0, 5] });
                    world.add(camera, Camera, { antialias: 0 });
                    world.add(camera, StandardRenderer);
                    world.add(camera, DepthPrepass);
                    world.add(camera, AmbientLight, { brightness: 249.52394 });
                    attachTexture(world, camera, { width: 64, height: 32 });
                    const sun = world.create();
                    world.add(sun, Transform, {
                        rotation: lookAtRotation(0, 0, 0, -0.4, -1, -0.3),
                    });
                    world.add(sun, DirectionalLight, {
                        shadowMapsEnabled: 1,
                        numCascades: 1,
                        maximumDistance: 10,
                    });
                    const textured = mode === "textured" || mode === "masked";
                    if (textured) {
                        const texture = world.gpu.device.createTexture({
                            size: [1, 1],
                            format: "rgba8unorm-srgb",
                            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
                        });
                        world.own(texture);
                        world.gpu.device.queue.writeTexture(
                            { texture },
                            new Uint8Array([180, 220, 140, 200]),
                            { bytesPerRow: 4 },
                            [1, 1],
                        );
                        world.resource(imageTextures).textures[0][0] = texture;
                    }
                    const type =
                        mode === "extended"
                            ? Extended
                            : mode === "skinned"
                              ? SkinnedMaterial
                              : Materials;
                    const values = {
                        unlit: mode === "unlit" || mode === "skinned",
                        baseColorTexture: textured ? 0 : -1,
                    };
                    const options = {
                        alphaMode:
                            mode === "masked"
                                ? AlphaMode.Mask(0.25)
                                : mode === "blended"
                                  ? AlphaMode.Blend
                                  : AlphaMode.Opaque,
                        alphaCutoff: 0.25,
                        cullMode: "none" as const,
                    };
                    const shared = world
                        .resource(type)
                        .add(
                            { ...StandardMaterial({ ...values, baseColor: [0.9, 0.1, 0.8, 0.8] }) },
                            options,
                        );
                    const reference = world
                        .resource(type)
                        .add({ ...StandardMaterial({ ...values, baseColor: color }) }, options);
                    let entity = -1;
                    if (mode === "skinned") {
                        const gltf = await loadGltfScene(world, skinningFixtureUrl());
                        for (const x of [-0.7, 0.7]) {
                            const instance = spawnGltfScene(world, gltf);
                            poseGltfScene(world, instance, gltf.animations[0], 1);
                            const eid = instance.entities[0];
                            world.storage(Transform).translation.set(eid, x, 0, 0);
                            world.storage(Transform).scale.set(eid, 0.4, 0.4, 0.4);
                            world.remove(eid, MeshMaterial);
                            world.add(eid, MeshMaterial, shared);
                            if (entity < 0) entity = eid;
                        }
                    } else {
                        for (const x of [-0.7, 0.7]) {
                            const eid = world.create();
                            world.add(eid, Transform, {
                                translation: [x, 0, 0],
                                scale: [0.5, 0.5, 0.5],
                            });
                            world.add(eid, MeshInstance);
                            world.add(eid, MeshMaterial, shared);
                            if (entity < 0) entity = eid;
                        }
                    }
                    world.add(entity, BaseColor, { color });
                    scenes[index] = { camera, entity, reference };
                    world.resource(DepthProbe);
                },
            },
        ],
    })),
);
beforeAll(() => {
    for (const { world } of apps()) {
        world.step(0);
        world.step(0);
    }
}, TIMEOUT.startup);

for (const [index, mode] of cases.entries()) {
    test(`${mode}: BaseColor replaces material RGBA beside an unchanged instance in one draw`, async () => {
        const { world } = apps()[index];
        const { camera, entity, reference } = scenes[index];
        const varied = (await captureTexture(world, camera)).rgba;
        const args = new Uint32Array(
            (
                await probeBuffer(
                    world,
                    world.gpu.root.unwrap(world.resource(MeshDrawBuffers).drawArgs!),
                )
            ).bytes,
        );
        const populated = [];
        for (let i = 0; i < args.length; i += 5) if (args[i + 1]) populated.push(args[i + 1]);
        if (mode !== "blended") expect(populated).toEqual([2, 2]);
        const variedDepths = await depths(world, camera);
        const values = new Float32Array(variedDepths);
        expect(values.subarray(0, SIDE * SIDE).some((v) => v > 0)).toBe(mode !== "blended");
        if (mode !== "blended") expect(values.subarray(SIDE * SIDE).some((v) => v > 0)).toBe(true);
        world.remove(entity, BaseColor);
        world.remove(entity, MeshMaterial);
        world.add(entity, MeshMaterial, reference);
        world.step(0);
        const expected = (await captureTexture(world, camera)).rgba;
        expect(varied).toEqual(expected);
        expect(variedDepths).toEqual(await depths(world, camera));
        expect(new Set(varied).size).toBeGreaterThan(4);
    });
}
