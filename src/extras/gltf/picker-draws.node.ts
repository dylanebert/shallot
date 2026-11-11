import { afterAll, expect, setDefaultTimeout, spyOn, test } from "bun:test";
import {
    MeshInstance,
    MeshPlugin,
    MeshRenderPlugin,
    StandardRenderingPlugin,
} from "@dylanebert/shallot";
import { GltfPlugin, loadGltfScene, spawnGltfScene } from "@dylanebert/shallot/gltf";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { modelPath, type Scene, scenes, truckPlacement } from "../../../examples/gltf/src/scenes";
import { fileFetch, syncAssets } from "../../../examples/gltf/tests/setup";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { DataImages, Images, imageTextures } from "../../core/rendering/image";
import { MeshMaterial } from "../../standard/rendering/material-data";
import { alphaPipelineKey, pipelineKey } from "../../standard/rendering/material-type";
import { isMeshInstanceDraw } from "../../standard/rendering/preprocess";
import { Draws } from "../../standard/rendering/registry";

setDefaultTimeout(TIMEOUT.node);
let restoreFetch: (() => void) | undefined;
afterAll(() => restoreFetch?.());
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            StandardRenderingPlugin,
            MeshRenderPlugin,
            MeshPlugin,
            GltfPlugin,
            {
                name: "PickerModels",
                dependencies: [GltfPlugin],
                async initialize(world) {
                    await syncAssets();
                    const fetch = spyOn(globalThis, "fetch").mockImplementation(fileFetch);
                    restoreFetch = () => fetch.mockRestore();
                    for (const scene of Object.keys(scenes) as Scene[]) {
                        models.set(
                            scene,
                            await loadGltfScene(
                                world,
                                new URL(
                                    `../../../examples/gltf/public/${modelPath(scene)}`,
                                    import.meta.url,
                                ),
                            ),
                        );
                    }
                    // Draw occupancy needs image identity, not decoded pixels on headless Dawn.
                    for (const [role, registry] of [Images, DataImages].entries()) {
                        const texture = world.gpu.device.createTexture({
                            size: [1, 1],
                            format: role === 0 ? "rgba8unorm-srgb" : "rgba8unorm",
                            usage: GPUTextureUsage.TEXTURE_BINDING,
                        });
                        world.own(texture);
                        world.resource(imageTextures).textures[role] = new Array(
                            world.resource(registry).size,
                        ).fill(texture);
                    }
                },
            },
        ],
    },
]);
const models = new Map<Scene, Awaited<ReturnType<typeof loadGltfScene>>>();

for (const scene of Object.keys(scenes) as Scene[]) {
    test(`${scene} registers no more draws than its material and mesh instance pairs`, () => {
        const { world } = subjects()[0];
        const instance = spawnGltfScene(
            world,
            models.get(scene)!,
            scene === "CesiumMilkTruck" ? truckPlacement : {},
        );
        try {
            world.step(0);
            const pairs = new Set<string>();
            const material = world.storage(MeshMaterial);
            for (const eid of world.query([MeshInstance])) {
                pairs.add(
                    `${material.type.get(eid)}:${material.material.get(eid)}:${pipelineKey(alphaPipelineKey(material.alphaMode.get(eid)), material.cullMode.get(eid))}:${world.storage(MeshInstance).mesh.get(eid)}`,
                );
            }
            if (scene === "CesiumMilkTruck") expect(pairs.size).toBe(4);
            if (scene === "TextureTransformMultiTest") expect(pairs.size).toBe(29);
            const draws = [...world.resource(Draws).values()].filter((draw) =>
                isMeshInstanceDraw(world, draw),
            );
            console.log(`${scene}: ${draws.length} draws / ${pairs.size} instance pairs`);
            expect(draws.length).toBeLessThanOrEqual(pairs.size);
        } finally {
            for (const eid of [...instance.entities, ...instance.lightEntities]) world.destroy(eid);
            world.step(0);
        }
    });
}
