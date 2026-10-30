import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { attachTexture, captureTexture } from "../../core/rendering";
import {
    type App,
    Camera,
    StandardRenderer,
    Tonemapping,
    TonemappingMethod,
    Transform,
} from "../../index";
import { openGpuApp } from "../../project/testing-gpu";
import { MeshRenderPlugin, StandardRenderingPlugin } from "../../standard/rendering";
import { loadGltfScene, spawnGltfScene } from "./index";

setDefaultTimeout(TIMEOUT.gpu);
const apps: App[] = [];
const cameras: number[] = [];
beforeAll(async () => {
    for (const mirrored of [false, true]) {
        const positions = new Float32Array([-0.8, -0.6, 0, 0.8, -0.6, 0, 0, 0.8, 0]);
        const fixture = {
            asset: { version: "2.0" },
            buffers: [
                {
                    uri: `data:application/octet-stream;base64,${Buffer.from(positions.buffer).toString("base64")}`,
                    byteLength: positions.byteLength,
                },
            ],
            bufferViews: [{ buffer: 0, byteLength: positions.byteLength }],
            accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: "VEC3" }],
            materials: [
                {
                    pbrMetallicRoughness: { baseColorFactor: [1, 0.2, 0.04, 1] },
                    // biome-ignore lint/style/useNamingConvention: glTF owns its extension names.
                    extensions: { KHR_materials_unlit: {} },
                },
            ],
            meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
            nodes: [{ scale: [mirrored ? -1 : 1, 1, 1], children: [1] }, { mesh: 0 }],
            scenes: [{ nodes: [0] }],
        };
        const app = await openGpuApp({
            defaults: false,
            plugins: [
                StandardRenderingPlugin,
                MeshRenderPlugin,
                {
                    name: "MirroredNode",
                    dependencies: [StandardRenderingPlugin, MeshRenderPlugin],
                    async initialize(world) {
                        const camera = world.create();
                        world.add(camera, Transform, { translation: [0, 0, 4] });
                        world.add(camera, Camera);
                        world.add(camera, StandardRenderer);
                        world.add(camera, Tonemapping, { method: TonemappingMethod.None });
                        attachTexture(world, camera, { width: 64, height: 64 });
                        cameras.push(camera);
                        const url = `data:model/gltf+json;base64,${Buffer.from(JSON.stringify(fixture)).toString("base64")}`;
                        spawnGltfScene(world, await loadGltfScene(world, url));
                    },
                },
            ],
        });
        console.log("glTF mirrored faces", app.world.gpu.adapter);
        apps.push(app);
        app.world.step(0);
        app.world.step(0);
    }
}, TIMEOUT.startup);
afterAll(() => {
    for (const app of apps) app.dispose();
});
test("a node mirrored by its parent draws its front faces wherever the unmirrored copy does", async () => {
    const pixels = await Promise.all(
        apps.map(async (app, i) => (await captureTexture(app.world, cameras[i])).rgba),
    );
    expect(pixels[1]).toEqual(pixels[0]);
    expect(pixels[0].some((v, i) => i % 4 === 0 && v > 200)).toBe(true);
});
