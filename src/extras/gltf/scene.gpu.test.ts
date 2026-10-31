import { afterAll, beforeAll, expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { attachTexture, captureTexture, ImageBindings } from "../../core/rendering";
import {
    type App,
    Camera,
    DirectionalLight,
    ImageAddressMode,
    lookAtRotation,
    Materials,
    Meshes,
    MeshInstance,
    MeshMaterial,
    type Plugin,
    registerImage,
    StandardMaterial,
    StandardRenderer,
    Tonemapping,
    TonemappingMethod,
    Transform,
    type World,
} from "../../index";
import { openGpuApp as createApp } from "../../project/testing-gpu";
import { MeshRenderPlugin, StandardRenderingPlugin } from "../../standard/rendering";
import { type Gltf, loadGltf, loadGltfScene, spawnGltfScene } from "./index";

setDefaultTimeout(TIMEOUT.gpu);
const vertices = new Float32Array([
    -0.8, -0.6, 0, 0, 0, 0, 1, 0, 0.8, -0.6, 0, 1, 0, 0, 1, 0, 0, 0.8, 0, 0.5, 0, 0, 1, 1,
]);
const positions = new Float32Array([-0.8, -0.6, 0, 0.8, -0.6, 0, 0, 0.8, 0]);
const fixture = {
    asset: { version: "2.0" },
    extensionsUsed: ["KHR_materials_unlit"],
    buffers: [
        {
            uri: `data:application/octet-stream;base64,${Buffer.from(positions.buffer).toString("base64")}`,
            byteLength: positions.byteLength,
        },
    ],
    bufferViews: [{ buffer: 0, byteLength: positions.byteLength }],
    accessors: [
        {
            bufferView: 0,
            componentType: 5126,
            count: 3,
            type: "VEC3",
            min: [-0.8, -0.6, 0],
            max: [0.8, 0.8, 0],
        },
    ],
    materials: [
        {
            pbrMetallicRoughness: {
                baseColorFactor: [0.8, 0.2, 0.04, 1],
                metallicFactor: 0.3,
                roughnessFactor: 0.7,
            },
            // biome-ignore lint/style/useNamingConvention: glTF owns its attribute and extension names.
            extensions: { KHR_materials_unlit: {} },
        },
    ],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
    nodes: [
        {
            name: "Triangle",
            mesh: 0,
            translation: [0.25, 0.1, 0],
            rotation: [0, 0, Math.sin(0.15), Math.cos(0.15)],
        },
    ],
    scenes: [{ nodes: [0] }],
    scene: 0,
};
const url = `data:model/gltf+json;base64,${Buffer.from(JSON.stringify(fixture)).toString("base64")}`;
const apps: App[] = [];
const cameras: number[] = [];
let loaded: Gltf;
function camera(world: World) {
    const eid = world.create();
    world.add(eid, Transform, { translation: [0, 0, 4] });
    world.add(eid, Camera);
    world.add(eid, StandardRenderer);
    world.add(eid, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, eid, { width: 64, height: 64 });
    cameras.push(eid);
}
beforeAll(async () => {
    for (const direct of [false, true]) {
        const subject: Plugin = {
            name: "Fixture",
            gpu: { features: ["indirect-first-instance"] },
            dependencies: [StandardRenderingPlugin, MeshRenderPlugin],
            async initialize(world) {
                camera(world);
                if (!direct) {
                    loaded = await loadGltfScene(world, url);
                    spawnGltfScene(world, loaded, {
                        translation: [-0.1, 0.2, 0],
                        scale: [1.2, 1.2, 1.2],
                    });
                } else {
                    const mesh = world.resource(Meshes).add({
                        name: "Authored triangle",
                        vertices,
                        indices: new Uint32Array([0, 1, 2]),
                    });
                    const material = world.resource(Materials).add(
                        StandardMaterial({
                            baseColor: [0.8, 0.2, 0.04, 1],
                            metallic: 0.3,
                            perceptualRoughness: 0.7,
                            unlit: true,
                        }),
                    );
                    const eid = world.create();
                    world.add(eid, Transform, {
                        translation: [0.2, 0.32, 0],
                        rotation: [0, 0, Math.sin(0.15), Math.cos(0.15)],
                        scale: [1.2, 1.2, 1.2],
                    });
                    world.add(eid, MeshInstance, { mesh });
                    world.add(eid, MeshMaterial, material);
                }
            },
        };
        const app = await createApp({
            defaults: false,
            plugins: [StandardRenderingPlugin, MeshRenderPlugin, subject],
        });
        if (!direct) {
            const { device, adapter } = app.world.gpu;
            console.log(
                JSON.stringify({ adapter, features: [...device.features], limits: device.limits }),
            );
        }
        apps.push(app);
        app.world.step(0);
        app.world.step(0);
        await captureTexture(app.world, cameras.at(-1)!);
    }
    const boxUrl = new URL("./fixtures/default-sampler.gltf", import.meta.url);
    for (const direct of [false, true]) {
        const subject: Plugin = {
            name: "DefaultSampler",
            dependencies: [StandardRenderingPlugin, MeshRenderPlugin],
            async initialize(world) {
                camera(world);
                if (!direct) {
                    spawnGltfScene(world, await loadGltfScene(world, boxUrl));
                    return;
                }
                const scene = await loadGltf(boxUrl);
                const image = registerImage(world, scene.images[0].uri!, {
                    sampler: {
                        addressModeU: ImageAddressMode.Repeat,
                        addressModeV: ImageAddressMode.Repeat,
                    },
                });
                const mesh = world.resource(Meshes).add(scene.meshes[0]);
                const source = scene.materials[0];
                const material = world.resource(Materials).add(
                    StandardMaterial({
                        baseColor: source.color,
                        baseColorTexture: image,
                        metallic: source.metallic,
                        perceptualRoughness: source.roughness,
                        unlit: true,
                    }),
                );
                const eid = world.create();
                world.add(eid, Transform, { rotation: [-Math.SQRT1_2, 0, 0, Math.SQRT1_2] });
                world.add(eid, MeshInstance, { mesh });
                world.add(eid, MeshMaterial, material);
            },
        };
        const app = await createApp({
            defaults: false,
            plugins: [StandardRenderingPlugin, MeshRenderPlugin, subject],
        });
        const { device } = app.world.gpu;
        // Headless Dawn has no bitmap decoder; retain registered settings with decoded fixture pixels.
        const texture = device.createTexture({
            size: [2, 2, 1],
            format: "rgba8unorm-srgb",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        app.world.own(texture);
        device.queue.writeTexture(
            { texture },
            new Uint8Array([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255]),
            { bytesPerRow: 8, rowsPerImage: 2 },
            [2, 2, 1],
        );
        app.world.resource(ImageBindings).images = texture;
        apps.push(app);
        app.world.step(0);
        app.world.step(0);
        await captureTexture(app.world, cameras.at(-1)!);
    }
}, TIMEOUT.startup);
afterAll(() => {
    for (const app of apps) app.dispose();
});
test("loaded primitive placement and material draw the same pixels as direct authoring", async () => {
    const pixels = await Promise.all(
        apps.slice(0, 2).map(async (app, i) => {
            app.world.step(0);
            return (await captureTexture(app.world, cameras[i])).rgba;
        }),
    );
    expect(pixels[0]).toEqual(pixels[1]);
    expect(pixels[0].some((value, i) => i % 4 === 0 && value > 150)).toBe(true);
});
test("a texture naming no sampler repeats outside [0, 1] like a directly registered image", async () => {
    const pixels = await Promise.all(
        apps.slice(2, 4).map(async (app, i) => {
            app.world.step(0);
            return (await captureTexture(app.world, cameras[i + 2])).rgba;
        }),
    );
    expect(pixels[0]).toEqual(pixels[1]);
    expect(new Set(pixels[0]).size).toBeGreaterThan(10);
});
test("two placements reuse the load's exposed primitive handles and model-space matrix", () => {
    const world = apps[0].world;
    const meshAdd = spyOn(world.resource(Meshes), "add");
    const materialAdd = spyOn(world.resource(Materials), "add");
    try {
        const first = spawnGltfScene(world, loaded, { translation: [3, 0, 0] });
        const second = spawnGltfScene(world, loaded, { translation: [-3, 0, 0] });
        expect(first).not.toEqual(second);
        expect(first.length).toBe(loaded.primitives.length);
        expect(second.length).toBe(loaded.primitives.length);
        for (const [i, primitive] of loaded.primitives.entries()) {
            expect(world.resource(Meshes).has(primitive.mesh)).toBe(true);
            expect(Array.from(primitive.matrix)).toEqual(
                Array.from(
                    new Float32Array([
                        Math.cos(0.3),
                        Math.sin(0.3),
                        0,
                        0,
                        -Math.sin(0.3),
                        Math.cos(0.3),
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0.25,
                        0.1,
                        0,
                        1,
                    ]),
                ),
            );
            for (const eid of [first[i], second[i]]) {
                expect(world.storage(MeshInstance).mesh.get(eid)).toBe(primitive.mesh);
                expect(world.storage(MeshMaterial).type.get(eid)).toBe(primitive.material.type);
                expect(world.storage(MeshMaterial).material.get(eid)).toBe(
                    primitive.material.material,
                );
            }
        }
        expect(meshAdd).not.toHaveBeenCalled();
        expect(materialAdd).not.toHaveBeenCalled();
    } finally {
        meshAdd.mockRestore();
        materialAdd.mockRestore();
    }
});
test("sheared placement refuses with its node named before spawning", () => {
    const world = apps[0].world;
    const before = [...world.query([MeshInstance])];
    expect(() => spawnGltfScene(world, loaded, { scale: [2, 1, 1] })).toThrow(/Triangle.*shear/);
    expect([...world.query([MeshInstance])]).toEqual(before);
});
test("a foreign world refuses the handle", () => {
    expect(() => spawnGltfScene(apps[1].world, loaded)).toThrow("another world");
});

// Constant-color mip levels isolate mip selection from filtering within a level.
// The loaded sampler/settings stay intact; only Dawn's missing bitmap decode is replaced.
const mipApps: { app: App; target: number }[] = [];
beforeAll(async () => {
    for (const minFilter of [9985, undefined]) {
        for (const lod of [0.25, 0.75]) {
            // Perspective at z=4, fov=60: 64 / (8*tan(30)) pixels per world unit.
            const scale = (64 / (8 * Math.tan(Math.PI / 6)) / 8) * 2 ** lod;
            const uv = new Float32Array(
                Array.from(
                    { length: 6 },
                    (_, i) => positions[Math.floor(i / 2) * 3 + (i % 2)] * scale,
                ),
            );
            const json = {
                ...fixture,
                buffers: [
                    ...fixture.buffers,
                    {
                        uri: `data:application/octet-stream;base64,${Buffer.from(uv.buffer).toString("base64")}`,
                        byteLength: uv.byteLength,
                    },
                ],
                bufferViews: [...fixture.bufferViews, { buffer: 1, byteLength: uv.byteLength }],
                accessors: [
                    ...fixture.accessors,
                    { bufferView: 1, componentType: 5126, count: 3, type: "VEC2" },
                ],
                meshes: [
                    { primitives: [{ attributes: { POSITION: 0, TEXCOORD_0: 1 }, material: 0 }] },
                ],
                nodes: [{ mesh: 0 }],
                materials: [
                    {
                        pbrMetallicRoughness: { baseColorTexture: { index: 0 } },
                        // biome-ignore lint/style/useNamingConvention: glTF owns its attribute and extension names.
                        extensions: { KHR_materials_unlit: {} },
                    },
                ],
                textures: [{ source: 0, sampler: 0 }],
                images: [{ uri: "data:image/png;base64,iVBORw0KGgo=" }],
                samplers: [{ minFilter }],
            };
            let target = 0;
            const app = await createApp({
                defaults: false,
                plugins: [
                    StandardRenderingPlugin,
                    MeshRenderPlugin,
                    {
                        name: "MipSelection",
                        dependencies: [StandardRenderingPlugin, MeshRenderPlugin],
                        async initialize(world) {
                            camera(world);
                            target = cameras.at(-1)!;
                            const source = `data:model/gltf+json;base64,${Buffer.from(JSON.stringify(json)).toString("base64")}`;
                            spawnGltfScene(world, await loadGltfScene(world, source));
                        },
                    },
                ],
            });
            const { device } = app.world.gpu;
            apps.push(app);
            {
                const texture = device.createTexture({
                    size: [8, 8, 1],
                    mipLevelCount: 4,
                    format: "rgba8unorm-srgb",
                    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
                });
                app.world.own(texture);
                for (let level = 0; level < 4; level++) {
                    const size = 8 >> level;
                    const pixels = new Uint8Array(size * size * 4);
                    for (let i = 0; i < pixels.length; i += 4)
                        pixels.set(
                            level === 0
                                ? [255, 0, 0, 255]
                                : level === 1
                                  ? [0, 255, 0, 255]
                                  : [0, 0, 255, 255],
                            i,
                        );
                    device.queue.writeTexture(
                        { texture, mipLevel: level },
                        pixels,
                        { bytesPerRow: size * 4, rowsPerImage: size },
                        [size, size, 1],
                    );
                }
                app.world.resource(ImageBindings).images = texture;
                app.world.step(0);
                app.world.step(0);
                await captureTexture(app.world, target);
                mipApps.push({ app, target });
            }
        }
    }
}, TIMEOUT.startup);
test("LINEAR_MIPMAP_NEAREST steps between levels while unspecified minFilter blends", async () => {
    const samples: number[][] = [];
    for (const { app, target } of mipApps) {
        app.world.step(0);
        const { rgba } = await captureTexture(app.world, target);
        samples.push(Array.from(rgba.slice((32 * 64 + 32) * 4, (32 * 64 + 32) * 4 + 3)));
    }
    expect(samples[0][0]).toBeGreaterThan(250);
    expect(samples[0][1]).toBeLessThan(5);
    expect(samples[1][0]).toBeLessThan(5);
    expect(samples[1][1]).toBeGreaterThan(250);
    for (const blended of samples.slice(2)) {
        expect(blended[0]).toBeGreaterThan(100);
        expect(blended[0]).toBeLessThan(240);
        expect(blended[1]).toBeGreaterThan(100);
        expect(blended[1]).toBeLessThan(240);
        expect(blended[2]).toBeLessThan(5);
    }
    expect(samples[2][0]).toBeGreaterThan(samples[3][0]);
    expect(samples[2][1]).toBeLessThan(samples[3][1]);
});

// The synthetic plane has no textures: lighting isolates the back-face normal flip.
const faceApps: { app: App; target: number; mode: string }[] = [];
beforeAll(async () => {
    for (const mode of ["loaded", "direct", "single", "unflipped"] as const) {
        let target = 0;
        const app = await createApp({
            defaults: false,
            plugins: [
                StandardRenderingPlugin,
                MeshRenderPlugin,
                {
                    name: "DoubleSided",
                    dependencies: [StandardRenderingPlugin, MeshRenderPlugin],
                    async initialize(world) {
                        camera(world);
                        target = cameras.at(-1)!;
                        world.add(target, Transform, {
                            translation: [0, 0, -4],
                            rotation: lookAtRotation(0, 0, -4, 0, 0, 0),
                        });
                        const sun = world.create();
                        world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, 0, 0, 1) });
                        world.add(sun, DirectionalLight, { illuminance: 1500 });
                        const source = new URL("./fixtures/double-sided.gltf", import.meta.url);
                        if (mode === "loaded" || mode === "single") {
                            const scene = await loadGltfScene(world, source);
                            spawnGltfScene(world, scene, {
                                translation: mode === "single" ? [-3, 0, 0] : [0, 0, 0],
                            });
                        } else {
                            const scene = await loadGltf(source);
                            const mesh = world.resource(Meshes).add(scene.meshes[0]);
                            const material = world.resource(Materials).add(
                                StandardMaterial({
                                    baseColor: [0.8, 0.6, 0.4, 1],
                                    metallic: 0,
                                    perceptualRoughness: 0.6,
                                    doubleSided: mode === "direct",
                                }),
                                { cullMode: "none" },
                            );
                            const eid = world.create();
                            world.add(eid, Transform);
                            world.add(eid, MeshInstance, { mesh });
                            world.add(eid, MeshMaterial, material);
                        }
                    },
                },
            ],
        });
        apps.push(app);
        app.world.step(0);
        app.world.step(0);
        await captureTexture(app.world, target);
        faceApps.push({ app, target, mode });
    }
}, TIMEOUT.startup);
test("loaded double-sided back faces draw and light like direct authoring; single-sided culls", async () => {
    const samples: number[][] = [];
    for (const { app, target, mode } of faceApps) {
        app.world.step(0);
        const { rgba } = await captureTexture(app.world, target);
        const center = Array.from(rgba.slice((32 * 64 + 32) * 4, (32 * 64 + 32) * 4 + 3));
        samples.push(center);
        if (mode === "single") expect(center).toEqual(Array.from(rgba.slice(0, 3)));
    }
    expect(samples[0]).toEqual(samples[1]);
    expect(samples[0][0]).toBeGreaterThan(samples[2][0] + 20);
    expect(samples[0][0]).toBeGreaterThan(samples[3][0] + 20);
});
