import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { Meshes, MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import type { App, World } from "../../engine";
import { probeTexture } from "../../engine/runtime";
import { openGpuApp } from "../../project/testing-gpu";
import {
    AlphaMode,
    MaterialPlugin,
    MeshMaterial,
    MeshRenderPlugin,
    materialLayout,
    materialType,
    StandardRenderer,
    StandardRenderingPlugin,
} from "../../standard/rendering";
import type { GltfScene } from "./gltf";
import { loadGltf } from "./load";
import { morphFixture, morphFixtureUrl } from "./morph.fixture";
import { type GltfSceneInstance, loadGltfScene, poseGltfScene, spawnGltfScene } from "./scene";
import {
    deformationBindings,
    deformationVertex,
    GltfPlugin,
    MorphedMesh,
    SkinPalette,
} from "./skinning";

setDefaultTimeout(TIMEOUT.gpu);
const layout = materialLayout(d.struct({ pad: d.u32 }), {
    ...deformationBindings,
    results: { type: "storage-texture", format: "rgba32uint", access: "write" },
});
const deform = deformationVertex(layout);
const Results = {
    create: (world: World) =>
        world.gpu.device.createTexture({
            size: [384, 64],
            format: "rgba32uint",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
        }),
};
const ReadMorph = materialType({
    name: "ReadMorph",
    vertexTangent: true,
    layout,
    defaults: { pad: 0 },
    bindings: (world) => ({
        ...world.resource(SkinPalette).bindings,
        results: world.resource(Results),
    }),
    varyings: {
        position: d.interpolate("flat", d.vec4u),
        normal: d.interpolate("flat", d.vec4u),
        posedTangent: d.interpolate("flat", d.vec4u),
        sourcePosition: d.interpolate("flat", d.vec4u),
        sourceNormal: d.interpolate("flat", d.vec4u),
        sourceTangent: d.interpolate("flat", d.vec4u),
    },
    vertex: (input) => {
        "use gpu";
        const posed = deform(
            input.eid,
            input.vidx,
            input.localPos,
            input.localNormal,
            input.localTangent,
            input.xform,
        );
        const corner = input.vidx % 3;
        return {
            world: d.vec4f(
                std.select(std.select(0, 1, corner === 1), -1, corner === 0),
                std.select(-1, 1, corner === 2),
                0,
                1,
            ),
            worldNormal: input.worldNormal,
            worldTangent: input.localTangent,
            position: std.bitcastF32toU32(posed.world),
            normal: std.bitcastF32toU32(d.vec4f(posed.worldNormal, 1)),
            posedTangent: std.bitcastF32toU32(posed.worldTangent),
            sourcePosition: std.bitcastF32toU32(d.vec4f(input.localPos, d.f32(input.vidx))),
            sourceNormal: std.bitcastF32toU32(d.vec4f(input.localNormal, 1)),
            sourceTangent: std.bitcastF32toU32(input.localTangent),
        };
    },
    fragment: (ctx) => {
        "use gpu";
        const vidx = d.u32(std.bitcastU32toF32(ctx.sourcePosition).w);
        const x = d.i32((vidx % 64) * 6);
        const y = d.i32(vidx / 64);
        std.textureStore(layout.$.results, d.vec2i(x, y), ctx.position);
        std.textureStore(layout.$.results, d.vec2i(x + 1, y), ctx.normal);
        std.textureStore(layout.$.results, d.vec2i(x + 2, y), ctx.posedTangent);
        std.textureStore(layout.$.results, d.vec2i(x + 3, y), ctx.sourcePosition);
        std.textureStore(layout.$.results, d.vec2i(x + 4, y), ctx.sourceNormal);
        std.textureStore(layout.$.results, d.vec2i(x + 5, y), ctx.sourceTangent);
        return d.vec4f(1);
    },
});
let app: App, cullApp: App, cullCamera: number;
let cullScene: GltfSceneInstance;
const scenes: {
    instance: GltfSceneInstance;
    decoded: GltfScene;
    base: number;
    skinned: boolean;
    fixedWeights?: number[];
}[] = [];
beforeAll(async () => {
    app = await openGpuApp({
        defaults: false,
        plugins: [
            {
                name: "MorphReadback",
                dependencies: [
                    StandardRenderingPlugin,
                    MeshRenderPlugin,
                    GltfPlugin,
                    MaterialPlugin(ReadMorph),
                ],
                async initialize(world) {
                    world.resource(Results);
                    const camera = world.create();
                    world.add(camera, Transform, { translation: [0, 0, 5] });
                    world.add(camera, Camera, { antialias: 0 });
                    world.add(camera, StandardRenderer);
                    attachTexture(world, camera, { width: 16, height: 16 });
                    const material = world
                        .resource(ReadMorph)
                        .add({ pad: 0 }, { alphaMode: AlphaMode.Blend, cullMode: "none" });
                    for (const [skinned, eight] of [
                        [false, false],
                        [true, false],
                        [false, true],
                    ]) {
                        const { json } = morphFixture();
                        if (!skinned) delete json.nodes![0].skin;
                        const fixedWeights = eight
                            ? [0.1, -0.2, 0.3, 0.4, 0.5, -0.6, 0.7, 0.8]
                            : undefined;
                        if (eight) {
                            json.meshes![0].primitives[0].targets = Array.from(
                                { length: 8 },
                                () => ({
                                    POSITION: json.meshes![0].primitives[0].targets![1].POSITION,
                                }),
                            );
                            json.nodes![0].weights = fixedWeights;
                            json.animations![0].channels.pop();
                        }
                        if (eight)
                            // Earlier registrations survive growth; this mesh's deltas cross a texture row.
                            world.resource(SkinPalette).morph.register({
                                name: "delta-growth",
                                vertices: new Float32Array(1010 * 8),
                                indices: new Uint32Array(0),
                                material: -1,
                                morphTargets: Array.from({ length: 8 }, () => ({
                                    position: new Float32Array(1010 * 3),
                                })),
                            });
                        const url = morphFixtureUrl(json);
                        const gltf = await loadGltfScene(world, url),
                            decoded = await loadGltf(url);
                        const instance = spawnGltfScene(world, gltf);
                        for (const eid of instance.entities) world.add(eid, MeshMaterial, material);
                        scenes.push({ instance, decoded, skinned, fixedWeights, base: 0 });
                    }
                },
            },
        ],
    });
    const world = app.world,
        meshes = world.resource(Meshes);
    for (const scene of scenes) {
        const mesh = meshes.get(scene.instance.gltf.primitives[0].mesh)!;
        const original = await mesh.indices.read();
        const base = Math.min(...original.slice(mesh.indexBase, mesh.indexBase + mesh.indexCount));
        scene.base = base;
        const count = scene.decoded.meshes[0].vertices.length / 8,
            indices: number[] = [];
        for (let v = base; v < base + count; v++) {
            const other = [base, base + 1, base + 2].filter((i) => i % 3 !== v % 3);
            indices.push(v, other[0], other[1]);
        }
        const indexBuffer = world.gpu.root
            .createBuffer(d.arrayOf(d.u32, indices.length), indices)
            .$usage("storage", "index");
        const handle = meshes.add({
            ...mesh,
            name: "morph-provoking",
            indices: indexBuffer,
            indexBase: 0,
            indexCount: indices.length,
            bounds: undefined,
        });
        world.storage(MeshInstance).mesh.set(scene.instance.entities[0], handle);
    }
    const palette = world.resource(SkinPalette);
    const oldWeights = palette.bindings.morphWeights,
        oldOffsets = palette.deformationOffsets;
    const reserved = world.create();
    world.add(reserved, MorphedMesh, { weightCount: 240 });
    while (world.create() < 300) {}
    const third = scenes[2],
        previous = third.instance;
    third.instance = spawnGltfScene(world, previous.gltf);
    world.add(
        third.instance.entities[0],
        MeshMaterial,
        world.get(previous.entities[0], MeshMaterial)!,
    );
    world
        .storage(MeshInstance)
        .mesh.set(
            third.instance.entities[0],
            world.storage(MeshInstance).mesh.get(previous.entities[0]),
        );
    world.remove(previous.entities[0], MeshInstance);
    world.remove(previous.entities[0], MorphedMesh);
    world.step(0);
    expect(palette.bindings.morphWeights).not.toBe(oldWeights);
    expect(palette.deformationOffsets).not.toBe(oldOffsets);
    // Eight contiguous weights straddle rows 0 and 1.
    expect(world.storage(MorphedMesh).firstWeight.get(third.instance.entities[0])).toBe(252);
    cullApp = await openGpuApp({
        defaults: false,
        plugins: [
            {
                name: "MorphCull",
                dependencies: [StandardRenderingPlugin, MeshRenderPlugin, GltfPlugin],
                async initialize(world) {
                    cullCamera = world.create();
                    world.add(cullCamera, Transform, { translation: [4, 0, 3] });
                    world.add(cullCamera, Camera, { antialias: 0 });
                    world.add(cullCamera, StandardRenderer);
                    world.add(cullCamera, Tonemapping, { method: TonemappingMethod.None });
                    attachTexture(world, cullCamera, { width: 32, height: 32 });
                    const { json } = morphFixture();
                    delete json.nodes![0].skin;
                    delete json.materials![0].normalTexture;
                    delete json.nodes![0].weights;
                    const staticPrimitive = structuredClone(json.meshes![0].primitives[0]);
                    delete staticPrimitive.targets;
                    json.meshes![0].primitives.push(staticPrimitive);
                    // biome-ignore lint/style/useNamingConvention: glTF owns the extension name.
                    json.materials![0].extensions = { KHR_materials_unlit: {} };
                    cullScene = spawnGltfScene(
                        world,
                        await loadGltfScene(world, morphFixtureUrl(json)),
                    );
                },
            },
        ],
    });
    expect(cullApp.world.has(cullScene.entities[0], MorphedMesh)).toBe(true);
    expect(cullApp.world.has(cullScene.entities[1], MorphedMesh)).toBe(false);
    const first = cullApp.world.storage(MorphedMesh).firstWeight.get(cullScene.entities[0]);
    expect(cullApp.world.resource(SkinPalette).morph.weights.slice(first, first + 2)).toEqual(
        new Float32Array([0.2, 0.1]),
    );
    console.info("morph floor", {
        adapter: world.gpu.adapter,
        features: [...world.gpu.device.features],
        storageBuffers: world.gpu.device.limits.maxStorageBuffersPerShaderStage,
        sampledTextures: world.gpu.device.limits.maxSampledTexturesPerShaderStage,
    });
}, TIMEOUT.startup);
afterAll(() => {
    app?.dispose();
    cullApp?.dispose();
});

// Absolute + relative 2e-4 permits float32 sum, normalization and matrix error. Echoed GPU inputs
// exclude packed base-stream quantization; target deltas are float32, never packed.
function close(actual: number, expected: number) {
    expect(Math.abs(actual - expected)).toBeLessThanOrEqual(2e-4 * Math.max(1, Math.abs(expected)));
}
const normalize = (v: number[]) => {
    const length = Math.hypot(...v);
    return v.map((x) => x / length);
};
for (const time of [0, 0.5, 1])
    test(`morph sum precedes skinning for positions, normals and tangents at recorded time ${time}, including a shared family`, async () => {
        const world = app.world,
            meshes = world.resource(Meshes);
        const a = meshes.get(scenes[0].instance.gltf.primitives[0].mesh)!,
            b = meshes.get(scenes[1].instance.gltf.primitives[0].mesh)!;
        expect(a.vertices).toBe(b.vertices);
        expect(a.attributes!.deformation).toBe(b.attributes!.deformation);
        expect(scenes[1].base).toBeGreaterThan(0);
        for (const scene of scenes) {
            poseGltfScene(world, scene.instance, scene.instance.gltf.animations[0], time);
            world.storage(MeshInstance).cullBounds.set(scene.instance.entities[0], 0, 0, 0, -1);
        }
        world.tick();
        world.step(1 / 60);
        const output = new Float32Array((await probeTexture(world, world.resource(Results))).bytes);
        for (const scene of scenes) {
            const angle = (time * Math.PI * 2) / 3,
                c = Math.cos(angle),
                s = Math.sin(angle);
            for (let v = 0; v < scene.decoded.meshes[0].vertices.length / 8; v++) {
                const o = (scene.base + v) * 24;
                expect(output[o + 15]).toBe(scene.base + v);
                expect(output[o + 19]).toBe(1);
                const values = [
                    Array.from(output.subarray(o + 12, o + 15)),
                    Array.from(output.subarray(o + 16, o + 19)),
                    Array.from(output.subarray(o + 20, o + 23)),
                ];
                for (const [m, target] of scene.decoded.meshes[0].morphTargets!.entries())
                    for (const [a, semantic] of (
                        ["position", "normal", "tangent"] as const
                    ).entries())
                        for (let k = 0; k < 3; k++)
                            values[a][k] +=
                                (scene.fixedWeights?.[m] ?? [time, time / 2][m]) *
                                (target[semantic]?.[v * 3 + k] ?? 0);
                if (scene.skinned)
                    for (let a = 0; a < 3; a++) {
                        const value = values[a],
                            x = value[0] * (a === 1 ? 1 / 1.5 : 1.5),
                            y = value[1] * (a === 1 ? 1 / 0.75 : 0.75);
                        value[0] = c * x - s * y;
                        value[1] = s * x + c * y;
                    }
                values[1] = normalize(values[1]);
                for (let a = 0; a < 3; a++)
                    for (let k = 0; k < 3; k++) close(output[o + a * 4 + k], values[a][k]);
                close(output[o + 11], output[o + 23]);
            }
            // Independent stage-1-style joint-global record for the fixture's authored scale/rotation.
            if (scene.skinned) {
                const expected = [
                    1.5 * c,
                    1.5 * s,
                    0,
                    0,
                    -0.75 * s,
                    0.75 * c,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                ];
                for (let i = 0; i < 16; i++) close(scene.instance.globals[1][i], expected[i]);
            }
        }
    });

test("a morphed vertex outside its base sphere remains visible", async () => {
    const world = cullApp.world;
    poseGltfScene(world, cullScene, cullScene.gltf.animations[0], 0);
    world.tick();
    world.step(1 / 60);
    world.step(1 / 60);
    const rest = await captureTexture(world, cullCamera);
    expect(rest.rgba.some((v, i) => i % 4 !== 3 && v > 200)).toBe(false);
    poseGltfScene(world, cullScene, cullScene.gltf.animations[0], 1);
    world.tick();
    world.step(1 / 60);
    world.step(1 / 60);
    const frame = await captureTexture(world, cullCamera);
    let white = 0;
    for (let i = 0; i < frame.rgba.length; i += 4)
        if (frame.rgba[i] > 200 && frame.rgba[i + 1] > 200 && frame.rgba[i + 2] > 200) white++;
    expect(white).toBeGreaterThan(10);
});

test("a detached morph retaining its material draws base geometry instead of another weight range", async () => {
    const world = cullApp.world,
        eid = cullScene.entities[0];
    poseGltfScene(world, cullScene, cullScene.gltf.animations[0], 1);
    world.remove(eid, MorphedMesh);
    try {
        world.tick();
        world.step(1 / 60);
        world.step(1 / 60);
        const frame = await captureTexture(world, cullCamera);
        expect(frame.rgba.some((v, i) => i % 4 !== 3 && v > 200)).toBe(false);
    } finally {
        world.add(eid, MorphedMesh, { weightCount: 2 });
        poseGltfScene(world, cullScene, cullScene.gltf.animations[0], 1);
    }
});

test("morphed joint boxes conservatively enclose every morphed-then-skinned vertex", () => {
    const world = app.world,
        scene = scenes[1];
    poseGltfScene(world, scene.instance, scene.instance.gltf.animations[0], 1);
    const eid = scene.instance.entities[0],
        bounds = world.storage(MeshInstance).cullBounds;
    const center = [bounds.x.get(eid), bounds.y.get(eid), bounds.z.get(eid)],
        radius = bounds.w.get(eid);
    const angle = (2 * Math.PI) / 3,
        c = Math.cos(angle),
        s = Math.sin(angle),
        mesh = scene.decoded.meshes[0];
    for (let v = 0; v < mesh.vertices.length / 8; v++) {
        const p = Array.from(mesh.vertices.subarray(v * 8, v * 8 + 3));
        for (let m = 0; m < 2; m++)
            for (let a = 0; a < 3; a++)
                p[a] += [1, 0.5][m] * (mesh.morphTargets![m].position?.[v * 3 + a] ?? 0);
        const x = 1.5 * p[0],
            y = 0.75 * p[1];
        expect(
            Math.hypot(c * x - s * y - center[0], s * x + c * y - center[1], p[2] - center[2]),
        ).toBeLessThanOrEqual(radius + 2e-4);
    }
});

test("weight membership owns reusable ranges; posing a detached morph cannot overwrite another spawn", () => {
    const world = app.world,
        gltf = scenes[0].instance.gltf,
        morph = world.resource(SkinPalette).morph,
        before = morph.ranges.extent;
    const a = spawnGltfScene(world, gltf),
        b = spawnGltfScene(world, gltf);
    const first = world.storage(MorphedMesh).firstWeight.get(a.entities[0]);
    world.remove(a.entities[0], MorphedMesh);
    expect(world.storage(MeshInstance).cullBounds.w.get(a.entities[0])).toBe(-1);
    const c = spawnGltfScene(world, gltf);
    expect(world.storage(MorphedMesh).firstWeight.get(c.entities[0])).toBe(first);
    poseGltfScene(world, c, gltf.animations[0], 1);
    const values = morph.weights.slice();
    poseGltfScene(world, a, gltf.animations[0], 0.5);
    expect(morph.weights).toEqual(values);
    for (const instance of [a, b, c]) for (const eid of instance.entities) world.destroy(eid);
    expect(morph.ranges.extent).toBe(before);
});
