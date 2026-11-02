import { afterAll, beforeAll, expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import gold from "../../../examples/gltf/tests/animations.gold.json";
import { fileFetch, syncAssets } from "../../../examples/gltf/tests/setup";
import { Meshes, MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    DepthPrepass,
    DirectionalLight,
    Tonemapping,
    TonemappingMethod,
    Views,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import type { App, World } from "../../engine";
import { probeBuffer, probeTexture } from "../../engine/runtime";
import { openGpuApp } from "../../project/testing-gpu";
import {
    AlphaMode,
    Draws,
    MaterialPlugin,
    MeshMaterial,
    MeshRenderPlugin,
    materialLayout,
    materialType,
    StandardRenderer,
    StandardRenderingPlugin,
} from "../../standard/rendering";
import { sunShadowView } from "../../standard/rendering/atlas";
import type { GltfScene } from "./gltf";
import { loadGltf } from "./load";
import {
    type Gltf,
    type GltfSceneInstance,
    loadGltfScene,
    poseGltfScene,
    spawnGltfScene,
} from "./scene";
import {
    deformationBindings,
    deformationVertex,
    GltfPlugin,
    SkinnedMesh,
    SkinPalette,
} from "./skinning";
import { skinningFixtureUrl } from "./skinning.fixture";

setDefaultTimeout(TIMEOUT.gpu);
// The mesh layout already uses all eight fragment storage buffers after skin. A storage texture
// keeps readback on the real draw path without requesting limits above the WebGPU floor.
const layout = materialLayout(d.struct({ pad: d.u32 }), {
    ...deformationBindings,
    results: { type: "storage-texture", format: "rgba32uint", access: "write" },
});
const skin = deformationVertex(layout);
const Results = {
    create: (world: World) =>
        world.gpu.device.createTexture({
            size: [256, 1024],
            format: "rgba32uint",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
        }),
};
const ReadSkin = materialType({
    name: "ReadSkin",
    layout,
    defaults: { pad: 0 },
    bindings: (world) => ({
        ...world.resource(SkinPalette).bindings,
        results: world.resource(Results),
    }),
    varyings: {
        position: d.interpolate("flat", d.vec4u),
        normal: d.interpolate("flat", d.vec4u),
        sourcePosition: d.interpolate("flat", d.vec4u),
        sourceNormal: d.interpolate("flat", d.vec4u),
    },
    vertex: (input) => {
        "use gpu";
        const posed = skin(
            input.eid,
            input.vidx,
            input.localPos,
            input.localNormal,
            input.localTangent,
            input.xform,
        );
        const corner = input.vidx % 3;
        const x = std.select(std.select(0, 1, corner === 1), -1, corner === 0);
        const y = std.select(-1, 1, corner === 2);
        return {
            world: d.vec4f(x, y, 0, 1),
            worldNormal: input.worldNormal,
            position: std.bitcastF32toU32(posed.world),
            normal: std.bitcastF32toU32(d.vec4f(posed.worldNormal, 1)),
            sourcePosition: std.bitcastF32toU32(d.vec4f(input.localPos, d.f32(input.vidx))),
            sourceNormal: std.bitcastF32toU32(d.vec4f(input.localNormal, 1)),
        };
    },
    fragment: (ctx) => {
        "use gpu";
        const vidx = d.u32(std.bitcastU32toF32(ctx.sourcePosition).w);
        const x = d.i32((vidx % 64) * 4);
        const y = d.i32(vidx / 64);
        std.textureStore(layout.$.results, d.vec2i(x, y), ctx.position);
        std.textureStore(layout.$.results, d.vec2i(x + 1, y), ctx.normal);
        std.textureStore(layout.$.results, d.vec2i(x + 2, y), ctx.sourcePosition);
        std.textureStore(layout.$.results, d.vec2i(x + 3, y), ctx.sourceNormal);
        return d.vec4f(1);
    },
});
let app: App;
let cullApp: App;
let cullCamera: number;
let cullScene: GltfSceneInstance;
let camera: number;
const scenes: {
    gltf: Gltf;
    decoded: GltfScene;
    instance: GltfSceneInstance;
    record: (typeof gold.records)[number];
    bases: number[];
}[] = [];
let fetchSpy: ReturnType<typeof spyOn>;
beforeAll(async () => {
    await syncAssets();
    fetchSpy = spyOn(globalThis, "fetch").mockImplementation(fileFetch);
    app = await openGpuApp({
        defaults: false,
        plugins: [
            {
                name: "SkinReadback",
                dependencies: [
                    StandardRenderingPlugin,
                    MeshRenderPlugin,
                    GltfPlugin,
                    MaterialPlugin(ReadSkin),
                ],
                async initialize(world) {
                    world.resource(Results);
                    camera = world.create();
                    world.add(camera, Transform, { translation: [0, 0, 5] });
                    world.add(camera, Camera, { antialias: 0 });
                    world.add(camera, StandardRenderer);
                    attachTexture(world, camera, { width: 16, height: 16 });
                    const material = world
                        .resource(ReadSkin)
                        .add({ pad: 0 }, { alphaMode: AlphaMode.Blend, cullMode: "none" });
                    for (const name of ["RiggedSimple", "RiggedFigure"]) {
                        const url = new URL(
                            `../../../examples/gltf/public/assets/Models/${name}/glTF/${name}.gltf`,
                            import.meta.url,
                        );
                        const gltf = await loadGltfScene(world, url);
                        const decoded = await loadGltf(url);
                        const instance = spawnGltfScene(world, gltf);
                        for (const eid of instance.entities) {
                            world.add(eid, MeshMaterial, material);
                            world.storage(MeshInstance).cullBounds.set(eid, 0, 0, 0, -1);
                        }
                        scenes.push({
                            gltf,
                            decoded,
                            instance,
                            record: gold.records.find((r) => r.fixture.includes(`/${name}/`))!,
                            bases: [],
                        });
                    }
                },
            },
        ],
    });
    const world = app.world,
        meshes = world.resource(Meshes);
    for (const scene of scenes)
        for (let p = 0; p < scene.gltf.primitives.length; p++) {
            const primitive = scene.gltf.primitives[p];
            const mesh = meshes.get(primitive.mesh)!;
            const original = await mesh.indices.read();
            const base = Math.min(
                ...original.slice(mesh.indexBase, mesh.indexBase + mesh.indexCount),
            );
            const count = scene.decoded.meshes[scene.decoded.instances[p].mesh].vertices.length / 8;
            scene.bases.push(base);
            const indices: number[] = [];
            // Every vertex is the provoking vertex of a triangle with one index of each residue.
            // All triangles rasterize the same area; integer varyings are flat and depth writes are off.
            for (let v = base; v < base + count; v++) {
                const other = [base, base + 1, base + 2].filter((i) => i % 3 !== v % 3);
                indices.push(v, other[0], other[1]);
            }
            const indexBuffer = world.gpu.root
                .createBuffer(d.arrayOf(d.u32, indices.length), indices)
                .$usage("storage", "index");
            const handle = meshes.add({
                ...mesh,
                name: `${mesh.name}-provoking`,
                indices: indexBuffer,
                indexBase: 0,
                indexCount: indices.length,
                bounds: undefined,
            });
            world.storage(MeshInstance).mesh.set(scene.instance.entities[p], handle);
        }
    // Replace both textures after pipeline warm. Read the relocated second scene through the
    // same material callback, at an eid beyond the old map and a joint on a later texture row.
    const palette = world.resource(SkinPalette);
    const oldPalette = palette.jointPalette,
        oldOffsets = palette.deformationOffsets;
    const reserved = world.create();
    world.add(reserved, SkinnedMesh, { jointCount: 65 });
    const columnLength = Math.max(
        world.fieldStorage(SkinnedMesh, "firstJoint").column.length,
        oldOffsets.width * oldOffsets.height,
    );
    while (world.create() < columnLength) {}
    const second = scenes[1];
    const previous = second.instance;
    second.instance = spawnGltfScene(world, second.gltf);
    for (let p = 0; p < previous.entities.length; p++) {
        const oldEid = previous.entities[p],
            eid = second.instance.entities[p];
        world.add(eid, MeshMaterial, world.get(oldEid, MeshMaterial)!);
        world.storage(MeshInstance).mesh.set(eid, world.storage(MeshInstance).mesh.get(oldEid));
        world.storage(MeshInstance).cullBounds.set(eid, 0, 0, 0, -1);
        world.remove(oldEid, MeshInstance);
    }
    world.step(1 / 60);
    expect(palette.jointPalette).not.toBe(oldPalette);
    expect(palette.deformationOffsets).not.toBe(oldOffsets);
    expect(
        world.storage(SkinnedMesh).firstJoint.get(second.instance.entities[0]),
    ).toBeGreaterThanOrEqual(64);
    cullApp = await openGpuApp({
        defaults: false,
        plugins: [
            {
                name: "SkinCull",
                dependencies: [StandardRenderingPlugin, MeshRenderPlugin, GltfPlugin],
                async initialize(world) {
                    cullCamera = world.create();
                    world.add(cullCamera, Transform, { translation: [20, 0, 5] });
                    world.add(cullCamera, Camera, { antialias: 0 });
                    world.add(cullCamera, StandardRenderer);
                    world.add(cullCamera, DepthPrepass);
                    world.add(cullCamera, Tonemapping, { method: TonemappingMethod.None });
                    attachTexture(world, cullCamera, { width: 32, height: 32 });
                    const sun = world.create();
                    world.add(sun, Transform);
                    world.add(sun, DirectionalLight, { shadowMapsEnabled: 1 });
                    const gltf = await loadGltfScene(world, skinningFixtureUrl());
                    cullScene = spawnGltfScene(world, gltf, {
                        translation: [20, 0, 0],
                        scale: [2, 3, 1],
                    });
                },
            },
        ],
    });
    console.info("skinning floor", {
        adapter: cullApp.world.gpu.adapter,
        features: [...cullApp.world.gpu.device.features],
        storageBuffers: cullApp.world.gpu.device.limits.maxStorageBuffersPerShaderStage,
        sampledTextures: cullApp.world.gpu.device.limits.maxSampledTexturesPerShaderStage,
        textureDimension: cullApp.world.gpu.device.limits.maxTextureDimension2D,
    });
}, TIMEOUT.startup);
afterAll(() => {
    app?.dispose();
    cullApp?.dispose();
    fetchSpy?.mockRestore();
});

// Absolute + relative 2e-4 admits float32 accumulation/inverse-transpose error, not mesh quantization:
// the reference starts from the decoded position/normal echoed by the actual vertex pull.
function close(actual: number, expected: number) {
    expect(Math.abs(actual - expected)).toBeLessThanOrEqual(2e-4 * Math.max(1, Math.abs(expected)));
}
function expectedVertex(m: number[], p: ArrayLike<number>, n: ArrayLike<number>) {
    const position = [0, 1, 2].map(
        (r) => m[r] * p[0] + m[r + 4] * p[1] + m[r + 8] * p[2] + m[r + 12],
    );
    const cross = (a: number[], b: number[]) => [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    const x = cross(m.slice(4, 7), m.slice(8, 11)),
        y = cross(m.slice(8, 11), m.slice(0, 3)),
        z = cross(m.slice(0, 3), m.slice(4, 7));
    const det = m[8] * z[0] + m[9] * z[1] + m[10] * z[2];
    const normal = [0, 1, 2].map((r) => (n[0] * x[r] + n[1] * y[r] + n[2] * z[r]) / det);
    const length = Math.hypot(...normal);
    return { position, normal: normal.map((v) => v / length) };
}
test("skinning through a game's materialLayout reads every family-absolute vertex and its entity palette at recorded times", async () => {
    expect(scenes[1].bases[0]).toBeGreaterThan(0);
    const meshes = app.world.resource(Meshes);
    const a = meshes.get(scenes[0].gltf.primitives[0].mesh)!,
        b = meshes.get(scenes[1].gltf.primitives[0].mesh)!;
    expect(a.vertices).toBe(b.vertices);
    expect(a.attributes!.deformation).toBe(b.attributes!.deformation);
    expect(
        app.world.storage(SkinnedMesh).firstJoint.get(scenes[1].instance.entities[0]),
    ).toBeGreaterThan(0);
    for (let s = 0; s < scenes.length; s++) {
        const scene = scenes[s];
        for (let animation = 0; animation < scene.record.animations.length; animation++)
            for (const sample of scene.record.animations[animation].samples) {
                poseGltfScene(
                    app.world,
                    scene.instance,
                    scene.gltf.animations[animation],
                    sample.time,
                );
                for (const skin of scene.gltf.skins)
                    for (const joint of skin.joints)
                        for (let c = 0; c < 16; c++)
                            close(scene.instance.globals[joint][c], sample.worlds[joint].matrix[c]);
                for (const eid of scene.instance.entities)
                    app.world.storage(MeshInstance).cullBounds.set(eid, 0, 0, 0, -1);
                app.world.tick();
                app.world.step(1 / 60);
                const result = await probeTexture(app.world, app.world.resource(Results));
                const output = new Float32Array(result.bytes);
                for (let p = 0; p < scene.gltf.primitives.length; p++) {
                    const primitive = scene.gltf.primitives[p];
                    const mesh = scene.decoded.meshes[scene.decoded.instances[p].mesh];
                    const skin = scene.gltf.skins[primitive.skin];
                    for (let v = 0; v < mesh.vertices.length / 8; v++) {
                        const o = (scene.bases[p] + v) * 16;
                        expect(output[o + 15]).toBe(1);
                        expect(output[o + 11]).toBe(scene.bases[p] + v);
                        const m = Array<number>(16).fill(0);
                        for (let k = 0; k < 4; k++) {
                            const joint = mesh.joints![v * 4 + k],
                                weight = mesh.weights![v * 4 + k];
                            const global = sample.worlds[skin.joints[joint]].matrix;
                            const bind = skin.inverseBindMatrices;
                            for (let c = 0; c < 4; c++)
                                for (let r = 0; r < 4; r++)
                                    for (let t = 0; t < 4; t++)
                                        m[c * 4 + r] +=
                                            weight *
                                            global[t * 4 + r] *
                                            bind[joint * 16 + c * 4 + t];
                        }
                        const expected = expectedVertex(
                            m,
                            output.subarray(o + 8, o + 11),
                            output.subarray(o + 12, o + 15),
                        );
                        for (let r = 0; r < 3; r++) {
                            close(output[o + r], expected.position[r]);
                            close(output[o + 4 + r], expected.normal[r]);
                        }
                    }
                }
            }
    }
});

function detachedInstances() {
    const world = cullApp.world;
    poseGltfScene(world, cullScene, cullScene.gltf.animations[0], 1);
    const b = spawnGltfScene(world, cullScene.gltf, { translation: [-80, 0, 0], scale: [2, 3, 1] });
    poseGltfScene(world, b, b.gltf.animations[0], 1);
    const released = world.storage(SkinnedMesh).firstJoint.get(b.entities[0]);
    world.remove(b.entities[0], SkinnedMesh);
    const c = spawnGltfScene(world, cullScene.gltf, { translation: [200, 0, 0] });
    poseGltfScene(world, c, c.gltf.animations[0], 0.5);
    expect(world.storage(SkinnedMesh).firstJoint.get(c.entities[0])).toBe(released);
    return { world, b, c };
}

test("posing a detached skin cannot write another entity's or its replacement's palette", () => {
    const { world, b, c } = detachedInstances();
    try {
        const palette = world.resource(SkinPalette);
        const firstA = world.storage(SkinnedMesh).firstJoint.get(cullScene.entities[0]);
        expect(firstA).toBe(0);
        const firstC = world.storage(SkinnedMesh).firstJoint.get(c.entities[0]);
        const aRows = palette.matrices.slice(firstA * 16, firstA * 16 + 16);
        const cRows = palette.matrices.slice(firstC * 16, firstC * 16 + 16);
        poseGltfScene(world, b, b.gltf.animations[0], 0.25);
        expect(palette.matrices.slice(firstA * 16, firstA * 16 + 16)).toEqual(aRows);
        expect(palette.matrices.slice(firstC * 16, firstC * 16 + 16)).toEqual(cRows);
    } finally {
        for (const instance of [b, c]) for (const eid of instance.entities) world.destroy(eid);
        poseGltfScene(world, cullScene, cullScene.gltf.animations[0], 0);
    }
});

test("a detached skin retaining SkinnedMaterial renders its bind pose, not palette zero", async () => {
    const { world, b, c } = detachedInstances();
    const a = cullScene.entities[0];
    world.remove(a, MeshInstance);
    try {
        expect(world.has(b.entities[0], SkinnedMesh)).toBe(false);
        expect(world.has(b.entities[0], MeshMaterial)).toBe(true);
        world.tick();
        world.step(1 / 60);
        world.step(1 / 60);
        const frame = await captureTexture(world, cullCamera);
        let white = 0;
        for (let i = 0; i < frame.rgba.length; i += 4)
            if (frame.rgba[i] > 200 && frame.rgba[i + 1] > 200 && frame.rgba[i + 2] > 200) white++;
        expect(white).toBeGreaterThan(20);
    } finally {
        world.add(a, MeshInstance, { mesh: cullScene.gltf.primitives[0].mesh });
        for (const instance of [b, c]) for (const eid of instance.entities) world.destroy(eid);
        poseGltfScene(world, cullScene, cullScene.gltf.animations[0], 0);
    }
});

test("a skinned asset refuses by name without GltfPlugin before registering its meshes", async () => {
    const absent = await openGpuApp({ defaults: false, plugins: [] });
    try {
        await expect(loadGltfScene(absent.world, skinningFixtureUrl())).rejects.toThrow(
            "requires GltfPlugin for skinning",
        );
    } finally {
        absent.dispose();
    }
});

test("destroy and respawn return palette allocation to its prior extent, including holes", () => {
    const world = app.world,
        palette = world.resource(SkinPalette),
        before = palette.extent;
    const a = spawnGltfScene(world, scenes[0].gltf),
        b = spawnGltfScene(world, scenes[1].gltf);
    const peak = palette.extent;
    for (const eid of a.entities) world.destroy(eid);
    const replacement = spawnGltfScene(world, scenes[0].gltf);
    expect(palette.extent).toBe(peak);
    for (const instance of [b, replacement])
        for (const eid of instance.entities) world.destroy(eid);
    expect(palette.extent).toBe(before);
    const again = spawnGltfScene(world, scenes[0].gltf);
    for (const eid of again.entities) world.remove(eid, SkinnedMesh);
    expect(palette.extent).toBe(before);
    for (const eid of again.entities) world.destroy(eid);
});

test("posed bounds stay entity-local under translated, nonuniformly scaled placement", async () => {
    const world = cullApp.world;
    world.tick();
    world.step(1 / 60);
    const rest = await captureTexture(world, cullCamera);
    expect(rest.rgba.some((value, i) => i % 4 !== 3 && value > 200)).toBe(false);
    poseGltfScene(world, cullScene, cullScene.gltf.animations[0], 1);
    expect(
        world.resource(Meshes).get(cullScene.gltf.primitives[0].mesh)!.bounds![0],
    ).toBeGreaterThan(40);
    world.tick();
    world.step(1 / 60);
    world.step(1 / 60);
    const frame = await captureTexture(world, cullCamera);
    let white = 0;
    for (let i = 0; i < frame.rgba.length; i += 4)
        if (frame.rgba[i] > 200 && frame.rgba[i + 1] > 200 && frame.rgba[i + 2] > 200) white++;
    expect(white).toBeGreaterThan(20);
    // Placement maps the local sphere to center x=20, radius 3*sqrt(2). A world-space
    // sphere would apply placement again and miss this view. At camera x=29 the correct
    // sphere misses the frustum, but scaling its radius twice would admit an indirect instance.
    try {
        world.storage(Transform).translation.set(cullCamera, 29, 0, 5);
        world.tick();
        world.step(1 / 60);
        world.step(1 / 60);
        const draw = [...world.resource(Draws)].find(
            (candidate) =>
                candidate.mesh === cullScene.gltf.primitives[0].mesh &&
                candidate.alphaPipelineKey === 0 &&
                candidate.material.type === cullScene.gltf.primitives[0].material.type &&
                candidate.cullMode === cullScene.gltf.primitives[0].material.cullMode,
        )!;
        const args = await probeBuffer(world, world.gpu.root.unwrap(draw.args.indirect), {
            offset: draw.args.offset ?? 0,
            size: 20,
        });
        expect(new Uint32Array(args.bytes)[1]).toBe(0);
    } finally {
        world.storage(Transform).translation.set(cullCamera, 20, 0, 5);
        world.tick();
        world.step(1 / 60);
    }
});

test("skinned StandardMaterial fits floor color, prepass and shadow layouts and instances batch into one indirect draw", async () => {
    const world = cullApp.world;
    expect(world.gpu.device.limits.maxStorageBuffersPerShaderStage).toBe(8);
    expect(world.gpu.device.limits.maxTextureDimension2D).toBe(8192);
    const extras = Array.from({ length: 4 }, () =>
        spawnGltfScene(world, cullScene.gltf, { translation: [20, 0, 0], scale: [2, 3, 1] }),
    );
    try {
        for (const instance of [cullScene, ...extras])
            poseGltfScene(world, instance, instance.gltf.animations[0], 1);
        world.tick();
        world.step(1 / 60);
        world.step(1 / 60);
        expect(world.resource(Views).get(cullCamera)!.depth).not.toBeNull();
        expect(sunShadowView(world)).not.toBeNull();
        const draws = [...world.resource(Draws)].filter(
            (draw) =>
                draw.mesh === cullScene.gltf.primitives[0].mesh &&
                draw.alphaPipelineKey === 0 &&
                draw.material.type ===
                    world.storage(MeshMaterial).type.get(cullScene.entities[0]) &&
                draw.cullMode === cullScene.gltf.primitives[0].material.cullMode,
        );
        expect(draws).toHaveLength(1);
        const draw = draws[0];
        const result = await probeBuffer(world, world.gpu.root.unwrap(draw.args.indirect), {
            offset: draw.args.offset ?? 0,
            size: 20,
        });
        const args = new Uint32Array(result.bytes);
        expect(args[0]).toBe(3);
        expect(args[1]).toBe(5);
    } finally {
        for (const instance of extras) for (const eid of instance.entities) world.destroy(eid);
    }
});
