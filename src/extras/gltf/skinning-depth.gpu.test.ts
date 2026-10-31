import { afterAll, beforeAll, expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import gold from "../../../examples/gltf/tests/animations.gold.json";
import { fileFetch, syncAssets } from "../../../examples/gltf/tests/setup";
import { Meshes, type MeshHandle, MeshInstance } from "../../core/mesh";
import { attachTexture, Camera, DepthPrepass, DirectionalLight, Views } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { type App, lookAtRotation, precompile, type World } from "../../engine";
import { probeBuffer } from "../../engine/runtime";
import { openGpuApp } from "../../project/testing-gpu";
import {
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "../../standard/rendering";
import { sunShadowView } from "../../standard/rendering/atlas";
import { loadGltf } from "./load";
import { type GltfSceneInstance, loadGltfScene, poseGltfScene, spawnGltfScene } from "./scene";
import { GltfPlugin } from "./skinning";

setDefaultTimeout(TIMEOUT.gpu);
const SIDE = 256;
const probeLayout = tgpu.bindGroupLayout({
    view: { texture: d.textureDepth2d() },
    sun: { texture: d.textureDepth2d() },
    out: { storage: d.arrayOf(d.f32), access: "mutable" },
});
// TypeGPU's textureDimensions overloads do not include depth textures.
const sampleDepth = tgpu
    .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [8, 8] })(`{
    let i = in.gid.y * ${SIDE}u + in.gid.x;
    bound.out[i] = textureLoad(bound.view, vec2i(in.gid.xy * textureDimensions(bound.view) / ${SIDE}u), 0);
    bound.out[${SIDE * SIDE}u + i] = textureLoad(bound.sun, vec2i(in.gid.xy * textureDimensions(bound.sun) / ${SIDE}u), 0);
}`)
    .$uses({ bound: probeLayout.$ });
const DepthProbe = {
    create(world: World) {
        const pipeline = world.gpu.root.createComputePipeline({ compute: sampleDepth });
        precompile(world, "gltf-skin-depth-probe", () => [pipeline]);
        return {
            pipeline: world.gpu.root.unwrap(pipeline),
            out: world.gpu.device.createBuffer({
                size: 2 * SIDE * SIDE * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            }),
        };
    },
};
let app: App;
let camera: number;
let neighbor: MeshHandle;
let actual: GltfSceneInstance;
const references: { eid: number; mesh: MeshHandle }[] = [];
const record = gold.records.find((r) => r.fixture.includes("/RiggedFigure/"))!;
const sample = record.animations[0].samples[Math.floor(record.animations[0].samples.length / 2)];
let fetchSpy: ReturnType<typeof spyOn>;
beforeAll(async () => {
    await syncAssets();
    fetchSpy = spyOn(globalThis, "fetch").mockImplementation(fileFetch);
    app = await openGpuApp({
        defaults: false,
        plugins: [
            {
                name: "SkinDepthParity",
                dependencies: [StandardRenderingPlugin, MeshRenderPlugin, GltfPlugin],
                async initialize(world) {
                    camera = world.create();
                    world.add(camera, Transform, {
                        translation: [2, 1.8, 4],
                        rotation: lookAtRotation(2, 1.8, 4, 0, 0.7, 0),
                    });
                    world.add(camera, Camera, { antialias: 0, far: 10 });
                    world.add(camera, StandardRenderer);
                    world.add(camera, DepthPrepass);
                    attachTexture(world, camera, { width: SIDE, height: SIDE });
                    const sun = world.create();
                    world.add(sun, Transform, {
                        rotation: lookAtRotation(0, 0, 0, -0.4, -1, -0.3),
                    });
                    world.add(sun, DirectionalLight, {
                        shadowMapsEnabled: 1,
                        numCascades: 1,
                        maximumDistance: 8,
                        firstCascadeFarBound: 8,
                        depthBias: 0,
                        shadowNormalBias: 0,
                    });
                    neighbor = (
                        await loadGltfScene(
                            world,
                            new URL(
                                "../../../examples/gltf/public/assets/Models/RiggedSimple/glTF/RiggedSimple.gltf",
                                import.meta.url,
                            ),
                        )
                    ).primitives[0].mesh;
                    const url = new URL(
                        "../../../examples/gltf/public/assets/Models/RiggedFigure/glTF/RiggedFigure.gltf",
                        import.meta.url,
                    );
                    const gltf = await loadGltfScene(world, url);
                    const decoded = await loadGltf(url);
                    actual = spawnGltfScene(world, gltf);
                    poseGltfScene(world, actual, gltf.animations[0], sample.time);
                    const material = world
                        .resource(Materials)
                        .add(StandardMaterial({ unlit: true }));
                    for (let p = 0; p < gltf.primitives.length; p++) {
                        const primitive = gltf.primitives[p],
                            skin = gltf.skins[primitive.skin];
                        const source = decoded.meshes[decoded.instances[p].mesh];
                        const vertices = source.vertices.slice();
                        for (let v = 0; v < vertices.length / 8; v++) {
                            const m = Array<number>(16).fill(0);
                            for (let k = 0; k < 4; k++) {
                                const joint = source.joints![v * 4 + k],
                                    weight = source.weights![v * 4 + k];
                                const global = sample.worlds[skin.joints[joint]].matrix;
                                for (let c = 0; c < 4; c++)
                                    for (let r = 0; r < 4; r++)
                                        for (let t = 0; t < 4; t++)
                                            m[c * 4 + r] +=
                                                weight *
                                                global[t * 4 + r] *
                                                skin.inverseBindMatrices[joint * 16 + c * 4 + t];
                            }
                            for (let r = 0; r < 3; r++)
                                vertices[v * 8 + r] =
                                    m[r] * source.vertices[v * 8] +
                                    m[r + 4] * source.vertices[v * 8 + 1] +
                                    m[r + 8] * source.vertices[v * 8 + 2] +
                                    m[r + 12];
                        }
                        const mesh = world
                            .resource(Meshes)
                            .add({ ...source, name: "CPU-skinned-silhouette", vertices });
                        const eid = world.create();
                        world.add(eid, Transform);
                        world.add(eid, MeshInstance, { mesh });
                        world.add(eid, MeshMaterial, material);
                        references.push({ eid, mesh });
                    }
                    for (const { eid } of references) world.remove(eid, MeshInstance);
                },
                warm(world) {
                    world.resource(DepthProbe);
                },
            },
        ],
    });
    const mesh = app.world.resource(Meshes).get(actual.gltf.primitives[0].mesh)!;
    const indices = await mesh.indices.read();
    expect(Math.min(...indices.slice(mesh.indexBase, mesh.indexBase + mesh.indexCount))).toBe(160);
    const first = app.world.resource(Meshes).get(neighbor)!;
    expect(mesh.vertices).toBe(first.vertices);
    expect(mesh.attributes!.deformation).toBe(first.attributes!.deformation);
}, TIMEOUT.startup);
afterAll(() => {
    app?.dispose();
    fetchSpy?.mockRestore();
});

async function depths(reference: boolean): Promise<Float32Array> {
    const world = app.world;
    for (let p = 0; p < actual.entities.length; p++) {
        const eid = actual.entities[p];
        if (reference) world.remove(eid, MeshInstance);
        else world.add(eid, MeshInstance, { mesh: actual.gltf.primitives[p].mesh });
    }
    for (const { eid, mesh } of references) {
        if (reference) world.add(eid, MeshInstance, { mesh });
        else world.remove(eid, MeshInstance);
    }
    if (!reference) poseGltfScene(world, actual, actual.gltf.animations[0], sample.time);
    world.tick();
    world.step(1 / 60);
    world.step(1 / 60);
    const probe = world.resource(DepthProbe);
    const group = world.gpu.root.createBindGroup(probeLayout, {
        view: world.resource(Views).get(camera)!.depth!,
        sun: sunShadowView(world)!,
        out: probe.out,
    });
    const result = await probeBuffer(world, probe.out, {
        encode(encoder) {
            const pass = encoder.beginComputePass();
            pass.setPipeline(probe.pipeline);
            pass.setBindGroup(0, world.gpu.root.unwrap(group));
            pass.dispatchWorkgroups(SIDE / 8, SIDE / 8);
            pass.end();
        },
    });
    return new Float32Array(result.bytes);
}

for (const [pass, label] of ["prepass", "sun shadow"].entries())
    test(`nonzero-family-base skinning matches CPU-skinned ${label} depth and silhouette`, async () => {
        const gpu = await depths(false),
            cpu = await depths(true);
        const offset = pass * SIDE * SIDE;
        let filled = 0,
            checkedFilled = 0,
            bad = 0;
        for (let y = 1; y < SIDE - 1; y++)
            for (let x = 1; x < SIDE - 1; x++) {
                const i = offset + y * SIDE + x,
                    value = cpu[i];
                if (value > 0) filled++;
                // The GPU skins packed bind positions; the CPU reference packs after skinning. Exclude
                // the one-pixel reference silhouette/discontinuity rim for that unorm16 ordering difference.
                // Interior and background depth must agree within 2e-4.
                if (
                    [i - 1, i + 1, i - SIDE, i + SIDE].some(
                        (n) => cpu[n] > 0 !== value > 0 || Math.abs(cpu[n] - value) > 1e-3,
                    )
                )
                    continue;
                if (value > 0) checkedFilled++;
                if (Math.abs(gpu[i] - value) > 2e-4) bad++;
            }
        console.info("skinning depth coverage", {
            pass: label,
            time: sample.time,
            filled,
            checkedFilled,
            bad,
        });
        expect(filled).toBeGreaterThan(10);
        expect(checkedFilled).toBeGreaterThan(2);
        expect(bad).toBe(0);
    });
