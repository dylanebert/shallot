import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { Meshes, type MeshHandle, MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    CameraMode,
    DepthPrepass,
    DirectionalLight,
    Views,
} from "../../core/rendering";
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
import { morphFixture, morphFixtureUrl } from "./morph.fixture";
import { type GltfSceneInstance, loadGltfScene, poseGltfScene, spawnGltfScene } from "./scene";
import { GltfPlugin } from "./skinning";

setDefaultTimeout(TIMEOUT.gpu);
const Side = 256;
const layout = tgpu.bindGroupLayout({
    view: { texture: d.textureDepth2d() },
    sun: { texture: d.textureDepth2d() },
    out: { storage: d.arrayOf(d.f32), access: "mutable" },
});
// TypeGPU's textureDimensions overloads do not include depth textures.
const sample = tgpu
    .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [8, 8] })(`{
 let i = in.gid.y * ${Side}u + in.gid.x;
 bound.out[i] = textureLoad(bound.view, vec2i(in.gid.xy * textureDimensions(bound.view) / ${Side}u), 0);
 bound.out[${Side * Side}u+i] = textureLoad(bound.sun, vec2i(in.gid.xy * textureDimensions(bound.sun) / ${Side}u), 0);
}`)
    .$uses({ bound: layout.$ });
const Probe = {
    create(world: World) {
        const pipeline = world.gpu.root.createComputePipeline({ compute: sample });
        precompile(world, "morph-depth-probe", () => [pipeline]);
        return {
            pipeline: world.gpu.root.unwrap(pipeline),
            out: world.gpu.device.createBuffer({
                size: Side * Side * 8,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            }),
        };
    },
};
let app: App, camera: number, actual: GltfSceneInstance, reference: number;
let referenceMesh: MeshHandle;
beforeAll(async () => {
    app = await openGpuApp({
        defaults: false,
        plugins: [
            {
                name: "MorphDepth",
                dependencies: [StandardRenderingPlugin, MeshRenderPlugin, GltfPlugin],
                async initialize(world) {
                    camera = world.create();
                    world.add(camera, Transform, { translation: [1.5, 2.6, 6] });
                    world.add(camera, Camera, {
                        mode: CameraMode.Orthographic,
                        size: 3,
                        antialias: 0,
                        far: 12,
                    });
                    world.add(camera, StandardRenderer);
                    world.add(camera, DepthPrepass);
                    attachTexture(world, camera, { width: Side, height: Side });
                    const sun = world.create();
                    world.add(sun, Transform, {
                        rotation: lookAtRotation(0, 0, 0, -0.4, -0.7, -1),
                    });
                    world.add(sun, DirectionalLight, {
                        shadowMapsEnabled: 1,
                        numCascades: 1,
                        maximumDistance: 10,
                        firstCascadeFarBound: 10,
                        depthBias: 0,
                        shadowNormalBias: 0,
                    });
                    const url = morphFixtureUrl(morphFixture().json),
                        decoded = await loadGltf(url);
                    actual = spawnGltfScene(world, await loadGltfScene(world, url));
                    poseGltfScene(world, actual, actual.gltf.animations[0], 0.5);
                    const source = decoded.meshes[0],
                        vertices = source.vertices.slice(),
                        c = Math.cos(Math.PI / 3),
                        s = Math.sin(Math.PI / 3);
                    for (let v = 0; v < vertices.length / 8; v++) {
                        const p = Array.from(source.vertices.subarray(v * 8, v * 8 + 3));
                        for (let m = 0; m < 2; m++)
                            for (let a = 0; a < 3; a++)
                                p[a] +=
                                    [0.5, 0.25][m] *
                                    (source.morphTargets![m].position?.[v * 3 + a] ?? 0);
                        vertices[v * 8] = 1.5 * c * p[0] - 0.75 * s * p[1];
                        vertices[v * 8 + 1] = 1.5 * s * p[0] + 0.75 * c * p[1];
                        vertices[v * 8 + 2] = p[2];
                    }
                    referenceMesh = world
                        .resource(Meshes)
                        .add({ ...source, name: "CPU-morphed-skinned-depth", vertices });
                    reference = world.create();
                    world.add(reference, Transform);
                    world.add(reference, MeshInstance, { mesh: referenceMesh });
                    world.add(
                        reference,
                        MeshMaterial,
                        world
                            .resource(Materials)
                            .add(StandardMaterial({ unlit: true }), { cullMode: "none" }),
                    );
                    world.remove(reference, MeshInstance);
                },
                warm(world) {
                    world.resource(Probe);
                },
            },
        ],
    });
}, TIMEOUT.startup);
afterAll(() => app?.dispose());
async function depths(cpu: boolean) {
    const world = app.world,
        eid = actual.entities[0];
    if (cpu) {
        world.remove(eid, MeshInstance);
        world.add(reference, MeshInstance, { mesh: referenceMesh });
    } else {
        world.remove(reference, MeshInstance);
        world.add(eid, MeshInstance, { mesh: actual.gltf.primitives[0].mesh });
        poseGltfScene(world, actual, actual.gltf.animations[0], 0.5);
    }
    world.tick();
    world.step(1 / 60);
    world.step(1 / 60);
    const probe = world.resource(Probe),
        group = world.gpu.root.createBindGroup(layout, {
            view: world.resource(Views).get(camera)!.depth!,
            sun: sunShadowView(world)!,
            out: probe.out,
        });
    return new Float32Array(
        (
            await probeBuffer(world, probe.out, {
                encode(encoder) {
                    const pass = encoder.beginComputePass();
                    pass.setPipeline(probe.pipeline);
                    pass.setBindGroup(0, world.gpu.root.unwrap(group));
                    pass.dispatchWorkgroups(Side / 8, Side / 8);
                    pass.end();
                },
            })
        ).bytes,
    );
}
for (const [pass, name] of ["prepass", "sun shadow"].entries())
    test(`morph then skin matches CPU silhouette and ${name} depth at recorded time 0.5`, async () => {
        const gpu = await depths(false),
            cpu = await depths(true),
            offset = pass * Side * Side;
        let filled = 0,
            checked = 0,
            bad = 0;
        for (let y = 1; y < Side - 1; y++)
            for (let x = 1; x < Side - 1; x++) {
                const i = offset + y * Side + x,
                    value = cpu[i];
                if (value > 0) filled++;
                // Exclude the reference silhouette's one-pixel rim: GPU packs before deformation,
                // CPU packs after it. Interior and background agree within 2e-4 in reverse-Z depth.
                if ([i - 1, i + 1, i - Side, i + Side].some((n) => cpu[n] > 0 !== value > 0))
                    continue;
                if (value > 0) checked++;
                if (Math.abs(gpu[i] - value) > 2e-4) bad++;
            }
        console.info("morph depth coverage", { pass: name, time: 0.5, filled, checked, bad });
        expect(filled).toBeGreaterThan(10);
        expect(checked).toBeGreaterThan(2);
        expect(bad).toBe(0);
    });
