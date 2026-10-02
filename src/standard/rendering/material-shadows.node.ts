import { expect, setDefaultTimeout, test } from "bun:test";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    DirectionalLight,
    NotShadowCaster,
    PointLight,
} from "../../core/rendering";
import { type Plugin, Transform, type World } from "../../engine";
import { probeBuffer } from "../../engine/runtime";
import { Xform } from "../../engine/utils";
import { pointAtlasView, sunShadowView } from "./atlas";
import { fsCtxSchema, registerSurface, Surfaces, surfaceLayout } from "./contract";
import { StandardRenderer } from "./forward";
import { StandardRenderingPlugin } from "./index";
import { Materials, MeshMaterial, materialTable, StandardMaterial } from "./material";
import { MeshRenderPlugin } from "./part-plugin";

setDefaultTimeout(CEILING.node);
const cutout: Plugin = {
    name: "MaterialShadowProof",
    dependencies: [MeshRenderPlugin, StandardRenderingPlugin],
    initialize(world) {
        const layout = surfaceLayout({
            eids: { type: "storage", element: d.vec4u },
            globalTransforms: { type: "storage", element: Xform },
        });
        registerSurface(world, {
            name: "material-cutout",
            layout,
            blend: "clip",
            fs: tgpu.fn(
                [fsCtxSchema()],
                d.vec4f,
            )((ctx) => {
                "use gpu";
                if (ctx.color.x < 0.5) std.discard();
                return d.vec4f(ctx.color);
            }),
        });
    },
    warm(world) {
        world.resource(depthProbe);
    },
};
const depthLayout = tgpu.bindGroupLayout({
    sun: { texture: d.textureDepth2d() },
    point: { texture: d.textureDepth2d() },
    counts: { storage: d.arrayOf(d.atomic(d.u32)), access: "mutable" },
});
// TypeGPU's textureDimensions overloads do not include depth textures.
const depthCount = tgpu
    .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [8, 8] })(/* wgsl */ `{
    let sunPx = vec2i(in.gid.xy * textureDimensions(bound.sun) / 64u);
    let pointPx = vec2i(in.gid.xy * textureDimensions(bound.point) / 64u);
    if (textureLoad(bound.sun, sunPx, 0) > 0.0) { atomicAdd(&bound.counts[0], 1u); }
    if (textureLoad(bound.point, pointPx, 0) > 0.0) { atomicAdd(&bound.counts[1], 1u); }
}`)
    .$uses({ bound: depthLayout.$ });
const depthProbe = {
    create: (world: World) =>
        world.gpu.root.unwrap(world.gpu.root.createComputePipeline({ compute: depthCount })),
};
const subjects = gpuApps(import.meta.path, [{ defaults: false, plugins: [cutout] }]);

test("NotShadowCaster preserves visibility and peers' shadows; shadow materials survive table growth", async () => {
    const { world } = subjects()[0];
    const device = world.gpu.device;
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: 16, height: 16 });
    const caster = world.create();
    world.add(caster, Transform);
    world.add(caster, MeshInstance);
    const materials = world.resource(Materials);
    const surface = world.resource(Surfaces).id("material-cutout")!;
    const values = StandardMaterial({ surface, baseColor: [1, 0, 0, 1] });
    world.add(caster, MeshMaterial, { material: materials.add(values) });
    const sun = world.create();
    world.add(sun, DirectionalLight, { direction: [-0.4, -0.8, -0.5, 0] });
    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);
    const point = world.create();
    world.add(point, Transform, { translation: [2, 2, 3, 0] });
    world.add(point, PointLight, { intensity: 8, range: 10 });
    world.storage(PointLight).shadowMapsEnabled.set(point, 1);
    const counts = device.createBuffer({
        size: 8,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    world.own(counts);
    const pipeline = world.resource(depthProbe);
    async function occupied() {
        device.pushErrorScope("validation");
        world.step(0);
        world.step(0);
        device.queue.writeBuffer(counts, 0, new Uint32Array(2));
        const group = world.gpu.root.createBindGroup(depthLayout, {
            sun: sunShadowView(world)!,
            point: pointAtlasView(world)!,
            counts,
        });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, world.gpu.root.unwrap(group));
        pass.dispatchWorkgroups(8, 8);
        pass.end();
        device.queue.submit([encoder.finish()]);
        const snapshot = await probeBuffer(world, counts, { size: 8 });
        expect(await device.popErrorScope()).toBeNull();
        return Array.from(new Uint32Array(snapshot.bytes));
    }
    const before = await occupied();
    expect(before[0]).toBeGreaterThan(0);
    expect(before[1]).toBeGreaterThan(0);
    const visibleBefore = (await captureTexture(world, camera)).rgba;
    world.add(caster, NotShadowCaster);
    expect(await occupied()).toEqual([0, 0]);
    expect((await captureTexture(world, camera)).rgba).toEqual(visibleBefore);
    const other = world.create();
    world.add(other, Transform);
    world.add(other, MeshInstance);
    world.add(other, MeshMaterial, {
        material: world.storage(MeshMaterial).material.get(caster),
    });
    expect(await occupied()).toEqual(before);
    world.destroy(other);
    expect(await occupied()).toEqual([0, 0]);
    world.remove(caster, NotShadowCaster);
    expect(await occupied()).toEqual(before);
    const table = materialTable(world);
    const generation = table.generation;
    const capacity = table.capacity;
    let material = 0;
    for (let i = 0; i < capacity * 4; i++) material = materials.add(values);
    expect(table.generation - generation).toBeGreaterThanOrEqual(2);
    expect(material).toBeGreaterThanOrEqual(capacity);
    world.storage(MeshMaterial).material.set(caster, material);
    expect(await occupied()).toEqual(before);
    materials.update(material, { baseColor: [0, 0, 0, 1] });
    expect(await occupied()).toEqual([0, 0]);
    materials.update(material, { baseColor: [1, 0, 0, 1] });
    expect(await occupied()).toEqual(before);
    console.log(
        `material shadows: generation ${generation} -> ${table.generation}; occupied sun/point samples ${before.join("/")} -> 0/0 -> ${before.join("/")}`,
    );
});
