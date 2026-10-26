import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { gpuScene } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    DepthPrepass,
    DirectionalLight,
    PointLight,
    Views,
} from "../../core/rendering";
import { xformPoint } from "../../core/rendering/xform";
import { Transform } from "../../core/transform";
import { lookAtRotation, type Plugin, precompile, type World } from "../../engine";
import { probeBuffer } from "../../engine/runtime";
import { pointAtlasView, sunShadowView } from "./atlas";
import { StandardRenderer } from "./forward";
import { StandardRenderingPlugin } from "./index";
import { MeshMaterial } from "./material-data";
import { AlphaMode, MaterialPlugin, materialType } from "./material-type";
import { MeshRenderPlugin } from "./mesh-render";

setDefaultTimeout(TIMEOUT.node);

// Inflates along the mesh normal and slides along its uv, so a pass that hands the vertex function
// any other normal or uv draws a different silhouette.
const DisplacedParameters = d.struct({ color: d.vec4f });
const DisplacedMaterial = materialType({
    name: "MaterialVertexDepth",
    parameters: DisplacedParameters,
    defaults: { color: d.vec4f(1) },
    vertex: (input) => {
        "use gpu";
        const local = std.add(
            std.add(input.localPos, std.mul(input.localNormal, 0.3)),
            d.vec3f(std.mul(input.uv, 0.4), 0),
        );
        return {
            world: d.vec4f(xformPoint(input.xform, local), 1),
            worldNormal: input.worldNormal,
        };
    },
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(ctx.parameters.color);
    },
    depthPass: { prepass: true, shadows: true },
});

const SIDE = 64;
const depthLayout = tgpu.bindGroupLayout({
    view: { texture: d.textureDepth2d() },
    sun: { texture: d.textureDepth2d() },
    point: { texture: d.textureDepth2d() },
    out: { storage: d.arrayOf(d.f32), access: "mutable" },
});
// TypeGPU's textureDimensions overloads do not include depth textures.
const depthSample = tgpu
    .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [8, 8] })(/* wgsl */ `{
    let i = in.gid.y * ${SIDE}u + in.gid.x;
    let viewPx = vec2i(in.gid.xy * textureDimensions(bound.view) / ${SIDE}u);
    let sunPx = vec2i(in.gid.xy * textureDimensions(bound.sun) / ${SIDE}u);
    let pointPx = vec2i(in.gid.xy * textureDimensions(bound.point) / ${SIDE}u);
    bound.out[i] = textureLoad(bound.view, viewPx, 0);
    bound.out[${SIDE * SIDE}u + i] = textureLoad(bound.sun, sunPx, 0);
    bound.out[${2 * SIDE * SIDE}u + i] = textureLoad(bound.point, pointPx, 0);
}`)
    .$uses({ bound: depthLayout.$ });
const depthProbe = {
    create(world: World) {
        const pipeline = world.gpu.root
            .createComputePipeline({ compute: depthSample })
            .$name("material-vertex-depth-probe");
        precompile(world, "material-vertex-depth-probe", () => [pipeline]);
        return world.gpu.root.unwrap(pipeline);
    },
};
const DisplacedPlugin: Plugin = {
    name: "MaterialVertexDepthProof",
    dependencies: [MeshRenderPlugin, StandardRenderingPlugin, MaterialPlugin(DisplacedMaterial)],
    warm(world) {
        world.resource(depthProbe);
    },
};

const subject = gpuScene(
    import.meta.path,
    { defaults: false, plugins: [DisplacedPlugin] },
    (world) => {
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 5, 0] });
        world.add(camera, Camera, { antialias: 0 });
        world.add(camera, StandardRenderer);
        world.add(camera, DepthPrepass);
        attachTexture(world, camera, { width: 32, height: 32 });
        const caster = world.create();
        world.add(caster, Transform, { rotation: lookAtRotation(0, 0, 0, 0.3, 0.4, 1) });
        world.add(caster, MeshInstance);
        const materials = world.resource(DisplacedMaterial);
        const opaque = materials.add({ color: d.vec4f(1) });
        const masked = materials.add({ color: d.vec4f(1) }, { alphaMode: AlphaMode.Mask(0.5) });
        world.add(caster, MeshMaterial, opaque);
        const sun = world.create();
        world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, -0.4, -0.8, -0.5) });
        world.add(sun, DirectionalLight, { shadowMapsEnabled: 1 });
        const point = world.create();
        world.add(point, Transform, { translation: [2, 2, 3, 0] });
        world.add(point, PointLight, { intensity: 315225.92, range: 10, shadowMapsEnabled: 1 });
        const out = world.gpu.device.createBuffer({
            size: 3 * SIDE * SIDE * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        world.own(out);
        return { camera, caster, opaque, masked, out };
    },
);

test("a material vertex function sees the mesh's normal and uv in every depth pass", async () => {
    const {
        world,
        scene: { camera, caster, masked, out },
    } = subject();
    const device = world.gpu.device;
    const pipeline = world.resource(depthProbe);
    async function depths() {
        device.pushErrorScope("validation");
        world.step(0);
        world.step(0);
        const group = world.gpu.root.createBindGroup(depthLayout, {
            view: world.resource(Views).get(camera)!.depth!,
            sun: sunShadowView(world)!,
            point: pointAtlasView(world)!,
            out,
        });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, world.gpu.root.unwrap(group));
        pass.dispatchWorkgroups(SIDE / 8, SIDE / 8);
        pass.end();
        device.queue.submit([encoder.finish()]);
        const snapshot = await probeBuffer(world, out, { size: 3 * SIDE * SIDE * 4 });
        expect(await device.popErrorScope()).toBeNull();
        const all = new Float32Array(snapshot.bytes);
        return [0, 1, 2].map((k) =>
            Array.from(all.subarray(k * SIDE * SIDE, (k + 1) * SIDE * SIDE)),
        );
    }
    const drawn = await depths();
    world.remove(caster, MeshMaterial);
    // Mask runs the main vertex stream in every depth pass, so its depths are the surface as drawn.
    world.add(caster, MeshMaterial, masked);
    const expected = await depths();
    const names = ["view prepass", "sun shadow", "point shadow"];
    for (let k = 0; k < 3; k++) {
        expect(
            expected[k]!.some((depth) => depth > 0),
            names[k],
        ).toBe(true);
    }
    const differing = names.map(
        (_, k) => drawn[k]!.filter((depth, i) => depth !== expected[k]![i]).length,
    );
    console.log(`material vertex depth: samples differing from Mask ${differing.join("/")}`);
    expect(differing).toEqual([0, 0, 0]);
});
