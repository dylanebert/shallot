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
    NormalPrepass,
    PointLight,
    Views,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { lookAtRotation, type Plugin, precompile, type World } from "../../engine";
import { probeBuffer } from "../../engine/runtime";
import { pointAtlasView, sunShadowView } from "./atlas";
import { StandardRenderer } from "./forward";
import { lit, StandardRenderingPlugin } from "./index";
import { MeshMaterial } from "./material-data";
import { AlphaMode, MaterialPlugin, materialType } from "./material-type";
import { MeshRenderPlugin } from "./mesh-render";

setDefaultTimeout(TIMEOUT.node);

const Parameters = d.struct({ color: d.vec4f });
const common = {
    parameters: Parameters,
    defaults: { color: d.vec4f(1) },
    fragmentInputs: { localPos: true },
    depthPass: { prepass: true, shadows: true },
} as const;
// Cuts the cube's -x half; any lit color lifts alpha past the cutoff.
const LitCut = materialType({
    ...common,
    name: "LitCut",
    fragment: (ctx) => {
        "use gpu";
        const shade = lit(ctx.parameters.color.xyz, ctx.worldNormal);
        return d.vec4f(
            ctx.parameters.color.xyz,
            ctx.localPos.x + 0.5 + std.select(d.f32(0), d.f32(1), shade.x + shade.y + shade.z > 0),
        );
    },
});
// LitCut with lit returning zero.
const UnlitCut = materialType({
    ...common,
    name: "UnlitCut",
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(ctx.parameters.color.xyz, ctx.localPos.x + 0.5);
    },
});

const SIDE = 64;
const depthLayout = tgpu.bindGroupLayout({
    view: { texture: d.textureDepth2d() },
    normalView: { texture: d.textureDepth2d() },
    sun: { texture: d.textureDepth2d() },
    point: { texture: d.textureDepth2d() },
    out: { storage: d.arrayOf(d.f32), access: "mutable" },
});
// TypeGPU's textureDimensions overloads do not include depth textures.
const depthSample = tgpu
    .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [8, 8] })(/* wgsl */ `{
    let i = in.gid.y * ${SIDE}u + in.gid.x;
    bound.out[i] = textureLoad(bound.view, vec2i(in.gid.xy * textureDimensions(bound.view) / ${SIDE}u), 0);
    bound.out[${SIDE * SIDE}u + i] = textureLoad(bound.normalView, vec2i(in.gid.xy * textureDimensions(bound.normalView) / ${SIDE}u), 0);
    bound.out[${2 * SIDE * SIDE}u + i] = textureLoad(bound.sun, vec2i(in.gid.xy * textureDimensions(bound.sun) / ${SIDE}u), 0);
    bound.out[${3 * SIDE * SIDE}u + i] = textureLoad(bound.point, vec2i(in.gid.xy * textureDimensions(bound.point) / ${SIDE}u), 0);
}`)
    .$uses({ bound: depthLayout.$ });
const depthProbe = {
    create(world: World) {
        const pipeline = world.gpu.root
            .createComputePipeline({ compute: depthSample })
            .$name("material-depth-lighting-probe");
        precompile(world, "material-depth-lighting-probe", () => [pipeline]);
        return world.gpu.root.unwrap(pipeline);
    },
};
const ProofPlugin: Plugin = {
    name: "MaterialDepthLightingProof",
    dependencies: [
        MeshRenderPlugin,
        StandardRenderingPlugin,
        MaterialPlugin(LitCut),
        MaterialPlugin(UnlitCut),
    ],
    warm(world) {
        world.resource(depthProbe);
    },
};

const subject = gpuScene(import.meta.path, { defaults: false, plugins: [ProofPlugin] }, (world) => {
    const cameras = [DepthPrepass, NormalPrepass].map((prepass) => {
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 5, 0] });
        world.add(camera, Camera, { antialias: 0 });
        world.add(camera, StandardRenderer);
        world.add(camera, prepass);
        attachTexture(world, camera, { width: 32, height: 32 });
        return camera;
    });
    const caster = world.create();
    world.add(caster, Transform, { rotation: lookAtRotation(0, 0, 0, 0.3, 0.4, 1) });
    world.add(caster, MeshInstance);
    const cut = { alphaMode: AlphaMode.Mask(0.5) };
    const litCut = world.resource(LitCut).add({ color: d.vec4f(1) }, cut);
    const unlitCut = world.resource(UnlitCut).add({ color: d.vec4f(1) }, cut);
    world.add(caster, MeshMaterial, litCut);
    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, -0.4, -0.8, -0.5) });
    world.add(sun, DirectionalLight, { shadowMapsEnabled: 1 });
    const point = world.create();
    world.add(point, Transform, { translation: [2, 2, 3, 0] });
    world.add(point, PointLight, { intensity: 315225.92, range: 10, shadowMapsEnabled: 1 });
    const out = world.gpu.device.createBuffer({
        size: 4 * SIDE * SIDE * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    world.own(out);
    return { cameras, caster, unlitCut, out };
});

test("a Mask material whose alpha derives from lit color cuts as unlit in every depth pass", async () => {
    const {
        world,
        scene: { cameras, caster, unlitCut, out },
    } = subject();
    const device = world.gpu.device;
    const pipeline = world.resource(depthProbe);
    async function depths() {
        device.pushErrorScope("validation");
        world.step(0);
        world.step(0);
        const views = world.resource(Views);
        const group = world.gpu.root.createBindGroup(depthLayout, {
            view: views.get(cameras[0]!)!.depth!,
            normalView: views.get(cameras[1]!)!.depth!,
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
        const snapshot = await probeBuffer(world, out, { size: 4 * SIDE * SIDE * 4 });
        expect(await device.popErrorScope()).toBeNull();
        const all = new Float32Array(snapshot.bytes);
        return [0, 1, 2, 3].map((k) =>
            Array.from(all.subarray(k * SIDE * SIDE, (k + 1) * SIDE * SIDE)),
        );
    }
    const drawn = await depths();
    world.remove(caster, MeshMaterial);
    world.add(caster, MeshMaterial, unlitCut);
    const expected = await depths();
    const names = ["depth prepass", "normal prepass", "sun shadow", "point shadow"];
    for (let k = 0; k < names.length; k++) {
        expect(
            expected[k]!.some((depth) => depth > 0),
            names[k],
        ).toBe(true);
    }
    const differing = names.map(
        (_, k) => drawn[k]!.filter((depth, i) => depth !== expected[k]![i]).length,
    );
    console.log(`material depth lighting: samples differing from unlit ${differing.join("/")}`);
    expect(differing).toEqual([0, 0, 0, 0]);
});
