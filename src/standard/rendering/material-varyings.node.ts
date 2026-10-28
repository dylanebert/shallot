import { afterAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { compileGpuFile, createTestGpuApp, disposeGpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    DepthPrepass,
    DirectionalLight,
    PointLight,
    Tonemapping,
    TonemappingMethod,
    Views,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { lookAtRotation, type Plugin, precompile, type World } from "../../engine";
import { probeBuffer } from "../../engine/runtime";
import { pointAtlasView, sunShadowView } from "./atlas";
import { StandardRenderer } from "./forward";
import { StandardRenderingPlugin } from "./index";
import { MeshMaterial } from "./material-data";
import { AlphaMode, type MaterialHandle, MaterialPlugin, materialType } from "./material-type";
import { MeshRenderPlugin } from "./mesh-render";

setDefaultTimeout(TIMEOUT.node);
const Parameters = d.struct({ color: d.vec4f });
const defaults = { color: d.vec4f(0.8, 0.2, 0.1, 1) };
const common = {
    parameters: Parameters,
    defaults,
    fragmentInputs: { uv: true, localPos: true },
    depthPass: { prepass: true, shadows: true },
} as const;
const Two = materialType({
    ...common,
    name: "TwoVaryings",
    varyings: { a: d.f32, b: d.f32 },
    vertex: (input) => {
        "use gpu";
        return {
            world: input.world,
            worldNormal: input.worldNormal,
            a: input.localPos.x + 1,
            b: input.localPos.x * 2 + 2,
        };
    },
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(ctx.parameters.color.xyz, std.select(0, 1, ctx.a + ctx.b * 2 - 5 > 0.625));
    },
});
const Six = materialType({
    ...common,
    name: "SixVaryings",
    varyings: { a: d.f32, b: d.f32, c: d.f32, e: d.f32, f: d.f32, g: d.f32 },
    vertex: (input) => {
        "use gpu";
        return {
            world: input.world,
            worldNormal: input.worldNormal,
            a: input.localPos.x + 8,
            b: input.localPos.x + 9,
            c: input.localPos.x + 10,
            e: input.localPos.x + 11,
            f: input.localPos.x + 12,
            g: input.localPos.x + 13,
        };
    },
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(
            ctx.parameters.color.xyz,
            std.select(
                0,
                1,
                ctx.a + ctx.b * 2 + ctx.c * 3 + ctx.e * 4 + ctx.f * 5 + ctx.g * 6 - 238 > 2.625,
            ),
        );
    },
});
const Reference = materialType({
    ...common,
    name: "VaryingReference",
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(
            ctx.parameters.color.xyz,
            std.select(
                0,
                1,
                ctx.localPos.x * ctx.parameters.color.w > ctx.parameters.color.w * 0.125,
            ),
        );
    },
});
const Row = materialType({
    ...common,
    name: "VaryingAlphaRow",
    varyings: { a: d.f32 },
    vertex: (input) => {
        "use gpu";
        return { world: input.world, worldNormal: input.worldNormal, a: input.localPos.x };
    },
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(
            ctx.alphaCutoff,
            d.f32(ctx.alphaMode) * 0.1,
            0,
            std.select(0, 1, ctx.a > ctx.alphaCutoff - 0.25),
        );
    },
});
const RowReference = materialType({
    ...common,
    name: "AlphaRowReference",
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(
            ctx.alphaCutoff,
            d.f32(ctx.alphaMode) * 0.1,
            0,
            std.select(0, 1, ctx.localPos.x > ctx.alphaCutoff - 0.25),
        );
    },
});
const Seven = materialType({
    ...common,
    name: "SevenVaryings",
    varyings: { a: d.f32, b: d.f32, c: d.f32, e: d.f32, f: d.f32, g: d.f32, h: d.f32 },
    vertex: (input) => {
        "use gpu";
        return {
            world: input.world,
            worldNormal: input.worldNormal,
            a: input.localPos.x + 8,
            b: input.localPos.x + 9,
            c: input.localPos.x + 10,
            e: input.localPos.x + 11,
            f: input.localPos.x + 12,
            g: input.localPos.x + 13,
            h: input.localPos.x + 14,
        };
    },
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(
            ctx.parameters.color.xyz,
            std.select(
                0,
                1,
                ctx.a +
                    ctx.b * 2 +
                    ctx.c * 3 +
                    ctx.e * 4 +
                    ctx.f * 5 +
                    ctx.g * 6 +
                    ctx.h * 7 -
                    336 >
                    3.5,
            ),
        );
    },
});
const Eight = materialType({
    ...common,
    name: "EightVaryings",
    varyings: { a: d.f32, b: d.f32, c: d.f32, e: d.f32, f: d.f32, g: d.f32, h: d.f32, k: d.f32 },
    vertex: (input) => {
        "use gpu";
        return {
            world: input.world,
            worldNormal: input.worldNormal,
            a: input.localPos.x,
            b: input.localPos.x,
            c: input.localPos.x,
            e: input.localPos.x,
            f: input.localPos.x,
            g: input.localPos.x,
            h: input.localPos.x,
            k: input.localPos.x,
        };
    },
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(ctx.a + ctx.b + ctx.c + ctx.e + ctx.f + ctx.g + ctx.h + ctx.k);
    },
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
    .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [8, 8] })(`{
    let i = in.gid.y * ${SIDE}u + in.gid.x;
    bound.out[i] = textureLoad(bound.view, vec2i(in.gid.xy * textureDimensions(bound.view) / ${SIDE}u), 0);
    bound.out[${SIDE * SIDE}u + i] = textureLoad(bound.sun, vec2i(in.gid.xy * textureDimensions(bound.sun) / ${SIDE}u), 0);
    bound.out[${2 * SIDE * SIDE}u + i] = textureLoad(bound.point, vec2i(in.gid.xy * textureDimensions(bound.point) / ${SIDE}u), 0);
}`)
    .$uses({ bound: depthLayout.$ });
const depthProbe = {
    create(world: World) {
        const pipeline = world.gpu.root.createComputePipeline({ compute: depthSample });
        precompile(world, "material-varyings-depth-probe", () => [pipeline]);
        return world.gpu.root.unwrap(pipeline);
    },
};
const cases = [
    {
        name: "two-varying Mask",
        material: Two,
        reference: Reference,
        weight: 5,
        mode: AlphaMode.Mask(0.5),
    },
    {
        name: "six-varying Mask",
        material: Six,
        reference: Reference,
        weight: 21,
        mode: AlphaMode.Mask(0.5),
    },
    {
        name: "six-varying AlphaToCoverage",
        material: Six,
        reference: Reference,
        weight: 21,
        mode: AlphaMode.AlphaToCoverage,
    },
    {
        name: "seven-varying Mask at the inter-stage limit",
        material: Seven,
        reference: Reference,
        weight: 28,
        mode: AlphaMode.Mask(0.5),
    },
    {
        name: "varying alpha row",
        material: Row,
        reference: RowReference,
        weight: 1,
        mode: AlphaMode.Mask(0.25),
    },
] as const;
type App = Awaited<ReturnType<typeof createTestGpuApp>>;
const apps: App[] = [];
const subjects = compileGpuFile(import.meta.path, async () => {
    const results: (
        | {
              world: World;
              camera: number;
              colorCamera: number;
              caster: number;
              actual: MaterialHandle;
              reference: MaterialHandle;
              out: GPUBuffer;
          }
        | Error
    )[] = [];
    for (const subject of cases) {
        let scene!: Exclude<(typeof results)[number], Error>;
        const plugin: Plugin = {
            name: "VaryingProof",
            dependencies: [
                MeshRenderPlugin,
                StandardRenderingPlugin,
                MaterialPlugin(subject.material),
                MaterialPlugin(subject.reference),
            ],
            initialize(world) {
                const camera = world.create();
                world.add(camera, Transform, { translation: [0, 0, 5, 0] });
                world.add(camera, Camera, { antialias: 0 });
                world.add(camera, StandardRenderer);
                world.add(camera, DepthPrepass);
                world.add(camera, Tonemapping, { method: TonemappingMethod.None });
                attachTexture(world, camera, { width: 32, height: 32 });
                const colorCamera = world.create();
                world.add(colorCamera, Transform, { translation: [0, 0, 5, 0] });
                world.add(colorCamera, Camera, { antialias: 1 });
                world.add(colorCamera, StandardRenderer);
                world.add(colorCamera, Tonemapping, { method: TonemappingMethod.None });
                attachTexture(world, colorCamera, { width: 32, height: 32 });
                const caster = world.create();
                world.add(caster, Transform, { rotation: lookAtRotation(0, 0, 0, 0.3, 0.4, 1) });
                world.add(caster, MeshInstance);
                const actual = world
                    .resource(subject.material)
                    .add(defaults, { alphaMode: subject.mode });
                const reference = world
                    .resource(subject.reference)
                    .add(
                        { color: d.vec4f(0.8, 0.2, 0.1, subject.weight) },
                        { alphaMode: subject.mode },
                    );
                world.add(caster, MeshMaterial, actual);
                const sun = world.create();
                world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, -0.4, -0.8, -0.5) });
                world.add(sun, DirectionalLight, { shadowMapsEnabled: 1 });
                const point = world.create();
                world.add(point, Transform, { translation: [2, 2, 3, 0] });
                world.add(point, PointLight, {
                    intensity: 315225.92,
                    range: 10,
                    shadowMapsEnabled: 1,
                });
                const out = world.gpu.device.createBuffer({
                    size: 3 * SIDE * SIDE * 4,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
                });
                world.own(out);
                scene = { world, camera, colorCamera, caster, actual, reference, out };
            },
            warm(world) {
                world.resource(depthProbe);
            },
        };
        try {
            apps.push(await createTestGpuApp({ defaults: false, plugins: [plugin] }));
            results.push(scene);
        } catch (error) {
            results.push(error as Error);
        }
    }
    let overflow: unknown;
    try {
        apps.push(
            await createTestGpuApp({
                defaults: false,
                plugins: [MeshRenderPlugin, StandardRenderingPlugin, MaterialPlugin(Eight)],
            }),
        );
    } catch (error) {
        overflow = error;
    }
    return { results, overflow };
});
afterAll(() => disposeGpuApps(apps));
async function observe(scene: Exclude<ReturnType<typeof subjects>["results"][number], Error>) {
    const { world, camera, out } = scene;
    const device = world.gpu.device;
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
    pass.setPipeline(world.resource(depthProbe));
    pass.setBindGroup(0, world.gpu.root.unwrap(group));
    pass.dispatchWorkgroups(SIDE / 8, SIDE / 8);
    pass.end();
    device.queue.submit([encoder.finish()]);
    const depths = new Float32Array(
        (await probeBuffer(world, out, { size: 3 * SIDE * SIDE * 4 })).bytes,
    );
    const color = (await captureTexture(world, scene.colorCamera)).rgba;
    expect(await device.popErrorScope()).toBeNull();
    return { depths, color };
}
for (const [index, subject] of cases.entries()) {
    for (const [pass, label] of ["color", "prepass depth", "sun depth", "point depth"].entries()) {
        test(`${subject.name} matches its localPos reference in ${label}`, async () => {
            const scene = subjects().results[index]!;
            if (scene instanceof Error) throw scene;
            expect(scene.world.gpu.device.limits.maxInterStageShaderVariables).toBe(16);
            scene.world.remove(scene.caster, MeshMaterial);
            scene.world.add(scene.caster, MeshMaterial, scene.actual);
            const actual = await observe(scene);
            scene.world.remove(scene.caster, MeshMaterial);
            scene.world.add(scene.caster, MeshMaterial, scene.reference);
            const reference = await observe(scene);
            if (pass === 0) {
                expect(actual.color).toEqual(reference.color);
            } else {
                const start = (pass - 1) * SIDE * SIDE;
                const end = pass * SIDE * SIDE;
                const expected = reference.depths.slice(start, end);
                expect(expected.some((v) => v > 0)).toBe(true);
                expect(actual.depths.slice(start, end)).toEqual(expected);
            }
        });
    }
}
test("a varying Mask fragment reads its row's cutoff and mode in color", async () => {
    const scene = subjects().results[4]!;
    if (scene instanceof Error) throw scene;
    scene.world.remove(scene.caster, MeshMaterial);
    scene.world.add(scene.caster, MeshMaterial, scene.actual);
    const actual = await observe(scene);
    scene.world.remove(scene.caster, MeshMaterial);
    scene.world.add(scene.caster, MeshMaterial, scene.reference);
    const reference = await observe(scene);
    const pixel = (rgba: Uint8Array | Uint8ClampedArray) => {
        for (let i = 0; i < rgba.length; i += 4) {
            if (rgba[i]! > 100 && rgba[i + 2] === 0) {
                const linear = (v: number) => {
                    const c = v / 255;
                    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
                };
                return [linear(rgba[i]!), linear(rgba[i + 1]!) * 10];
            }
        }
        throw new Error("alpha-row material drew no interior pixel");
    };
    const observed = pixel(actual.color);
    console.log(`alpha row: expected cutoff 0.25 and Mask code 1; observed ${observed.join(", ")}`);
    expect(observed[0]).toBeCloseTo(0.25, 2);
    expect(observed[1]).toBeCloseTo(1, 1);
    expect(observed).toEqual(pixel(reference.color));
    expect(actual.color).toEqual(reference.color);
});
test("eight varyings with uv and localPos are refused against the granted device limit", () => {
    expect(String(subjects().overflow)).toContain(
        'standard: material "EightVaryings" uses 17 inter-stage variables (7 engine, uv, localPos, 8 varyings); device.limits.maxInterStageShaderVariables is 16',
    );
});
