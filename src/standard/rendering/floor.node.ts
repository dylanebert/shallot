import { expect, setDefaultTimeout, test } from "bun:test";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { create, globals } from "webgpu";
import { CEILING } from "../../../scripts/test-tiers";
import { MeshInstance } from "../../core/mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    captureTexture,
    DepthPrepass,
    DirectionalLight,
    EffectPasses,
    PointLight,
    RenderContext,
    SpotLight,
    Views,
} from "../../core/rendering";
import { PointsPlugin } from "../../core/rendering/points.fixture";
import { Transform } from "../../core/transform";
import type { Plugin } from "../../engine";
import { createApp } from "../../engine/app";
import { probeBuffer } from "../../engine/runtime";
import { Xform } from "../../engine/utils";
import { Sprite, SpritePlugin } from "../../extras/sprite";
import { internText, registerFont, Text, TextPlugin } from "../../extras/text";
import { isolationFont } from "../../extras/text/font.fixture";
import { Vignette, VignettePlugin } from "../../extras/vignette";
import {
    CLUSTER_COUNT,
    LIGHT_GRID_OFFSET,
    LIGHT_INDICES_OFFSET,
    LIGHT_POOL,
    LightCull,
} from "./cluster";
import {
    BackgroundContext,
    backgroundLayout,
    CameraBackground,
    Draws,
    fsCtxSchema,
    Materials,
    MeshMaterial,
    registerBackground,
    registerSurface,
    StandardMaterial,
    StandardRenderer,
    Surfaces,
    surfaceLayout,
} from "./index";
import "../../standard";

setDefaultTimeout(CEILING.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();
const gpu = create([]);
const proof: Plugin = {
    name: "FloorProof",
    dependencies: [SpritePlugin, TextPlugin, VignettePlugin, PointsPlugin],
    initialize(world) {
        registerFont(
            world,
            `data:font/ttf;base64,${Buffer.from(isolationFont()).toString("base64")}`,
            "floor",
        );
        const layout = surfaceLayout({
            eids: { type: "storage", element: d.vec4u },
            globalTransforms: { type: "storage", element: Xform },
        });
        registerSurface(world, {
            name: "floor-custom",
            layout,
            fs: tgpu.fn(
                [fsCtxSchema()],
                d.vec4f,
            )((ctx) => {
                "use gpu";
                return d.vec4f(ctx.color);
            }),
        });
    },
};

test("the standard composition and points render every variant at the declared ten-buffer limit", async () => {
    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new Error("WebGPU adapter unavailable");
    const requiredFeatures = ["indirect-first-instance", "rg11b10ufloat-renderable"] as const;
    const device = await adapter.requestDevice({
        requiredFeatures: [...requiredFeatures],
        requiredLimits: { maxStorageBuffersPerShaderStage: 10 },
    });
    console.log(
        "floor adapter",
        adapter.info.vendor,
        adapter.info.device,
        adapter.info.description,
    );
    console.log("floor features", [...device.features]);
    console.log(
        "floor limits",
        JSON.stringify(
            Object.fromEntries(
                Object.getOwnPropertyNames(Object.getPrototypeOf(device.limits))
                    .filter((k) => k !== "constructor")
                    .map((k) => [k, device.limits[k as keyof GPUSupportedLimits]]),
            ),
        ),
    );
    expect(device.limits.maxStorageBuffersPerShaderStage).toBe(10);
    expect(
        [...device.features].filter((feature) => feature !== "core-features-and-limits").sort(),
    ).toEqual([...requiredFeatures].sort());
    device.pushErrorScope("validation");
    device.createPipelineLayout({
        bindGroupLayouts: [6, 5].map((length) =>
            device.createBindGroupLayout({
                entries: Array.from({ length }, (_, binding) => ({
                    binding,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: "read-only-storage" as const },
                })),
            }),
        ),
    });
    const refused = await device.popErrorScope();
    expect(refused).not.toBeNull();
    expect(refused!.message).toMatch(/storage.*(10|limit)|11.*storage/i);
    console.log("eleven-binding control", refused!.message);
    const work = { uploadedBytes: 0, renderPasses: 0, computePasses: 0, dispatches: 0 };
    const writeBuffer = device.queue.writeBuffer.bind(device.queue);
    device.queue.writeBuffer = (buffer, offset, data, dataOffset, size) => {
        const unit =
            ArrayBuffer.isView(data) && "BYTES_PER_ELEMENT" in data
                ? Number(data.BYTES_PER_ELEMENT)
                : 1;
        work.uploadedBytes +=
            size === undefined ? data.byteLength - (dataOffset ?? 0) * unit : size * unit;
        writeBuffer(buffer, offset, data, dataOffset, size);
    };
    const createEncoder = device.createCommandEncoder.bind(device);
    device.createCommandEncoder = (descriptor) => {
        const encoder = createEncoder(descriptor);
        const render = encoder.beginRenderPass.bind(encoder);
        encoder.beginRenderPass = (descriptor) => {
            work.renderPasses++;
            return render(descriptor);
        };
        const compute = encoder.beginComputePass.bind(encoder);
        encoder.beginComputePass = (descriptor) => {
            work.computePasses++;
            const pass = compute(descriptor);
            const dispatch = pass.dispatchWorkgroups.bind(pass);
            pass.dispatchWorkgroups = (x, y, z) => {
                work.dispatches++;
                dispatch(x, y, z);
            };
            return pass;
        };
        return encoder;
    };
    const previousGlobals = Object.fromEntries(
        Object.keys(globals).map((key) => [key, Reflect.get(globalThis, key)]),
    );
    Object.assign(globalThis, globals);
    device.pushErrorScope("validation");
    let app: Awaited<ReturnType<typeof createApp>> | undefined;
    try {
        app = await createApp({ device, plugins: [proof] });
        const { world } = app;
        expect(world.resource(RenderContext).cullVolumes.size).toBe(7168);
        expect(world.resource(LightCull).lights!.size).toBe(
            LIGHT_INDICES_OFFSET + (LIGHT_POOL + 2) * 4,
        );
        console.log("floor storage bytes", {
            cullVolumes: world.resource(RenderContext).cullVolumes.size,
            lightList: LIGHT_GRID_OFFSET,
            grid: LIGHT_INDICES_OFFSET - LIGHT_GRID_OFFSET,
            pool: (LIGHT_POOL + 2) * 4,
            clusters: CLUSTER_COUNT,
        });
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 5, 0] });
        world.add(camera, Camera);
        world.add(camera, StandardRenderer);
        attachTexture(world, camera, { width: 32, height: 32 });
        const captureCompletedFrame = async () => {
            await device.queue.onSubmittedWorkDone();
            return captureTexture(world, camera);
        };
        const bg = registerBackground(world, {
            name: "floor-background",
            layout: backgroundLayout({}),
            fs: tgpu.fn(
                [BackgroundContext],
                d.vec3f,
            )(() => {
                "use gpu";
                return d.vec3f(0.1, 0.2, 0.3);
            }),
        });
        world.add(camera, CameraBackground, { name: bg });
        for (const [index, surface] of ["default", "vertex", "unlit", "floor-custom"].entries()) {
            const material = world.resource(Materials).add(
                StandardMaterial({
                    surface: world.resource(Surfaces).id(surface)!,
                    baseColor: [1, 0.3, 0.2, 1],
                }),
            );
            for (let i = 0; i < (index === 0 ? 192 : 1); i++) {
                const mesh = world.create();
                world.add(mesh, Transform, {
                    translation: [(index - 1.5) * 0.65, (i % 8) * 0.01, -i * 0.01, 0],
                    scale: [0.4, 0.4, 0.4, 0],
                });
                world.add(mesh, MeshInstance);
                world.add(mesh, MeshMaterial, { material });
            }
        }
        world.add(world.create(), AmbientLight, { intensity: 0.2 });
        world.add(world.create(), DirectionalLight, { shadowMapsEnabled: 1 });
        const point = world.create();
        world.add(point, Transform, { translation: [1, 1, 2, 0] });
        world.add(point, PointLight, { shadowMapsEnabled: 1, range: 10 });
        const spot = world.create();
        world.add(spot, Transform, { translation: [-1, 1, 3, 0] });
        world.add(spot, SpotLight, { shadowMapsEnabled: 1, range: 10 });
        for (const billboard of [0, 1, 2]) {
            for (const blend of [0, 1]) {
                const sprite = world.create();
                world.add(sprite, Transform);
                world.add(sprite, Sprite, { billboard, blend });
            }
        }
        const text = world.create();
        world.add(text, Transform);
        world.add(text, Text, { content: internText(world, "isolation"), fontSize: 0.2 });
        for (const aa of [0, 1]) {
            world.storage(Camera).antialias.set(camera, aa);
            for (const depth of [0, 1]) {
                if (depth) world.add(camera, DepthPrepass);
                else world.remove(camera, DepthPrepass);
                world.step(0);
                work.uploadedBytes = work.renderPasses = work.computePasses = work.dispatches = 0;
                world.step(0);
                console.log(
                    `floor work meshes=195 sprites=6 glyphs=9 AA=${aa} depth=${depth}`,
                    JSON.stringify(work),
                );
                const shot = await captureCompletedFrame();
                expect(shot.rgba.some((v, i) => i % 4 !== 3 && v > 0)).toBe(true);
            }
        }
        const view = world.resource(Views).get(camera)!;
        for (const [surface, count] of [
            ["default", 192],
            ["vertex", 1],
            ["unlit", 1],
            ["floor-custom", 1],
        ] as const) {
            const draw = [...world.resource(Draws)].find(
                (draw) => draw.surface === surface && draw.mesh === "cube",
            )!;
            const result = await probeBuffer(world, world.gpu.root.unwrap(draw.args.indirect), {
                offset: (draw.args.offset ?? 0) + view.slot * (draw.args.viewStride ?? 0),
                size: 20,
            });
            const args = new Uint32Array(result.bytes);
            expect(args[1]).toBe(count);
            expect(args[3]).toBe(0);
        }
        let spriteAndTextDraws = 0;
        for (const draw of world.resource(Draws)) {
            if (
                draw.name !== draw.surface ||
                (!draw.surface.startsWith("sprite-") && draw.surface !== "text0")
            )
                continue;
            const result = await probeBuffer(world, world.gpu.root.unwrap(draw.args.indirect), {
                offset: (draw.args.offset ?? 0) + view.slot * (draw.args.viewStride ?? 0),
                size: 20,
            });
            expect(new Uint32Array(result.bytes)[1]).toBe(draw.surface === "text0" ? 9 : 1);
            spriteAndTextDraws++;
        }
        expect(spriteAndTextDraws).toBe(7);
        const draw = [...world.resource(Draws)].find(
            (draw) => draw.surface === "default" && draw.mesh === "cube",
        )!;
        const records = await probeBuffer(world, world.gpu.root.unwrap(draw.args.indirect));
        const words = new Uint32Array(records.bytes);
        for (let i = 3; i < words.length; i += 5) expect(words[i]).toBe(0);
        const plain = (await captureCompletedFrame()).rgba;
        let points = 0;
        for (let i = 0; i < plain.length; i += 4)
            if (plain[i + 1] > plain[i] && plain[i + 1] > plain[i + 2]) points++;
        expect(points).toBeGreaterThan(0);
        world.add(camera, Vignette, { intensity: 0.5 });
        world.step(0);
        world.step(0);
        const vignette = (await captureCompletedFrame()).rgba;
        expect(vignette).not.toEqual(plain);
        // Identity after-tonemap pass; cache by input view, as built-in effects do.
        const module = device.createShaderModule({
            code: `
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
 let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u)); return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}
@group(0) @binding(0) var input: texture_2d<f32>;
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f { return textureLoad(input, vec2u(p.xy), 0); }`,
        });
        const pipeline = device.createRenderPipeline({
            layout: "auto",
            vertex: { module, entryPoint: "vs" },
            fragment: {
                module,
                entryPoint: "fs",
                targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
            },
        });
        const groups = new Map<GPUTextureView, GPUBindGroup>();
        const stack = world.resource(EffectPasses).get(camera)!;
        stack.after.push((world, _eid, _view, input, output) => {
            let group = groups.get(input);
            if (!group) {
                group = device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [{ binding: 0, resource: input }],
                });
                groups.set(input, group);
            }
            const pass = world.frameEncoder()!.beginRenderPass({
                colorAttachments: [{ view: output, loadOp: "clear", storeOp: "store" }],
            });
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, group);
            pass.draw(3);
            pass.end();
        });
        world.step(0);
        world.step(0);
        expect((await captureCompletedFrame()).rgba).toEqual(vignette);
        let groupsCreated = 0;
        const original = device.createBindGroup;
        device.createBindGroup = function (descriptor) {
            groupsCreated++;
            return original.call(this, descriptor);
        };
        try {
            for (let frame = 0; frame < 10; frame++) world.step(1 / 60);
            expect(groupsCreated).toBe(0);
        } finally {
            device.createBindGroup = original;
        }
        await device.queue.onSubmittedWorkDone();
    } finally {
        app?.dispose();
        expect(await device.popErrorScope()).toBeNull();
        device.destroy();
        Object.assign(globalThis, previousGlobals);
    }
});
