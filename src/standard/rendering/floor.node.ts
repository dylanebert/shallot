import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import tgpu, { readFromArrayBuffer } from "typegpu";
import * as d from "typegpu/data";
import { create, globals } from "webgpu";
import {
    createTestGpuApp as createApp,
    createTestGpuApp,
    expectAdapterNotice,
} from "../../../scripts/gpu.fixture";
import { expectError } from "../../../scripts/test-console";
import { Meshes, MeshInstance } from "../../core/mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    CULL_VOLUME_FLOATS,
    captureTexture,
    DepthPrepass,
    DirectionalLight,
    EffectPasses,
    MAX_SLOTS,
    PointLight,
    RenderContext,
    SpotLight,
    Views,
} from "../../core/rendering";
import { PointsPlugin } from "../../core/rendering/points.fixture";
import { Transform } from "../../core/transform";
import type { Plugin } from "../../engine";
import { probeBuffer, stampAdapter } from "../../engine/runtime";
import { type AdapterInfoFacts, classifyAdapter } from "../../engine/runtime/adapter";
import { packColor } from "../../engine/utils";
import { Line, LinesPlugin } from "../../extras/lines";
import { lineLayout, viewLayout as lineViewLayout } from "../../extras/lines/pipeline";
import { Sprite, SpriteMaterialInput, SpriteMaterialType, SpritePlugin } from "../../extras/sprite";
import { packSpriteFill } from "../../extras/sprite/pack";
import { internText, registerFont, Text, TextPlugin } from "../../extras/text";
import { isolationFont } from "../../extras/text/font.fixture";
import { textLayout, viewLayout as textViewLayout } from "../../extras/text/pipeline";
import { Vignette, VignettePlugin } from "../../extras/vignette";
import {
    CLUSTER_COUNT,
    LIGHT_GRID_OFFSET,
    LIGHT_INDICES_OFFSET,
    LIGHT_POOL,
    LightCull,
} from "./cluster";
import {
    AlphaMode,
    AlphaPipelineKey,
    BackgroundContext,
    backgroundLayout,
    CameraBackground,
    Draws,
    engineLayout,
    MaterialPlugin,
    Materials,
    MeshMaterial,
    materialLayout,
    materialType,
    materialTypeId,
    registerBackground,
    StandardMaterial,
    StandardRenderer,
    VertexMaterialType,
} from "./index";
import "../../standard";

const FloorParameters = d.struct({ color: d.vec4f });
const attributeLayout = materialLayout(FloorParameters, {
    stream: { type: "attribute", element: d.u32 },
});
const FloorMaterial = materialType({
    name: "FloorCustomMaterial",
    parameters: FloorParameters,
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(ctx.parameters.color);
    },
    defaults: { color: d.vec4f(1) },
});
const FloorMaterialPlugin = MaterialPlugin(FloorMaterial);

setDefaultTimeout(TIMEOUT.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();
const gpu = create([]);

function storageLimitPlugin(bindings: number): Plugin {
    return {
        name: "StandardRendering",
        gpu: {},
        async warm(world) {
            const device = world.gpu.device!;
            // Same owner, pipeline label and storage count as a compact-layout mutation, without
            // poisoning later builds by mutating the shared built-in schema.
            const group = device.createBindGroupLayout({
                label: "shallot-light-compact-mutation",
                entries: Array.from({ length: bindings }, (_, binding) => ({
                    binding,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" as const },
                })),
            });
            const layout = device.createPipelineLayout({ bindGroupLayouts: [group] });
            await device.createComputePipelineAsync({
                label: "shallot-light-compact",
                layout,
                compute: {
                    module: device.createShaderModule({
                        code: "@compute @workgroup_size(1) fn main() {}",
                    }),
                    entryPoint: "main",
                },
            });
        },
    };
}

function storageCount(layout: unknown, stage: "vertex" | "fragment"): number {
    const entries = (
        layout as {
            entries: Record<string, { storage?: unknown; visibility?: readonly string[] }>;
        }
    ).entries;
    return Object.values(entries).filter(
        (entry) => entry.storage && (entry.visibility ?? ["vertex", "fragment"]).includes(stage),
    ).length;
}
const proof: Plugin = {
    name: "FloorProof",
    dependencies: [
        SpritePlugin,
        TextPlugin,
        VignettePlugin,
        PointsPlugin,
        LinesPlugin,
        FloorMaterialPlugin,
    ],
    initialize(world) {
        registerFont(
            world,
            `data:font/ttf;base64,${Buffer.from(isolationFont()).toString("base64")}`,
            "floor",
        );
    },
};

test("pipeline creation refuses the ten-binding light-compact mutation and builds at eight", async () => {
    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new Error("WebGPU adapter unavailable");
    const device = await adapter.requestDevice({
        requiredLimits: { maxStorageBuffersPerShaderStage: 8 },
    });
    expect(device.limits.maxStorageBuffersPerShaderStage).toBe(8);
    try {
        const app = await createTestGpuApp({
            defaults: false,
            plugins: [storageLimitPlugin(8)],
            device,
            adapter,
        });
        app.dispose();

        // The refused build stamps the supplied adapter before it throws.
        expectAdapterNotice(
            classifyAdapter({ present: true, info: adapter.info as AdapterInfoFacts }),
        );
        const message = await createApp({
            defaults: false,
            plugins: [storageLimitPlugin(10)],
            device,
            adapter,
        }).then(
            (app) => {
                app.dispose();
                return "build unexpectedly succeeded";
            },
            (error: unknown) => (error instanceof Error ? error.message : String(error)),
        );
        expect(message).toContain(
            'Plugin "StandardRendering" pipeline "shallot-light-compact" uses 10 compute storage-buffer bindings; maxStorageBuffersPerShaderStage is 8',
        );
    } finally {
        expectError(/GPU device lost destroyed: device was destroyed/);
        device.destroy();
    }
});

test("standard, custom-material and extras layouts stay within eight buffers", async () => {
    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new Error("WebGPU adapter unavailable");
    const requiredFeatures = ["indirect-first-instance", "rg11b10ufloat-renderable"] as const;
    const device = await adapter.requestDevice({
        requiredFeatures: [...requiredFeatures],
    });
    const classification = stampAdapter(adapter);
    expectAdapterNotice(classification);
    console.log("layout adapter", adapter.info, "classification", classification);
    console.log("layout device features", [...device.features]);
    console.log(
        "layout device limits",
        JSON.stringify(
            Object.fromEntries(
                Object.getOwnPropertyNames(Object.getPrototypeOf(device.limits))
                    .filter((k) => k !== "constructor")
                    .map((k) => [k, device.limits[k as keyof GPUSupportedLimits]]),
            ),
        ),
    );
    expect(device.limits.maxStorageBuffersPerShaderStage).toBe(8);
    for (const feature of requiredFeatures) expect(device.features.has(feature)).toBe(true);
    device.pushErrorScope("validation");
    device.createPipelineLayout({
        bindGroupLayouts: [5, 6].map((length) =>
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
        app = await createTestGpuApp({ device, adapter, plugins: [proof] });
        const { world } = app;
        const layouts = {
            standardVertex:
                storageCount(engineLayout, "vertex") + storageCount(Materials.layout, "vertex"),
            standardFragment:
                storageCount(engineLayout, "fragment") + storageCount(Materials.layout, "fragment"),
            floorVertex:
                storageCount(engineLayout, "vertex") + storageCount(FloorMaterial.layout, "vertex"),
            vertexMaterialVertex:
                storageCount(engineLayout, "vertex") +
                storageCount(VertexMaterialType.layout, "vertex"),
            vertexMaterialFragment:
                storageCount(engineLayout, "fragment") +
                storageCount(VertexMaterialType.layout, "fragment"),
            attributeVertex:
                storageCount(engineLayout, "vertex") + storageCount(attributeLayout, "vertex"),
            attributeFragment:
                storageCount(engineLayout, "fragment") + storageCount(attributeLayout, "fragment"),
            spriteVertex:
                storageCount(engineLayout, "vertex") +
                storageCount(SpriteMaterialType.layout, "vertex"),
            textVertex: storageCount(textViewLayout, "vertex") + storageCount(textLayout, "vertex"),
            textFragment: storageCount(textLayout, "fragment"),
            linesVertex:
                storageCount(lineViewLayout, "vertex") + storageCount(lineLayout, "vertex"),
            linesFragment: storageCount(lineLayout, "fragment"),
        };
        expect(layouts).toEqual({
            standardVertex: 7,
            standardFragment: 7,
            floorVertex: 7,
            vertexMaterialVertex: 7,
            vertexMaterialFragment: 7,
            attributeVertex: 8,
            attributeFragment: 8,
            spriteVertex: 7,
            textVertex: 5,
            textFragment: 0,
            linesVertex: 1,
            linesFragment: 0,
        });
        expect(layouts.standardVertex).toBeLessThanOrEqual(8);
        expect(layouts.standardFragment).toBeLessThanOrEqual(8);
        expect(() =>
            materialLayout(FloorParameters, {
                first: { type: "storage", element: d.u32 },
                second: { type: "storage", element: d.u32 },
            }),
        ).toThrow("material layout uses 9 vertex storage buffers; WebGPU's limit is 8");
        console.log("storage buffers per changed pipeline layout", layouts);
        expect(world.resource(RenderContext).cullVolumes.size).toBe(
            MAX_SLOTS * CULL_VOLUME_FLOATS * Float32Array.BYTES_PER_ELEMENT,
        );
        expect(world.resource(LightCull).lights!.size).toBe(LIGHT_GRID_OFFSET + CLUSTER_COUNT * 8);
        console.log("floor storage bytes", {
            cullVolumes: world.resource(RenderContext).cullVolumes.size,
            lightList: LIGHT_INDICES_OFFSET,
            pool: (LIGHT_POOL + 2) * 4,
            grid: LIGHT_GRID_OFFSET,
            clusters: CLUSTER_COUNT,
        });
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 5, 0] });
        world.add(camera, Camera);
        world.add(camera, AmbientLight, { brightness: 199.61915 });
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
        const standardMaterial = world
            .resource(Materials)
            .add(StandardMaterial({ baseColor: [1, 0.3, 0.2, 1] }));
        for (let i = 0; i < 192; i++) {
            const mesh = world.create();
            world.add(mesh, Transform, {
                translation: [-0.65, (i % 8) * 0.01, -i * 0.01, 0],
                scale: [0.4, 0.4, 0.4, 0],
            });
            world.add(mesh, MeshInstance);
            world.add(mesh, MeshMaterial, standardMaterial);
        }
        const customMaterial = world
            .resource(FloorMaterial)
            .add({ color: d.vec4f(1, 0.3, 0.2, 1) });
        const vertexMaterial = world
            .resource(VertexMaterialType)
            .add(StandardMaterial({ baseColor: [0, 1, 0, 1] }), { alphaMode: AlphaMode.Mask(0.5) });
        const unlitMaterial = world
            .resource(Materials)
            .add(StandardMaterial({ baseColor: [0, 0, 1, 1], unlit: true }));
        const customMesh = world.create();
        world.add(customMesh, Transform, {
            translation: [0.65, 0, 0, 0],
            scale: [0.4, 0.4, 0.4, 0],
        });
        world.add(customMesh, MeshInstance);
        world.add(customMesh, MeshMaterial, customMaterial);
        for (const [x, material] of [
            [-1.6, vertexMaterial],
            [1.6, unlitMaterial],
        ] as const) {
            const mesh = world.create();
            world.add(mesh, Transform, {
                translation: [x, 0, 0, 0],
                scale: [0.4, 0.4, 0.4, 0],
            });
            world.add(mesh, MeshInstance);
            world.add(mesh, MeshMaterial, material);
        }
        const sun = world.create();
        world.add(sun, Transform);
        world.add(sun, DirectionalLight, { shadowMapsEnabled: 1 });
        const point = world.create();
        world.add(point, Transform, { translation: [1, 1, 2, 0] });
        world.add(point, PointLight, { shadowMapsEnabled: 1, range: 10 });
        const spot = world.create();
        world.add(spot, Transform, { translation: [-1, 1, 3, 0] });
        world.add(spot, SpotLight, { shadowMapsEnabled: 1, range: 10 });
        const spriteEids: number[] = [];
        for (const billboard of [0, 1, 2]) {
            for (const blend of [0, 1]) {
                const sprite = world.create();
                world.add(sprite, Transform);
                world.add(sprite, Sprite, {
                    billboard,
                    blend,
                    color: 0xff0000 + billboard * 0x000101,
                    fill: 0.5,
                    fillMode: billboard % 3,
                });
                spriteEids.push(sprite);
            }
        }
        const text = world.create();
        world.add(text, Transform);
        world.add(text, Text, { content: internText(world, "isolation"), fontSize: 0.2 });
        const line = world.create();
        world.add(line, Transform, { translation: [0, 1, 0, 0] });
        world.add(line, Line, { offset: [1, 0, 0, 0], thickness: 3, color: 0xff00ff });
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
                if (aa === 0 && depth === 0) {
                    const pixel = (x: number, y: number) =>
                        Array.from(shot.rgba.subarray((y * 32 + x) * 4, (y * 32 + x) * 4 + 4));
                    const vertex = pixel(7, 16);
                    const unlit = pixel(25, 16);
                    expect(vertex[1]).toBeGreaterThan(vertex[0]);
                    expect(unlit[2]).toBeGreaterThan(unlit[0]);
                }
            }
        }
        for (const eid of spriteEids) {
            const typeId = world.storage(MeshMaterial).type.get(eid);
            const row = world.storage(MeshMaterial).material.get(eid);
            expect(row).toBe(eid);
            const type = SpriteMaterialType;
            expect(typeId).toBe(materialTypeId(world, type));
            const bytes = world
                .resource(type)
                .table.bytes.slice(
                    row * d.sizeOf(SpriteMaterialInput),
                    (row + 1) * d.sizeOf(SpriteMaterialInput),
                );
            const parameters = readFromArrayBuffer(bytes.buffer, SpriteMaterialInput);
            const sprite = world.storage(Sprite);
            expect(parameters.layer).toBe(sprite.image.get(eid));
            expect(parameters.color).toBe(
                packColor(sprite.color.get(eid), sprite.opacity.get(eid)),
            );
            expect(parameters.fill).toBe(
                packSpriteFill(sprite.fill.get(eid), sprite.fillMode.get(eid)),
            );
        }
        const view = world.resource(Views).get(camera)!;
        const cube = [...world.resource(Meshes).entries()].find(
            ([, mesh]) => mesh.name === "cube",
        )![0];
        for (const [materialType, alphaPipelineKey, count] of [
            [0, AlphaPipelineKey.Opaque, 193],
            [materialTypeId(world, VertexMaterialType), AlphaPipelineKey.Mask, 1],
            [materialTypeId(world, FloorMaterial), AlphaPipelineKey.Opaque, 1],
        ] as const) {
            const draw = [...world.resource(Draws)].find(
                (draw) =>
                    draw.materialType === materialType &&
                    draw.alphaPipelineKey === alphaPipelineKey &&
                    draw.mesh === cube,
            )!;
            const result = await probeBuffer(world, world.gpu.root.unwrap(draw.args.indirect), {
                offset: (draw.args.offset ?? 0) + view.slot * (draw.args.viewStride ?? 0),
                size: 20,
            });
            const args = new Uint32Array(result.bytes);
            expect(args[1]).toBe(count);
            expect(args[3]).toBe(0);
        }
        const spriteMesh = [...world.resource(Meshes).entries()].find(
            ([, mesh]) => mesh.name === "spriteQuad",
        )![0];
        for (const [alphaPipelineKey, count] of [
            [AlphaPipelineKey.Mask, 3],
            [AlphaPipelineKey.Blend, 3],
        ] as const) {
            const draw = [...world.resource(Draws)].find(
                (draw) =>
                    draw.materialType === materialTypeId(world, SpriteMaterialType) &&
                    draw.alphaPipelineKey === alphaPipelineKey &&
                    draw.mesh === spriteMesh,
            )!;
            const result = await probeBuffer(world, world.gpu.root.unwrap(draw.args.indirect), {
                offset: (draw.args.offset ?? 0) + view.slot * (draw.args.viewStride ?? 0),
                size: 20,
            });
            expect(new Uint32Array(result.bytes)[1]).toBe(count);
        }
        const defaultDraw = [...world.resource(Draws)].find(
            (draw) => draw.materialType === 0 && draw.mesh === cube,
        )!;
        const records = await probeBuffer(world, world.gpu.root.unwrap(defaultDraw.args.indirect));
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
        const finalError = await device.popErrorScope();
        expect(finalError, finalError?.message).toBeNull();
        expectError(/GPU device lost destroyed: device was destroyed/);
        device.destroy();
        Object.assign(globalThis, previousGlobals);
    }
});
