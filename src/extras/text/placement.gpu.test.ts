import { afterAll, expect, setDefaultTimeout, test } from "bun:test";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { compileGpuFile } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { VsIn } from "../../core/rendering";
import { createApp, globalTransformTable, Transform } from "../../engine";
import { prepareGlobalTransformFrame } from "../../engine/ecs/global-transform";
import { probeTexture } from "../../engine/runtime";
import { encodePos } from "../../engine/utils";
import { maskLayoutPlain, maskVertex } from "../outline/passes";
import { typedTextSurface } from "./index";

setDefaultTimeout(CEILING.gpu);
const subject = compileGpuFile(import.meta.path, async () => {
    const app = await createApp({ defaults: false, plugins: [] });
    const world = app.world;
    const device = world.gpu.device;
    console.info("placement adapter:", world.gpu.adapter);
    // A later eid still gets the first dense row. Its eid addresses a decoy placement.
    world.create();
    const eid = world.create();
    const table = globalTransformTable(world);
    world.add(eid, Transform);
    world.storage(Transform).translation.set(eid, -0.75, -0.25, 0.5, 0);
    for (let i = 0; i < eid; i++) {
        const decoy = world.create();
        world.add(decoy, Transform);
        world.storage(Transform).translation.set(decoy, 0.25, -0.25, 0.5, 0);
    }
    world.step(1 / 60);
    expect(table.rowIndex(eid)).toBe(0);
    expect(table.rowIndex(eid)).not.toBe(eid);
    // The small composition has no renderer; record its placement pass exactly once at warm-up.
    const placementEncoder = device.createCommandEncoder();
    prepareGlobalTransformFrame(world, placementEncoder);
    const runtime = world.globalTransformRuntime!;
    device.queue.writeBuffer(runtime.params!, 0, new Float32Array([1, table.count, 0, 0]));
    const placementPass = placementEncoder.beginComputePass();
    placementPass.setPipeline(runtime.pipeline!);
    placementPass.setBindGroup(0, runtime.group!);
    placementPass.dispatchWorkgroups(1);
    placementPass.end();
    device.queue.submit([placementEncoder.finish()]);
    const buffers: Record<string, GPUBuffer> = {
        globalTransforms: table.buffer,
        globalTransformRows: table.eidToRowBuffer!,
    };
    function buffer(name: string, data: ArrayBufferView, uniform = false) {
        const b = device.createBuffer({
            size: data.byteLength,
            usage:
                (uniform ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE) |
                GPUBufferUsage.COPY_DST,
        });
        world.own(b);
        device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
        buffers[name] = b;
    }
    const glyph = new ArrayBuffer(48);
    new Uint32Array(glyph)[3] = eid;
    new Float32Array(glyph).set([0.5, 0.5], 8);
    buffer("textGlyphs", new Uint8Array(glyph));
    buffer("maskEids", new Uint32Array([eid]));
    const quant = { posOffset: d.vec4f(0), posScale: d.vec4f(0.5, 0.5, 1, 0), uvScale: d.vec4f(0) };
    buffer("meshQuant", new Float32Array([0, 0, 0, 0, 0.5, 0.5, 1, 0, 0, 0, 0, 0]));
    const positions = [d.vec3f(0), d.vec3f(0.5, 0, 0), d.vec3f(0, 0.5, 0)].flatMap((p) => [
        ...encodePos(p, 0, quant),
    ]);
    buffer("position", new Uint32Array(positions));
    buffer("indices", new Uint32Array([0, 1, 2]));
    const view = new Float32Array(52);
    for (const i of [0, 5, 10, 15]) view[i] = 1;
    buffer("view", view, true);
    const text = typedTextSurface(0);
    const textCode = tgpu.resolve({
        names: "strict",
        externals: { VsIn, textVs: text.vs },
        template: `
@vertex fn main(@builtin(vertex_index) vidx: u32) -> @builtin(position) vec4f {
    var input: VsIn;
    input.localPos = array<vec3f, 3>(vec3f(0), vec3f(1, 0, 0), vec3f(0, 1, 0))[vidx];
    return textVs(input).world;
}`,
    });
    const outlineCode = tgpu.resolve([maskVertex(maskLayoutPlain)], { names: "strict" });
    const texture = device.createTexture({
        size: [32, 16],
        format: "rgba8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    world.own(texture);
    const pipelines = await Promise.all(
        [textCode, outlineCode].map(async (code, index) => {
            const module = device.createShaderModule({
                code: code + `\n@fragment fn white() -> @location(0) vec4f { return vec4f(1); }`,
            });
            const pipeline = await device.createRenderPipelineAsync({
                layout: "auto",
                vertex: { module, entryPoint: index === 0 ? "main" : "maskVs" },
                fragment: { module, entryPoint: "white", targets: [{ format: "rgba8unorm" }] },
            });
            const entries = new Map<number, GPUBindGroupEntry[]>();
            for (const match of code.matchAll(
                /@group\((\d+)\)\s*@binding\((\d+)\)\s*var(?:<[^>]+>)?\s+(\w+)\s*:/g,
            )) {
                const group = Number(match[1]);
                const list = entries.get(group) ?? [];
                list.push({ binding: Number(match[2]), resource: { buffer: buffers[match[3]]! } });
                entries.set(group, list);
            }
            const max = Math.max(...entries.keys());
            const groups = Array.from({ length: max + 1 }, (_, group) =>
                device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(group),
                    entries: entries.get(group) ?? [],
                }),
            );
            return { pipeline, groups };
        }),
    );
    return { app, texture, pipelines };
});
afterAll(() => subject().app.dispose());

for (const [index, name] of ["label", "outline"].entries()) {
    test(`${name} lands at its own dense transform row, not the placement indexed by eid`, async () => {
        const { app, texture, pipelines } = subject();
        const { pipeline, groups } = pipelines[index];
        const encoder = app.world.gpu.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: texture.createView(),
                    clearValue: [0, 0, 0, 0],
                    loadOp: "clear",
                    storeOp: "store",
                },
            ],
        });
        pass.setPipeline(pipeline);
        groups.forEach((group, i) => {
            pass.setBindGroup(i, group);
        });
        pass.draw(3);
        pass.end();
        app.world.gpu.device.queue.submit([encoder.finish()]);
        const bytes = new Uint8Array((await probeTexture(app.world, texture)).bytes);
        const pixel = (x: number, y: number) => bytes[(y * 32 + x) * 4];
        expect(
            [pixel(5, 8), pixel(21, 8)],
            `${name} coverage [own row 0, eid-indexed decoy]`,
        ).toEqual([255, 0]);
    });
}
