import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import type { World } from "../../engine";
import {
    AlphaMode,
    MaterialPlugin,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    materialType,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "../../standard/rendering";
import { MeshDrawBuffers } from "../../standard/rendering/preprocess";
import { Meshes, MeshInstance } from "../mesh";
import { Transform } from "../transform";
import {
    attachTexture,
    Camera,
    CameraMode,
    captureTexture,
    DEPTH_FORMAT,
    RenderContext,
    RenderPhases,
    Tonemapping,
    TonemappingMethod,
    type TransparentPhase,
    TransparentPhases,
    type View,
    Views,
} from ".";
import { computeViewProj } from "./camera";

setDefaultTimeout(TIMEOUT.node);
const TransitionParameters = d.struct({ color: d.vec4f });
const TransitionMaterial = materialType({
    name: "TransparentQualificationTransition",
    parameters: TransitionParameters,
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(ctx.parameters.color);
    },
    defaults: { color: d.vec4f(1) },
});
const subjects = gpuApps(import.meta.path, [
    ...Array.from({ length: 6 }, () => ({
        defaults: false,
        plugins: [StandardRenderingPlugin, MeshRenderPlugin],
    })),
    {
        defaults: false,
        plugins: [StandardRenderingPlugin, MeshRenderPlugin, MaterialPlugin(TransitionMaterial)],
    },
]);

function addCamera(
    world: World,
    transform: {
        translation?: [number, number, number, number];
        rotation?: [number, number, number, number];
    } = {},
) {
    const eid = world.create();
    world.add(eid, Transform, transform);
    world.add(eid, Camera);
    world.add(eid, StandardRenderer);
    world.add(eid, Tonemapping, { method: TonemappingMethod.None });
    world.storage(Camera).mode.set(eid, CameraMode.Orthographic);
    world.storage(Camera).size.set(eid, 3);
    world.storage(Camera).clearColor.set(eid, 0);
    world.storage(Camera).antialias.set(eid, 0);
    attachTexture(world, eid, { width: 64, height: 64 });
    return eid;
}

function panePipeline(world: World, camera: number, z: number, color: readonly number[]) {
    const matrix = new Float32Array(16);
    computeViewProj(world, camera, 1, matrix);
    const clipZ = matrix[10]! * z + matrix[14]!;
    const w = matrix[11]! * z + matrix[15]!;
    const format = world.resource(RenderContext).format;
    const module = world.gpu.device!.createShaderModule({
        label: `phase-qualification-${camera}-${z}`,
        code: `
            @vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
                var xy = vec2f(-1.0, -1.0);
                if (index == 1u) { xy = vec2f(3.0, -1.0); }
                if (index == 2u) { xy = vec2f(-1.0, 3.0); }
                return vec4f(xy * ${w}, ${clipZ}, ${w});
            }
            @fragment fn fragment() -> @location(0) vec4f {
                return vec4f(${color[0]}, ${color[1]}, ${color[2]}, 0.5);
            }
        `,
    });
    return world.gpu.device!.createRenderPipeline({
        label: `phase-qualification-${camera}-${z}`,
        layout: "auto",
        vertex: { module, entryPoint: "vertex" },
        fragment: {
            module,
            entryPoint: "fragment",
            targets: [
                {
                    format,
                    blend: {
                        color: {
                            operation: "add",
                            srcFactor: "src-alpha",
                            dstFactor: "one-minus-src-alpha",
                        },
                        alpha: {
                            operation: "add",
                            srcFactor: "one",
                            dstFactor: "one-minus-src-alpha",
                        },
                    },
                },
            ],
        },
        primitive: { topology: "triangle-list", cullMode: "none" },
        depthStencil: {
            format: DEPTH_FORMAT,
            depthWriteEnabled: false,
            depthCompare: "greater-equal",
        },
    });
}

function centerPixel(rgba: Uint8ClampedArray): number[] {
    return Array.from(rgba.subarray((32 * 64 + 32) * 4, (32 * 64 + 32) * 4 + 4));
}

async function frame(world: World, cameras: readonly number[]) {
    world.step(0);
    world.step(0);
    return Promise.all(cameras.map(async (camera) => (await captureTexture(world, camera)).rgba));
}

async function validatedFrame(
    world: World,
    camera: number,
    sentinel: readonly [number, number, number, number],
): Promise<Uint8ClampedArray> {
    const device = world.gpu.device;
    const texture = world.resource(Views).get(camera)?.texture;
    if (!texture) throw new Error(`camera ${camera} has no final texture to clear`);
    const uncaptured: string[] = [];
    const listener = (event: GPUUncapturedErrorEvent) => uncaptured.push(event.error.message);
    device.addEventListener("uncapturederror", listener);
    device.pushErrorScope("validation");
    let scopeOpen = true;
    try {
        const encoder = device.createCommandEncoder({
            label: "transparent qualification sentinel",
        });
        const pass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: texture.createView(),
                    loadOp: "clear",
                    storeOp: "store",
                    clearValue: { r: sentinel[0], g: sentinel[1], b: sentinel[2], a: sentinel[3] },
                },
            ],
        });
        pass.end();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        world.step(0);
        if (!world.frameFence) throw new Error("measured render frame did not submit");
        await world.frameFence;
        const rgba = (await captureTexture(world, camera)).rgba;
        await device.queue.onSubmittedWorkDone();
        const validation = await device.popErrorScope();
        scopeOpen = false;
        if (validation)
            throw new Error(`uncaptured render validation error: ${validation.message}`);
        if (uncaptured.length) throw new Error(`uncaptured GPU errors: ${uncaptured.join("; ")}`);
        return rgba;
    } finally {
        if (scopeOpen) await device.popErrorScope();
        device.removeEventListener("uncapturederror", listener);
    }
}

function expectPixelNear(actual: readonly number[], expected: readonly number[], tolerance = 1) {
    expect(actual).toHaveLength(expected.length);
    for (let channel = 0; channel < expected.length; channel++)
        expect(Math.abs(actual[channel]! - expected[channel]!)).toBeLessThanOrEqual(tolerance);
}

function srgbByte(linear: number): number {
    const encoded = linear <= 0.0031308 ? 12.92 * linear : 1.055 * linear ** (1 / 2.4) - 0.055;
    return Math.round(encoded * 255);
}

function blendOverBlackPixel(
    color: readonly [number, number, number],
    alpha: number,
    layers: number,
) {
    const result = [0, 0, 0];
    for (let layer = 0; layer < layers; layer++)
        for (let channel = 0; channel < 3; channel++)
            result[channel] = color[channel]! * alpha + result[channel]! * (1 - alpha);
    return result.map(srgbByte);
}

function expectedLayerComposite(
    layers: readonly { eid: number; z: number; color: readonly [number, number, number] }[],
    reverseView: boolean,
): number[] {
    const ordered = [...layers].sort(
        (a, b) => (reverseView ? b.z - a.z : a.z - b.z) || a.eid - b.eid,
    );
    const result = [0, 0, 0];
    for (const layer of ordered)
        for (let channel = 0; channel < 3; channel++)
            result[channel] = layer.color[channel]! * 0.25 + result[channel]! * 0.75;
    return [...result.map(srgbByte), 255];
}

test("equal-depth composites are invariant to contribution order and snapshot restore", async () => {
    const { world } = subjects()[0];
    const camera = addCamera(world, { translation: [0, 0, 5, 0] });
    const first = world.create();
    const second = world.create();
    await frame(world, [camera]);
    let reverse = false;
    const pipelines = [
        panePipeline(world, camera, 0, [1, 0, 0]),
        panePipeline(world, camera, 0, [0, 0, 1]),
    ];
    world.resource(RenderPhases).push({
        transparent(_world, _eid, _view, phase: TransparentPhase) {
            if (reverse) {
                phase.add(1, 0, 0, 0, 0, second);
                phase.add(0, 0, 0, 0, 0, first);
            } else {
                phase.add(0, 0, 0, 0, 0, first);
                phase.add(1, 0, 0, 0, 0, second);
            }
        },
        drawTransparent(
            _world,
            _eid,
            _view: View,
            phase: TransparentPhase,
            start: number,
            end: number,
            pass: GPURenderPassEncoder,
        ) {
            for (let i = start; i < end; i++) {
                pass.setPipeline(pipelines[phase.item(i).draw]!);
                pass.draw(3);
            }
        },
    });

    const [initial] = await frame(world, [camera]);
    const snapshot = world.snapshot();
    const phase = world.resource(TransparentPhases).get(camera)!;
    expect([phase.item(0).stableId, phase.item(1).stableId]).toEqual([first, second]);
    reverse = true;
    const [permuted] = await frame(world, [camera]);
    expect(permuted).toEqual(initial);
    world.restore(snapshot);
    reverse = true;
    const [restored] = await frame(world, [camera]);
    expect(restored).toEqual(initial);
});

test("transparent item and staging capacities grow mid-run while opposite views reverse the order", async () => {
    const { world } = subjects()[1];
    const cameras = [
        addCamera(world, { translation: [0, 0, 5, 0] }),
        addCamera(world, { translation: [0, 0, -5, 0], rotation: [0, 1, 0, 0] }),
    ];
    const materials = world.resource(Materials);
    const red = materials.add(StandardMaterial({ baseColor: [1, 0.05, 0.02, 0.25], unlit: true }), {
        alphaMode: AlphaMode.Blend,
    });
    const blue = materials.add(StandardMaterial({ baseColor: [0.02, 0.1, 1, 0.25], unlit: true }), {
        alphaMode: AlphaMode.Blend,
    });
    const entities: number[] = [];
    const depths = new Map<number, number>();
    const layers: { eid: number; z: number; color: readonly [number, number, number] }[] = [];
    const redColor = [1, 0.05, 0.02] as const;
    const blueColor = [0.02, 0.1, 1] as const;
    const addPane = (
        z: number,
        material: typeof red,
        color: typeof redColor | typeof blueColor,
    ) => {
        const eid = world.create();
        world.add(eid, Transform, { translation: [0, 0, z, 0], scale: [2, 2, 0.02, 0] });
        world.add(eid, MeshInstance);
        world.add(eid, MeshMaterial, material);
        entities.push(eid);
        depths.set(eid, z);
        layers.push({ eid, z, color });
    };
    addPane(0, red, redColor);
    await frame(world, cameras);
    const before = world.gpu.root.unwrap(world.resource(MeshDrawBuffers).transparentEids!).size;

    for (let i = 0; i < 70; i++)
        addPane(-0.8 + i * 0.02, i < 35 ? blue : red, i < 35 ? blueColor : redColor);
    const [front, reverse] = await frame(world, cameras);
    const after = world.gpu.root.unwrap(world.resource(MeshDrawBuffers).transparentEids!).size;
    expect(after).toBeGreaterThan(before);
    const frontPhase = world.resource(TransparentPhases).get(cameras[0])!;
    const reversePhase = world.resource(TransparentPhases).get(cameras[1])!;
    const order = (phase: TransparentPhase) =>
        Array.from({ length: phase.length }, (_, index) => phase.item(index).stableId);
    expect(frontPhase.length).toBe(71);
    const expected = [...entities].sort((a, b) => depths.get(a)! - depths.get(b)! || a - b);
    expect(order(frontPhase)).toEqual(expected);
    expect(order(reversePhase)).toEqual(
        [...entities].sort((a, b) => depths.get(b)! - depths.get(a)! || a - b),
    );
    expect(frontPhase.metrics.draws).toBe(1);
    expect(reversePhase.metrics.draws).toBe(1);
    expectPixelNear(centerPixel(front), expectedLayerComposite(layers, false), 6);
    expectPixelNear(centerPixel(reverse), expectedLayerComposite(layers, true), 6);
    expect(centerPixel(front)).not.toEqual(centerPixel(reverse));
});

test("transparent pipelines follow MSAA changes and remove entities or alpha-mode changes in the same run", async () => {
    const { world } = subjects()[2];
    const camera = addCamera(world, { translation: [0, 0, 5, 0] });
    const materials = world.resource(Materials);
    const blend = materials.add(StandardMaterial({ baseColor: [1, 0.1, 0.1, 0.3], unlit: true }), {
        alphaMode: AlphaMode.Blend,
    });
    const opaque = materials.add(StandardMaterial({ baseColor: [0.1, 1, 0.1, 1], unlit: true }));
    const multiply = materials.add(
        StandardMaterial({ baseColor: [0.1, 0.1, 1, 0.5], unlit: true }),
        { alphaMode: AlphaMode.Multiply },
    );
    const entities: number[] = [];
    for (let i = 0; i < 3; i++) {
        const eid = world.create();
        world.add(eid, Transform, {
            translation: [0, 0, i * 0.2 - 0.2, 0],
            scale: [2, 2, 0.02, 0],
        });
        world.add(eid, MeshInstance);
        world.add(eid, MeshMaterial, blend);
        entities.push(eid);
    }
    await frame(world, [camera]);
    const expected = blendOverBlackPixel([1, 0.1, 0.1], 0.3, 3);
    const sentinel = [0, 1, 0, 1] as const;
    const phase = world.resource(TransparentPhases).get(camera)!;
    const blendBatch = phase.item(0).batch;
    world.storage(Camera).antialias.set(camera, 1);
    const multisample = await validatedFrame(world, camera, sentinel);
    expectPixelNear(centerPixel(multisample), [...expected, 255]);
    expect(centerPixel(multisample)).not.toEqual([0, 255, 0, 255]);
    expect(world.resource(TransparentPhases).get(camera)!.length).toBe(3);

    world.storage(Camera).antialias.set(camera, 0);
    const singleSample = await validatedFrame(world, camera, sentinel);
    expectPixelNear(centerPixel(singleSample), [...expected, 255]);

    world.destroy(entities[0]!);
    const meshMaterial = world.storage(MeshMaterial);
    meshMaterial.material.set(entities[1]!, opaque.material);
    meshMaterial.type.set(entities[1]!, opaque.type);
    meshMaterial.alphaMode.set(entities[1]!, opaque.alphaMode);
    meshMaterial.alphaCutoff.set(entities[1]!, opaque.alphaCutoff);
    meshMaterial.material.set(entities[2]!, multiply.material);
    meshMaterial.type.set(entities[2]!, multiply.type);
    meshMaterial.alphaMode.set(entities[2]!, multiply.alphaMode);
    meshMaterial.alphaCutoff.set(entities[2]!, multiply.alphaCutoff);
    const [alphaChanged] = await frame(world, [camera]);
    const remaining = world.resource(TransparentPhases).get(camera)!;
    expectPixelNear(centerPixel(alphaChanged), [0.055, 0.55, 0.1].map(srgbByte).concat(255), 2);
    expect(remaining.length).toBe(1);
    expect(remaining.item(0).stableId).toBe(entities[2]);
    expect(remaining.item(0).batch).not.toBe(blendBatch);
});

test("transparent type and alpha-mode transitions render from their new draws", async () => {
    const { world } = subjects()[6];
    const camera = addCamera(world, { translation: [0, 0, 5, 0] });
    const standard = world.resource(Materials);
    const red = standard.add(StandardMaterial({ baseColor: [1, 0, 0, 0.5], unlit: true }), {
        alphaMode: AlphaMode.Blend,
    });
    const blue = world
        .resource(TransitionMaterial)
        .add({ color: d.vec4f(0, 0, 1, 0.5) }, { alphaMode: AlphaMode.Blend });
    const greenOpaque = standard.add(StandardMaterial({ baseColor: [0, 1, 0, 0.25], unlit: true }));
    const eid = world.create();
    world.add(eid, Transform, { scale: [2, 2, 0.02, 0] });
    world.add(eid, MeshInstance);
    world.add(eid, MeshMaterial, red);

    const [before] = await frame(world, [camera]);
    expectPixelNear(centerPixel(before), [...blendOverBlackPixel([1, 0, 0], 0.5, 1), 255]);
    const phase = world.resource(TransparentPhases).get(camera)!;
    const redBatch = phase.item(0).batch;
    const meshMaterial = world.storage(MeshMaterial);
    const assign = (material: typeof red) => {
        meshMaterial.type.set(eid, material.type);
        meshMaterial.material.set(eid, material.material);
        meshMaterial.alphaMode.set(eid, material.alphaMode);
        meshMaterial.alphaCutoff.set(eid, material.alphaCutoff);
    };

    assign(blue);
    const [afterType] = await frame(world, [camera]);
    expectPixelNear(centerPixel(afterType), [...blendOverBlackPixel([0, 0, 1], 0.5, 1), 255]);
    expect(phase.length).toBe(1);
    expect(phase.item(0).batch).not.toBe(redBatch);
    const blueBatch = phase.item(0).batch;

    assign(greenOpaque);
    const [afterAlphaMode] = await frame(world, [camera]);
    expect(centerPixel(afterAlphaMode).slice(0, 3)).toEqual([0, 255, 0]);
    expect(phase.length).toBe(0);
    expect(blueBatch).not.toBe(redBatch);
});

test("transparent mesh instances are culled independently for each view", async () => {
    const { world } = subjects()[3];
    const cameras = [
        addCamera(world, { translation: [0, 0, 5, 0] }),
        addCamera(world, { translation: [0, 0, -5, 0] }),
    ];
    const material = world
        .resource(Materials)
        .add(StandardMaterial({ baseColor: [0.2, 0.8, 0.3, 0.5], unlit: true }), {
            alphaMode: AlphaMode.Blend,
        });
    const eid = world.create();
    world.add(eid, Transform, { scale: [2, 2, 0.02, 0] });
    world.add(eid, MeshInstance);
    world.add(eid, MeshMaterial, material);
    const [visible, culled] = await frame(world, cameras);
    expect(world.resource(TransparentPhases).get(cameras[0])!.length).toBe(1);
    expect(world.resource(TransparentPhases).get(cameras[1])!.length).toBe(0);
    expect(centerPixel(visible)).not.toEqual(centerPixel(culled));
});

test("only consecutive compatible items merge, and one steady frame submits once", async () => {
    const { world } = subjects()[4];
    const camera = addCamera(world, { translation: [0, 0, 5, 0] });
    await frame(world, [camera]);
    const pipelines = [
        panePipeline(world, camera, -1, [1, 0, 0]),
        panePipeline(world, camera, -0.5, [0, 1, 0]),
        panePipeline(world, camera, 0, [0, 0, 1]),
    ];
    const ranges: [number, number][] = [];
    world.resource(RenderPhases).push({
        transparent(_world, _eid, _view, phase: TransparentPhase) {
            phase.add(0, 10, 0, 0, -1, 10);
            phase.add(1, 20, 0, 0, -0.5, 20);
            phase.add(2, 10, 0, 0, 0, 30);
        },
        drawTransparent(
            _world,
            _eid,
            _view: View,
            phase: TransparentPhase,
            start: number,
            end: number,
            pass: GPURenderPassEncoder,
        ) {
            ranges.push([start, end]);
            for (let i = start; i < end; i++) {
                pass.setPipeline(pipelines[phase.item(i).draw]!);
                pass.draw(3);
            }
        },
    });
    await frame(world, [camera]);
    ranges.length = 0;
    const queue = world.gpu.device.queue;
    const descriptor = Object.getOwnPropertyDescriptor(queue, "submit");
    const submit = queue.submit.bind(queue);
    const sortDescriptor = Object.getOwnPropertyDescriptor(Array.prototype, "sort");
    const originalSort = Array.prototype.sort;
    let submissions = 0;
    let sortCalls = 0;
    let comparatorCalls = 0;
    Object.defineProperty(Array.prototype, "sort", {
        configurable: true,
        writable: true,
        value: function <T>(this: T[], compareFn?: (a: T, b: T) => number): T[] {
            sortCalls++;
            if (compareFn) comparatorCalls++;
            return originalSort.call(this, compareFn);
        },
    });
    Object.defineProperty(queue, "submit", {
        configurable: true,
        value: (...args: Parameters<GPUQueue["submit"]>) => {
            submissions++;
            return submit(...args);
        },
    });
    try {
        world.step(0);
        expect(submissions).toBe(1);
        expect(sortCalls).toBe(0);
        expect(comparatorCalls).toBe(0);
        expect(world.frameFence).toBeDefined();
        expect(ranges).toEqual([
            [0, 1],
            [1, 2],
            [2, 3],
        ]);
        expect(world.resource(TransparentPhases).get(camera)!.metrics.batches).toBe(3);
        await world.frameFence;
    } finally {
        if (descriptor) Object.defineProperty(queue, "submit", descriptor);
        else Reflect.deleteProperty(queue, "submit");
        if (sortDescriptor) Object.defineProperty(Array.prototype, "sort", sortDescriptor);
        else Reflect.deleteProperty(Array.prototype, "sort");
    }
});

test("a standard transparent surface cannot occlude a later, farther fragment with its own depth write", async () => {
    const { world } = subjects()[5];
    const camera = addCamera(world, { translation: [0, 0, 5, 0] });
    const vertices = (slope: number) =>
        new Float32Array([
            -2,
            -2,
            -2 * slope,
            0,
            0,
            0,
            1,
            0,
            2,
            -2,
            2 * slope,
            1,
            0,
            0,
            1,
            0,
            2,
            2,
            2 * slope,
            1,
            0,
            0,
            1,
            1,
            -2,
            2,
            -2 * slope,
            0,
            0,
            0,
            1,
            1,
        ]);
    const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);
    const sloped = world.resource(Meshes).add({
        name: "depth-write-slope",
        vertices: vertices(0.25),
        indices,
    });
    const flat = world
        .resource(Meshes)
        .add({ name: "depth-write-flat", vertices: vertices(0), indices });
    const materials = world.resource(Materials);
    const red = materials.add(StandardMaterial({ baseColor: [1, 0, 0, 0.5], unlit: true }), {
        alphaMode: AlphaMode.Blend,
    });
    const blue = materials.add(StandardMaterial({ baseColor: [0, 0, 1, 0.5], unlit: true }), {
        alphaMode: AlphaMode.Blend,
    });
    const first = world.create();
    world.add(first, Transform, { translation: [0, 0, -1, 0] });
    world.add(first, MeshInstance, { mesh: sloped });
    world.add(first, MeshMaterial, red);
    const [onlyFirst] = await frame(world, [camera]);
    const sample = (rgba: Uint8ClampedArray) =>
        Array.from(rgba.subarray((32 * 64 + 48) * 4, (32 * 64 + 48) * 4 + 4));
    const before = sample(onlyFirst);

    const second = world.create();
    world.add(second, Transform, { translation: [0, 0, -0.75, 0] });
    world.add(second, MeshInstance, { mesh: flat });
    world.add(second, MeshMaterial, blue);
    const [both] = await frame(world, [camera]);
    expect(sample(both)).not.toEqual(before);
});
