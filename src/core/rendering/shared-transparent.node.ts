import { expect, setDefaultTimeout, test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import type { World } from "../../engine";
import {
    AlphaMode,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "../../standard/rendering";
import { MeshInstance } from "../mesh";
import { Transform } from "../transform";
import {
    attachTexture,
    Camera,
    captureTexture,
    DEPTH_FORMAT,
    RenderContext,
    RenderPhases,
    Tonemapping,
    TonemappingMethod,
    type TransparentPhase,
    TransparentPhases,
    type View,
} from ".";
import { computeViewProj } from "./camera";

setDefaultTimeout(CEILING.node);
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [StandardRenderingPlugin, MeshRenderPlugin] },
]);

const CLEAR = [0x2e, 0x2b, 0x28].map((channel) => {
    const value = channel / 255;
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
});

function srgbByte(linear: number): number {
    const encoded = linear <= 0.0031308 ? linear * 12.92 : 1.055 * linear ** (1 / 2.4) - 0.055;
    return Math.round(encoded * 255);
}

function composite(colors: readonly (readonly number[])[]): number[] {
    let red = CLEAR[0];
    let green = CLEAR[1];
    let blue = CLEAR[2];
    for (const color of colors) {
        red = color[0] * 0.25 + red * 0.75;
        green = color[1] * 0.25 + green * 0.75;
        blue = color[2] * 0.25 + blue * 0.75;
    }
    return [srgbByte(red), srgbByte(green), srgbByte(blue), 255];
}

function panePipeline(world: World, eid: number, z: number, color: readonly number[]) {
    const matrix = new Float32Array(16);
    const view = new Float32Array(16);
    computeViewProj(world, eid, 1, matrix, view);
    const clipZ = matrix[10]! * z + matrix[14]!;
    const w = matrix[11]! * z + matrix[15]!;
    const format = world.resource(RenderContext).format;
    const module = world.gpu.device!.createShaderModule({
        label: `transparent-witness-${eid}-${z}`,
        code: `
            @vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
                var xy = vec2f(-1.0, -1.0);
                if (index == 1u) { xy = vec2f(3.0, -1.0); }
                if (index == 2u) { xy = vec2f(-1.0, 3.0); }
                return vec4f(xy * ${w}, ${clipZ}, ${w});
            }
            @fragment fn fragment() -> @location(0) vec4f {
                return vec4f(${color[0]}, ${color[1]}, ${color[2]}, 0.25);
            }
        `,
    });
    return world.gpu.device!.createRenderPipeline({
        label: `transparent-witness-${eid}-${z}`,
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

test("transparent standard instances and another renderer composite in per-view depth order", async () => {
    const { world } = subjects()[0];
    const cameras = [world.create(), world.create()];
    world.add(cameras[0], Transform, { translation: [0, 0, 5, 0] });
    world.add(cameras[1], Transform, {
        translation: [0, 0, -5, 0],
        rotation: [0, 1, 0, 0],
    });
    for (const camera of cameras) {
        world.add(camera, Camera);
        world.storage(Camera).antialias.set(camera, 0);
        world.add(camera, StandardRenderer);
        world.add(camera, Tonemapping, { method: TonemappingMethod.None });
        attachTexture(world, camera, { width: 64, height: 64 });
    }

    const materials = world.resource(Materials);
    const red = materials.add(StandardMaterial({ baseColor: [1, 0, 0, 0.25], unlit: true }), {
        alphaMode: AlphaMode.Blend,
    });
    const blue = materials.add(StandardMaterial({ baseColor: [0, 0, 1, 0.25], unlit: true }), {
        alphaMode: AlphaMode.Blend,
    });
    const panes = [
        { z: -2, material: red },
        { z: -1, material: blue },
        { z: 0, material: red },
        { z: 1, material: blue },
    ];
    const meshEntities: number[] = [];
    for (const pane of panes) {
        const eid = world.create();
        world.add(eid, Transform, {
            translation: [0, 0, pane.z, 0],
            scale: [1, 1, 0.04, 0],
        });
        world.add(eid, MeshInstance);
        world.add(eid, MeshMaterial, pane.material);
        meshEntities.push(eid);
    }

    world.step(0);
    world.step(0);
    const green = [0, 1, 0] as const;
    const yellow = [1, 1, 0] as const;
    const panesByView = new Map<number, [GPURenderPipeline, GPURenderPipeline]>();
    for (const camera of cameras)
        panesByView.set(camera, [
            panePipeline(world, camera, -1.5, green),
            panePipeline(world, camera, 0.5, yellow),
        ]);
    world.resource(RenderPhases).push({
        transparent(_world, _eid, _view, phase: TransparentPhase) {
            phase.add(0, 0, 0, 0, -1.5, 0);
            phase.add(1, 1, 0, 0, 0.5, 1);
        },
        drawTransparent(
            _world,
            eid,
            _view: View,
            phase: TransparentPhase,
            start: number,
            end: number,
            pass: GPURenderPassEncoder,
        ) {
            const pipelines = panesByView.get(eid)!;
            for (let i = start; i < end; i++) {
                pass.setPipeline(pipelines[phase.item(i).draw]!);
                pass.draw(3);
            }
        },
    });

    world.step(0);
    world.step(0);
    const pixels = [];
    for (const camera of cameras) {
        const rgba = (await captureTexture(world, camera)).rgba;
        pixels.push(Array.from(rgba.subarray((32 * 64 + 32) * 4, (32 * 64 + 32) * 4 + 4)));
    }
    const redColor = [1, 0, 0];
    const greenColor = [0, 1, 0];
    const blueColor = [0, 0, 1];
    const yellowColor = [1, 1, 0];
    const firstPhase = world.resource(TransparentPhases).get(cameras[0])!;
    const firstOrder = Array.from({ length: firstPhase.length }, (_, i) => firstPhase.item(i).draw);
    const secondPhase = world.resource(TransparentPhases).get(cameras[1])!;
    const secondOrder = Array.from(
        { length: secondPhase.length },
        (_, i) => secondPhase.item(i).draw,
    );
    expect(firstPhase.metrics.draws).toBe(3);
    expect(firstPhase.metrics.pipelineChanges).toBe(3);
    expect(firstPhase.metrics.bindGroupChanges).toBe(9);
    expect(secondPhase.metrics.draws).toBe(3);
    expect(secondPhase.metrics.pipelineChanges).toBe(3);
    expect(secondPhase.metrics.bindGroupChanges).toBe(9);
    expect(firstOrder).toEqual([
        meshEntities[0],
        0,
        meshEntities[1],
        meshEntities[2],
        1,
        meshEntities[3],
    ]);
    expect(secondOrder).toEqual([...firstOrder].reverse());
    const near = (actual: number[], expected: number[]) => {
        expect(
            Math.max(...actual.map((channel, i) => Math.abs(channel - expected[i]))),
        ).toBeLessThanOrEqual(2);
    };
    near(pixels[0], composite([redColor, greenColor, blueColor, redColor, yellowColor, blueColor]));
    near(pixels[1], composite([blueColor, yellowColor, redColor, blueColor, greenColor, redColor]));

    const occluder = world.create();
    world.add(occluder, Transform, {
        translation: [0, 0, 2, 0],
        scale: [1.5, 1.5, 0.06, 0],
    });
    world.add(occluder, MeshInstance);
    world.add(
        occluder,
        MeshMaterial,
        materials.add(StandardMaterial({ baseColor: [1, 0, 0, 1], unlit: true })),
    );
    world.step(0);
    world.step(0);
    const occluded = (await captureTexture(world, cameras[0])).rgba;
    expect(Array.from(occluded.subarray((32 * 64 + 32) * 4, (32 * 64 + 32) * 4 + 4))).toEqual([
        255, 0, 0, 255,
    ]);
});
