import { expect, setDefaultTimeout, test } from "bun:test";
import { FogPlugin } from "@dylanebert/shallot/extras";
import { MeshInstance } from "@dylanebert/shallot/mesh";
import { attachTexture, Camera, captureTexture } from "@dylanebert/shallot/rendering";
import {
    AlphaMode,
    extendedMaterial,
    extensionLayout,
    type MaterialBinding,
    MaterialPlugin,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "@dylanebert/shallot/standard/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { Transform } from "@dylanebert/shallot/transform";
import * as d from "typegpu/data";

setDefaultTimeout(TIMEOUT.node);
const Parameters = d.struct({ gain: d.f32 });
const Uniform = d.struct({ value: d.vec4f });
function extension(kind: "texture" | "uniform" | "sampler", count: number) {
    const binding: MaterialBinding =
        kind === "uniform"
            ? { type: kind, struct: Uniform }
            : kind === "texture"
              ? { type: "texture" }
              : { type: "sampler" };
    const declarations = Object.fromEntries(
        Array.from({ length: count }, (_, i) => [`extra${i}`, binding]),
    );
    const layout = extensionLayout(Materials, Parameters, declarations);
    return extendedMaterial(Materials, {
        name: `Budget-${kind}-${count}`,
        layout,
        defaults: { gain: 1 },
        bindings: (world) => world.resource(Resources)[kind],
        fragment: (_ctx, extension) => {
            "use gpu";
            return d.vec4f(extension.gain, 0, 0, 1);
        },
    });
}
const Resources = {
    create(world: import("@dylanebert/shallot").World) {
        const device = world.gpu.device;
        const uniform = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const texture = device.createTexture({
            size: [1, 1],
            format: "rgba8unorm",
            usage: GPUTextureUsage.TEXTURE_BINDING,
        });
        const sampler = device.createSampler();
        return Object.fromEntries(
            (["texture", "uniform", "sampler"] as const).map((kind) => [
                kind,
                Object.fromEntries(
                    Array.from({ length: 16 }, (_, i) => [
                        `extra${i}`,
                        { uniform, texture, sampler }[kind],
                    ]),
                ),
            ]),
        ) as Record<
            "texture" | "uniform" | "sampler",
            Record<string, GPUBuffer | GPUTexture | GPUSampler>
        >;
    },
};
async function draw(kind: "texture" | "uniform" | "sampler", count: number, fog: boolean) {
    const type = extension(kind, count);
    let camera = 0;
    const app = await openGpuApp({
        defaults: false,
        plugins: [
            {
                name: "BudgetScene",
                dependencies: [
                    StandardRenderingPlugin,
                    MeshRenderPlugin,
                    MaterialPlugin(type),
                    ...(fog ? [FogPlugin] : []),
                ],
                initialize(world) {
                    world.resource(Resources);
                    camera = world.create();
                    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
                    world.add(camera, Camera, { antialias: 0 });
                    world.add(camera, StandardRenderer);
                    attachTexture(world, camera, { width: 16, height: 16 });
                    const eid = world.create();
                    world.add(eid, Transform);
                    world.add(eid, MeshInstance);
                    world.add(
                        eid,
                        MeshMaterial,
                        world
                            .resource(type)
                            .add(
                                { ...StandardMaterial(), gain: 1 },
                                { alphaMode: AlphaMode.Blend },
                            ),
                    );
                },
            },
        ],
    });
    try {
        app.world.step(0);
        app.world.step(0);
        const frame = await captureTexture(app.world, camera);
        expect(frame.rgba[(8 * 16 + 8) * 4]).toBeGreaterThan(200);
    } finally {
        app.dispose();
    }
}
test("transparent extension budget includes registered fog's sampled textures", async () => {
    await expect(draw("texture", 9, true)).rejects.toThrow(
        'material "Budget-texture-9" uses 19 fragment sampled textures; WebGPU\'s limit is 16',
    );
});
test("extension budget refuses uniform buffers above the default per-stage limit", async () => {
    await expect(draw("uniform", 8, false)).rejects.toThrow(
        "material layout uses 13 vertex uniform buffers; WebGPU's limit is 12",
    );
});
test("extension budget refuses samplers above the default per-stage limit", async () => {
    await expect(draw("sampler", 15, false)).rejects.toThrow(
        "material layout uses 17 vertex samplers; WebGPU's limit is 16",
    );
});
for (const [kind, count, fog] of [
    ["texture", 6, true],
    ["uniform", 6, false],
    ["sampler", 14, false],
] as const) {
    test(`extension at the ${kind} budget draws${fog ? " with transparent fog" : ""}`, async () => {
        await draw(kind, count, fog);
    });
}
