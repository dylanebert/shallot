import { expect, setDefaultTimeout, test } from "bun:test";
import {
    attachTexture,
    Camera,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
} from "@dylanebert/shallot/rendering";
import {
    type Background,
    BackgroundContext,
    backgroundLayout,
    CameraBackground,
    type MaterialBinding,
    MeshRenderPlugin,
    registerBackground,
    StandardRenderer,
    StandardRenderingPlugin,
} from "@dylanebert/shallot/standard/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { Transform } from "@dylanebert/shallot/transform";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";

setDefaultTimeout(TIMEOUT.node);
const Uniform = d.struct({ value: d.vec4f });
function declaration(
    kind: "texture" | "uniform" | "sampler",
    count: number,
): Pick<Background, "layout" | "fs"> {
    const entries = <B extends MaterialBinding>(binding: B) =>
        Object.fromEntries(Array.from({ length: count }, (_, i) => [`extra${i}`, binding]));
    if (kind === "uniform") {
        const layout = backgroundLayout(entries({ type: "uniform", struct: Uniform }));
        return {
            layout,
            fs: tgpu.fn(
                [BackgroundContext],
                d.vec3f,
            )(() => {
                "use gpu";
                return layout.$.extra0.value.rgb;
            }),
        };
    }
    if (kind === "texture") {
        const layout = backgroundLayout(entries({ type: "texture" }));
        return {
            layout,
            fs: tgpu.fn(
                [BackgroundContext],
                d.vec3f,
            )(() => {
                "use gpu";
                return std.textureLoad(layout.$.extra0, d.vec2i(0), 0).rgb;
            }),
        };
    }
    const layout = backgroundLayout({
        ...entries({ type: "sampler" }),
        extra0: { type: "sampler" },
        signal: { type: "texture" },
    });
    return {
        layout,
        fs: tgpu.fn(
            [BackgroundContext],
            d.vec3f,
        )(() => {
            "use gpu";
            return std.textureSampleLevel(layout.$.signal, layout.$.extra0, d.vec2f(0.5), 0).rgb;
        }),
    };
}
async function draw(kind: "texture" | "uniform" | "sampler", count: number) {
    const definition = declaration(kind, count);
    let camera = 0;
    const app = await openGpuApp({
        defaults: false,
        plugins: [
            {
                name: "BackgroundBudgetScene",
                dependencies: [StandardRenderingPlugin, MeshRenderPlugin],
                initialize(world) {
                    const device = world.gpu.device;
                    const uniform = device.createBuffer({
                        size: 16,
                        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                    });
                    device.queue.writeBuffer(uniform, 0, new Float32Array([1, 0, 0, 1]));
                    const texture = device.createTexture({
                        size: [1, 1],
                        format: "rgba8unorm",
                        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
                    });
                    device.queue.writeTexture(
                        { texture },
                        new Uint8Array([255, 0, 0, 255]),
                        { bytesPerRow: 4 },
                        [1, 1],
                    );
                    const sampler = device.createSampler();
                    const resources = {
                        ...Object.fromEntries(
                            Array.from({ length: count }, (_, i) => [
                                `extra${i}`,
                                { uniform, texture, sampler }[kind],
                            ]),
                        ),
                        signal: texture,
                    };
                    const background = registerBackground(world, {
                        name: `Budget-${kind}-${count}`,
                        ...definition,
                        bindings: () => resources,
                    });
                    camera = world.create();
                    world.add(camera, Transform);
                    world.add(camera, Camera, { antialias: 0 });
                    world.add(camera, StandardRenderer);
                    world.add(camera, CameraBackground, { background });
                    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
                    attachTexture(world, camera, { width: 8, height: 8 });
                },
            },
        ],
    });
    try {
        app.world.step(0);
        app.world.step(0);
        const { rgba } = await captureTexture(app.world, camera);
        expect(Array.from(rgba.slice((4 * 8 + 4) * 4, (4 * 8 + 4) * 4 + 4))).toEqual([
            255, 0, 0, 255,
        ]);
    } finally {
        app.dispose();
    }
}
for (const [kind, count, message] of [
    ["texture", 12, "background layout uses 17 fragment sampled textures; WebGPU's limit is 16"],
    ["uniform", 7, "background layout uses 13 fragment uniform buffers; WebGPU's limit is 12"],
    ["sampler", 16, "background layout uses 17 vertex samplers; WebGPU's limit is 16"],
] as const) {
    test(`background ${kind} budget includes engine and shadow bindings`, async () => {
        await expect(draw(kind, count)).rejects.toThrow(message);
    });
    test(`background at the ${kind} limit draws`, async () => {
        await draw(kind, count - 1);
    });
}
