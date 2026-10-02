import tgpu, { type TgpuRenderPipeline } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import type { World } from "../../engine";
import { GradingConfig, grade } from "./color-grading";
import { tmLuma, tonemap } from "./tonemap";
import { tonyLut } from "./tony-lut";
import { linearToSrgb3 } from "./view";

export const fullscreenVertex = tgpu.vertexFn({
    in: { index: d.builtin.vertexIndex },
    out: { position: d.builtin.position },
})((input) => {
    "use gpu";
    const uv = d.vec2f(d.f32((input.index << 1) & 2), d.f32(input.index & 2));
    return { position: d.vec4f(uv.x * 2 - 1, uv.y * 2 - 1, 0, 1) };
});

export function tonemappingLayout() {
    return tgpu
        .bindGroupLayout({
            input: { texture: d.texture2d(d.f32), visibility: ["fragment"] },
            grading: { uniform: GradingConfig, visibility: ["fragment"] },
            lut: { texture: d.texture3d(d.f32), visibility: ["fragment"] },
            sampler: { sampler: "filtering", visibility: ["fragment"] },
        })
        .$idx(0);
}

export function tonemappingKernel(layout: ReturnType<typeof tonemappingLayout>) {
    return tgpu
        .fragmentFn({ in: { position: d.builtin.position }, out: d.vec4f })((input) => {
            "use gpu";
            let color = grade(
                std.textureLoad(layout.$.input, d.vec2u(input.position.xy), 0).xyz,
                layout.$.grading,
            );
            if (layout.$.grading.tonemapMode === 0) {
                const encoded = std.div(color, std.add(color, 1));
                const uv = std.add(std.mul(encoded, 47 / 48), 0.5 / 48);
                color = std.textureSampleLevel(
                    layout.$.lut,
                    layout.$.sampler,
                    std.saturate(uv),
                    0,
                ).xyz;
            } else color = tonemap(layout.$.grading.tonemapMode, color);
            color = std.mix(d.vec3f(tmLuma(color)), color, layout.$.grading.postSaturation);
            return d.vec4f(linearToSrgb3(std.max(color, d.vec3f(0))), 1);
        })
        .$name("tonemapping");
}

type Composite = {
    layout: ReturnType<typeof tonemappingLayout>;
    pipeline: TgpuRenderPipeline;
    lut: GPUTextureView;
    sampler: GPUSampler;
};
export const compositeCacheKey = { create: () => new Map<GPUTextureFormat, Composite>() };

export function composite(world: World, format: GPUTextureFormat): Composite {
    if (format !== "bgra8unorm" && format !== "rgba8unorm")
        throw new Error(`Tonemapping: unsupported output format ${format}`);
    const cache = world.resource(compositeCacheKey);
    const cached = cache.get(format);
    if (cached) return cached;
    const device = world.gpu.device;
    const shared = cache.values().next().value;
    let lut = shared?.lut;
    if (!lut) {
        const texture = device.createTexture({
            label: "tony-mc-mapface",
            size: [48, 48, 48],
            dimension: "3d",
            format: "rgb9e5ufloat",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        world.own(texture);
        device.queue.writeTexture(
            { texture },
            tonyLut,
            { bytesPerRow: 48 * 4, rowsPerImage: 48 },
            [48, 48, 48],
        );
        lut = texture.createView();
    }
    const layout = shared?.layout ?? tonemappingLayout();
    const pipeline = world.gpu.root
        .createRenderPipeline({
            vertex: fullscreenVertex,
            fragment: tonemappingKernel(layout),
            targets: { format },
        })
        .$name("tonemapping");
    const result = {
        layout,
        pipeline,
        lut,
        sampler:
            shared?.sampler ?? device.createSampler({ minFilter: "linear", magFilter: "linear" }),
    };
    cache.set(format, result);
    return result;
}
