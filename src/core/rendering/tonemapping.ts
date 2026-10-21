import tgpu, { type TgpuRenderPipeline } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import type { World } from "../../engine";
import { agxLut } from "./agx-lut";
import { GradingConfig, grade } from "./color-grading";
import { tmAgxLog, tmLuma, tonemap } from "./tonemap";
import { tonyLut } from "./tony-lut";
import { linearToSrgb3 } from "./view";

/** Bevy's `fullscreen_vertex_shader`: one triangle covering the viewport at depth 0, drawn with
 * three vertices and no vertex buffer. `uv` is (0, 0) at the top-left corner and (1, 1) at the
 * bottom-right. Bloom and the effect stack hold Bevy parity through this exact triangle: another one
 * covering the viewport interpolates uv with different rounding. */
export const fullscreenVertex = tgpu.vertexFn({
    in: { index: d.builtin.vertexIndex },
    out: { position: d.builtin.position, uv: d.vec2f },
})((input) => {
    "use gpu";
    const uv = std.mul(d.vec2f(d.f32(input.index >>> 1), d.f32(input.index & 1)), 2);
    return { position: d.vec4f(std.add(std.mul(uv, d.vec2f(2, -2)), d.vec2f(-1, 1)), 0, 1), uv };
});

export function tonemappingLayout() {
    return tgpu
        .bindGroupLayout({
            input: { texture: d.texture2d(d.f32), visibility: ["fragment"] },
            grading: { uniform: GradingConfig, visibility: ["fragment"] },
            lut: { texture: d.texture3d(d.f32), visibility: ["fragment"] },
            sampler: { sampler: "filtering", visibility: ["fragment"] },
            agx: { texture: d.texture3d(d.f32), visibility: ["fragment"] },
        })
        .$idx(0);
}

export function tonemappingKernel(layout: ReturnType<typeof tonemappingLayout>) {
    return tgpu
        .fragmentFn({ in: { position: d.builtin.position }, out: d.vec4f })((input) => {
            "use gpu";
            const texel = std.textureLoad(layout.$.input, d.vec2u(input.position.xy), 0);
            let source = texel.xyz;
            let alpha = d.f32(1);
            if (layout.$.grading.premultipliedAlpha !== 0) {
                alpha = std.saturate(texel.w);
                if (alpha > 0) source = std.div(source, alpha);
            }
            if (layout.$.grading.tonemapMode === 1)
                return d.vec4f(std.mul(linearToSrgb3(std.max(source, d.vec3f(0))), alpha), alpha);
            let color = grade(source, layout.$.grading);
            if (layout.$.grading.tonemapMode === 0) {
                const encoded = std.div(color, std.add(color, 1));
                const uv = std.add(std.mul(encoded, 47 / 48), 0.5 / 48);
                color = std.textureSampleLevel(
                    layout.$.lut,
                    layout.$.sampler,
                    std.saturate(uv),
                    0,
                ).xyz;
            } else if (layout.$.grading.tonemapMode === 5) {
                const uv = std.add(std.mul(tmAgxLog(color), 31 / 32), 0.5 / 32);
                color = std.textureSampleLevel(layout.$.agx, layout.$.sampler, uv, 0).xyz;
            } else color = tonemap(layout.$.grading.tonemapMode, color);
            color = std.mix(d.vec3f(tmLuma(color)), color, layout.$.grading.postSaturation);
            if (layout.$.grading.debandDither !== 0) {
                color = std.mul(std.pow(std.abs(color), d.vec3f(1 / 2.2)), std.sign(color));
                const seed = std.dot(d.vec2f(171, 231), input.position.xy);
                const noise = std.div(
                    std.sub(std.fract(std.div(d.vec3f(seed), d.vec3f(103, 71, 97))), 0.5),
                    255,
                );
                color = std.add(color, noise);
                color = std.mul(std.pow(std.abs(color), d.vec3f(2.2)), std.sign(color));
            }
            return d.vec4f(std.mul(linearToSrgb3(std.max(color, d.vec3f(0))), alpha), alpha);
        })
        .$name("tonemapping");
}

type Composite = {
    layout: ReturnType<typeof tonemappingLayout>;
    pipeline: TgpuRenderPipeline;
    lut: GPUTextureView;
    agx: GPUTextureView;
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
    let agx = shared?.agx;
    if (!agx) {
        const texture = device.createTexture({
            label: "agx",
            size: [32, 32, 32],
            dimension: "3d",
            format: "rgba16float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        world.own(texture);
        device.queue.writeTexture(
            { texture },
            agxLut,
            { bytesPerRow: 32 * 8, rowsPerImage: 32 },
            [32, 32, 32],
        );
        agx = texture.createView();
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
        agx,
        sampler:
            shared?.sampler ?? device.createSampler({ minFilter: "linear", magFilter: "linear" }),
    };
    cache.set(format, result);
    return result;
}
