import { type Plugin, precompile } from "@dylanebert/shallot";
import { MainPassSystem } from "@dylanebert/shallot/rendering";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { FIELD, groundAt } from "./hills";

export const RESOLUTION = 512;
export const DEPTH = 0.45;
export const Trail = {
    create: () => ({
        capture: null as GPUTexture | null,
        image: null as GPUTexture | null,
        previous: null as GPUTexture | null,
        ground: null as GPUTexture | null,
        pipeline: null as GPUComputePipeline | null,
        group: null as GPUBindGroup | null,
    }),
};
const layout = tgpu.bindGroupLayout({
    capture: { texture: d.texture2d(d.f32) },
    previous: { texture: d.texture2d(d.f32), sampleType: "unfilterable-float" },
    next: { storageTexture: d.textureStorage2d("r32float", "write-only") },
    ground: { texture: d.texture2d(d.f32), sampleType: "unfilterable-float" },
});
/** The steepest snow wall, as rise over run: above every hill's slope, so untouched snow stays. */
const TALUS = 1;
/** The least press at `xy` that keeps its snow top within TALUS of the neighbor at `n`'s. */
const slump = tgpu.fn(
    [d.vec2i, d.vec2i],
    d.f32,
)((xy, n) => {
    "use gpu";
    const m = std.clamp(n, d.vec2i(0), d.vec2i(RESOLUTION - 1));
    const rise =
        std.textureLoad(layout.$.ground, xy, 0).r -
        std.textureLoad(layout.$.ground, m, 0).r -
        (TALUS * FIELD) / RESOLUTION;
    return std.textureLoad(layout.$.previous, m, 0).r + rise / DEPTH;
});
const accumulate = tgpu.computeFn({
    workgroupSize: [8, 8],
    in: { gid: d.builtin.globalInvocationId },
})((input) => {
    "use gpu";
    const xy = d.vec2i(input.gid.xy);
    const encoded = std.textureLoad(layout.$.capture, xy, 0).r;
    const press = std.select(
        encoded / 12.92,
        std.pow((encoded + 0.055) / 1.055, 2.4),
        encoded > 0.04045,
    );
    const previous = std.textureLoad(layout.$.previous, xy, 0).r;
    // Snow steeper than TALUS slumps outward, one texel a frame.
    const slumped = std.max(
        std.max(slump(xy, xy.add(d.vec2i(1, 0))), slump(xy, xy.sub(d.vec2i(1, 0)))),
        std.max(slump(xy, xy.add(d.vec2i(0, 1))), slump(xy, xy.sub(d.vec2i(0, 1)))),
    );
    const next = std.min(std.max(std.max(previous, press), slumped), 1);
    std.textureStore(layout.$.next, xy, d.vec4f(next, 0, 0, 1));
});
export const TrailPlugin: Plugin = {
    name: "Trail",
    gpu: {},
    initialize(world) {
        const s = world.resource(Trail);
        const device = world.gpu.device;
        s.image = device.createTexture({
            label: "snow:trail",
            size: [RESOLUTION, RESOLUTION],
            format: "r32float",
            usage:
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC,
        });
        s.previous = device.createTexture({
            label: "snow:previous",
            size: [RESOLUTION, RESOLUTION],
            format: "r32float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        s.ground = device.createTexture({
            label: "snow:ground",
            size: [RESOLUTION, RESOLUTION],
            format: "r32float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        world.own(s.image);
        world.own(s.previous);
        world.own(s.ground);
        // The ground under each texel's center; row v runs toward -z, as the capture looks.
        const ground = new Float32Array(RESOLUTION * RESOLUTION);
        for (let v = 0; v < RESOLUTION; v++)
            for (let u = 0; u < RESOLUTION; u++)
                ground[v * RESOLUTION + u] = groundAt(
                    ((u + 0.5) / RESOLUTION - 0.5) * FIELD,
                    (0.5 - (v + 0.5) / RESOLUTION) * FIELD,
                );
        device.queue.writeTexture({ texture: s.ground }, ground, { bytesPerRow: RESOLUTION * 4 }, [
            RESOLUTION,
            RESOLUTION,
        ]);
        s.pipeline = world.gpu.root.unwrap(
            world.gpu.root.createComputePipeline({ compute: accumulate }).$name("snow:accumulate"),
        );
        precompile(world, "snow trail", () => [s.pipeline!]);
    },
    systems: [
        {
            name: "accumulate-trail",
            group: "draw",
            before: [MainPassSystem],
            update(world) {
                const s = world.resource(Trail);
                s.group ??= world.gpu.device.createBindGroup({
                    layout: s.pipeline!.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: s.capture!.createView() },
                        { binding: 1, resource: s.previous!.createView() },
                        { binding: 2, resource: s.image!.createView() },
                        { binding: 3, resource: s.ground!.createView() },
                    ],
                });
                const encoder = world.frameEncoder()!;
                const pass = encoder.beginComputePass({ label: "snow:accumulate" });
                pass.setPipeline(s.pipeline!);
                pass.setBindGroup(0, s.group);
                pass.dispatchWorkgroups(RESOLUTION / 8, RESOLUTION / 8);
                pass.end();
                encoder.copyTextureToTexture({ texture: s.image! }, { texture: s.previous! }, [
                    RESOLUTION,
                    RESOLUTION,
                ]);
            },
        },
    ],
};
