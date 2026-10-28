import { type Plugin, precompile } from "@dylanebert/shallot";
import { MainPassSystem } from "@dylanebert/shallot/rendering";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";

export const FIELD = 24;
export const RESOLUTION = 512;
export const DEPTH = 0.45;
export const Trail = {
    create: () => ({
        capture: null as GPUTexture | null,
        image: null as GPUTexture | null,
        previous: null as GPUTexture | null,
        pipeline: null as GPUComputePipeline | null,
        group: null as GPUBindGroup | null,
    }),
};
const layout = tgpu.bindGroupLayout({
    capture: { texture: d.texture2d(d.f32) },
    previous: { texture: d.texture2d(d.f32), sampleType: "unfilterable-float" },
    next: { storageTexture: d.textureStorage2d("r32float", "write-only") },
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
    std.textureStore(layout.$.next, xy, d.vec4f(std.max(previous, press), 0, 0, 1));
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
        world.own(s.image);
        world.own(s.previous);
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
