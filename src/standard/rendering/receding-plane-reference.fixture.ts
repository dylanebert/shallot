import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { fullscreenVertex } from "../../core/rendering";
import { probeTexture, type World } from "../../engine";
import { planeUv } from "./receding-plane.fixture";

const point = d.struct({ uv: d.vec2f, dx: d.vec2f, dy: d.vec2f });
const layout = tgpu.bindGroupLayout({
    images: { texture: d.texture2dArray() },
    sampler: { sampler: "filtering" },
    points: { storage: d.arrayOf(point), access: "readonly" },
    width: { uniform: d.u32 },
});
const fragment = tgpu.fragmentFn({ in: { pos: d.builtin.position }, out: d.vec4f })((input) => {
    "use gpu";
    const p = layout.$.points[d.u32(input.pos.y) * layout.$.width + d.u32(input.pos.x)];
    return std.textureSampleGrad(layout.$.images, layout.$.sampler, p.uv, 0, p.dx, p.dy);
});

/** Direct hardware filtering at independently intersected pixel centres and quad gradients. */
export async function hardwarePlaneReference(
    world: World,
    images: GPUTexture,
    width: number,
    height: number,
) {
    const { device, root } = world.gpu;
    const points = root
        .createBuffer(
            d.arrayOf(point, width * height),
            Array.from({ length: width * height }, (_, i) => {
                const x = i % width;
                const y = Math.floor(i / width);
                const c = planeUv(width, height, (x & ~1) + 0.5, (y & ~1) + 0.5) ?? [0, 0];
                const r = planeUv(width, height, (x & ~1) + 1.5, (y & ~1) + 0.5) ?? c;
                const b = planeUv(width, height, (x & ~1) + 0.5, (y & ~1) + 1.5) ?? c;
                const uv = planeUv(width, height, x + 0.5, y + 0.5) ?? [0, 0];
                return {
                    uv: d.vec2f(...uv),
                    dx: d.vec2f(r[0] - c[0], r[1] - c[1]),
                    dy: d.vec2f(b[0] - c[0], b[1] - c[1]),
                };
            }),
        )
        .$usage("storage");
    const output = device.createTexture({
        size: [width, height],
        format: "rgba8unorm-srgb",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    world.own(output);
    const group = root.createBindGroup(layout, {
        images: images.createView(),
        sampler: device.createSampler({
            addressModeU: "clamp-to-edge",
            addressModeV: "clamp-to-edge",
            magFilter: "linear",
            minFilter: "linear",
            mipmapFilter: "linear",
        }),
        points,
        width: root.createBuffer(d.u32, width).$usage("uniform"),
    });
    const pipeline = root.createRenderPipeline({
        vertex: fullscreenVertex,
        fragment,
        targets: { format: "rgba8unorm-srgb" },
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: output.createView(), loadOp: "clear", storeOp: "store" }],
    });
    pipeline.with(group).with(pass).draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
    return {
        rgba: new Uint8ClampedArray((await probeTexture(world, output)).bytes),
        width,
        height,
    };
}
