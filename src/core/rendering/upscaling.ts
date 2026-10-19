import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import type { World } from "../../engine";
import {
    beginRenderPass,
    draw,
    endRenderPass,
    setRenderBindGroup,
    setRenderPipeline,
} from "../../engine/runtime";
import type { View } from "./view";

const upscalingLayout = tgpu.bindGroupLayout({
    source: { texture: d.texture2d(d.f32), visibility: ["fragment"] },
    sampler: { sampler: "non-filtering", visibility: ["fragment"] },
});

const upscalingVertex = tgpu.vertexFn({
    in: { index: d.builtin.vertexIndex },
    out: { position: d.builtin.position, uv: d.vec2f },
})((input) => {
    "use gpu";
    const uv = d.vec2f(d.f32((input.index << 1) & 2), d.f32(input.index & 2));
    return { position: d.vec4f(uv.x * 2 - 1, uv.y * 2 - 1, 0, 1), uv: d.vec2f(uv.x, 1 - uv.y) };
});

const upscalingFragment = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })((input) => {
    "use gpu";
    return std.textureSample(upscalingLayout.$.source, upscalingLayout.$.sampler, input.uv);
});

// Premultiplied over: a pixel the camera left at zero alpha keeps the target's value exactly.
const OVER: GPUBlendState = {
    color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
};

function createUpscaling(world: World) {
    const device = world.gpu.device;
    const pipeline = world.gpu.root
        .createRenderPipeline({
            vertex: upscalingVertex,
            fragment: upscalingFragment,
            targets: { format: navigator.gpu.getPreferredCanvasFormat(), blend: OVER },
        })
        .$name("upscaling");
    const attachment: GPURenderPassColorAttachment = {
        view: null!,
        loadOp: "load",
        storeOp: "store",
    };
    return {
        pipeline: world.gpu.root.unwrap(pipeline),
        layout: world.gpu.root.unwrap(upscalingLayout),
        sampler: device.createSampler({ minFilter: "nearest", magFilter: "nearest" }),
        groups: new WeakMap<GPUTextureView, GPUBindGroup>(),
        attachment,
        pass: { label: "upscaling", colorAttachments: [attachment] } as GPURenderPassDescriptor,
    };
}
const upscalingKey = { create: createUpscaling };

/**
 * Blends each presented camera above the bottom of its target over that target, bottom to top, as Bevy's
 * upscaling node writes every camera after the first with alpha blending. The bottom camera presents into
 * the target directly, so it takes no blit. `views` is this frame's presenting views sorted by target,
 * Camera.order and eid. Runs after every draw system, so a replacement presentation has written its image.
 */
export function encodeUpscaling(world: World, views: readonly View[]): void {
    let state: ReturnType<typeof createUpscaling> | undefined;
    for (let i = 0; i < views.length; i++) {
        const view = views[i]!;
        if (view.sortedIndexForTarget === 0 || !view.present || !view.surfaceView) continue;
        state ??= world.resource(upscalingKey);
        let group = state.groups.get(view.present);
        if (!group) {
            group = world.gpu.device.createBindGroup({
                layout: state.layout,
                entries: [
                    { binding: 0, resource: view.present },
                    { binding: 1, resource: state.sampler },
                ],
            });
            state.groups.set(view.present, group);
        }
        state.attachment.view = view.surfaceView;
        const pass = beginRenderPass(world.frameEncoder()!, state.pass);
        setRenderPipeline(pass, state.pipeline);
        setRenderBindGroup(pass, 0, group);
        draw(pass, 3);
        endRenderPass(pass);
    }
}
