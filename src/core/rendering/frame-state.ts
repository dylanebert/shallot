import type { System, World } from "../../engine";
import { Render } from "./render";
import { MAX_SLOTS, type View, Views } from "./view";

export const VIEW_KEY_FLOATS = 26;

export const renderFrameKey = {
    create: () => ({
        camWorld: new Float32Array(16),
        submit: [] as GPUCommandBuffer[],
        depthOnlyEids: [] as number[],
        depthOnlyViews: [] as import("./view").View[],
        viewProjs: [] as Float32Array[],
        invViewProjs: [] as Float32Array[],
        viewKeys: new Float64Array(MAX_SLOTS * VIEW_KEY_FLOATS).fill(Number.NaN),
        viewKeyNext: new Float64Array(VIEW_KEY_FLOATS),
    }),
};

export function initializeRenderFrameState(world: World): void {
    world.resource(renderFrameKey);
}

function clearTargets(view: View): void {
    view.framebuffer = null;
    view.framebufferFormat = undefined;
    view.present = null;
}

/** closes the frame: submits the encoder, advances `world.gpu.frame` */
export const EndFrameSystem: System = {
    group: "draw",
    terminal: true,
    update(world) {
        const _render = world.resource(Render);
        const _renderFrame = world.resource(renderFrameKey);

        const device = world.gpu.device;
        if (!device) return;
        const encoder = _render.encoder;
        if (!encoder)
            throw new Error("render submission requires BeginFrameSystem to open an encoder");
        _renderFrame.submit[0] = encoder.finish();
        device.queue.submit(_renderFrame.submit);
        for (const view of world.resource(Views).values()) {
            if (view.texture && view.present) view.presented = true;
        }
        world.endGpuFrame();
        _render.encoder = null;
        world.resource(Views).forEach(clearTargets);
    },
};
