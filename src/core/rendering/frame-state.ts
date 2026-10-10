import type { System, World } from "../../engine";
import { encodeFrameCapture } from "./capture";
import { MAX_SLOTS, type View, Views } from "./view";

export const VIEW_KEY_FLOATS = 25;

export const renderFrameKey = {
    create: () => ({
        camWorld: new Float32Array(16),
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

/** Completes per-frame view state after every draw system; engine submission follows the group. */
export const EndFrameSystem: System = {
    group: "draw",
    boundary: "after",
    update(world) {
        const views = world.resource(Views);
        for (const view of views.values()) {
            encodeFrameCapture(world, view);
            if (view.texture && view.present) view.presented = true;
        }
        views.forEach(clearTargets);
    },
};
