import type { System, World } from "../../engine";
import { encodeFrameCapture } from "./capture";
import { encodeUpscaling } from "./upscaling";
import { INITIAL_VIEW_CAPACITY, type View, Views } from "./view";

export const VIEW_KEY_FLOATS = 29;

export const renderFrameKey = {
    create: () => ({
        camWorld: new Float32Array(16),
        depthOnlyEids: [] as number[],
        depthOnlyViews: [] as import("./view").View[],
        presentingEids: [] as number[],
        presentingViews: [] as import("./view").View[],
        // canvas and texture views sorted by target, Camera.order and eid
        targetEids: [] as number[],
        targetViews: [] as import("./view").View[],
        clipFromWorlds: [] as Float32Array[],
        worldFromClips: [] as Float32Array[],
        viewFromWorlds: [] as Float32Array[],
        clipFromViews: [] as Float32Array[],
        // the View each slot was last packed for: a new View starts with an unfilled viewFromWorld
        slotViews: [] as (import("./view").View | undefined)[],
        viewKeys: new Float64Array(INITIAL_VIEW_CAPACITY * VIEW_KEY_FLOATS).fill(Number.NaN),
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

/** Completes per-frame view state after every draw system: blends each camera above the bottom of a
 * shared target over it, then copies captures. Engine submission follows the group. */
export const EndFrameSystem: System = {
    group: "draw",
    boundary: "after",
    update(world) {
        const views = world.resource(Views);
        encodeUpscaling(world, world.resource(renderFrameKey).presentingViews);
        for (const view of views.values()) {
            encodeFrameCapture(world, view);
            if (view.texture && view.present) view.presented = true;
        }
        views.forEach(clearTargets);
    },
};
