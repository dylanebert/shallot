import type { World } from "../../engine";
import { MAX_SLOTS } from "./view";

export const VIEW_KEY_FLOATS = 26;

export const renderFrameKey = {
    create: () => ({
        camWorld: new Float32Array(16),
        submit: [] as GPUCommandBuffer[],
        depthOnlyEids: [] as number[],
        depthOnlyViews: [] as import("./view").View[],
        viewProjs: [] as Float32Array[],
        invViewProjs: [] as Float32Array[],
        lightViews: [] as Float32Array[],
        viewKeys: new Float64Array(MAX_SLOTS * VIEW_KEY_FLOATS).fill(Number.NaN),
        viewKeyNext: new Float64Array(VIEW_KEY_FLOATS),
    }),
};

export function initializeRenderFrameState(world: World): void {
    world.resource(renderFrameKey);
}
