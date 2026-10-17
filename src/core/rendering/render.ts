import type { World } from "../../engine";

/**
 * World-owned renderer context. Record through World.frameEncoder; submission follows the draw group.
 * viewBuffers grows from one to MAX_VIEWS, one ViewUniforms buffer per presenting slot.
 * viewCount is the active cull-slot count; shadeCount is its presenting prefix. Depth-only slots
 * [shadeCount, viewCount) have cull volumes and staging but no view buffer.
 * cullVolumes holds tagged six-plane frusta.
 */
export interface RenderContext {
    format: GPUTextureFormat;
    viewBuffers: GPUBuffer[];
    viewStaging: Float32Array;
    cullVolumes: GPUBuffer;
    cullVolumeStaging: Float32Array;
    viewCount: number;
    shadeCount: number;
}

export const renderKey = { create: createRender };

function createRender(): RenderContext {
    return {
        format: "" as GPUTextureFormat,
        viewBuffers: [],
        viewStaging: null!,
        cullVolumes: null!,
        cullVolumeStaging: null!,
        viewCount: 0,
        shadeCount: 0,
    };
}

export function initializeRenderState(world: World): void {
    world.resource(renderKey);
}

/** World-owned rendering state, resolved with `world.resource(RenderContext)`. */
export const RenderContext: import("../../engine").Resource<RenderContext> = {
    create: (world) => world.resource(renderKey),
};
