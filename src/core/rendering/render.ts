import type { World } from "../../engine";

/**
 * World-owned GPU context shared by renderers. BeginFrameSystem opens `encoder`;
 * every frame pass records on it before the terminal submission.
 * `viewBuffers` holds one 208-byte ViewUniforms buffer per presenting slot, bounded
 * by MAX_VIEWS. Depth-only slots have staging and cull volumes but no view buffer.
 * `cullVolumes`, published under that name in world.gpu.buffers, contains one
 * tagged six-plane frustum per active slot for producers to test their bounds.
 * `viewCount` is the populated slot count; `shadeCount` is its presenting prefix.
 * Depth-only views occupy [shadeCount, viewCount). No draw or light layout is assumed.
 */
export interface RenderContext {
    format: GPUTextureFormat;
    encoder: GPUCommandEncoder | null;
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
        encoder: null,
        viewBuffers: [],
        viewStaging: null!,
        cullVolumes: null!,
        cullVolumeStaging: null!,
        viewCount: 0,
        shadeCount: 0,
    };
}

/** Create this world's render state during RenderingPlugin initialization. */
export function initializeRenderState(world: World): void {
    world.resource(renderKey);
}

/** World-owned rendering state, resolved with `world.resource(RenderContext)`. */
export const RenderContext: import("../../engine").Resource<RenderContext> = {
    create: (world) => world.resource(renderKey),
};
