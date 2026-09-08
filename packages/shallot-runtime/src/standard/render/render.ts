/**
 * device-level render state owned by `RenderPlugin`. `encoder` is transient
 * per-frame state set by `BeginFrameSystem`; `viewBuffers` is one static
 * `View`-struct uniform buffer per shading slot (`MAX_VIEWS` of them — the
 * per-slot-buffer design, replacing the old single dynamic-offset UBO: a
 * depth-only slot's shadow-atlas passes never read `view`, so only the shading
 * prefix needs a real buffer). `viewStaging` stays the full per-slot pack
 * (shading + depth-only), unchanged — the writer sources each shading slot's
 * 208 B from the same subrange it always did. `cullVolumes` packs one per-slot cull
 * volume per active view (a tagged descriptor carrying a frustum's six clip-space planes;
 * published to `Compute.buffers` as `"cullVolumes"`); a GPU cull pass tests instance bounds
 * for `cullVolumes[slot]`.
 * `viewCount` is how many slots `BeginFrameSystem` populated this frame: the
 * view dimension a producer's cull dispatches over. `shadeCount` is the shading
 * prefix of those slots (presenting cameras, the views that carry clustered-light
 * state); depth-only views (shadow light cameras) fill `[shadeCount, viewCount)`,
 * so the cluster + light-cull passes dispatch over `shadeCount` alone.
 * Renderer-agnostic: knows
 * nothing about how draws are issued. Lives as a leaf (no intra-module imports)
 * so the frame loop and the view binding both depend on it inward
 * @expand
 */
export interface Render {
    format: GPUTextureFormat;
    encoder: GPUCommandEncoder | null;
    viewBuffers: GPUBuffer[];
    viewStaging: Float32Array;
    cullVolumes: GPUBuffer;
    cullVolumeStaging: Float32Array;
    viewCount: number;
    shadeCount: number;
}

export const Render: Render = {
    format: "" as GPUTextureFormat,
    encoder: null,
    viewBuffers: [],
    viewStaging: null!,
    cullVolumes: null!,
    cullVolumeStaging: null!,
    viewCount: 0,
    shadeCount: 0,
};
