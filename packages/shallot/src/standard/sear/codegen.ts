// Sear's shared render constants and relocatable clustered-light WGSL.

import { chunk, octEncodeWgsl, spliceNs } from "../../engine/utils/core";
import type { View } from "../render/core";
import { clusterCell, distanceAttenuation, pointLightsWgsl, spotFactor } from "../render/core";

// the depth format shared by the color pass's own 4× MSAA depth, the 1× prepass depth, and the shadow
// map. depth32float is sampleable and the reverse-Z precision win needs a float buffer (an integer depth
// gains nothing from reverse-Z); one source of truth for every depth-stencil state — and the shadow map
// renders through sear's compiled prepass depth pipeline, so one format keeps them sharing it
/** the depth-stencil format for every sear depth target (color pass, prepass, shadow atlases): one format so they share the compiled prepass depth pipeline */
export const DEPTH_FORMAT: GPUTextureFormat = "depth32float";

// the geometric-AA sample count when a camera's `Camera.antialias` is on (the default); off renders
// single-sample straight into the offscreen, no resolve. Per-camera + runtime-toggleable, sample-based
// only (no post-process blur). The prepass + shadow map stay single-sample regardless — an id can't
// MSAA-resolve and a render pass can't mix counts, so they own their own 1× depth, never varying by this
export const SAMPLE_COUNT = 4;

// the id lane's screen-space target, published per camera as `view.tag`. r32uint holds the front-most
// fragment's tag per pixel, filled by the single-sample prepass (not the color MRT — see PrepassSystem)
// for cameras carrying the `Tag` marker. The tag is surface-authored — a mutable fs local defaulting to
// the entity's eid for an instanced surface and TAG_NONE otherwise, which a surface's fs overrides
// (terrain → `capacity + cell`). A consumer reads `view.tag` to know which surface owns each pixel
// (hover, outline, debug)
/** the id-lane texture format (`r32uint`): an integer id can't MSAA-resolve, which forces the single-sample prepass */
export const TAG_FORMAT: GPUTextureFormat = "r32uint";

// the reserved tag sentinel: the default for a non-instanced surface that authors no tag, and the
// value the tag target clears to (the background). eids are bounded by `capacity`, so 0xffffffff
// never collides with a real one — a reader takes any other value as a literal surface tag. Exported
// so a consumer interprets the readback without re-deriving the sentinel
/** the reserved id-lane sentinel: a pixel no surface owns. A consumer decoding `view.tag` reads any other value as a literal surface tag */
export const TAG_NONE = 0xffffffff;

/**
 * opt a Sear camera into the **id lane**: the `view.tag` target {@link PrepassSystem} fills. Unreal's
 * `CustomStencil` generalized from an 8-bit stencil to a u32 lane; Bevy's prepass carries no id (it
 * CPU-raycasts), so the id rides this single-sample pass because it's the same rasterization. A marker
 * in the spirit of Bevy's `DepthPrepass` / `NormalPrepass`: add it to enable one extra camera output;
 * omit it and the lane is absent (no target allocated). Per-camera, so a minimap opts out while the main
 * view opts in. A consumer that reads `view.tag` (hover, outline, picking) wants this on its camera; the
 * engine itself stays tag-agnostic.
 *
 * @example
 * ```
 * <a camera sear tag transform />
 * ```
 */
export const Tag = {};

// ---- prepass lanes: opt-in screen-space outputs, a closed engine-owned union (not a consumer registry).
// Each lane is gated by a camera marker (Bevy's DepthPrepass / NormalPrepass shape). Two ship: the `depth`
// lane (the depth-stencil itself, marker `Depth`, published as `view.depth`) and the id lane (a color
// attachment, marker `Tag`, published as `view.tag`). normal / motion are the future rows — adding one is
// a COLOR_LANES entry + a `View.*` field; the prepass already iterates it, never a new pass. `depth` isn't
// a color attachment (it's the depth-stencil), so it's stored or discarded by the `Depth` marker, separate
// from COLOR_LANES (the color attachments the prepass MRTs)
export interface ColorLane {
    // the `view.*` field + lane identity ("tag"); `set` publishes the rendered texture onto that field
    name: string;
    marker: object;
    format: GPUTextureFormat;
    usage: number;
    clear: GPUColor;
    // the mutable fs local a surface authors (symmetric with `col`), its WGSL type, and its per-surface
    // default (the fs chunk may override it)
    local: string;
    type: string;
    init(instanced: boolean): string;
    set(view: View, texture: GPUTexture): void;
}

// the id lane: the front-most opaque / `clip` surface's tag per pixel. r32uint (an integer id can't
// MSAA-resolve — what forces the prepass single-sample), COPY_SRC for a hover readback + TEXTURE_BINDING
// for an outline sample, cleared to TAG_NONE (the "no surface" background). The tag local defaults to the
// instance's eid (instanced) or TAG_NONE (a world-space producer like terrain authors its own)
export const COLOR_LANES: ColorLane[] = [
    {
        name: "tag",
        marker: Tag,
        format: TAG_FORMAT,
        usage:
            GPUTextureUsage.RENDER_ATTACHMENT |
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_SRC,
        clear: { r: TAG_NONE, g: 0, b: 0, a: 0 },
        local: "tag",
        type: "u32",
        init: (instanced) => (instanced ? "eid" : `${TAG_NONE}u`),
        set: (view, texture) => {
            view.tag = texture;
        },
    },
];

// a lane-set's stable key (the prepass pipeline-map key): "" for the empty depth-only set, "tag" for the
// id lane, "tag-normal" when normal lands
export function laneKey(lanes: ColorLane[]): string {
    return lanes.map((l) => l.name).join("-");
}

/** relocatable clustered-light WGSL (`distanceAttenuation` / `spotFactor` / `clusterCell`) so a screen-space consumer evaluates the same froxel lights sear's color FS does */
export function lightEvalWgsl(): string {
    // force the base chunks first: `PointLightGpu` belongs to the light-list chunk and `octDecodeNormal`
    // to the oct chunk, and every consumer splices both ahead of this one
    pointLightsWgsl();
    octEncodeWgsl();
    return lightEvalChunk();
}

const lightEvalChunk = chunk(
    "lightEvalWgsl",
    [distanceAttenuation, spotFactor, clusterCell],
    spliceNs,
);
