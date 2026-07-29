// sear's non-public extension surface: the surface chunk environment, the backdrop seam, the opt-in
// prepass lanes, and the relocatable shading chunks a screen-space consumer splices. Rendering's
// contract (custom producers, renderers, the registries) is `render/core`; this is what makes a surface
// shade — and what a screen-space effect samples — under the default renderer.

export type { Background } from "./forward";
export {
    Backgrounds,
    ColorSystem,
    DEPTH_FORMAT,
    Depth,
    lightEvalWgsl,
    PrepassSystem,
    pointAtlasView,
    shadowSampler,
    sunShadowParams,
    sunShadowView,
    TAG_FORMAT,
    TAG_NONE,
    Tag,
} from "./forward";
// the relocatable shadow chunks + the uniform layouts they resolve from: the sun pair (struct then
// sampler) and the point/spot pair (caster structs then receiver), each spliced around the group-1
// declarations the consumer makes
export {
    casterWgsl,
    pointShadowWgsl,
    SHADOW_PARAMS_BYTES,
    sunShadowWgsl,
    sunStructWgsl,
} from "./shade";
// the shadow-caster diagnostic surface: the pooled cascade/combo cull-slot eids + resolved atlas sizing a
// GPU-readback oracle pins per-cascade / per-combo survivor counts against (a custom shadow tool reads the
// same). sear owns the render; these are read-only introspection, so they live at the extension tier.
export {
    cascadeComboEids,
    cascadeCount,
    pointAtlasSize,
    pointCasters,
    pointComboCount,
    pointComboEids,
} from "./shadows";
