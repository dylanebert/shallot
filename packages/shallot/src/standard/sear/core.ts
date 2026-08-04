// sear's non-public extension surface: the surface chunk environment, the backdrop seam, the opt-in
// prepass lanes, and the relocatable shading chunks a screen-space consumer splices. Rendering's
// contract (custom producers, renderers, the registries) is `render/core`; this is what makes a surface
// shade — and what a screen-space effect samples — under the default renderer.

export { pointAtlasView, shadowSampler, sunShadowParams, sunShadowView } from "./atlas";
export { DEPTH_FORMAT, lightEvalWgsl, TAG_FORMAT, TAG_NONE, Tag } from "./codegen";
// the `Surfaces`/`Backgrounds` contract: schema-carrying bindings + TGSL-fn code chunks.
// `surfaceLayout`/`registerSurface`/
// `backgroundLayout` carry their namespace (`layout`/`register` are too generic for a barrel, exports.md);
// the IO schemas (`VsIn`/`vsPatchSchema`/`fsCtxSchema`/`BgCtx`) are what a typed chunk is authored against
export type {
    Background,
    BgFn,
    BgLayout,
    Binding,
    FsFn,
    Specialize,
    Surface,
    SurfaceLayout,
    TagFn,
    VsFn,
} from "./contract";
export {
    Backgrounds,
    BgCtx,
    backgroundLayout,
    fsCtxSchema,
    registerBackground,
    registerSurface,
    Surfaces,
    surfaceLayout,
    VsIn,
    vsPatchSchema,
} from "./contract";
// the canonical typed engine substrate: the pass-invariant group-0 layout + the shading scaffold authored
// once against it and used by every typed sear pipeline
export {
    clusterOf,
    engineLayout,
    engineScaffoldWgsl,
    fragCoord,
    fragWorld,
    lightFactor,
    lit,
    litPbr,
    pointFactor,
    pointScale,
    sunVisibility,
} from "./engine";
export { ColorSystem, Depth, PrepassSystem } from "./forward";
/** compiled surface-variant cache introspection for renderer diagnostics and real-device gates. */
export { getCompiledSurface } from "./pipelines";
// the relocatable shadow chunks + the uniform layouts they resolve from: the sun pair (struct then
// sampler) and the point/spot pair (caster structs then receiver), each spliced around the group-1
// declarations the consumer makes
export {
    casterWgsl,
    Pbr,
    pointCastersSchema,
    pointShadowRef,
    pointShadowWgsl,
    SHADOW_PARAMS_BYTES,
    SunShadow,
    sampleSunShadow,
    sunShadowWgsl,
    sunStructWgsl,
    tileRectsSchema,
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
