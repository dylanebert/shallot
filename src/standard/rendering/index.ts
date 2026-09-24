// Sear's author barrel — the game-author surface of the default renderer. The renderer itself is
// `forward.ts` (the GPU-driven forward pass); this file re-exports only what a scene author touches: the
// `Sear` camera marker + its opt-in prepass lanes (`Tag` / `Depth`), the `Material` / `Backdrop`
// components, the `Shadow` cast opt-in + its `SunShadows` / `PointShadows` config, and `SearPlugin`. The
// extension surface (surface codegen, the relocatable shading chunks, the backdrop registry, the ordering
// anchors) follows below, drawn from the same `forward.ts` impl.
// re-export each name from its definition site (one hop, so the reference generator resolves the JSDoc):
// the renderer + most of its components live in forward.ts, `Tag` in codegen.ts (it's part of the
// COLOR_LANES table there), the shadow config + cast opt-in in shadows.ts.

import type { Plugin } from "../../engine";
import { createSearPlugin } from "./forward";

export { Tag } from "./codegen";
export { Backdrop, Depth, Material, Sear } from "./forward";
/**
 * Sear: the one shallot renderer. A GPU-driven raster forward pass: a 4× MSAA color pass (opaque draws
 * then `blend` draws composited over them, fused into one render pass) and an opt-in single-sample
 * prepass emitting per-camera lanes (the {@link Tag} → `view.tag` id lane, the {@link Depth} →
 * `view.depth` lane), with sun shadows sampled inline in the FS. Add `SearPlugin` and give a Camera the
 * {@link Sear} marker and the happy path renders. Sun shadows are data-gated on the {@link Shadow}
 * component on a `DirectionalLight`: add it to cast (and tune), omit it for the fully-lit bare path (no
 * shadow map allocated), exactly like a camera without a lane marker runs no prepass. No separate shadow
 * plugin, no coordination singleton: sear owns its own shadow map and binds it (Bevy's clustered-forward
 * shape). Sear renders into the offscreen (`view.framebuffer`) and never the swapchain; presenting it is
 * a separate **composite** the consumer picks: glaze (the default postfx composite) or a custom one. So
 * sear depends only on {@link RenderPlugin}; list a composite alongside it or nothing reaches the
 * swapchain. `ColorSystem` still orders before glaze so glaze, when present, composites after the resolve.
 */
export const SearPlugin: Plugin = { ...createSearPlugin(), device: "required" };
export { MAX_CASCADES, MAX_POINT_CASTERS, PointShadows, Shadow, SunShadows } from "./shadows";

// sear's non-public extension surface: the surface chunk environment, the backdrop seam, the opt-in
// prepass lanes, and the relocatable shading chunks a screen-space consumer splices. Rendering's
// contract (custom producers, renderers, the registries) is the render barrel; this is what makes a surface
// shade — and what a screen-space effect samples — under the default renderer.

export { pointAtlasView, shadowSampler, sunShadowParams, sunShadowView } from "./atlas";
export { DEPTH_FORMAT, lightEvalWgsl, TAG_FORMAT, TAG_NONE } from "./codegen";
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
export { ColorSystem, PrepassSystem } from "./forward";
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
