// sear's non-public extension surface: the surface chunk environment, the backdrop seam, the opt-in
// prepass lanes, and the relocatable shading chunks a screen-space consumer splices. Rendering's
// contract (custom producers, renderers, the registries) is `render/core`; this is what makes a surface
// shade — and what a screen-space effect samples — under the default renderer.

// #doc:dev
// ## Shading under sear
//
// A surface's `vs` / `fs` chunks are spliced into sear's shader — sear owns the vertex pull, the frame /
// view / lighting uniforms, and the entry points, and provides the prelude a chunk shades against: the
// standard per-instance transform (declare the `eids` + `transforms` bindings, no `vs` chunk needed), the
// quantized-vertex decode, and the lighting helpers `lit(base, normal)` / `lightFactor(normal)` /
// `litPbr(pbr, normal, world)` reading the `Lighting` uniform. A chunk shades by calling them or writes
// `col` directly to stay unlit. `sunVisibility` (the sun shadow factor) and the clustered point-light
// loop are filled by the color scaffold before the chunk runs, so per-pixel `lit()` gets sun shadows +
// local lights for free — a hand-written BRDF that reads `sunVisibility` does too. The full prelude (every
// local, every helper, the interpolator budget) is the render contract's *Surface authoring*.

// #doc:dev
// ## The backdrop seam
//
// `Backgrounds.register({ name, bindings?, preamble?, fs })` is the `Surfaces` analogue for the
// background: a view-ray → HDR color recipe sear draws on the un-rendered pixels (the infinite-skybox
// technique), the per-camera `Backdrop` component selecting one by name. The `fs` writes `col` from
// `dir`, the world-space view ray sear reconstructs per pixel. The engine names no sky concept — a plugin
// (`extras/sky`) owns its sky math behind this seam.

// #doc:dev
// ## Screen-space consumers
//
// Beyond surfaces, sear exposes two surfaces to a screen-space effect. The opt-in **prepass lanes** —
// `Tag` (a per-pixel id, published `view.tag`) and `Depth` (`view.depth`) — are gated per-camera by their
// marker component; a consumer reads the published `view.*` field for picking, outlines, or AO, owns the
// readback, and decodes the id itself (the engine knows nothing of picking). And the **relocatable
// shading chunks** — `LIGHT_EVAL_WGSL`, the point / sun shadow WGSL + the `pointAtlasView` /
// `shadowSampler` / `sunShadowView` resource getters — let a consumer evaluate the same lit, shadowed
// lights sear's color FS does, one source of truth. Sear still owns the maps and params (read-only, not a
// writable seam); the `fog` volumetric march is the worked case.

export type { Background } from "./forward";
export {
    Backgrounds,
    ColorSystem,
    casterWgsl,
    DEPTH_FORMAT,
    Depth,
    LIGHT_EVAL_WGSL,
    PrepassSystem,
    pointAtlasView,
    pointShadowWgsl,
    SAMPLE_SUN_SHADOW_WGSL,
    SHADOW_PARAMS_BYTES,
    SUN_SHADOW_STRUCT_WGSL,
    shadowSampler,
    sunShadowParams,
    sunShadowView,
    TAG_FORMAT,
    TAG_NONE,
    Tag,
} from "./forward";
