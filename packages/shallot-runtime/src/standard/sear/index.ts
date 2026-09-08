// Sear's author barrel — the game-author surface of the default renderer. The renderer itself is
// `forward.ts` (the GPU-driven forward pass); this file re-exports only what a scene author touches: the
// `Sear` camera marker + its opt-in prepass lanes (`Tag` / `Depth`), the `Material` / `Backdrop`
// components, the `Shadow` cast opt-in + its `SunShadows` / `PointShadows` config, and `SearPlugin`. The
// extension surface (surface codegen, the relocatable shading chunks, the backdrop registry, the ordering
// anchors) lives behind `sear/core`, drawn from the same `forward.ts` impl.
// re-export each name from its definition site (one hop, so the reference generator resolves the JSDoc):
// the renderer + most of its components live in forward.ts, `Tag` in codegen.ts (it's part of the
// COLOR_LANES table there), the shadow config + cast opt-in in shadows.ts.

export { Tag } from "./codegen";
export { Backdrop, Depth, Material, Sear, SearPlugin } from "./forward";
export { MAX_CASCADES, MAX_POINT_CASTERS, PointShadows, Shadow, SunShadows } from "./shadows";
