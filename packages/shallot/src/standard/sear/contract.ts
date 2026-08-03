// Compatibility boundary for Sear internals and existing public imports. The concrete contract and
// mutable registries are render-owned, so render/core can expose their exact types without importing Sear.
export type {
    Background,
    BgFn,
    BgLayout,
    Binding,
    FsFn,
    Specialize,
    Surface,
    SurfaceLayout,
    VsFn,
} from "../render/core";
export {
    Backgrounds,
    BgCtx,
    backgroundLayout,
    fsCtxSchema,
    registerBackground,
    registerSurface,
    SURFACE_GROUP,
    Surfaces,
    surfaceLayout,
    VsIn,
    vsPatchSchema,
} from "../render/core";
