/// <reference types="@webgpu/types" />

export { pointAtlasView, shadowSampler, sunShadowParams, sunShadowView } from "./atlas";
export { Clusters, clusterCell, LightClusters, LightCull } from "./cluster";
export { lightEvalWgsl } from "./codegen";
export type { Background, Surface } from "./contract";
export {
    BackgroundContext,
    Backgrounds,
    backgroundLayout,
    fsCtxSchema,
    MeshInstanceInput,
    registerBackground,
    registerSurface,
    Surfaces,
    surfaceLayout,
    VsIn,
    vsPatchSchema,
} from "./contract";
export { engineLayout, lit } from "./engine";
export { CameraBackground, StandardRenderer, StandardRenderingPlugin } from "./forward";
export {
    distanceAttenuation,
    Lighting,
    LightingGpu,
    PointLightGpu,
    pointLightsWgsl,
    spotFactor,
} from "./lighting";
export { Materials, MeshMaterial, StandardMaterial } from "./material";
export { MeshRenderPlugin } from "./mesh-render";
export type { Draw } from "./registry";
export { DrawIndexedIndirect, Draws } from "./registry";
export {
    pointCastersSchema,
    pointShadowRef,
    SunShadow,
    sampleSunShadow,
    tileRectsSchema,
} from "./shade";
export {
    cascadeComboEids,
    MAX_CASCADES,
    MAX_POINT_CASTERS,
    PointShadows,
    pointCasters,
    pointComboEids,
    SunShadows,
} from "./shadows";
