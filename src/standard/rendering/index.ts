/// <reference types="@webgpu/types" />

export { pointAtlasView, shadowSampler, sunShadowParams, sunShadowView } from "./atlas";
export { Clusters, clusterCell, LightClusters, LightCull } from "./cluster";
export type { Background } from "./contract";
export { BackgroundContext, Backgrounds, backgroundLayout, registerBackground } from "./contract";
export { engineLayout, lit } from "./engine";
export { CameraBackground, StandardRenderer, StandardRenderingPlugin } from "./forward";
export {
    distanceAttenuation,
    Lighting,
    LightingGpu,
    PointLightGpu,
    spotFactor,
} from "./lighting";
export {
    MaterialPlugin,
    Materials,
    MaterialTypes,
    MeshMaterial,
    materialAssets,
    materialType,
    materialTypeId,
    materialTypes,
    StandardMaterial,
    StandardMaterialType,
} from "./material";
export { MeshInstanceInput, StandardMaterialInput } from "./material-data";
export type {
    MaterialBinding,
    MaterialFragmentFn,
    MaterialHandle,
    MaterialLayout,
    MaterialType,
    MaterialVertexFn,
} from "./material-type";
export {
    MaterialAssets,
    MaterialVertexInput,
    materialFragmentContext,
    materialLayout,
    materialVertexOutput,
} from "./material-type";
export { MeshRenderPlugin } from "./mesh-render";
export { MeshPreprocessSystem } from "./preprocess";
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
    DirectionalLightShadowMap,
    MAX_CASCADES,
    MAX_POINT_CASTERS,
    PointShadows,
    pointComboEids,
} from "./shadows";
