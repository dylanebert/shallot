/// <reference types="@webgpu/types" />

export { ShadowFilteringMethod, ShadowFilteringMethodMode } from "../../core/rendering";
export {
    pointAtlasView,
    pointShadowCasters,
    pointShadowRects,
    shadowSampler,
    sunShadowParams,
    sunShadowView,
} from "./atlas";
export { Clusters, clusterCell, LightClusters, LightCull } from "./cluster";
export type { Background, BackgroundHandle } from "./contract";
export { BackgroundContext, Backgrounds, backgroundLayout, registerBackground } from "./contract";
export { engineLayout, lit } from "./engine";
export { CameraBackground, StandardRenderer, StandardRenderingPlugin } from "./forward";
export type { IndirectLightSource } from "./indirect";
export {
    IndirectLightInput,
    indirectLight,
    indirectLightSlot,
    indirectLightSources,
    registerIndirectLightSource,
} from "./indirect";
export {
    DirectionalLightGpu,
    distanceAttenuation,
    Lighting,
    LightingGpu,
    MAX_DIRECTIONAL_LIGHTS,
    PointLightGpu,
    spotFactor,
} from "./lighting";
export {
    AlphaMode,
    AlphaModeCode,
    AlphaPipelineKey,
    MaterialPlugin,
    Materials,
    MaterialTypes,
    MeshMaterial,
    materialType,
    materialTypeId,
    StandardMaterial,
    VertexMaterialType,
} from "./material";
export { MeshInstanceInput, StandardMaterialInput } from "./material-data";
export type {
    MaterialBinding,
    MaterialFragmentFn,
    MaterialHandle,
    MaterialLayout,
    MaterialSource,
    MaterialType,
    MaterialVertexFn,
} from "./material-type";
export { MaterialAssets, materialLayout } from "./material-type";
export { MeshRenderPlugin } from "./mesh-render";
export { MeshPreprocessSystem, meshInstanceTable } from "./preprocess";
export { type ProducerTable, registerProducerTable } from "./producer-table";
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
