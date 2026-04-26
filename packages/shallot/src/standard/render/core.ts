export {
    SURFACE_DATA_STRUCT_WGSL,
    SCENE_STRUCT_WGSL,
    SKY_STRUCT_WGSL,
    DATA_STRUCT_WGSL,
    OKLAB_WGSL,
    SPACE_CONVERT_WGSL,
    POINT_LIGHT_STRUCT_WGSL,
    POINT_SHADOW_STRUCT_WGSL,
    SHADOW_STRUCT_WGSL,
    VERTEX_PULL_WGSL,
    WGSL_STRUCTS,
} from "./surface/structs";

export {
    NOISE_WGSL,
    SKY_WGSL,
    HAZE_WGSL,
    SKY_DIR_WGSL,
    SPECULAR_WGSL,
    POINT_LIGHT_EVAL_WGSL,
    SHADOW_SAMPLE_WGSL,
    POINT_SHADOW_SAMPLE_WGSL,
    SURFACE_HELPERS_WGSL,
    compileVertexBody,
    WGSL_LIGHTING_CALC,
    REFLECTION_WGSL,
} from "./surface/shaders";

export {
    compileSurfaceBlock,
    compileVertexVariant,
    compileVertexDispatch,
    OPACITY_GUARD_WGSL,
} from "./surface/compile";
export type { PipelineVariantConfig } from "./surface/compile";
export { hasProperties, instanceStructWGSL, instanceBindingWGSL } from "./surface";
export type { SurfaceData } from "./surface";

export { drawBatches, CULL_ENTITY_STRIDE, INDIRECT_STRIDE } from "./batch";
export type { Batching } from "./batch";

export { CULL_SHARED_WGSL, CULL_WORKGROUP_SIZE, SHAPE_AABB_STRIDE, packShapeAABBs } from "./cull";

export type { SharedPassContext, OverlayDraw } from "./pass";

export { COLOR_FORMAT, Z_FORMAT, SCENE_UNIFORM_SIZE } from "./scene";

export { projectActiveSun, projectSunToScreen } from "./camera";

export {
    packLightUniforms,
    packPointLights,
    POINT_LIGHT_BUFFER_SIZE,
    POINT_LIGHT_STRIDE,
    MAX_RASTER_POINT_LIGHTS,
} from "./light";

export type { ShapeAtlas, MeshData, AABB, DynamicInfo } from "./mesh";
export {
    MAX_BATCH_SLOTS,
    MAX_SHAPES,
    computeShapeAABB,
    isDynamic,
    dynamicInfo,
    getMeshId,
    getMesh,
    PartSizes,
    PartShapes,
    MeshGeometryData,
} from "./mesh";
