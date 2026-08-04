// extension API for renderer + producer authors: the contract registries, the
// per-frame uniform singletons + their WGSL structs, the vertex-pull contract,
// canvas binding, and the frame-loop ordering anchor. The typical-user surface
// (components, plugin, public types, mesh()) lives in the index barrel. `VIEW_STRIDE`
// + `MAX_VIEWS` size a per-view uniform a consumer packs slot-major (glaze's postfx
// config); the buffer sizes and the cull-volume packer stay internal — a consumer reads
// the packed `Render.cullVolumes` buffer, never re-packs it. A producer that runs its own
// cull (Part's pack) reads the per-slot layout constants below to index + dispatch on the tag.

export { computeViewProj } from "./camera";
export type { ClusterView } from "./cluster";
export {
    CLUSTER_COUNT,
    CLUSTER_X,
    CLUSTER_Y,
    CLUSTER_Z,
    Clusters,
    clusterAabb,
    clusterCell,
    clusterCoord,
    clusterIndex,
    clusterView,
    LIGHT_POOL,
    LightCull,
    lightClusters,
    sliceDepth,
    zSlice,
} from "./cluster";
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
    SURFACE_GROUP,
    Surfaces,
    surfaceLayout,
    VsIn,
    vsPatchSchema,
} from "./contract";
export { Frame, FrameGpu, frameWgsl } from "./frame";
export { CULL_FRUSTUM, CULL_VOLUME_FLOATS, FRUSTUM_FLOATS, frustumPlanes } from "./frustum";
// the shared image→`texture_2d_array` upload path — the producer substrate glTF baseColor + the sprite atlas
// both sample, inward of both extras so neither reaches sideways into the other
export {
    allocArray,
    arrayFromBitmaps,
    commonSize,
    imageArray,
    mipLevels,
    uploadLayer,
} from "./image";
export { BeginFrameSystem, OverlaySystem } from "./index";

export {
    distanceAttenuation,
    LIGHTING_UNIFORM_SIZE,
    Lighting,
    LightingGpu,
    lightingWgsl,
    MAX_POINT_LIGHTS,
    PointLightGpu,
    PointLights,
    pointLightsWgsl,
    spotFactor,
    spotParams,
} from "./lighting";
export type { Mesh, MeshBinding, MeshIndex, MeshStorage, QuantStreams } from "./mesh";
export {
    Meshes,
    meshBounds,
    packMeshes,
    quantizeMeshes,
    VERTEX_FLOATS,
    VERTEX_STRIDE,
} from "./mesh";
export type { Draw, DrawIndirectBuffer } from "./registry";
export { DrawIndexedIndirect, Draws } from "./registry";
export { Render } from "./render";
export {
    attachCanvas,
    attachView,
    detachCanvas,
    linearToSrgb,
    linearToSrgbWgsl,
    MAX_SLOTS,
    MAX_VIEWS,
    sceneTransform,
    VIEW_BYTES,
    VIEW_STRIDE,
    View,
    Views,
    viewWgsl,
} from "./view";
