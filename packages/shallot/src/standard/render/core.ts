// extension API for renderer + producer authors: the contract registries, the
// per-frame uniform singletons + their WGSL structs, the vertex-pull contract,
// canvas binding, and the frame-loop ordering anchor. The typical-user surface
// (components, plugin, public types, mesh()) lives in the index barrel. `VIEW_STRIDE`
// + `MAX_VIEWS` size a per-view uniform a consumer packs slot-major (glaze's postfx
// config); the buffer sizes and the cull-volume packer stay internal — a consumer reads
// the packed `Render.cullVolumes` buffer, never re-packs it. A producer that runs its own
// cull (Part's pack) reads the per-slot layout constants below to index + dispatch on the tag.

// #doc:dev
// ## Custom producers and renderers
//
// `render/` is renderer-agnostic: it defines the `Surfaces` / `Meshes` / `Draws` registries and the
// per-frame uniforms, but iterates none of them. A **producer** registers a surface, a geometry slice,
// and a draw record, publishing its per-instance buffers by name; a **renderer** consumes those draws. Because
// the contract sits between them, a custom producer (terrain, particles) and a custom renderer are peers of
// the built-in `Part` and sear — build either against the registries below, no engine change needed.

// #doc:dev
// ## Ordering anchors
//
// Systems order against no-op anchor systems, not registration order. A producer whose compute writes
// geometry a renderer reads for position pins `before: [PrepassSystem]`, so the prepass, shadow, and color
// passes all read the same frame's data; a screen-space effect slots around `OverlaySystem` (scene-space
// transforms `before:`, screen-space overlays `after:`). Never mark a renderer `last: true` — that slot
// belongs to the frame submit.

export { computeViewProj } from "./camera";
export type { ClusterView } from "./cluster";
export {
    CLUSTER_COUNT,
    CLUSTER_X,
    CLUSTER_Y,
    CLUSTER_Z,
    Clusters,
    clusterAabb,
    clusterCoord,
    clusterIndex,
    clusterView,
    LIGHT_POOL,
    LightCull,
    lightClusters,
    sliceDepth,
    zSlice,
} from "./cluster";
export { FRAME_STRUCT_WGSL, Frame } from "./frame";
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
    LIGHTING_STRUCT_WGSL,
    LIGHTING_UNIFORM_SIZE,
    Lighting,
    MAX_POINT_LIGHTS,
    POINT_LIGHTS_STRUCT_WGSL,
    spotParams,
} from "./lighting";
export type { Mesh, QuantStreams } from "./mesh";
export {
    Meshes,
    meshBounds,
    packMeshes,
    quantizeMeshes,
    VERTEX_FLOATS,
    VERTEX_STRIDE,
} from "./mesh";
export type { Binding, Draw, Surface } from "./registry";
export { Draws, Surfaces } from "./registry";
export { Render } from "./render";
export type { View } from "./view";
export {
    attachCanvas,
    attachView,
    detachCanvas,
    LINEAR_TO_SRGB_WGSL,
    MAX_SLOTS,
    MAX_VIEWS,
    sceneTransform,
    VIEW_BYTES,
    VIEW_STRIDE,
    VIEW_STRUCT_WGSL,
    Views,
} from "./view";
