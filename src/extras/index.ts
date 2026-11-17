/// <reference types="@webgpu/types" />

// the extras barrel lists each module's author names explicitly; extension names stay on the module.

export { Bloom, BloomCompositeMode, BloomPlugin } from "./bloom";
export {
    ChromaticAberration,
    EffectStackPlugin,
    LensDistortion,
    Vignette,
    VignettePlugin,
} from "./effect-stack";
export {
    Fog,
    FogPlugin,
    fogReadBindGroup,
    fogReadLayout,
    NoFog,
    readFog,
    readFogInScatter,
} from "./fog";
export {
    deformationBindings,
    deformationVertex,
    type Gltf,
    type GltfPlacement,
    GltfPlugin,
    type GltfPrimitive,
    type GltfSceneInstance,
    loadGltfScene,
    MorphedMesh,
    poseGltfScene,
    SkinnedMaterial,
    SkinnedMesh,
    SkinPalette,
    spawnGltfScene,
    VertexDeformation,
} from "./gltf";
export {
    drawArrow,
    drawLine,
    drawWireBox,
    Gizmo,
    GizmoAsset,
    LinesPlugin,
    registerGizmo,
} from "./lines";
export {
    type AcceptedIntent,
    ClientOf,
    ClientPrediction,
    type Connection,
    client,
    contactOutputKey,
    FixedOutputs,
    type Game,
    IntentMessageKind,
    IntentReceiver,
    type IntentRecord,
    IntentSender,
    type LinkFaults,
    local,
    MultiplayerPlugin,
    PredictionMotion,
    PredictionPlugin,
    PredictionTarget,
    Replicate,
    type ReplicatedRows,
    ReplicationMessageKind,
    ReplicationReceiver,
    type ReplicationRules,
    type ReplicationSchema,
    ReplicationSender,
    ServerSession,
    seededLink,
    server,
} from "./multiplayer";
export {
    Orbit,
    OrbitMode,
    OrbitPlugin,
} from "./orbit";
export {
    Outline,
    OutlinePlugin,
} from "./outline";
export {
    DiagnosticsOverlayPlugin,
    ProfilePlugin,
    profilerBounds,
    registerDebugChord,
    showProfiler,
} from "./profile";
export {
    Sky,
    SkyPlugin,
    skyBackgroundHandle,
} from "./sky";
export {
    Sprite,
    SpriteAlphaMode,
    SpriteFill,
    SpritePlugin,
} from "./sprite";
export {
    Content,
    Fonts,
    internText,
    Justify,
    registerFont,
    Text,
    TextPlugin,
} from "./text";
