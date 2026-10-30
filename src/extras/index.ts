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
    type Gltf,
    type GltfPlacement,
    type GltfPrimitive,
    loadGltfScene,
    spawnGltfScene,
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
