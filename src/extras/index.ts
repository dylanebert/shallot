/// <reference types="@webgpu/types" />

// the extras barrel lists each module's author names explicitly; extension names stay on the module.

export { Fog, FogPlugin } from "./fog";
export {
    Arrow,
    drawArrow,
    drawLine,
    drawWireBox,
    Line,
    LinesPlugin,
} from "./lines";
export {
    Orbit,
    OrbitMode,
    OrbitOverlayPlugin,
    OrbitPick,
    OrbitPlugin,
} from "./orbit";
export {
    Outline,
    OutlinePlugin,
} from "./outline";
export {
    DrivePlayerSystem,
    LocalPlayer,
    Player,
    PlayerInput,
    PlayerPlugin,
    type PointerLockStatus,
    UpdatePlayerControlSystem,
} from "./player";
export {
    type BenchmarkAPI,
    type BenchmarkCompileStats,
    type BenchmarkCpuStats,
    type BenchmarkFrameStats,
    type BenchmarkGpuStats,
    type BenchmarkMeasurement,
    type BenchmarkMemoryStats,
    Profile,
    ProfilePlugin,
    showProfiler,
} from "./profile";
export {
    Sky,
    SkyPlugin,
} from "./sky";
export {
    Images,
    registerImage,
    Sprite,
    SpriteBillboard,
    SpriteBlend,
    SpriteFill,
    SpritePlugin,
} from "./sprite";
export {
    Content,
    Fonts,
    internText,
    registerFont,
    Text,
    TextPlugin,
} from "./text";
export { Vignette, VignettePlugin } from "./vignette";
