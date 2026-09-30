/// <reference types="@webgpu/types" />

// the extras barrel lists each module's author names explicitly; extension names stay on the module.

export { Fog, FogPlugin } from "./fog";
export {
    Arrow,
    arrow,
    box,
    Line,
    LinesPlugin,
    segment,
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
    Player,
    PlayerControlSystem,
    PlayerPlugin,
    type PointerLockStatus,
    pointerLockRefusal,
    pointerLockStatus,
} from "./player";
export {
    type BenchmarkAPI,
    type BenchmarkCompileStats,
    type BenchmarkCpuStats,
    type BenchmarkFrameStats,
    type BenchmarkGpuStats,
    type BenchmarkMeasurement,
    type BenchmarkMemoryStats,
    PhysicsProfilePlugin,
    Profile,
    ProfilePlugin,
    showProfiler,
    timingClock,
} from "./profile";
export {
    Sky,
    SkyPlugin,
} from "./sky";
export {
    Images,
    image,
    Sprite,
    SpriteBillboard,
    SpriteBlend,
    SpriteFill,
    SpritePlugin,
} from "./sprite";
export {
    Content,
    Fonts,
    font,
    Text,
    TextPlugin,
    text,
} from "./text";
