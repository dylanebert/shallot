/// <reference types="@webgpu/types" />

export * from "./adapter";
export {
    Compute,
    checkStorageBinding,
    checkTextureLimits,
    checkTgsl,
    currentWorld,
    deviceLost,
    type LazyAlloc,
    PIPELINE_COMPILE_MEASURE_PREFIX,
    precompile,
    precompileAll,
    precompileScope,
    requestGPU,
    type ShaderArtifact,
    stampAdapter,
    tgslCanary,
    UnsupportedError,
    validateGpu,
    withCompute,
    withComputeAsync,
    worldResource,
} from "./gpu";
export { drainLog, type GpuLog } from "./log";
export { now, Runtime, readBinary, readFile, requestFrame } from "./platform";
export {
    type BufferProbe,
    type BufferProbeOptions,
    type ProbeEncode,
    probeBuffer,
    probeTexture,
    type TextureProbe,
    type TextureProbeOptions,
} from "./probe";
