/// <reference types="@webgpu/types" />

export * from "./adapter";
export {
    checkStorageBinding,
    checkTextureLimits,
    checkTgsl,
    deviceLost,
    type LazyAlloc,
    PIPELINE_COMPILE_MEASURE_PREFIX,
    precompile,
    precompileAll,
    precompileScope,
    rawDevice,
    requestGPU,
    type ShaderArtifact,
    stampAdapter,
    tgslCanary,
    UnsupportedError,
    validateGpu,
    type WorldGpu,
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
export { ReadbackPool, type ReadbackStamp } from "./readback";
