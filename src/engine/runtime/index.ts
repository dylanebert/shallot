// GPU policy the device's consumers share. Quantization waits until a pass is five times off its bandwidth
// floor and a per-field audit is clean, because precision loss is permanent and a small bandwidth win isn't
// worth it. NaN propagates and is fixed at its source, since a clamp downstream hides the bug it came from.

export {
    type AdapterClass,
    type AdapterFacts,
    type AdapterInfoFacts,
    type AdapterVerdict,
    adapterIdentity,
    classifyAdapter,
} from "./adapter";
export {
    Compute,
    checkStorageBinding,
    checkTextureLimits,
    checkTgsl,
    deviceLost,
    type LazyAlloc,
    PIPELINE_COMPILE_MEASURE_PREFIX,
    precompile,
    precompileAll,
    precompileScope,
    requestGPU,
    resetCompute,
    type ShaderArtifact,
    stampAdapter,
    tgslCanary,
    UnsupportedError,
    validateGpu,
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
