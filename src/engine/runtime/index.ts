/// <reference types="@webgpu/types" />

export * from "./adapter";
export {
    beginComputePass,
    beginRenderPass,
    continueSubmissionFence,
    createFrameEncoder,
    dispatchWorkgroups,
    dispatchWorkgroups2,
    dispatchWorkgroups3,
    dispatchWorkgroupsIndirect,
    draw,
    drawIndexedIndirect,
    endComputePass,
    endRenderPass,
    executeRenderBundles,
    finishFrameEncoder,
    setComputeBindGroup,
    setComputePipeline,
    setIndexBuffer,
    setRenderBindGroup,
    setRenderPipeline,
    setViewport,
    submissionFence,
    submitFrame,
} from "./allocation-site";
export {
    adapterNotice,
    checkStorageBinding,
    checkTextureLimits,
    checkTgsl,
    deviceLost,
    type GpuRequirements,
    type NamedGpuRequirements,
    PIPELINE_COMPILE_MEASURE_PREFIX,
    PLUGIN_HOOK_MEASURE_PREFIX,
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
    withPipelineOwner,
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

export { resizeViewport, type Viewport, Viewports } from "./viewport";
