// Each function isolates one WebGPU API expression for allocation-stack attribution.
export function createFrameEncoder(
    device: GPUDevice,
    descriptor: GPUCommandEncoderDescriptor,
): GPUCommandEncoder {
    return device.createCommandEncoder(descriptor);
}

export function finishFrameEncoder(encoder: GPUCommandEncoder): GPUCommandBuffer {
    return encoder.finish();
}

export function submitFrame(queue: GPUQueue, buffers: readonly GPUCommandBuffer[]): void {
    queue.submit(buffers);
}

export function submissionFence(queue: GPUQueue): Promise<void> {
    return queue.onSubmittedWorkDone();
}

export function continueSubmissionFence(
    fence: Promise<void>,
    complete: () => void,
    settle: () => void,
): void {
    fence.then(complete, settle);
}

export function beginRenderPass(encoder: GPUCommandEncoder, descriptor: GPURenderPassDescriptor) {
    return encoder.beginRenderPass(descriptor);
}

export function beginComputePass(encoder: GPUCommandEncoder, descriptor: GPUComputePassDescriptor) {
    return encoder.beginComputePass(descriptor);
}

export function setComputePipeline(
    pass: GPUComputePassEncoder,
    pipeline: GPUComputePipeline,
): void {
    pass.setPipeline(pipeline);
}

export function setComputeBindGroup(
    pass: GPUComputePassEncoder,
    index: number,
    group: GPUBindGroup,
): void {
    pass.setBindGroup(index, group);
}

export function executeRenderBundles(
    pass: GPURenderPassEncoder,
    bundles: readonly GPURenderBundle[],
): void {
    pass.executeBundles(bundles);
}

export function dispatchWorkgroups(pass: GPUComputePassEncoder, x: number): void {
    pass.dispatchWorkgroups(x);
}

export function dispatchWorkgroups2(pass: GPUComputePassEncoder, x: number, y: number): void {
    pass.dispatchWorkgroups(x, y);
}

export function setIndexBuffer(
    pass: GPURenderPassEncoder,
    buffer: GPUBuffer,
    format: GPUIndexFormat,
): void {
    pass.setIndexBuffer(buffer, format);
}

export function setViewport(
    pass: GPURenderPassEncoder,
    x: number,
    y: number,
    width: number,
    height: number,
    minDepth: number,
    maxDepth: number,
): void {
    pass.setViewport(x, y, width, height, minDepth, maxDepth);
}

export function drawIndexedIndirect(
    pass: GPURenderPassEncoder,
    buffer: GPUBuffer,
    offset: number,
): void {
    pass.drawIndexedIndirect(buffer, offset);
}

export function setRenderPipeline(pass: GPURenderPassEncoder, pipeline: GPURenderPipeline): void {
    pass.setPipeline(pipeline);
}

export function setRenderBindGroup(
    pass: GPURenderPassEncoder,
    index: number,
    group: GPUBindGroup,
): void {
    pass.setBindGroup(index, group);
}

export function draw(pass: GPURenderPassEncoder, vertexCount: number): void {
    pass.draw(vertexCount);
}

export function endRenderPass(pass: GPURenderPassEncoder): void {
    pass.end();
}

export function endComputePass(pass: GPUComputePassEncoder): void {
    pass.end();
}

export const gpuAllocationExpressions = [
    {
        fn: createFrameEncoder,
        site: "src/engine/runtime/allocation-site.ts:6:12",
        expression: "device.createCommandEncoder(descriptor)",
    },
    {
        fn: finishFrameEncoder,
        site: "src/engine/runtime/allocation-site.ts:10:12",
        expression: "encoder.finish()",
    },
    {
        fn: submitFrame,
        site: "src/engine/runtime/allocation-site.ts:14:5",
        expression: "queue.submit(buffers)",
    },
    {
        fn: submissionFence,
        site: "src/engine/runtime/allocation-site.ts:18:12",
        expression: "queue.onSubmittedWorkDone()",
    },
    {
        fn: continueSubmissionFence,
        site: "src/engine/runtime/allocation-site.ts:26:5",
        expression: "fence.then(complete, settle)",
    },
    {
        fn: beginRenderPass,
        site: "src/engine/runtime/allocation-site.ts:30:12",
        expression: "encoder.beginRenderPass(descriptor)",
    },
    {
        fn: beginComputePass,
        site: "src/engine/runtime/allocation-site.ts:34:12",
        expression: "encoder.beginComputePass(descriptor)",
    },
    {
        fn: endRenderPass,
        site: "src/engine/runtime/allocation-site.ts:112:5",
        expression: "pass.end()",
    },
    {
        fn: endComputePass,
        site: "src/engine/runtime/allocation-site.ts:116:5",
        expression: "pass.end()",
    },
] as const;
