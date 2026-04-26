import { beginComputePass, type ComputeNode, type ExecutionContext } from "../compute";
import { bindView, type BufferView, type GBuf } from "../compute";
import { capacity } from "../../engine";
import { extractFrustumPlanes } from "../../engine";
import {
    type Batching,
    CULL_SHARED_WGSL,
    CULL_WORKGROUP_SIZE,
    packShapeAABBs,
    MAX_BATCH_SLOTS,
    MAX_SHAPES,
    SHAPE_AABB_STRIDE,
} from "../render/core";

const TOTAL_INDIRECT_SLOTS = MAX_BATCH_SLOTS * 2;

export const ZERO_INSTANCE_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> indirect: array<u32>;

const TOTAL_SLOTS: u32 = ${TOTAL_INDIRECT_SLOTS}u;
const INDIRECT_STRIDE: u32 = 5u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= TOTAL_SLOTS) { return; }
    indirect[gid.x * INDIRECT_STRIDE + 1u] = 0u;
}
`;

export function makeFrustumCullShader(): string {
    return /* wgsl */ `
${CULL_SHARED_WGSL}

@compute @workgroup_size(${CULL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.entityCount) { return; }

    let sphere = computeWorldSphere(gid.x);

    if (!frustumTest(sphere.center, sphere.radius)) { return; }

    emitVisible(sphere);
}
`;
}

export interface CullPipelineResult {
    nodes: ComputeNode[];
}

export function createCullPipeline(render: {
    matrices: GBuf;
    sizes: BufferView;
    batching: Batching;
    viewProj: Float32Array;
}): CullPipelineResult {
    let frustumPipeline: GPUComputePipeline | null = null;
    let zeroInstancePipeline: GPUComputePipeline | null = null;
    let zeroInstanceBG: GPUBindGroup | null = null;
    let paramsBuffer: GPUBuffer | null = null;
    let shapeAABBsBuffer: GPUBuffer | null = null;
    let inputBindGroup: GPUBindGroup | null = null;
    let outputBindGroup: GPUBindGroup | null = null;
    let cachedCapacity = capacity();

    const paramsData = new ArrayBuffer(112);
    const paramsF32 = new Float32Array(paramsData);
    const paramsU32 = new Uint32Array(paramsData);
    const planesScratch = new Float32Array(24);
    const shapeAABBGPU = new Float32Array(MAX_SHAPES * SHAPE_AABB_STRIDE);

    const frustumCullNode: ComputeNode = {
        name: "frustum-cull",
        inputs: ["shadow-atlas", "point-shadow-atlas"],
        outputs: ["culled"],

        async prepare(device: GPUDevice) {
            const shaderCode = makeFrustumCullShader();
            const module = device.createShaderModule({ code: shaderCode });
            const bgl0 = device.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "read-only-storage" },
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "read-only-storage" },
                    },
                    {
                        binding: 3,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "read-only-storage" },
                    },
                    {
                        binding: 4,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "read-only-storage" },
                    },
                ],
            });
            const bgl1 = device.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                ],
            });

            frustumPipeline = await device.createComputePipelineAsync({
                label: "frustum-cull",
                layout: device.createPipelineLayout({ bindGroupLayouts: [bgl0, bgl1] }),
                compute: { module, entryPoint: "main" },
            });

            paramsBuffer = device.createBuffer({
                label: "frustum-cull-params",
                size: 112,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

            shapeAABBsBuffer = device.createBuffer({
                label: "frustum-shape-aabbs",
                size: MAX_SHAPES * SHAPE_AABB_STRIDE * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });

            const zeroInstModule = device.createShaderModule({ code: ZERO_INSTANCE_SHADER });
            zeroInstancePipeline = await device.createComputePipelineAsync({
                label: "zero-instance",
                layout: "auto",
                compute: { module: zeroInstModule, entryPoint: "main" },
            });

            zeroInstanceBG = device.createBindGroup({
                layout: zeroInstancePipeline.getBindGroupLayout(0),
                entries: [{ binding: 0, resource: { buffer: render.batching.indirect } }],
            });

            inputBindGroup = device.createBindGroup({
                layout: bgl0,
                entries: [
                    { binding: 0, resource: { buffer: paramsBuffer } },
                    { binding: 1, resource: { buffer: render.matrices.buffer } },
                    bindView(2, render.sizes),
                    { binding: 3, resource: { buffer: shapeAABBsBuffer } },
                    { binding: 4, resource: { buffer: render.batching.cullEntities.buffer } },
                ],
            });

            outputBindGroup = device.createBindGroup({
                layout: bgl1,
                entries: [
                    { binding: 0, resource: { buffer: render.batching.indirect } },
                    { binding: 1, resource: { buffer: render.batching.entityIds.buffer } },
                ],
            });
        },

        execute(ctx: ExecutionContext) {
            if (
                !frustumPipeline ||
                !paramsBuffer ||
                !shapeAABBsBuffer ||
                !zeroInstancePipeline ||
                !zeroInstanceBG
            )
                return;

            if (capacity() !== cachedCapacity) {
                cachedCapacity = capacity();
                inputBindGroup = null;
                outputBindGroup = null;
            }
            if (!inputBindGroup || !outputBindGroup) {
                inputBindGroup = ctx.device.createBindGroup({
                    layout: frustumPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: paramsBuffer } },
                        { binding: 1, resource: { buffer: render.matrices.buffer } },
                        bindView(2, render.sizes),
                        { binding: 3, resource: { buffer: shapeAABBsBuffer } },
                        { binding: 4, resource: { buffer: render.batching.cullEntities.buffer } },
                    ],
                });
                outputBindGroup = ctx.device.createBindGroup({
                    layout: frustumPipeline.getBindGroupLayout(1),
                    entries: [
                        { binding: 0, resource: { buffer: render.batching.indirect } },
                        { binding: 1, resource: { buffer: render.batching.entityIds.buffer } },
                    ],
                });
            }

            const batching = render.batching;
            const entityCount = batching.cullEntityCount;
            if (entityCount === 0) return;

            packShapeAABBs(batching.shapeAABBs, shapeAABBGPU);
            ctx.device.queue.writeBuffer(shapeAABBsBuffer, 0, shapeAABBGPU);

            extractFrustumPlanes(render.viewProj, planesScratch);
            paramsF32.set(planesScratch);
            paramsU32[24] = entityCount;
            ctx.device.queue.writeBuffer(paramsBuffer, 0, paramsData);

            const zeroPass = beginComputePass(ctx.encoder);
            zeroPass.setPipeline(zeroInstancePipeline);
            zeroPass.setBindGroup(0, zeroInstanceBG);
            zeroPass.dispatchWorkgroups(Math.ceil(TOTAL_INDIRECT_SLOTS / 64));
            zeroPass.end();

            const pass = beginComputePass(ctx.encoder, ctx.timestampWrites?.("frustum-cull"));
            pass.setPipeline(frustumPipeline);
            pass.setBindGroup(0, inputBindGroup);
            pass.setBindGroup(1, outputBindGroup);
            pass.dispatchWorkgroups(Math.ceil(entityCount / CULL_WORKGROUP_SIZE));
            pass.end();
        },
    };

    return { nodes: [frustumCullNode] };
}
