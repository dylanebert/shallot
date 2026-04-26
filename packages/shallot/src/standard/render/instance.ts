import { capacity, write } from "../../engine";
import { beginComputePass, type ComputeNode, type ExecutionContext } from "../compute";
import { bindView, type BufferView, type GBuf, gbuf } from "../compute";
import {
    instanceEntries,
    propertyCount,
    instanceStride,
    instancePackingShader,
    hasProperties,
} from "./surface";

const WORKGROUP_SIZE = 64;

export function createInstanceNode(render: {
    instanceDataBuffer: GBuf | null;
    entityCountBuffer: BufferView;
    entityCount: number;
}): ComputeNode {
    let sourceBuffer: GBuf | null = null;
    let pipeline: GPUComputePipeline | null = null;
    let bindGroup: GPUBindGroup | null = null;
    let cachedCapacity = capacity();

    function rebuildBindGroup(device: GPUDevice): GPUBindGroup {
        return device.createBindGroup({
            layout: pipeline!.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: sourceBuffer!.buffer } },
                { binding: 1, resource: { buffer: render.instanceDataBuffer!.buffer } },
                bindView(2, render.entityCountBuffer),
            ],
        });
    }

    return {
        name: "instance-data",
        scope: "frame",
        inputs: [],
        outputs: ["instance-data"],

        async prepare(device: GPUDevice) {
            if (!hasProperties()) return;

            const StorageDst = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

            if (!sourceBuffer) {
                sourceBuffer = gbuf(
                    device,
                    "instance-data-source",
                    StorageDst,
                    (c) => c * propertyCount() * 4,
                );
                render.instanceDataBuffer = gbuf(
                    device,
                    "instance-data",
                    StorageDst,
                    (c) => c * instanceStride(),
                );
            }

            const shader = instancePackingShader();
            if (!shader) return;

            const module = device.createShaderModule({ code: shader });
            pipeline = await device.createComputePipelineAsync({
                label: "instance",
                layout: "auto",
                compute: { module, entryPoint: "main" },
            });

            bindGroup = rebuildBindGroup(device);
        },

        execute(ctx: ExecutionContext) {
            if (!pipeline || !sourceBuffer || !render.instanceDataBuffer) return;

            if (capacity() !== cachedCapacity) {
                cachedCapacity = capacity();
                bindGroup = null;
            }
            if (!bindGroup) {
                bindGroup = rebuildBindGroup(ctx.device);
            }

            const entityCount = render.entityCount;
            const entries = instanceEntries();

            for (let f = 0; f < entries.length; f++) {
                write(
                    ctx.device.queue,
                    sourceBuffer.buffer,
                    f * entityCount * 4,
                    entries[f].data,
                    entityCount,
                );
            }

            const workgroups = Math.ceil(entityCount / WORKGROUP_SIZE);
            const pass = beginComputePass(ctx.encoder, ctx.timestampWrites?.("instance-upload"));
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(workgroups);
            pass.end();
        },
    };
}
