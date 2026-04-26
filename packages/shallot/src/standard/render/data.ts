import { capacity } from "../../engine";
import { beginComputePass, type ComputeNode, type ExecutionContext } from "../compute";
import { bindView, type BufferView, type GBuf } from "../compute";
import { DATA_STRUCT_WGSL } from "./surface/structs";

const WORKGROUP_SIZE = 64;

const shader = /* wgsl */ `
${DATA_STRUCT_WGSL}

@group(0) @binding(0) var<storage, read> colors: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> pbr: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> emission: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> surfaces: array<u32>;
@group(0) @binding(4) var<storage, read> entityCount: array<u32>;
@group(0) @binding(5) var<storage, read_write> data: array<Data>;
@group(0) @binding(6) var<storage, read> sizes: array<vec4<f32>>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let eid = gid.x;
    let count = entityCount[0];
    if (eid >= count) { return; }

    let s = sizes[eid];
    var d: Data;
    d.baseColor = colors[eid];
    d.pbr = pbr[eid];
    d.emission = emission[eid];
    d.flags = surfaces[eid];
    d.sizeX = s.x;
    d.sizeY = s.y;
    d.sizeZ = s.z;
    data[eid] = d;
}
`;

export function createDataNode(render: {
    colors: BufferView;
    pbr: BufferView;
    emission: BufferView;
    surfaces: BufferView;
    entityCountBuffer: BufferView;
    data: GBuf;
    sizes: BufferView;
    entityCount: number;
}): ComputeNode {
    let pipeline: GPUComputePipeline | null = null;
    let bindGroup: GPUBindGroup | null = null;
    let cachedCapacity = capacity();

    function buildBindGroup(device: GPUDevice): GPUBindGroup {
        return device.createBindGroup({
            layout: pipeline!.getBindGroupLayout(0),
            entries: [
                bindView(0, render.colors),
                bindView(1, render.pbr),
                bindView(2, render.emission),
                bindView(3, render.surfaces),
                bindView(4, render.entityCountBuffer),
                { binding: 5, resource: { buffer: render.data.buffer } },
                bindView(6, render.sizes),
            ],
        });
    }

    return {
        name: "data",
        scope: "frame",
        inputs: [],
        outputs: ["data"],

        async prepare(device: GPUDevice) {
            const module = device.createShaderModule({ code: shader });

            pipeline = await device.createComputePipelineAsync({
                label: "upload-data",
                layout: "auto",
                compute: { module, entryPoint: "main" },
            });

            bindGroup = buildBindGroup(device);
        },

        execute(ctx: ExecutionContext) {
            if (!pipeline) return;

            if (capacity() !== cachedCapacity) {
                cachedCapacity = capacity();
                bindGroup = null;
            }
            if (!bindGroup) {
                bindGroup = buildBindGroup(ctx.device);
            }

            const workgroups = Math.ceil(render.entityCount / WORKGROUP_SIZE);

            const pass = beginComputePass(ctx.encoder, ctx.timestampWrites?.("data-upload"));
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(workgroups);
            pass.end();
        },
    };
}
