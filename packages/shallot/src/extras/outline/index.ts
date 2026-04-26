import { resource, unpackColor, type Plugin, type State } from "../../engine";
import {
    beginComputePass,
    Compute,
    ComputePlugin,
    type ComputeNode,
    type ExecutionContext,
} from "../../standard/compute";
import { RenderPlugin } from "../../standard/render";
import { COLOR_FORMAT } from "../../standard/render/core";

export interface Outline {
    getEntities: () => number[];
    color: number;
    thickness: number;
}

export const Outline = resource<Outline>("outline");

const OUTLINE_SHADER = /* wgsl */ `
struct Uniforms {
    outlineColor: vec3f,
    thickness: f32,
    selectedCount: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var entityIdTexture: texture_2d<u32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;
@group(0) @binding(3) var<storage, read> selectedEntities: array<u32>;
@group(0) @binding(4) var output: texture_storage_2d<${COLOR_FORMAT}, write>;

fn isSelected(eid: u32) -> bool {
    for (var i = 0u; i < uniforms.selectedCount; i++) {
        if (selectedEntities[i] == eid) { return true; }
    }
    return false;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(output));
    let coord = vec2i(gid.xy);
    if (coord.x >= dims.x || coord.y >= dims.y) { return; }

    let centerEid = textureLoad(entityIdTexture, coord, 0).r;
    let pxThickness = max(uniforms.thickness * f32(dims.y), 1.0);
    let radius = i32(ceil(pxThickness));

    let isCenterSelected = isSelected(centerEid);
    var minDistSq = pxThickness * pxThickness + 1.0;

    if (isCenterSelected) {
        for (var dy = -radius; dy <= radius; dy++) {
            for (var dx = -radius; dx <= radius; dx++) {
                let nc = clamp(coord + vec2i(dx, dy), vec2i(0), dims - 1);
                let nid = textureLoad(entityIdTexture, nc, 0).r;
                if (nid != centerEid) {
                    minDistSq = min(minDistSq, f32(dx * dx + dy * dy));
                }
            }
        }
    } else {
        for (var dy = -radius; dy <= radius; dy++) {
            for (var dx = -radius; dx <= radius; dx++) {
                let nc = clamp(coord + vec2i(dx, dy), vec2i(0), dims - 1);
                let nid = textureLoad(entityIdTexture, nc, 0).r;
                if (isSelected(nid)) {
                    minDistSq = min(minDistSq, f32(dx * dx + dy * dy));
                }
            }
        }
    }

    var color: vec3f;
    if (minDistSq <= pxThickness * pxThickness) {
        color = uniforms.outlineColor;
    } else {
        color = textureLoad(inputTexture, coord, 0).rgb;
    }

    textureStore(output, coord, vec4f(color, 1.0));
}
`;

const MAX_SELECTED = 256;

function createOutlineNode(state: State): ComputeNode {
    let pipeline: GPUComputePipeline | null = null;
    let layout: GPUBindGroupLayout | null = null;
    let uniformBuffer: GPUBuffer | null = null;
    let selectedBuffer: GPUBuffer | null = null;
    let intermediate: GPUTexture | null = null;
    let intermediateView: GPUTextureView | null = null;
    let bindGroup: GPUBindGroup | null = null;
    let cachedInputView: GPUTextureView | null = null;
    let cachedEidView: GPUTextureView | null = null;
    let cachedWidth = 0;
    let cachedHeight = 0;
    const uniformData = new ArrayBuffer(32);
    const uniformFloats = new Float32Array(uniformData);
    const uniformUints = new Uint32Array(uniformData);
    const selectedScratch = new Uint32Array(MAX_SELECTED);

    return {
        name: "outline",
        inputs: ["color", "eid"],
        outputs: ["color"],

        async prepare(device: GPUDevice) {
            layout = device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        texture: { sampleType: "float" },
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        texture: { sampleType: "uint" },
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "uniform" },
                    },
                    {
                        binding: 3,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "read-only-storage" },
                    },
                    {
                        binding: 4,
                        visibility: GPUShaderStage.COMPUTE,
                        storageTexture: { format: COLOR_FORMAT, access: "write-only" },
                    },
                ],
            });
            const module = device.createShaderModule({ code: OUTLINE_SHADER });
            pipeline = await device.createComputePipelineAsync({
                label: "outline",
                layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
                compute: { module, entryPoint: "main" },
            });
            uniformBuffer = device.createBuffer({
                label: "outline-uniforms",
                size: 32,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            selectedBuffer = device.createBuffer({
                label: "outline-selected",
                size: MAX_SELECTED * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
        },

        execute(ctx: ExecutionContext) {
            if (!pipeline || !layout || !uniformBuffer || !selectedBuffer) return;

            const outline = Outline.from(state);
            if (!outline) return;
            const entities = outline.getEntities();
            if (entities.length === 0) return;

            const inputView = ctx.getTextureView("color");
            const eidView = ctx.getTextureView("eid");
            const colorTex = ctx.getTexture("color");
            if (!inputView || !eidView || !colorTex) return;

            const { device, encoder } = ctx;
            const width = colorTex.width;
            const height = colorTex.height;

            if (!intermediate || width !== cachedWidth || height !== cachedHeight) {
                intermediate?.destroy();
                intermediate = device.createTexture({
                    label: "outline-color",
                    size: { width, height },
                    format: COLOR_FORMAT,
                    usage:
                        GPUTextureUsage.STORAGE_BINDING |
                        GPUTextureUsage.TEXTURE_BINDING |
                        GPUTextureUsage.RENDER_ATTACHMENT,
                });
                intermediateView = intermediate.createView();
                cachedWidth = width;
                cachedHeight = height;
                bindGroup = null;
            }

            const { color, thickness } = outline;
            const { r, g, b } = unpackColor(color);
            uniformFloats[0] = r;
            uniformFloats[1] = g;
            uniformFloats[2] = b;
            uniformFloats[3] = thickness / height;
            const count = Math.min(entities.length, MAX_SELECTED);
            uniformUints[4] = count;
            device.queue.writeBuffer(uniformBuffer, 0, uniformData);
            for (let i = 0; i < count; i++) selectedScratch[i] = entities[i];
            device.queue.writeBuffer(selectedBuffer, 0, selectedScratch, 0, count);

            if (!bindGroup || inputView !== cachedInputView || eidView !== cachedEidView) {
                bindGroup = device.createBindGroup({
                    layout,
                    entries: [
                        { binding: 0, resource: inputView },
                        { binding: 1, resource: eidView },
                        { binding: 2, resource: { buffer: uniformBuffer } },
                        { binding: 3, resource: { buffer: selectedBuffer } },
                        { binding: 4, resource: intermediateView! },
                    ],
                });
                cachedInputView = inputView;
                cachedEidView = eidView;
            }

            const ts = ctx.timestampWrites?.("outline");
            const pass = beginComputePass(encoder, ts as GPUComputePassTimestampWrites | undefined);
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
            pass.end();

            ctx.setTextureView("color", intermediateView!);
        },
    };
}

export const OutlinePlugin: Plugin = {
    name: "Outline",
    dependencies: [ComputePlugin, RenderPlugin],
    initialize(state) {
        const compute = Compute.from(state);
        if (!compute) return;
        compute.graph.add(createOutlineNode(state));
    },
};
