import {
    resource,
    capacity,
    buf,
    traits,
    write,
    CHUNK_SHIFT,
    CHUNK_MASK,
    type Plugin,
    type State,
    type System,
} from "../../engine";
import { createFieldProxy, type FieldProxy } from "../../engine/ecs/core";
import { Compute, ComputePlugin } from "../../standard/compute";
import { gbuf, type GBuf } from "../../standard/compute";
import { Render, RenderPlugin } from "../../standard/render";
import { Z_FORMAT, SCENE_STRUCT_WGSL, type SharedPassContext } from "../../standard/render/core";
import { Transform } from "../../standard/transforms";
import { Line, LineData, Lines, LinesPlugin } from "../lines";

export const ArrowData = buf(Float32Array, 4, 0);

export const Arrow: {
    start: FieldProxy;
    end: FieldProxy;
    size: FieldProxy;
} = {
    start: createFieldProxy(ArrowData, 4, 0),
    end: createFieldProxy(ArrowData, 4, 1),
    size: createFieldProxy(ArrowData, 4, 2),
};

traits(Arrow, {
    requires: [Line],
    defaults: () => ({
        start: 0,
        end: 1,
        size: 1,
    }),
});

const END_FLAG = 0x80000000;

const arrowShader = /* wgsl */ `
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) entityId: u32,
}

${SCENE_STRUCT_WGSL}

struct ArrowData {
    start: f32,
    end: f32,
    size: f32,
    _pad: f32,
}

struct LineData {
    offset: vec3<f32>,
    thickness: f32,
    visible: f32,
    _pad1: f32,
    _pad2: f32,
    opacity: f32,
    color: vec4<f32>,
}

const END_FLAG: u32 = 0x80000000u;

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<storage, read> entityIds: array<u32>;
@group(0) @binding(2) var<storage, read> arrows: array<ArrowData>;
@group(0) @binding(3) var<storage, read> lines: array<LineData>;
@group(0) @binding(4) var<storage, read> matrices: array<mat4x4<f32>>;

@vertex
fn vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOutput {
    let packed = entityIds[iid];
    let isEnd = (packed & END_FLAG) != 0u;
    let eid = packed & ~END_FLAG;

    let arrow = arrows[eid];
    let line = lines[eid];
    let transform = matrices[eid];

    let scale = length(transform[0].xyz);

    let start = transform[3].xyz;
    let rotation = mat3x3<f32>(transform[0].xyz, transform[1].xyz, transform[2].xyz);
    let end = start + rotation * line.offset;

    let startClip = scene.viewProj * vec4(start, 1.0);
    let endClip = scene.viewProj * vec4(end, 1.0);

    let startScreen = (startClip.xy / startClip.w) * scene.viewport * 0.5;
    let endScreen = (endClip.xy / endClip.w) * scene.viewport * 0.5;

    let anchorScreen = select(startScreen, endScreen, isEnd);
    let anchorDepth = select(startClip.z / startClip.w, endClip.z / endClip.w, isEnd);

    let dir = endScreen - startScreen;
    let len = length(dir);
    let normDir = select(vec2(1.0, 0.0), dir / len, len > 0.0001);
    let perp = vec2(-normDir.y, normDir.x);

    let arrowDir = select(-normDir, normDir, isEnd);

    let viewportScale = scene.viewport.y / 1080.0;
    let arrowLengthPx = arrow.size * line.thickness * 4.0 * scale * viewportScale;
    let arrowWidthPx = arrow.size * line.thickness * 2.0 * scale * viewportScale;

    var posScreen: vec2<f32>;
    switch vid {
        case 0u: { posScreen = anchorScreen; }
        case 1u: { posScreen = anchorScreen - arrowDir * arrowLengthPx + perp * arrowWidthPx; }
        case 2u: { posScreen = anchorScreen - arrowDir * arrowLengthPx - perp * arrowWidthPx; }
        default: { posScreen = anchorScreen; }
    }

    let pos = posScreen / (scene.viewport * 0.5);

    var out: VertexOutput;
    out.position = vec4(pos, anchorDepth, 1.0);
    out.color = vec4(line.color.rgb, line.color.a * line.opacity);
    out.entityId = eid;
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) mask: f32,
    @location(2) entityId: u32,
}

@fragment
fn fs(input: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    out.color = input.color;
    out.mask = 1.0;
    out.entityId = input.entityId;
    return out;
}
`;

function createArrowsPipeline(
    device: GPUDevice,
    format: GPUTextureFormat,
    maskFormat: GPUTextureFormat,
    eidFormat: GPUTextureFormat,
): GPURenderPipeline {
    const module = device.createShaderModule({ code: arrowShader });

    return device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module,
            entryPoint: "vs",
        },
        fragment: {
            module,
            entryPoint: "fs",
            targets: [
                {
                    format,
                    blend: {
                        color: {
                            srcFactor: "src-alpha",
                            dstFactor: "one-minus-src-alpha",
                            operation: "add",
                        },
                        alpha: {
                            srcFactor: "one",
                            dstFactor: "one-minus-src-alpha",
                            operation: "add",
                        },
                    },
                },
                {
                    format: maskFormat,
                    writeMask: GPUColorWrite.RED,
                },
                {
                    format: eidFormat,
                },
            ],
        },
        primitive: {
            topology: "triangle-list",
        },
        depthStencil: {
            format: Z_FORMAT,
            depthCompare: "less",
            depthWriteEnabled: false,
        },
    });
}

export interface Arrows {
    buffer: GBuf;
    entityIds: GBuf;
    count: number;
}

export const Arrows = resource<Arrows>("arrows");

const ArrowEntityIdArray = buf(Uint32Array, 2, 0);

const ArrowsSystem: System = {
    group: "draw",
    annotations: { mode: "always" },

    update(state: State) {
        const compute = Compute.from(state);
        const arrows = Arrows.from(state);
        const lines = Lines.from(state);
        if (!compute || !arrows || !lines) return;

        const { device } = compute;

        let count = 0;
        const ldChunks = LineData.chunks;
        const adChunks = ArrowData.chunks;
        const eidChunks = ArrowEntityIdArray.chunks;
        for (const eid of state.query([Arrow, Line, Transform])) {
            const ld = ldChunks[eid >>> CHUNK_SHIFT];
            const local = eid & CHUNK_MASK;
            if (!ld[local * 12 + 4]) continue;

            const ad = adChunks[eid >>> CHUNK_SHIFT];
            const adOff = local * 4;
            if (ad[adOff]) {
                const slot = count++;
                eidChunks[slot >>> CHUNK_SHIFT][(slot & CHUNK_MASK) * 2] = eid;
            }
            if (ad[adOff + 1]) {
                const slot = count++;
                eidChunks[slot >>> CHUNK_SHIFT][(slot & CHUNK_MASK) * 2] = eid | END_FLAG;
            }
        }

        const uploadCount = state.max + 1;
        write(device.queue, arrows.buffer.buffer, 0, ArrowData, uploadCount);
        write(device.queue, arrows.entityIds.buffer, 0, ArrowEntityIdArray, count);
        arrows.count = count;
    },
};

export const ArrowsPlugin: Plugin = {
    name: "Arrows",
    systems: [ArrowsSystem],
    components: { Arrow },
    dependencies: [ComputePlugin, RenderPlugin, LinesPlugin],

    initialize(state: State) {
        const compute = Compute.from(state);
        const render = Render.from(state);
        const lines = Lines.from(state);
        if (!compute || !render || !lines) return;

        const { device } = compute;

        const arrowsState: Arrows = {
            buffer: gbuf(
                device,
                "arrows",
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                (cap) => cap * 4 * 4,
            ),
            entityIds: gbuf(
                device,
                "arrow-entityIds",
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
                (cap) => cap * 2 * 4,
            ),
            count: 0,
        };

        state.setResource(Arrows, arrowsState);

        let pipeline: GPURenderPipeline | null = null;
        let bindGroup: GPUBindGroup | null = null;
        let cachedCapacity = capacity();

        render.effects.overlay.push({
            order: 1,

            draw(pass: GPURenderPassEncoder, ctx: SharedPassContext) {
                if (capacity() !== cachedCapacity) {
                    cachedCapacity = capacity();
                    bindGroup = null;
                }
                if (arrowsState.count === 0) return;

                if (!pipeline) {
                    pipeline = createArrowsPipeline(
                        ctx.device,
                        ctx.format,
                        ctx.maskFormat,
                        ctx.eidFormat,
                    );
                }

                if (!bindGroup) {
                    bindGroup = ctx.device.createBindGroup({
                        layout: pipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: { buffer: render.scene } },
                            { binding: 1, resource: { buffer: arrowsState.entityIds.buffer } },
                            { binding: 2, resource: { buffer: arrowsState.buffer.buffer } },
                            { binding: 3, resource: { buffer: lines.buffer.buffer } },
                            { binding: 4, resource: { buffer: render.matrices.buffer } },
                        ],
                    });
                }

                pass.setPipeline(pipeline);
                pass.setBindGroup(0, bindGroup);
                pass.draw(3, arrowsState.count);
            },
        });
    },
};
