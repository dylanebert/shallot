import {
    resource,
    traits,
    capacity,
    buf,
    write,
    CHUNK_SHIFT,
    CHUNK_MASK,
    type Plugin,
    type State,
    type System,
} from "../../engine";
import {
    createColorProxy,
    createFieldProxy,
    formatHex,
    type FieldProxy,
} from "../../engine/ecs/core";
import { Compute, ComputePlugin } from "../../standard/compute";
import { gbuf, type GBuf } from "../../standard/compute";
import { Render, RenderPlugin } from "../../standard/render";
import { Z_FORMAT, SCENE_STRUCT_WGSL, type SharedPassContext } from "../../standard/render/core";
import { Transform } from "../../standard/transforms";

export const LineData = buf(Float32Array, 12, 0);

export const Line: {
    offsetX: FieldProxy;
    offsetY: FieldProxy;
    offsetZ: FieldProxy;
    thickness: FieldProxy;
    visible: FieldProxy;
    overdraw: FieldProxy;
    opacity: FieldProxy;
    color: FieldProxy;
    colorR: FieldProxy;
    colorG: FieldProxy;
    colorB: FieldProxy;
} = {
    offsetX: createFieldProxy(LineData, 12, 0),
    offsetY: createFieldProxy(LineData, 12, 1),
    offsetZ: createFieldProxy(LineData, 12, 2),
    thickness: createFieldProxy(LineData, 12, 3),
    visible: createFieldProxy(LineData, 12, 4),
    overdraw: createFieldProxy(LineData, 12, 5),
    opacity: createFieldProxy(LineData, 12, 7),
    color: createColorProxy(LineData, 12, 8),
    colorR: createFieldProxy(LineData, 12, 8),
    colorG: createFieldProxy(LineData, 12, 9),
    colorB: createFieldProxy(LineData, 12, 10),
};

traits(Line, {
    requires: [Transform],
    defaults: () => ({
        offsetX: 1,
        offsetY: 0,
        offsetZ: 0,
        thickness: 2,
        visible: 1,
        opacity: 1,
        color: 0xffffff,
    }),
    format: { color: formatHex },
});

const lineShader = /* wgsl */ `
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) dist: f32,
    @location(2) halfWidth: f32,
    @location(3) @interpolate(flat) entityId: u32,
}

${SCENE_STRUCT_WGSL}

struct LineData {
    offset: vec3<f32>,
    thickness: f32,
    visible: f32,
    _pad1: f32,
    _pad2: f32,
    opacity: f32,
    color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<storage, read> entityIds: array<u32>;
@group(0) @binding(2) var<storage, read> lines: array<LineData>;
@group(0) @binding(3) var<storage, read> matrices: array<mat4x4<f32>>;

@vertex
fn vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOutput {
    let eid = entityIds[iid];
    let line = lines[eid];
    let transform = matrices[eid];

    let start = transform[3].xyz;
    let rotation = mat3x3<f32>(transform[0].xyz, transform[1].xyz, transform[2].xyz);
    let end = start + rotation * line.offset;

    var startClip = scene.viewProj * vec4(start, 1.0);
    var endClip = scene.viewProj * vec4(end, 1.0);

    let nearW = 0.001;
    if (startClip.w < nearW && endClip.w < nearW) {
        var out: VertexOutput;
        out.position = vec4(0.0, 0.0, -1.0, 1.0);
        out.color = vec4(0.0);
        out.dist = 0.0;
        out.halfWidth = 0.0;
        out.entityId = eid;
        return out;
    }
    if (startClip.w < nearW) {
        let t = (nearW - startClip.w) / (endClip.w - startClip.w);
        startClip = mix(startClip, endClip, t);
    } else if (endClip.w < nearW) {
        let t = (nearW - endClip.w) / (startClip.w - endClip.w);
        endClip = mix(endClip, startClip, t);
    }

    let startNDC = startClip.xy / startClip.w;
    let endNDC = endClip.xy / endClip.w;

    let dir = endNDC - startNDC;
    let len = length(dir);
    let normDir = select(vec2(1.0, 0.0), dir / len, len > 0.0001);

    let scale = scene.viewport.y / 1080.0;
    let halfWidth = line.thickness * 0.5 * scale;
    let aaPadding = 1.0;
    let totalHalf = halfWidth + aaPadding;
    let perpNDC = vec2(-normDir.y, normDir.x) * totalHalf * 2.0 / scene.viewport;

    var pos: vec2<f32>;
    var t: f32;
    var edge: f32;
    switch vid {
        case 0u: { pos = startNDC - perpNDC; t = 0.0; edge = -1.0; }
        case 1u: { pos = startNDC + perpNDC; t = 0.0; edge = 1.0; }
        case 2u: { pos = endNDC + perpNDC; t = 1.0; edge = 1.0; }
        case 3u: { pos = startNDC - perpNDC; t = 0.0; edge = -1.0; }
        case 4u: { pos = endNDC + perpNDC; t = 1.0; edge = 1.0; }
        case 5u: { pos = endNDC - perpNDC; t = 1.0; edge = -1.0; }
        default: { pos = startNDC; t = 0.0; edge = 0.0; }
    }

    let depth = mix(startClip.z / startClip.w, endClip.z / endClip.w, t);

    let pixelDist = edge * totalHalf;

    var out: VertexOutput;
    out.position = vec4(pos, depth, 1.0);
    out.color = vec4(line.color.rgb, line.color.a * line.opacity);
    out.dist = pixelDist;
    out.halfWidth = halfWidth;
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
    let dist = abs(input.dist);
    let aaWidth = fwidth(input.dist);
    let aa = 1.0 - smoothstep(input.halfWidth - aaWidth, input.halfWidth + aaWidth, dist);
    var out: FragmentOutput;
    out.color = vec4(input.color.rgb, input.color.a * aa);
    out.mask = select(0.0, 1.0, aa > 0.01);
    out.entityId = input.entityId;
    return out;
}
`;

function createLinesPipeline(
    device: GPUDevice,
    format: GPUTextureFormat,
    maskFormat: GPUTextureFormat,
    eidFormat: GPUTextureFormat,
    depthCompare: GPUCompareFunction = "less",
): GPURenderPipeline {
    const module = device.createShaderModule({ code: lineShader });

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
            depthCompare,
            depthWriteEnabled: false,
        },
    });
}

export interface Lines {
    buffer: GBuf;
    entityIds: GBuf;
    overdrawEntityIds: GBuf;
    count: number;
    overdrawCount: number;
}

export const Lines = resource<Lines>("lines");

const EntityIdArray = buf(Uint32Array, 1, 0);
const OverdrawEntityIdArray = buf(Uint32Array, 1, 0);

const LinesSystem: System = {
    group: "draw",
    annotations: { mode: "always" },

    update(state: State) {
        const compute = Compute.from(state);
        const lines = Lines.from(state);
        if (!compute || !lines) return;

        const { device } = compute;

        let count = 0;
        let overdrawCount = 0;
        const ldChunks = LineData.chunks;
        const eidChunks = EntityIdArray.chunks;
        const odChunks = OverdrawEntityIdArray.chunks;
        for (const eid of state.query([Line, Transform])) {
            const ld = ldChunks[eid >>> CHUNK_SHIFT];
            const local = eid & CHUNK_MASK;
            if (!ld[local * 12 + 4]) continue;
            if (ld[local * 12 + 5]) {
                const slot = overdrawCount++;
                odChunks[slot >>> CHUNK_SHIFT][slot & CHUNK_MASK] = eid;
            } else {
                const slot = count++;
                eidChunks[slot >>> CHUNK_SHIFT][slot & CHUNK_MASK] = eid;
            }
        }

        const uploadCount = state.max + 1;
        write(device.queue, lines.buffer.buffer, 0, LineData, uploadCount);
        write(device.queue, lines.entityIds.buffer, 0, EntityIdArray, count);
        lines.count = count;
        if (overdrawCount > 0) {
            write(
                device.queue,
                lines.overdrawEntityIds.buffer,
                0,
                OverdrawEntityIdArray,
                overdrawCount,
            );
        }
        lines.overdrawCount = overdrawCount;
    },
};

export const LinesPlugin: Plugin = {
    name: "Lines",
    systems: [LinesSystem],
    components: { Line },
    dependencies: [ComputePlugin, RenderPlugin],

    initialize(state: State) {
        const compute = Compute.from(state);
        const render = Render.from(state);
        if (!compute || !render) return;

        const { device } = compute;

        const linesState: Lines = {
            buffer: gbuf(
                device,
                "lines",
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                (cap) => cap * 12 * 4,
            ),
            entityIds: gbuf(
                device,
                "line-entityIds",
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
                (cap) => cap * 4,
            ),
            overdrawEntityIds: gbuf(
                device,
                "line-overdraw-entityIds",
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
                (cap) => cap * 4,
            ),
            count: 0,
            overdrawCount: 0,
        };

        state.setResource(Lines, linesState);

        let pipeline: GPURenderPipeline | null = null;
        let overdrawPipeline: GPURenderPipeline | null = null;
        let bindGroup: GPUBindGroup | null = null;
        let overdrawBindGroup: GPUBindGroup | null = null;
        let cachedCapacity = capacity();

        render.effects.overlay.push({
            order: 0,

            draw(pass: GPURenderPassEncoder, ctx: SharedPassContext) {
                if (capacity() !== cachedCapacity) {
                    cachedCapacity = capacity();
                    bindGroup = null;
                    overdrawBindGroup = null;
                }
                const count = linesState.count;
                const overdrawCount = linesState.overdrawCount;
                if (count === 0 && overdrawCount === 0) return;

                if (!pipeline) {
                    pipeline = createLinesPipeline(
                        ctx.device,
                        ctx.format,
                        ctx.maskFormat,
                        ctx.eidFormat,
                    );
                }

                if (count > 0) {
                    if (!bindGroup) {
                        bindGroup = ctx.device.createBindGroup({
                            layout: pipeline.getBindGroupLayout(0),
                            entries: [
                                { binding: 0, resource: { buffer: render.scene } },
                                { binding: 1, resource: { buffer: linesState.entityIds.buffer } },
                                { binding: 2, resource: { buffer: linesState.buffer.buffer } },
                                { binding: 3, resource: { buffer: render.matrices.buffer } },
                            ],
                        });
                    }
                    pass.setPipeline(pipeline);
                    pass.setBindGroup(0, bindGroup);
                    pass.draw(6, count);
                }

                if (overdrawCount > 0) {
                    if (!overdrawPipeline) {
                        overdrawPipeline = createLinesPipeline(
                            ctx.device,
                            ctx.format,
                            ctx.maskFormat,
                            ctx.eidFormat,
                            "always",
                        );
                    }
                    if (!overdrawBindGroup) {
                        overdrawBindGroup = ctx.device.createBindGroup({
                            layout: overdrawPipeline.getBindGroupLayout(0),
                            entries: [
                                { binding: 0, resource: { buffer: render.scene } },
                                {
                                    binding: 1,
                                    resource: { buffer: linesState.overdrawEntityIds.buffer },
                                },
                                { binding: 2, resource: { buffer: linesState.buffer.buffer } },
                                { binding: 3, resource: { buffer: render.matrices.buffer } },
                            ],
                        });
                    }
                    pass.setPipeline(overdrawPipeline);
                    pass.setBindGroup(0, overdrawBindGroup);
                    pass.draw(6, overdrawCount);
                }
            },
        });
    },
};
