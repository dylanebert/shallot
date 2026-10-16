import tgpu, { type TgpuRoot } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { DEPTH_FORMAT, ViewUniforms } from "../../core/rendering";
import { unpackLdrColor } from "../../engine/utils";

/** A pixel-width shaft, or one world-space triangle with vertices a, b and c. */
export const Segment = d
    .struct({ a: d.vec3f, width: d.f32, b: d.vec3f, color: d.u32, c: d.vec3f, triangle: d.u32 })
    .$name("Segment");

export const SEGMENT_BYTES = d.sizeOf(Segment);
export const SEGMENT_FLOATS = SEGMENT_BYTES / 4;

export const triangleVertex = tgpu.fn(
    [d.vec3f, d.vec3f, d.vec3f, d.u32],
    d.vec3f,
)((a, b, c, index) => {
    "use gpu";
    if (index === 1) return d.vec3f(b);
    if (index === 2) return d.vec3f(c);
    return d.vec3f(a);
});

const viewLayout = tgpu
    .bindGroupLayout({ view: { uniform: ViewUniforms, visibility: ["vertex"] } })
    .$idx(0);
export const lineLayout = tgpu
    .bindGroupLayout({
        segments: { storage: d.arrayOf(Segment), access: "readonly", visibility: ["vertex"] },
    })
    .$idx(1);

const LineVertex = tgpu.vertexFn({
    in: { vertex: d.builtin.vertexIndex, instance: d.builtin.instanceIndex },
    out: { position: d.builtin.position, rgba: d.vec4f, edge: d.vec2f },
})((input) => {
    "use gpu";
    const seg = Segment(lineLayout.$.segments[input.instance]);
    const index = input.vertex % 6;
    let t = d.f32(0);
    let edge = d.f32(-1);
    if (index === 1) edge = d.f32(1);
    if (index === 2 || index === 4) {
        t = d.f32(1);
        edge = d.f32(1);
    }
    if (index === 5) {
        t = d.f32(1);
        edge = d.f32(-1);
    }
    const view = viewLayout.$.view;
    if (seg.triangle !== 0) {
        const point = triangleVertex(seg.a, seg.b, seg.c, index);
        return {
            position: std.mul(view.viewProj, d.vec4f(point, 1)),
            rgba: unpackLdrColor(seg.color),
            edge: d.vec2f(0, 1),
        };
    }
    const quad = lineQuad(
        std.mul(view.viewProj, d.vec4f(seg.a, 1)),
        std.mul(view.viewProj, d.vec4f(seg.b, 1)),
        view.resolution,
        t,
        edge,
        seg.width,
    );
    return {
        position: quad.clip,
        rgba: std.mul(unpackLdrColor(seg.color), quad.tint),
        edge: quad.edge,
    };
});

const LineFragment = tgpu.fragmentFn({ in: { rgba: d.vec4f, edge: d.vec2f }, out: d.vec4f })(
    (input) => {
        "use gpu";
        const width = std.fwidth(input.edge.x);
        const alpha =
            d.f32(1) -
            std.smoothstep(input.edge.y - width, input.edge.y + width, std.abs(input.edge.x));
        return d.vec4f(input.rgba.xyz, std.mul(input.rgba.w, alpha));
    },
);

const ALPHA_BLEND: GPUBlendState = {
    color: { operation: "add", srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
    alpha: { operation: "add", srcFactor: "one", dstFactor: "one-minus-src-alpha" },
};

export function createLinePipeline(root: TgpuRoot, format: GPUTextureFormat, sampleCount: number) {
    return root.createRenderPipeline({
        vertex: LineVertex,
        fragment: LineFragment,
        primitive: { topology: "triangle-list", cullMode: "none" },
        depthStencil: {
            format: DEPTH_FORMAT,
            depthWriteEnabled: false,
            depthCompare: "greater-equal",
        },
        multisample: { count: sampleCount },
        targets: { format, blend: ALPHA_BLEND },
    });
}

export const lineQuad = tgpu.fn(
    [d.vec4f, d.vec4f, d.vec2f, d.f32, d.f32, d.f32],
    d.struct({ clip: d.vec4f, edge: d.vec2f, tint: d.vec4f }),
)((sClip, eClip, res, t, edge, widthPx) => {
    "use gpu";
    const nearW = d.f32(1e-5);
    let start = d.vec4f(sClip);
    let end = d.vec4f(eClip);
    if (start.w < nearW && end.w < nearW) {
        return { clip: d.vec4f(0, 0, -1, 1), edge: d.vec2f(0), tint: d.vec4f(0) };
    }
    if (start.w < nearW) {
        start = d.vec4f(std.mix(start, end, (nearW - start.w) / (end.w - start.w)));
    } else if (end.w < nearW) {
        end = d.vec4f(std.mix(end, start, (nearW - end.w) / (start.w - end.w)));
    }
    const startNdc = std.div(start.xy, start.w);
    const endNdc = std.div(end.xy, end.w);
    const directionPx = std.mul(std.sub(endNdc, startNdc), res);
    const lengthPx = std.length(directionPx);
    const direction = std.select(
        d.vec2f(1, 0),
        std.div(directionPx, lengthPx),
        lengthPx > d.f32(1e-4),
    );
    const perpendicular = d.vec2f(-direction.y, direction.x);
    const halfWidth = std.max(widthPx, d.f32(1)) * d.f32(0.5);
    const total = halfWidth + d.f32(1);
    const useEnd = t > d.f32(0.5);
    const baseNdc = std.select(startNdc, endNdc, useEnd);
    const baseClip = std.select(start, end, useEnd);
    const ndc = std.add(
        baseNdc,
        std.div(std.mul(std.mul(perpendicular, edge * total), d.f32(2)), res),
    );
    return {
        clip: d.vec4f(ndc, baseClip.z / baseClip.w, 1),
        edge: d.vec2f(edge * total, halfWidth),
        tint: d.vec4f(1, 1, 1, std.min(widthPx, d.f32(1))),
    };
});

export { LineFragment, LineVertex, viewLayout };
