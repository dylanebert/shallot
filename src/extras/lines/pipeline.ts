import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { biasClipDepth, DEPTH_FORMAT, ViewUniforms, Xform, xformMat } from "../../core/rendering";
import { precompile, type World } from "../../engine";
import { unpackLdrColor } from "../../engine/utils";

/** A pixel-width shaft, or one world-space triangle with vertices a, b and c. */
export const Segment = d
    .struct({
        a: d.vec3f,
        width: d.f32,
        b: d.vec3f,
        color: d.u32,
        c: d.vec3f,
        triangle: d.u32,
        depthBias: d.f32,
        layers: d.u32,
    })
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

const polygonVertex = tgpu.fn(
    [d.vec3f, d.vec3f, d.vec3f, d.vec3f, d.u32],
    d.vec3f,
)((a, b, c, last, index) => {
    "use gpu";
    if (index === 5) return d.vec3f(last);
    if (index === 4) return d.vec3f(c);
    return triangleVertex(a, b, c, index);
});

// Near clipping precedes bias so behind-eye vertices cannot cut away visible head geometry.
// A triangle cut by one plane has at most four vertices, fitting the segment's six corners.
export const nearTriangleVertex = tgpu.fn(
    [d.vec3f, d.vec3f, d.vec3f, d.vec3f, d.u32],
    d.vec4f,
)((a, b, c, distance, index) => {
    "use gpu";
    if (distance.x < 0 && distance.y < 0 && distance.z < 0) return d.vec4f(a, 0);
    if (distance.x >= 0 && distance.y >= 0 && distance.z >= 0)
        return d.vec4f(triangleVertex(a, b, c, index), 1);
    let ab = d.vec3f(a);
    let bc = d.vec3f(b);
    let ca = d.vec3f(c);
    if (distance.x < 0 !== distance.y < 0)
        ab = std.mix(a, b, distance.x / (distance.x - distance.y));
    if (distance.y < 0 !== distance.z < 0)
        bc = std.mix(b, c, distance.y / (distance.y - distance.z));
    if (distance.z < 0 !== distance.x < 0)
        ca = std.mix(c, a, distance.z / (distance.z - distance.x));
    if (distance.x < 0) {
        if (distance.y < 0) return d.vec4f(triangleVertex(ca, bc, c, index), 1);
        if (distance.z < 0) return d.vec4f(triangleVertex(ab, b, bc, index), 1);
        return d.vec4f(polygonVertex(ab, b, c, ca, index), 1);
    }
    if (distance.y < 0) {
        if (distance.z < 0) return d.vec4f(triangleVertex(a, ab, ca, index), 1);
        return d.vec4f(polygonVertex(a, ab, bc, c, index), 1);
    }
    return d.vec4f(polygonVertex(a, b, bc, ca, index), 1);
});

const viewLayout = tgpu
    .bindGroupLayout({ view: { uniform: ViewUniforms, visibility: ["vertex", "fragment"] } })
    .$idx(0);
export const lineLayout = tgpu
    .bindGroupLayout({
        segments: { storage: d.arrayOf(Segment), access: "readonly", visibility: ["vertex"] },
    })
    .$idx(1);

/** Placement is the entity's interpolated global-transform row; a negative width marks
 * immediate lines, which are already in world space. */
export const LineInstance = d.struct({
    eid: d.u32,
    width: d.f32,
    opacity: d.f32,
    depthBias: d.f32,
    layers: d.u32,
});
export const instanceLayout = tgpu
    .bindGroupLayout({
        instances: { storage: d.arrayOf(LineInstance), access: "readonly", visibility: ["vertex"] },
        globalTransforms: { storage: d.arrayOf(Xform), access: "readonly", visibility: ["vertex"] },
        globalTransformRows: {
            storage: d.arrayOf(d.u32),
            access: "readonly",
            visibility: ["vertex"],
        },
    })
    .$idx(2);

const LineVertex = tgpu.vertexFn({
    in: { vertex: d.builtin.vertexIndex, instance: d.builtin.instanceIndex },
    out: { position: d.builtin.position, rgba: d.vec4f, edge: d.vec2f, cameraDepth: d.f32 },
})((input) => {
    "use gpu";
    const instance = instanceLayout.$.instances[input.instance];
    const retained = instance.width >= 0;
    let worldFromLocal = d.mat4x4f(
        d.vec4f(1, 0, 0, 0),
        d.vec4f(0, 1, 0, 0),
        d.vec4f(0, 0, 1, 0),
        d.vec4f(0, 0, 0, 1),
    );
    if (retained) {
        const row = instanceLayout.$.globalTransformRows[instance.eid];
        if (row === 0) {
            return {
                position: d.vec4f(0),
                rgba: d.vec4f(0),
                edge: d.vec2f(0),
                cameraDepth: d.f32(0),
            };
        }
        worldFromLocal = xformMat(Xform(instanceLayout.$.globalTransforms[row - 1]));
    }
    const seg = Segment(lineLayout.$.segments[input.vertex / 6]);
    const index = input.vertex % 6;
    const a = std.mul(worldFromLocal, d.vec4f(seg.a, 1)).xyz;
    const b = std.mul(worldFromLocal, d.vec4f(seg.b, 1)).xyz;
    const c = std.mul(worldFromLocal, d.vec4f(seg.c, 1)).xyz;
    const width = std.select(seg.width, instance.width, retained);
    const bias = std.select(seg.depthBias, instance.depthBias, retained);
    let rgba = unpackLdrColor(seg.color);
    rgba = d.vec4f(rgba.xyz, rgba.w * instance.opacity);
    if (!retained && (seg.layers & instance.layers) === 0) rgba = d.vec4f(0);
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
        const depths = d.vec3f(
            -std.mul(view.viewFromWorld, d.vec4f(a, 1)).z,
            -std.mul(view.viewFromWorld, d.vec4f(b, 1)).z,
            -std.mul(view.viewFromWorld, d.vec4f(c, 1)).z,
        );
        const point = nearTriangleVertex(
            a,
            b,
            c,
            std.sub(depths, d.vec3f(view.projection.x)),
            index,
        );
        return {
            position: biasClipDepth(std.mul(view.clipFromWorld, d.vec4f(point.xyz, 1)), bias),
            rgba: std.mul(rgba, point.w),
            edge: d.vec2f(0, 1),
            cameraDepth: -std.mul(view.viewFromWorld, d.vec4f(point.xyz, 1)).z,
        };
    }
    const quad = cameraLineQuad(
        std.mul(view.clipFromWorld, d.vec4f(a, 1)),
        std.mul(view.clipFromWorld, d.vec4f(b, 1)),
        view.resolution,
        t,
        edge,
        width,
        view.projection,
    );
    return {
        position: biasClipDepth(quad.clip, bias),
        rgba: std.mul(rgba, quad.tint),
        edge: quad.edge,
        cameraDepth: view.projection.x,
    };
});

const LineFragment = tgpu.fragmentFn({
    in: { rgba: d.vec4f, edge: d.vec2f, cameraDepth: d.f32 },
    out: d.vec4f,
})((input) => {
    "use gpu";
    // Arrow triangles retain perspective-correct depth independently of depth bias.
    if (input.cameraDepth > viewLayout.$.view.projection.y) std.discard();
    const width = std.fwidth(input.edge.x);
    const alpha =
        d.f32(1) -
        std.smoothstep(input.edge.y - width, input.edge.y + width, std.abs(input.edge.x));
    return d.vec4f(input.rgba.xyz, std.mul(input.rgba.w, alpha));
});

const ALPHA_BLEND: GPUBlendState = {
    color: { operation: "add", srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
    alpha: { operation: "add", srcFactor: "one", dstFactor: "one-minus-src-alpha" },
};

export function createLinePipeline(world: World, format: GPUTextureFormat, sampleCount: number) {
    const pipeline = world.gpu.root
        .createRenderPipeline({
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
        })
        .$name(`lines-${sampleCount}`);
    precompile(world, `lines-${sampleCount}`, () => pipeline);
    return pipeline;
}

const LineQuad = d.struct({ clip: d.vec4f, edge: d.vec2f, tint: d.vec4f });

// Perspective uses infinite reverse-Z; its authored far bound is not a homogeneous clip plane.
export const cameraLineQuad = tgpu.fn(
    [d.vec4f, d.vec4f, d.vec2f, d.f32, d.f32, d.f32, d.vec4f],
    LineQuad,
)((sClip, eClip, res, t, edge, widthPx, projection) => {
    "use gpu";
    let start = d.vec4f(sClip);
    let end = d.vec4f(eClip);
    if (projection.z === 1) {
        const startFar = projection.y - start.w;
        const endFar = projection.y - end.w;
        if (startFar < 0 && endFar < 0) {
            return { clip: d.vec4f(0, 0, -1, 1), edge: d.vec2f(0), tint: d.vec4f(0) };
        }
        if (startFar < 0) {
            const clipped = std.mix(start, end, startFar / (startFar - endFar));
            start = d.vec4f(clipped.xyz, projection.y);
        } else if (endFar < 0) {
            const clipped = std.mix(end, start, endFar / (endFar - startFar));
            end = d.vec4f(clipped.xyz, projection.y);
        }
    }
    return lineQuad(start, end, res, t, edge, widthPx);
});

export const lineQuad = tgpu.fn(
    [d.vec4f, d.vec4f, d.vec2f, d.f32, d.f32, d.f32],
    LineQuad,
)((sClip, eClip, res, t, edge, widthPx) => {
    "use gpu";
    let start = d.vec4f(sClip);
    let end = d.vec4f(eClip);
    // Clip to the reverse-Z near plane before dividing by w. An eye-plane epsilon can round
    // to zero on long segments, and their projected depth then loses the surface offset.
    const startNear = start.w - start.z;
    const endNear = end.w - end.z;
    if (startNear < 0 && endNear < 0) {
        return { clip: d.vec4f(0, 0, -1, 1), edge: d.vec2f(0), tint: d.vec4f(0) };
    }
    if (startNear < 0) {
        const clipped = std.mix(start, end, startNear / (startNear - endNear));
        start = d.vec4f(clipped.xy, clipped.z, clipped.z);
    } else if (endNear < 0) {
        const clipped = std.mix(end, start, endNear / (endNear - startNear));
        end = d.vec4f(clipped.xy, clipped.z, clipped.z);
    }
    // Reverse-Z puts a finite far plane at z = 0; the clipped end is set on it exactly, since the
    // mix may round past it and depth bias treats the plane itself as a limit.
    if (start.z < 0 && end.z < 0) {
        return { clip: d.vec4f(0, 0, -1, 1), edge: d.vec2f(0), tint: d.vec4f(0) };
    }
    if (start.z < 0) {
        const clipped = std.mix(start, end, -start.z / (end.z - start.z));
        start = d.vec4f(clipped.xy, 0, clipped.w);
    } else if (end.z < 0) {
        const clipped = std.mix(end, start, -end.z / (start.z - end.z));
        end = d.vec4f(clipped.xy, 0, clipped.w);
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
