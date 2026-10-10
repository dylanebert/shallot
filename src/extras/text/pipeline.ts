import tgpu, { type TgpuRoot } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { DEPTH_FORMAT, ViewUniforms } from "../../core/rendering";
import { decodePos, MeshQuant, meshIdOf, Xform, xformPoint } from "../../engine/utils";
import { Glyph, sdfToSignedDistance, textSrgbToLinear } from "./glyph";

const viewLayout = tgpu
    .bindGroupLayout({
        view: { uniform: ViewUniforms, visibility: ["vertex"] },
        meshQuant: { storage: d.arrayOf(MeshQuant), access: "readonly", visibility: ["vertex"] },
    })
    .$idx(0);

export const textLayout = tgpu
    .bindGroupLayout({
        vertices: { storage: d.arrayOf(d.vec4u), access: "readonly", visibility: ["vertex"] },
        glyphs: { storage: d.arrayOf(Glyph), access: "readonly", visibility: ["vertex"] },
        globalTransforms: { storage: d.arrayOf(Xform), access: "readonly", visibility: ["vertex"] },
        globalTransformRows: {
            storage: d.arrayOf(d.u32),
            access: "readonly",
            visibility: ["vertex"],
        },
        atlas: { texture: d.texture2d(d.f32), visibility: ["fragment"] },
        sampler: { sampler: "filtering", visibility: ["fragment"] },
    })
    .$idx(1);

export const TextVertex = tgpu.vertexFn({
    in: { vertex: d.builtin.vertexIndex, instance: d.builtin.instanceIndex },
    out: {
        position: d.builtin.position,
        uvSize: d.vec4f,
        color: d.vec4f,
        local: d.vec2f,
    },
})((input) => {
    "use gpu";
    const packed = textLayout.$.vertices[input.vertex];
    const quant = MeshQuant(viewLayout.$.meshQuant[meshIdOf(packed.y)]);
    const corner = decodePos(packed.x, packed.y, quant).xy;
    const glyph = Glyph(textLayout.$.glyphs[input.instance]);
    const row = textLayout.$.globalTransformRows[glyph.eid];
    if (row === 0) {
        return {
            position: d.vec4f(0),
            uvSize: d.vec4f(0),
            color: d.vec4f(0),
            local: d.vec2f(0),
        };
    }
    const transform = Xform(textLayout.$.globalTransforms[row - 1]);
    const local = d.vec3f(
        glyph.pos.x + corner.x * glyph.size.x,
        glyph.pos.y + corner.y * glyph.size.y,
        glyph.pos.z,
    );
    const uv = std.mix(glyph.uvRect.xy, glyph.uvRect.zw, corner);
    const world = xformPoint(transform, local);
    return {
        position: std.mul(viewLayout.$.view.viewProj, d.vec4f(world, 1)),
        uvSize: d.vec4f(uv, glyph.size),
        color: std.unpack4x8unorm(glyph.color),
        local: corner,
    };
});

export const TextFragment = tgpu.fragmentFn({
    in: { uvSize: d.vec4f, color: d.vec4f, local: d.vec2f },
    out: d.vec4f,
})((input) => {
    "use gpu";
    const sdf = std.textureSample(textLayout.$.atlas, textLayout.$.sampler, input.uvSize.xy).x;
    const maxDim = std.max(input.uvSize.z, input.uvSize.w);
    const signedDistance = sdfToSignedDistance(sdf, maxDim);
    const aa = std.length(std.mul(std.fwidth(input.local), input.uvSize.zw)) * 0.5;
    const alpha = std.smoothstep(aa, -aa, signedDistance);
    if (alpha < 0.01) std.discard();
    return d.vec4f(textSrgbToLinear(input.color.xyz), std.mul(input.color.w, alpha));
});

const ALPHA_BLEND: GPUBlendState = {
    color: { operation: "add", srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
    alpha: { operation: "add", srcFactor: "one", dstFactor: "one-minus-src-alpha" },
};

export function createTextPipeline(root: TgpuRoot, format: GPUTextureFormat, sampleCount: number) {
    return root.createRenderPipeline({
        vertex: TextVertex,
        fragment: TextFragment,
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

export { viewLayout };
