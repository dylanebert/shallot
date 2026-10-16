import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { litPbr } from "./engine";
import { StandardMaterial, StandardMaterialInput } from "./material-data";
import {
    materialFragmentContext,
    materialLayout,
    materialType,
    materialVertexInput,
    materialVertexOutput,
} from "./material-type";
import { Pbr } from "./shade";

const standardLayout = materialLayout(StandardMaterialInput, {
    images: { type: "texture-2d-array" },
    imageSampler: { type: "sampler" },
});
const StandardFragmentContext = materialFragmentContext(StandardMaterialInput);

const standardFragment = tgpu.fn(
    [StandardFragmentContext],
    d.vec4f,
)((ctx) => {
    "use gpu";
    const material = ctx.parameters;
    let baseColor = d.vec4f(material.baseColor);
    const uvDx = std.dpdx(ctx.uv);
    const uvDy = std.dpdy(ctx.uv);
    if (material.baseColorTexture >= 0) {
        baseColor = std.mul(
            baseColor,
            std.textureSampleGrad(
                standardLayout.$.images,
                standardLayout.$.imageSampler,
                ctx.uv,
                material.baseColorTexture,
                uvDx,
                uvDy,
            ),
        );
    }
    if (material.unlit !== 0) return baseColor;
    const pbr = Pbr({
        albedo: baseColor.xyz,
        metallic: material.metallic,
        roughness: material.perceptualRoughness,
        occlusion: material.occlusion,
        dielectric: 0,
        diffuseWrap: material.diffuseWrap,
    });
    const emissive = material.emissive;
    return d.vec4f(std.add(litPbr(pbr, ctx.worldNormal, ctx.world), emissive), baseColor.a);
});

/** The built-in mesh material type. Its type-local table is the sole source of its PBR parameters. */
export const Materials = materialType({
    name: "StandardMaterial",
    layout: standardLayout,
    fragmentInputs: { uv: true },
    fragment: standardFragment,
    defaults: StandardMaterial(),
    depthPass: { prepass: true, shadows: true },
});

const vertexVaryings = { litColor: d.vec4f };
const VertexOutput = materialVertexOutput(vertexVaryings);
const VertexContext = materialFragmentContext(StandardMaterialInput, vertexVaryings);
const vertexVertex = tgpu.fn(
    [materialVertexInput(StandardMaterialInput)],
    VertexOutput,
)((input) => {
    "use gpu";
    const material = input.parameters;
    let litColor = d.vec4f(material.baseColor);
    if (material.unlit === 0) {
        const pbr = Pbr({
            albedo: material.baseColor.xyz,
            metallic: material.metallic,
            roughness: material.perceptualRoughness,
            occlusion: material.occlusion,
            dielectric: 0,
            diffuseWrap: material.diffuseWrap,
        });
        litColor = d.vec4f(
            std.add(
                litPbr(pbr, std.normalize(input.worldNormal), input.world.xyz),
                material.emissive,
            ),
            material.baseColor.a,
        );
    }
    return VertexOutput({ world: input.world, worldNormal: input.worldNormal, litColor });
});
const vertexFragment = tgpu.fn(
    [VertexContext],
    d.vec4f,
)((ctx) => {
    "use gpu";
    return d.vec4f(ctx.litColor);
});
/**
 * Built-in Gouraud material: standard lighting is evaluated once per vertex and interpolated.
 * Ignores baseColorTexture; its interpolated colour uses baseColor alone.
 */
export const VertexMaterialType = materialType({
    name: "VertexMaterial",
    parameters: StandardMaterialInput,
    varyings: vertexVaryings,
    vertex: vertexVertex,
    fragment: vertexFragment,
    defaults: StandardMaterial(),
    depthPass: { prepass: true, shadows: true },
});
