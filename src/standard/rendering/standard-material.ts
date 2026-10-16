import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { litPbr } from "./engine";
import { StandardMaterial, StandardMaterialInput } from "./material-data";
import { AlphaModeCode, materialLayout, materialType } from "./material-type";
import { Pbr } from "./shade";

const standardLayout = materialLayout(StandardMaterialInput, {
    images: { type: "texture-2d-array" },
    imageSampler: { type: "sampler" },
});
/** The built-in mesh material type. Its type-local table is the sole source of its PBR parameters. */
export const Materials = materialType({
    name: "StandardMaterial",
    layout: standardLayout,
    fragmentInputs: { uv: true },
    defaults: StandardMaterial(),
    depthPass: { prepass: true, shadows: true },
    fragment: (ctx) => {
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
        const alphaWidth = std.fwidth(baseColor.a);
        if (material.baseColorTexture >= 0 && ctx.alphaMode === AlphaModeCode.AlphaToCoverage) {
            baseColor.a = (baseColor.a - ctx.alphaCutoff) / std.max(alphaWidth, 0.0001) + 0.5;
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
    },
});

const vertexVaryings = { litColor: d.vec4f };
/**
 * Built-in Gouraud material: standard lighting is evaluated once per vertex and interpolated.
 * Ignores baseColorTexture; its interpolated colour uses baseColor alone.
 */
export const VertexMaterialType = materialType({
    name: "VertexMaterial",
    parameters: StandardMaterialInput,
    varyings: vertexVaryings,
    defaults: StandardMaterial(),
    depthPass: { prepass: true, shadows: true },
    vertex: (input) => {
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
        return { world: input.world, worldNormal: input.worldNormal, litColor };
    },
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(ctx.litColor);
    },
});
