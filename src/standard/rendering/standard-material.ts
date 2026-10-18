import tgpu, { type TgpuFn } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { ImageBindings, sampleImage } from "../../core/rendering";
import { engineLayout, litPbr } from "./engine";
import { StandardMaterial, StandardMaterialInput } from "./material-data";
import {
    AlphaModeCode,
    MaterialAssets,
    type MaterialFragmentFn,
    materialLayout,
    materialType,
} from "./material-type";
import { Pbr } from "./shade";

const standardLayout = materialLayout(StandardMaterialInput, {
    images: { type: "texture-2d-array" },
    imageSampler: { type: "sampler" },
    imageSettings: { type: "texture-2d" },
});
/** The built-in mesh material type. Its type-local table is the sole source of its PBR parameters. */
export const Materials = materialType({
    name: "StandardMaterial",
    layout: standardLayout,
    bindings: (world) => world.resource(ImageBindings),
    fragmentInputs: { uv: true },
    defaults: StandardMaterial(),
    depthPass: { prepass: true, shadows: true },
    fragment: (ctx): d.v4f => {
        "use gpu";
        return applyPbrLighting(pbrInputFromStandardMaterial(ctx));
    },
});

type StandardContextSchema =
    MaterialFragmentFn<typeof StandardMaterialInput> extends TgpuFn<infer F>
        ? Parameters<F>[0]
        : never;
const StandardContext: StandardContextSchema = (
    Materials.fragment as unknown as {
        shell: { argTypes: [StandardContextSchema] };
    }
).shell.argTypes[0];

/** Resolved standard surface values. baseColor includes the sampled texture and sharpened coverage alpha.
 * perceptualRoughness, metallic and occlusion are unitless; emissive is added only for lit surfaces. */
export const PbrInput = d.struct({
    worldPosition: d.vec3f,
    worldNormal: d.vec3f,
    view: d.vec3f,
    baseColor: d.vec4f,
    metallic: d.f32,
    perceptualRoughness: d.f32,
    occlusion: d.f32,
    emissive: d.vec3f,
    unlit: d.u32,
});

/** Initialize a lit white surface from any material fragment context. Normals are world-space;
 * view points from the surface toward the camera. Material terms remain mutable before lighting. */
export function pbrInputFromMaterial(ctx: { world: d.v3f; worldNormal: d.v3f }) {
    "use gpu";
    return PbrInput({
        worldPosition: ctx.world,
        worldNormal: ctx.worldNormal,
        view: std.normalize(std.sub(engineLayout.$.view.eye.xyz, ctx.world)),
        baseColor: d.vec4f(1),
        metallic: 0,
        perceptualRoughness: 0.5,
        occlusion: 1,
        emissive: d.vec3f(0),
        unlit: 0,
    });
}

/** Sample standard's parameters into mutable surface values before lighting. Does not change depth-pass coverage. */
export const pbrInputFromStandardMaterial = tgpu.fn(
    [StandardContext],
    PbrInput,
)((ctx) => {
    "use gpu";
    const material = ctx.parameters;
    let baseColor = d.vec4f(material.baseColor);
    const uvDx = std.dpdx(ctx.uv);
    const uvDy = std.dpdy(ctx.uv);
    if (material.baseColorTexture >= 0) {
        baseColor = std.mul(
            baseColor,
            sampleImage(
                standardLayout.$.images,
                standardLayout.$.imageSampler,
                standardLayout.$.imageSettings,
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
    const input = pbrInputFromMaterial(ctx);
    input.baseColor = d.vec4f(baseColor);
    input.metallic = material.metallic;
    input.perceptualRoughness = material.perceptualRoughness;
    input.occlusion = material.occlusion;
    input.emissive = d.vec3f(material.emissive);
    input.unlit = material.unlit;
    return input;
});

/** Return camera-exposed linear lit colour, including emissive, before fog and presentation.
 * Unlit input returns baseColor unchanged. Alpha is preserved. */
export const applyPbrLighting = tgpu.fn(
    [PbrInput],
    d.vec4f,
)((input) => {
    "use gpu";
    if (input.unlit !== 0) return d.vec4f(input.baseColor);
    const pbr = Pbr({
        albedo: input.baseColor.xyz,
        metallic: input.metallic,
        roughness: input.perceptualRoughness,
        occlusion: input.occlusion,
        dielectric: 0,
    });
    return d.vec4f(
        std.add(litPbr(pbr, input.worldNormal, input.worldPosition, input.view), input.emissive),
        input.baseColor.a,
    );
});

/** Define a standard-derived material with independently owned rows. Shares the base's parameter schema,
 * layout, bindings, fragment context identity and depth passes. The fragment returns camera-exposed
 * linear colour before fog. Extension parameters/bindings and vertex/prepass overrides are not supported. */
export function extendedMaterial(
    base: typeof Materials,
    spec: {
        name: string;
        fragment: ((context: d.InferGPU<typeof StandardContext>) => d.v4f) & {
            readonly resourceType?: never;
        };
    },
): typeof Materials {
    if (!spec.name) throw new Error("material type name must not be empty");
    if ("resourceType" in spec.fragment) {
        throw new Error(
            `material type "${spec.name}" fragment shader must be a plain "use gpu" function, not a prebuilt TypeGPU function`,
        );
    }
    const Context = (base.fragment as unknown as { shell: { argTypes: [StandardContextSchema] } })
        .shell.argTypes[0];
    const definition: typeof Materials = {
        ...base,
        name: spec.name,
        fragment: tgpu.fn([Context], d.vec4f)(spec.fragment),
        create: (world) => new MaterialAssets(world, definition),
    };
    return definition;
}

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
            });
            litColor = d.vec4f(
                std.add(
                    litPbr(
                        pbr,
                        std.normalize(input.worldNormal),
                        input.world.xyz,
                        std.normalize(std.sub(engineLayout.$.view.eye.xyz, input.world.xyz)),
                    ),
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
