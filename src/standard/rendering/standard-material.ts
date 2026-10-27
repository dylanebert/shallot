import tgpu, { type TgpuFn } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { ImageBindings, sampleImage } from "../../core/rendering";
import type { World } from "../../engine";
import { engineLayout, litPbr } from "./engine";
import { StandardMaterial, StandardMaterialInput } from "./material-data";
import {
    AlphaModeCode,
    MaterialAssets,
    type MaterialBinding,
    type MaterialFragmentFn,
    type MaterialLayout,
    type MaterialType,
    type MaterialVertexFn,
    materialLayout,
    materialType,
    materialVertexInput,
    materialVertexOutput,
} from "./material-type";
import type { BindResource } from "./pipelines";
import { Pbr } from "./shade";

const standardLayout = materialLayout(StandardMaterialInput, {
    images: { type: "texture", dimension: "2d-array" },
    imageSampler: { type: "sampler" },
    imageSettings: { type: "texture" },
});
const standardImages = tgpu.accessor(d.texture2dArray(d.f32), () => standardLayout.$.images);
const standardSampler = tgpu.accessor(d.sampler(), () => standardLayout.$.imageSampler);
const standardImageSettings = tgpu.accessor(
    d.texture2d(d.f32),
    () => standardLayout.$.imageSettings,
);
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

/** Sample standard's parameters into mutable surface values before lighting; a doubleSided row's back
 * face takes the flipped normal. Does not change depth-pass coverage. */
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
                standardImages.$,
                standardSampler.$,
                standardImageSettings.$,
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
    // glTF doubleSided: a back face shades with the reversed normal.
    if (material.doubleSided !== 0 && !ctx.frontFacing)
        input.worldNormal = std.mul(input.worldNormal, -1);
    input.baseColor = d.vec4f(baseColor);
    input.metallic = material.metallic;
    input.perceptualRoughness = material.perceptualRoughness;
    input.occlusion = material.occlusion;
    input.emissive = d.vec3f(material.emissive);
    input.unlit = material.unlit;
    return input;
});

/** Return camera-exposed linear lit colour, including emissive, before fog and presentation.
 * Unlit input returns baseColor unchanged. Alpha is preserved. In every depth pass, lit input
 * returns emissive alone, and the normal prepass writes its worldNormal as the fragment's prepass
 * normal; see `lit`. */
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

type StandardParameters = typeof StandardMaterialInput.propTypes;
type ExtendedParameters<E extends d.AnyWgslStruct> = d.WgslStruct<
    StandardParameters & E["propTypes"]
>;
type StandardBindings = typeof Materials.layout.bindingSpecs;
export type ExtensionLayout<
    B extends Record<string, MaterialBinding>,
    E extends d.AnyWgslStruct,
> = MaterialLayout<StandardBindings & B, ExtendedParameters<E>> & { readonly extension: E };

/** Extend standard's group 2 with bindings numbered from 100 and parameters folded into its row.
 * Base binding numbers are preserved in colour and depth layouts. Refuses parameter/binding name
 * collisions and merged bindings exceeding WebGPU's default budgets: one extension storage/attribute
 * per stage, nine sampled textures and four fragment-only storage textures. */
export function extensionLayout<
    E extends d.AnyWgslStruct,
    B extends Record<string, MaterialBinding>,
>(base: typeof Materials, parameters: E, bindings: B): ExtensionLayout<B, E> {
    for (const name of Object.keys(parameters.propTypes)) {
        if (name in base.parameters.propTypes)
            throw new Error(`extension parameter "${name}" collides with a base parameter`);
    }
    for (const name of Object.keys(bindings)) {
        if (name in base.layout.entries)
            throw new Error(`extension binding "${name}" collides with a base or reserved binding`);
    }
    const merged = d.struct({ ...base.parameters.propTypes, ...parameters.propTypes });
    const checked = materialLayout(merged, { ...base.layout.bindingSpecs, ...bindings });
    const entries = {
        ...base.layout.entries,
        materialParameters: checked.entries.materialParameters,
    };
    let index = Object.keys(entries).length;
    const padding: Record<string, null> = {};
    while (index < 100) {
        let key = `_extensionPad${index}`;
        while (key in bindings || key in entries || key in padding) key += "_";
        padding[key] = null;
        index++;
    }
    const extensionEntries = Object.fromEntries(
        Object.keys(bindings).map((key) => [key, checked.entries[key]]),
    );
    const color = tgpu.bindGroupLayout({ ...entries, ...padding, ...extensionEntries }).$idx(2);
    const depth = tgpu
        .bindGroupLayout({
            ...entries,
            vertices: checked.depthVariant.entries.vertices,
            ...padding,
            ...extensionEntries,
        })
        .$idx(2);
    return Object.assign(color, {
        parameters: merged,
        bindingSpecs: checked.bindingSpecs,
        attributes: checked.attributes,
        depthVariant: depth,
        extension: parameters,
    }) as ExtensionLayout<B, E>;
}

type StandardVertex =
    MaterialVertexFn<typeof StandardMaterialInput> extends TgpuFn<infer F> ? F : never;
type Plain<F> = F & { readonly resourceType?: never };
type ExtensionSpec<E extends d.AnyWgslStruct> = {
    name: string;
    defaults?: d.InferInput<NoInfer<E>>;
    bindings?: (world: World) => Record<string, BindResource>;
    vertex?: Plain<
        (
            input: d.InferGPU<Parameters<StandardVertex>[0]>,
            extension: d.InferGPU<E>,
        ) => d.InferGPU<ReturnType<StandardVertex>>
    >;
    fragment: Plain<
        (context: d.InferGPU<typeof StandardContext>, extension: d.InferGPU<E>) => d.v4f
    >;
};

/** Define independently owned standard-material rows with optional extension parameters and resources.
 * Parameters share the base row; defaults merge with standard's defaults. Layout bindings start at
 * 100 and live resources supply extension bindings only. Fragment and optional vertex functions receive
 * the unchanged standard input and extension parameters. Refuses empty names, prebuilt TypeGPU
 * functions, name collisions and bindings beyond WebGPU's default budgets: one extension
 * storage/attribute per stage, nine sampled textures and four fragment-only storage textures. */
export function extendedMaterial<
    E extends d.AnyWgslStruct = d.WgslStruct<Record<never, never>>,
    B extends Record<string, MaterialBinding> = Record<never, never>,
>(
    base: typeof Materials,
    spec: ExtensionSpec<E> &
        (
            | { layout: ExtensionLayout<B, E>; parameters?: never }
            | { parameters?: E; layout?: never }
        ),
): MaterialType<ExtendedParameters<E>, StandardBindings & B> {
    if (!spec.name) throw new Error("material type name must not be empty");
    for (const stage of ["fragment", "vertex"] as const) {
        if (spec[stage] && "resourceType" in spec[stage])
            throw new Error(
                `material type "${spec.name}" ${stage} shader must be a plain "use gpu" function, not a prebuilt TypeGPU function`,
            );
    }
    const extension = spec.layout?.extension ?? spec.parameters;
    const layout = spec.layout ?? (extension ? extensionLayout(base, extension, {}) : base.layout);
    const Context = (base.fragment as unknown as { shell: { argTypes: [StandardContextSchema] } })
        .shell.argTypes[0];
    const resources = new WeakMap<World, Record<string, BindResource>>();
    const definition: MaterialType<ExtendedParameters<E>, StandardBindings & B> = {
        ...base,
        name: spec.name,
        layout: layout as ExtensionLayout<B, E>,
        parameters: layout.parameters as ExtendedParameters<E>,
        extension,
        defaults: { ...base.defaults, ...spec.defaults } as d.InferInput<ExtendedParameters<E>>,
        bindings: spec.bindings
            ? (world) => {
                  let merged = resources.get(world);
                  if (!merged) {
                      merged = {};
                      resources.set(world, merged);
                  }
                  return Object.assign(merged, base.bindings?.(world), spec.bindings!(world));
              }
            : base.bindings,
        slots: [
            [standardImages, () => layout.$.images],
            [standardSampler, () => layout.$.imageSampler],
            [standardImageSettings, () => layout.$.imageSettings],
        ],
        fragment: (extension
            ? tgpu.fn([Context, extension], d.vec4f)(spec.fragment)
            : tgpu.fn([Context], d.vec4f)(spec.fragment as any)) as any,
        vertex: spec.vertex
            ? ((extension
                  ? tgpu.fn(
                        [materialVertexInput(StandardMaterialInput), extension],
                        materialVertexOutput(),
                    )(spec.vertex as any)
                  : tgpu.fn(
                        [materialVertexInput(StandardMaterialInput)],
                        materialVertexOutput(),
                    )(spec.vertex as any)) as any)
            : undefined,
        create: (world) => new MaterialAssets(world, definition),
    };
    return definition;
}

const vertexVaryings = { litColor: d.vec4f };
/**
 * Built-in Gouraud material: standard lighting is evaluated once per vertex and interpolated.
 * Does not light point or spot lights or receive sun shadows (litPbr's vertex-stage limit).
 * Ignores baseColorTexture and doubleSided; its interpolated colour uses baseColor alone.
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
        const litColor = applyPbrLighting(
            PbrInput({
                worldPosition: input.world.xyz,
                worldNormal: std.normalize(input.worldNormal),
                view: std.normalize(std.sub(engineLayout.$.view.eye.xyz, input.world.xyz)),
                baseColor: d.vec4f(material.baseColor),
                metallic: material.metallic,
                perceptualRoughness: material.perceptualRoughness,
                occlusion: material.occlusion,
                emissive: d.vec3f(material.emissive),
                unlit: material.unlit,
            }),
        );
        return { world: input.world, worldNormal: input.worldNormal, litColor };
    },
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(ctx.litColor);
    },
});
