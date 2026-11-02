import tgpu, { type TgpuFn } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import type { World } from "../../engine";
import { engineLayout, litPbr } from "./engine";
import type { InstanceData } from "./instance-data";
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
    baseColor: { type: "image", field: "baseColorTexture" },
    normal: { type: "image", field: "normalMapTexture", isSrgb: false },
    metallicRoughness: { type: "image", field: "metallicRoughnessTexture", isSrgb: false },
    occlusion: { type: "image", field: "occlusionTexture", isSrgb: false },
    emissive: { type: "image", field: "emissiveTexture" },
});
const baseColorImage = tgpu.accessor(d.texture2d(d.f32), () => standardLayout.$.baseColor);
const baseColorSampler = tgpu.accessor(d.sampler(), () => standardLayout.$.baseColorSampler);
const normalImage = tgpu.accessor(d.texture2d(d.f32), () => standardLayout.$.normal);
const normalSampler = tgpu.accessor(d.sampler(), () => standardLayout.$.normalSampler);
const metallicRoughnessImage = tgpu.accessor(
    d.texture2d(d.f32),
    () => standardLayout.$.metallicRoughness,
);
const metallicRoughnessSampler = tgpu.accessor(
    d.sampler(),
    () => standardLayout.$.metallicRoughnessSampler,
);
const occlusionImage = tgpu.accessor(d.texture2d(d.f32), () => standardLayout.$.occlusion);
const occlusionSampler = tgpu.accessor(d.sampler(), () => standardLayout.$.occlusionSampler);
const emissiveImage = tgpu.accessor(d.texture2d(d.f32), () => standardLayout.$.emissive);
const emissiveSampler = tgpu.accessor(d.sampler(), () => standardLayout.$.emissiveSampler);
/** The built-in mesh material type. Its type-local table is the sole source of its PBR parameters. */
export const Materials = materialType({
    name: "StandardMaterial",
    layout: standardLayout,
    fragmentInputs: { uv: true, uv1: true, color: true, tangent: true },
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

const transformUv = tgpu.fn(
    [d.vec2f, StandardMaterialInput.propTypes.baseColorTransform],
    d.vec2f,
)((uv, transform) => {
    "use gpu";
    let result = std.mul(uv, d.vec2f(transform.scaleX, transform.scaleY));
    if (transform.rotation !== 0) {
        const c = std.cos(transform.rotation);
        const s = std.sin(transform.rotation);
        result = d.vec2f(c * result.x - s * result.y, s * result.x + c * result.y);
    }
    return std.add(result, d.vec2f(transform.offsetX, transform.offsetY));
});

/** Sample standard's parameters into mutable surface values before lighting; a doubleSided row's back
 * face takes the flipped normal. Does not change depth-pass coverage. */
export const pbrInputFromStandardMaterial = tgpu.fn(
    [StandardContext],
    PbrInput,
)((ctx) => {
    "use gpu";
    const material = ctx.parameters;
    let baseColor = std.mul(material.baseColor, ctx.color);
    const baseUv = transformUv(
        std.select(ctx.uv, ctx.uv1, material.baseColorChannel === 1),
        material.baseColorTransform,
    );
    const normalUv = transformUv(
        std.select(ctx.uv, ctx.uv1, material.normalMapChannel === 1),
        material.normalMapTransform,
    );
    const metalUv = transformUv(
        std.select(ctx.uv, ctx.uv1, material.metallicRoughnessChannel === 1),
        material.metallicRoughnessTransform,
    );
    const occlusionUv = transformUv(
        std.select(ctx.uv, ctx.uv1, material.occlusionChannel === 1),
        material.occlusionTransform,
    );
    const emissiveUv = transformUv(
        std.select(ctx.uv, ctx.uv1, material.emissiveChannel === 1),
        material.emissiveTransform,
    );
    const uvDx = std.dpdx(baseUv);
    const uvDy = std.dpdy(baseUv);
    const normalDx = std.dpdx(normalUv);
    const normalDy = std.dpdy(normalUv);
    const metalDx = std.dpdx(metalUv);
    const metalDy = std.dpdy(metalUv);
    const occlusionDx = std.dpdx(occlusionUv);
    const occlusionDy = std.dpdy(occlusionUv);
    const emissiveDx = std.dpdx(emissiveUv);
    const emissiveDy = std.dpdy(emissiveUv);
    if (material.baseColorTexture >= 0) {
        baseColor = std.mul(
            baseColor,
            std.textureSampleGrad(baseColorImage.$, baseColorSampler.$, baseUv, uvDx, uvDy),
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
    if (material.normalMapTexture >= 0 && ctx.tangent.w !== 0) {
        const tex = std.textureSampleGrad(
            normalImage.$,
            normalSampler.$,
            normalUv,
            normalDx,
            normalDy,
        );
        const normal = ctx.rawWorldNormal;
        const tangent = ctx.tangent.xyz;
        const bitangent = std.mul(std.cross(normal, tangent), ctx.tangent.w);
        const mapped = std.sub(std.mul(tex.xyz, 2), d.vec3f(1));
        mapped.x *= material.normalMapScale;
        mapped.y *= material.normalMapScale;
        input.worldNormal = std.normalize(
            std.add(
                std.add(std.mul(tangent, mapped.x), std.mul(bitangent, mapped.y)),
                std.mul(normal, mapped.z),
            ),
        );
    }
    // glTF doubleSided reverses the complete mapped normal on back faces.
    if (material.doubleSided !== 0 && !ctx.frontFacing)
        input.worldNormal = std.mul(input.worldNormal, -1);
    if (material.metallicRoughnessTexture >= 0) {
        const tex = std.textureSampleGrad(
            metallicRoughnessImage.$,
            metallicRoughnessSampler.$,
            metalUv,
            metalDx,
            metalDy,
        );
        input.perceptualRoughness *= tex.g;
        input.metallic *= tex.b;
    }
    if (material.occlusionTexture >= 0) {
        const tex = std.textureSampleGrad(
            occlusionImage.$,
            occlusionSampler.$,
            occlusionUv,
            occlusionDx,
            occlusionDy,
        );
        input.occlusion *= 1 + material.occlusionStrength * (tex.r - 1);
    }
    if (material.emissiveTexture >= 0) {
        const tex = std.textureSampleGrad(
            emissiveImage.$,
            emissiveSampler.$,
            emissiveUv,
            emissiveDx,
            emissiveDy,
        );
        input.emissive = std.mul(input.emissive, tex.xyz);
    }
    return input;
});

/** Return camera-exposed linear lit color, including emissive, before fog and presentation.
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
 * Base binding numbers are preserved in color and depth layouts. Refuses parameter/binding name
 * collisions and per-stage bindings exceeding WebGPU's default budgets, including the renderer's
 * group-0 and group-1 entries. Transparent pipelines also count registered fog's group-3 entries at
 * preparation. Storage textures are fragment-only. Declare instanceData here when supplying
 * this layout to extendedMaterial; declarations from the base are inherited. */
export function extensionLayout<
    E extends d.AnyWgslStruct,
    B extends Record<string, MaterialBinding>,
>(
    base: typeof Materials,
    parameters: E,
    bindings: B,
    options: { instanceData?: readonly InstanceData[] } = {},
): ExtensionLayout<B, E> {
    for (const name of Object.keys(parameters.propTypes)) {
        if (name in base.parameters.propTypes)
            throw new Error(`extension parameter "${name}" collides with a base parameter`);
    }
    for (const name of Object.keys(bindings)) {
        if (name in base.layout.entries)
            throw new Error(`extension binding "${name}" collides with a base or reserved binding`);
    }
    const merged = d.struct({ ...base.parameters.propTypes, ...parameters.propTypes });
    const declarations = [...(base.instanceData ?? []), ...(options.instanceData ?? [])];
    const checked = materialLayout(
        merged,
        { ...base.layout.bindingSpecs, ...bindings },
        { instanceData: declarations },
    );
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
        Object.entries(checked.entries).filter(([key]) => !(key in base.layout.entries)),
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
        instanceData: declarations,
        bindingSpecs: checked.bindingSpecs,
        attributes: checked.attributes,
        depthVariant: depth,
        extension: parameters,
    }) as unknown as ExtensionLayout<B, E>;
}

type StandardVertex =
    MaterialVertexFn<typeof StandardMaterialInput> extends TgpuFn<infer F> ? F : never;
type Plain<F> = F & { readonly resourceType?: never };
type ExtensionSpec<E extends d.AnyWgslStruct> = {
    name: string;
    /** Without an explicit layout, add component bindings here; otherwise use extensionLayout. */
    instanceData?: readonly InstanceData[];
    vertexTangent?: true;
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
 * functions, name collisions and per-stage bindings beyond WebGPU's default budgets. Budgets include
 * renderer groups 0 and 1; transparent pipelines also count registered fog's group 3 at preparation.
 * Storage textures are fragment-only. */
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
    if (spec.vertexTangent && !spec.vertex)
        throw new Error(`material type "${spec.name}" vertexTangent requires a vertex function`);
    for (const stage of ["fragment", "vertex"] as const) {
        if (spec[stage] && "resourceType" in spec[stage])
            throw new Error(
                `material type "${spec.name}" ${stage} shader must be a plain "use gpu" function, not a prebuilt TypeGPU function`,
            );
    }
    const extension = spec.layout?.extension ?? spec.parameters;
    if (spec.layout && spec.instanceData !== undefined)
        throw new Error(
            `Material type "${spec.name}": declare instanceData in materialLayout (extensionLayout for extensions), not the spec with an explicit layout`,
        );
    const declarations = [...(base.instanceData ?? []), ...(spec.instanceData ?? [])];
    const layout =
        spec.layout ??
        (extension
            ? extensionLayout(base, extension, {}, { instanceData: spec.instanceData })
            : declarations.length
              ? materialLayout(base.parameters, base.layout.bindingSpecs, {
                    instanceData: declarations,
                })
              : base.layout);
    const Context = (base.fragment as unknown as { shell: { argTypes: [StandardContextSchema] } })
        .shell.argTypes[0];
    const resources = new WeakMap<World, Record<string, BindResource>>();
    const definition: MaterialType<ExtendedParameters<E>, StandardBindings & B> = {
        ...base,
        name: spec.name,
        instanceData: layout.instanceData,
        layout: layout as ExtensionLayout<B, E>,
        parameters: layout.parameters as ExtendedParameters<E>,
        extension,
        vertexTangent: spec.vertexTangent,
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
            [baseColorImage, () => layout.$.baseColor],
            [baseColorSampler, () => layout.$.baseColorSampler],
            [normalImage, () => layout.$.normal],
            [normalSampler, () => layout.$.normalSampler],
            [metallicRoughnessImage, () => layout.$.metallicRoughness],
            [metallicRoughnessSampler, () => layout.$.metallicRoughnessSampler],
            [occlusionImage, () => layout.$.occlusion],
            [occlusionSampler, () => layout.$.occlusionSampler],
            [emissiveImage, () => layout.$.emissive],
            [emissiveSampler, () => layout.$.emissiveSampler],
        ],
        fragment: (extension
            ? tgpu.fn([Context, extension], d.vec4f)(spec.fragment)
            : tgpu.fn([Context], d.vec4f)(spec.fragment as any)) as any,
        vertex: spec.vertex
            ? ((extension
                  ? tgpu.fn(
                        [materialVertexInput(StandardMaterialInput), extension],
                        materialVertexOutput({}, spec.vertexTangent),
                    )(spec.vertex as any)
                  : tgpu.fn(
                        [materialVertexInput(StandardMaterialInput)],
                        materialVertexOutput({}, spec.vertexTangent),
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
 * Ignores baseColorTexture and doubleSided; its interpolated color uses baseColor alone.
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
