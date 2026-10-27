import type { TgpuBindGroupLayout, TgpuFn } from "typegpu";
import tgpu, { readFromArrayBuffer, writeToArrayBuffer } from "typegpu";
import type { AnyWgslData, AnyWgslStruct, WgslArray } from "typegpu/data";
import * as d from "typegpu/data";
import { MeshInstance } from "../../core/mesh";
import { Xform } from "../../core/rendering";
import { GlobalTransform } from "../../core/transform";
import type { GpuTable, Plugin, Resource, World } from "../../engine";
import { type ComponentStorage, type LaneFormat, laneLayout, writeLane } from "../../engine/ecs";
import { MeshInstanceInput, MeshMaterial } from "./material-data";

/** Add the material contract to TypeGPU's unfilled-slot error at a pipeline resolution boundary. @internal */
export function rethrowMaterialError(name: string, error: unknown): never {
    const message = String(error);
    for (const read of ["Depth", "Normal"]) {
        if (message.includes(`Missing value for 'slot:prepass${read}'`)) {
            throw new Error(
                `Material type "${name}" requires readsPrepass.${read.toLowerCase()} for prepass${read}`,
                { cause: error },
            );
        }
    }
    throw error;
}

type ShaderStage = "vertex" | "fragment";
const VS_FS: ShaderStage[] = ["vertex", "fragment"];
type Dimension = "1d" | "2d" | "2d-array" | "cube" | "cube-array" | "3d";
type DepthDimension = "2d" | "2d-array" | "cube" | "cube-array";

/** A material-owned resource or mesh attribute made available to its shaders. A texture is a 2d float
 * texture and a sampler filters unless their options say otherwise; a sint or uint texture is read with
 * textureLoad. A texture binds as one view of its dimension over every mip and layer, so a cube texture
 * has six layers and a cube array a multiple of six. A multisampled texture is 2d, read with textureLoad
 * by sample index; a float one is never filtered, so it binds `unfilterable-float`. */
export type MaterialBinding =
    | { type: "uniform"; struct: AnyWgslStruct }
    | { type: "attribute"; element: AnyWgslData }
    | {
          type: "storage";
          element: AnyWgslData;
          access?: "read" | "read_write";
          visibility?: readonly ShaderStage[];
      }
    | {
          type: "texture";
          dimension?: Dimension;
          sampleType?: "float";
          /** False binds `unfilterable-float`, which formats such as r32float require. */
          filterable?: boolean;
          multisampled?: boolean;
      }
    | {
          type: "texture";
          dimension?: Dimension;
          sampleType: "sint" | "uint";
          multisampled?: boolean;
      }
    | {
          type: "texture";
          dimension?: DepthDimension;
          sampleType: "depth";
          multisampled?: boolean;
      }
    | { type: "sampler"; samplerType?: "filtering" | "non-filtering" | "comparison" };

/** The values a binding's option may take, with undefined where it may be absent. */
type OptionOf<B, K extends string> = B extends { [P in K]: infer V }
    ? V
    : B extends { [P in K]?: infer V }
      ? V | undefined
      : undefined;
type TextureEntry<T> = { texture: T; visibility: ShaderStage[] };
type FloatEntry<T> = TextureEntry<T> & { sampleType?: "unfilterable-float" };
type ChannelOf<S> = S extends "sint" ? d.I32 : d.U32;
type SampledOf<D, C extends d.F32 | d.I32 | d.U32> = D extends "1d"
    ? d.WgslTexture1d<C>
    : D extends "2d-array"
      ? d.WgslTexture2dArray<C>
      : D extends "cube"
        ? d.WgslTextureCube<C>
        : D extends "cube-array"
          ? d.WgslTextureCubeArray<C>
          : D extends "3d"
            ? d.WgslTexture3d<C>
            : d.WgslTexture2d<C>;
type DepthOf<D> = D extends "2d-array"
    ? d.WgslTextureDepth2dArray
    : D extends "cube"
      ? d.WgslTextureDepthCube
      : D extends "cube-array"
        ? d.WgslTextureDepthCubeArray
        : d.WgslTextureDepth2d;
type TextureEntryOf<S, D, M> = S extends "depth"
    ? TextureEntry<M extends true ? d.WgslTextureDepthMultisampled2d : DepthOf<D>>
    : S extends "sint" | "uint"
      ? TextureEntry<
            M extends true ? d.WgslTextureMultisampled2d<ChannelOf<S>> : SampledOf<D, ChannelOf<S>>
        >
      : FloatEntry<M extends true ? d.WgslTextureMultisampled2d<d.F32> : SampledOf<D, d.F32>>;
type SamplerEntryOf<T> = T extends "comparison"
    ? { sampler: "comparison"; visibility: ShaderStage[] }
    : {
          sampler: T extends "non-filtering" ? "non-filtering" : "filtering";
          visibility: ShaderStage[];
      };

/** @internal The TypeGPU layout entry a material or background binding declares. */
export type EntryFor<B extends MaterialBinding> = B extends { type: "uniform" }
    ? { uniform: B["struct"]; visibility: ShaderStage[] }
    : B extends { type: "storage" | "attribute" }
      ? {
            storage: (count: number) => WgslArray<B["element"]>;
            access: "mutable" | "readonly";
            visibility: ShaderStage[];
        }
      : B extends { type: "texture" }
        ? TextureEntryOf<
              OptionOf<B, "sampleType">,
              OptionOf<B, "dimension">,
              OptionOf<B, "multisampled">
          >
        : SamplerEntryOf<OptionOf<B, "samplerType">>;

const sampled: Record<Dimension, (channel: d.F32 | d.I32 | d.U32) => d.WgslTexture> = {
    "1d": d.texture1d,
    "2d": d.texture2d,
    "2d-array": d.texture2dArray,
    cube: d.textureCube,
    "cube-array": d.textureCubeArray,
    "3d": d.texture3d,
};
const depth: Record<DepthDimension, () => d.WgslTexture> = {
    "2d": d.textureDepth2d,
    "2d-array": d.textureDepth2dArray,
    cube: d.textureDepthCube,
    "cube-array": d.textureDepthCubeArray,
};

/** @internal */
export function bindingEntry<B extends MaterialBinding>(binding: B): EntryFor<B> {
    switch (binding.type) {
        case "uniform":
            return { uniform: binding.struct, visibility: VS_FS } as EntryFor<B>;
        case "attribute":
            return {
                storage: d.arrayOf(binding.element),
                access: "readonly",
                visibility: VS_FS,
            } as EntryFor<B>;
        case "storage":
            return {
                storage: d.arrayOf(binding.element),
                access: binding.access === "read_write" ? "mutable" : "readonly",
                visibility: binding.visibility
                    ? [...binding.visibility]
                    : binding.access === "read_write"
                      ? ["fragment"]
                      : VS_FS,
            } as EntryFor<B>;
        case "texture": {
            if (binding.sampleType === "depth")
                return {
                    texture: binding.multisampled
                        ? d.textureDepthMultisampled2d()
                        : depth[binding.dimension ?? "2d"](),
                    visibility: VS_FS,
                } as EntryFor<B>;
            const channel =
                binding.sampleType === "sint"
                    ? d.i32
                    : binding.sampleType === "uint"
                      ? d.u32
                      : d.f32;
            return {
                texture: binding.multisampled
                    ? d.textureMultisampled2d(channel)
                    : sampled[binding.dimension ?? "2d"](channel),
                ...(channel === d.f32 &&
                    (binding.multisampled ||
                        ("filterable" in binding && binding.filterable === false)) && {
                        sampleType: "unfilterable-float",
                    }),
                visibility: VS_FS,
            } as EntryFor<B>;
        }
        case "sampler":
            return {
                sampler: binding.samplerType ?? "filtering",
                visibility: VS_FS,
            } as EntryFor<B>;
    }
}

/** A material layout includes its typed parameter table and the mesh streams required by its shaders. */
type MaterialBuiltins = {
    eids: {
        storage: (count: number) => WgslArray<d.Vec4u>;
        access: "readonly";
        visibility: ShaderStage[];
    };
    globalTransforms: {
        storage: (count: number) => WgslArray<typeof Xform>;
        access: "readonly";
        visibility: ShaderStage[];
    };
};
type MaterialStreams<P extends AnyWgslStruct, Vertex extends AnyWgslData> = {
    materialParameters: {
        storage: (count: number) => WgslArray<P>;
        access: "readonly";
        visibility: ShaderStage[];
    };
    meshInstances: {
        storage: (count: number) => WgslArray<typeof MeshInstanceInput>;
        access: "readonly";
        visibility: ShaderStage[];
    };
    vertices: {
        storage: (count: number) => WgslArray<Vertex>;
        access: "readonly";
        visibility: ShaderStage[];
    };
};
export type MaterialLayout<
    B extends Record<string, MaterialBinding>,
    P extends AnyWgslStruct,
> = TgpuBindGroupLayout<
    { [K in keyof B]: EntryFor<B[K]> } & MaterialBuiltins & MaterialStreams<P, d.Vec4u>
> & {
    readonly parameters: P;
    readonly attributes: Readonly<Record<string, AnyWgslData>>;
    readonly depthVariant: TgpuBindGroupLayout<
        { [K in keyof B]: EntryFor<B[K]> } & MaterialBuiltins & MaterialStreams<P, d.Vec2u>
    >;
};

const meshInstancesEntry = {
    storage: d.arrayOf(MeshInstanceInput),
    access: "readonly" as const,
    visibility: VS_FS,
};
const colorVerticesEntry = {
    storage: d.arrayOf(d.vec4u),
    access: "readonly" as const,
    visibility: VS_FS,
};
const depthVerticesEntry = {
    storage: d.arrayOf(d.vec2u),
    access: "readonly" as const,
    visibility: VS_FS,
};

/** Declare owned bindings before shader functions so they can close over the layout's typed resources. */
export function materialLayout<B extends Record<string, MaterialBinding>, P extends AnyWgslStruct>(
    parameters: P,
    bindings: B,
): MaterialLayout<B, P> {
    const reserved = [
        "eids",
        "globalTransforms",
        "meshInstances",
        "vertices",
        "materialParameters",
    ];
    for (const name of reserved) {
        if (name in bindings)
            throw new Error(`material binding "${name}" is reserved by the mesh renderer`);
    }
    // Five built-in group-2 storage bindings (eids, transforms, parameters, instances, vertices), plus
    // at most one owned binding and the two group-0 storage slots, stay within WebGPU's eight per stage.
    for (const stage of ["vertex", "fragment"] as const) {
        const count =
            5 +
            Object.values(bindings).filter(
                (binding) =>
                    binding.type === "attribute" ||
                    (binding.type === "storage" &&
                        (
                            binding.visibility ??
                            (binding.access === "read_write" ? ["fragment"] : VS_FS)
                        ).includes(stage)),
            ).length;
        if (count + 2 > 8) {
            throw new Error(
                `material layout uses ${count + 2} ${stage} storage buffers; WebGPU's limit is 8`,
            );
        }
    }
    // Group 0 contributes AO and two prepass textures; group 1 contributes two shadow atlases.
    const textures = Object.values(bindings).filter((binding) => binding.type === "texture").length;
    if (textures + 5 > 16) {
        throw new Error(
            `material layout uses ${textures + 5} fragment sampled textures; WebGPU's limit is 16`,
        );
    }
    const builtins = {
        eids: bindingEntry({ type: "storage", element: d.vec4u }),
        globalTransforms: bindingEntry({ type: "storage", element: Xform }),
    };
    const own = {
        ...builtins,
        ...Object.fromEntries(
            Object.entries(bindings).map(([name, binding]) => [name, bindingEntry(binding)]),
        ),
    };
    const materialParameters = {
        storage: d.arrayOf(parameters),
        access: "readonly" as const,
        visibility: VS_FS,
    };
    const color = tgpu
        .bindGroupLayout({
            ...own,
            materialParameters,
            meshInstances: meshInstancesEntry,
            vertices: colorVerticesEntry,
        })
        .$idx(2);
    const depth = tgpu
        .bindGroupLayout({
            ...own,
            materialParameters,
            meshInstances: meshInstancesEntry,
            vertices: depthVerticesEntry,
        })
        .$idx(2);
    const attributes = Object.fromEntries(
        Object.entries(bindings)
            .filter(([, binding]) => binding.type === "attribute")
            .map(([name, binding]) => [name, (binding as { element: AnyWgslData }).element]),
    );
    return Object.assign(color, { parameters, attributes, depthVariant: depth }) as MaterialLayout<
        B,
        P
    >;
}

/** Material vertex fields. eid is the entity id, or 0xFFFFFFFF for a producer row.
 * row is the index within a producer table, or 0xFFFFFFFF for an entity. */
export const MeshVertexInput = d
    .struct({
        localPos: d.vec3f,
        localNormal: d.vec3f,
        uv: d.vec2f,
        vidx: d.u32,
        eid: d.u32,
        row: d.u32,
        iid: d.u32,
        xform: Xform,
        world: d.vec4f,
        worldNormal: d.vec3f,
        material: d.u32,
    })
    .$name("MeshVertexInput");

/** Vertex input with this material's typed parameter row. */
function materialVertexInput<P extends AnyWgslStruct>(parameters: P) {
    return d.struct({ parameters, ...MeshVertexInput.propTypes });
}

/** World-space fields a material vertex function may override, plus its declared varyings. */
function materialVertexOutput<V extends Record<string, AnyWgslData> = Record<never, never>>(
    varyings: V = {} as V,
) {
    return d.struct({ world: d.vec4f, worldNormal: d.vec3f, ...varyings });
}

/** Fragment context with typed parameters and varyings. eid is the entity id, or 0xFFFFFFFF
 * for a producer row; row is the producer-table index, or 0xFFFFFFFF for an entity. */
function materialFragmentContext<
    P extends AnyWgslStruct,
    V extends Record<string, AnyWgslData> = Record<never, never>,
>(parameters: P, varyings: V = {} as V) {
    return d.struct({
        parameters,
        eid: d.u32,
        row: d.u32,
        world: d.vec3f,
        worldNormal: d.vec3f,
        uv: d.vec2f,
        localPos: d.vec3f,
        material: d.u32,
        alphaMode: d.u32,
        alphaCutoff: d.f32,
        /** Whether the fragment lies on its triangle's front face, counter-clockwise in the pass's view.
         * Constant true where the row culls back faces and false where it culls front faces; read from
         * `@builtin(front_facing)` only where it culls none, which counts one inter-stage variable. */
        frontFacing: d.bool,
        /** Framebuffer pixel centres in the view's render target at its Resolution, origin top-left.
         * z and w retain WGSL @builtin(position) semantics: depth and interpolated reciprocal clip w.
         * Depth and shadow passes use their own render target's coordinates. */
        fragCoord: d.vec4f,
        ...varyings,
    });
}

export type MaterialVertexFn<
    P extends AnyWgslStruct,
    V extends Record<string, AnyWgslData> = Record<never, never>,
> = TgpuFn<
    (input: ReturnType<typeof materialVertexInput<P>>) => ReturnType<typeof materialVertexOutput<V>>
>;
export type MaterialFragmentFn<
    P extends AnyWgslStruct,
    V extends Record<string, AnyWgslData> = Record<never, never>,
> = TgpuFn<(context: ReturnType<typeof materialFragmentContext<P, V>>) => d.Vec4f>;

export type AlphaMode =
    | { readonly kind: "opaque" }
    | { readonly kind: "mask"; readonly cutoff: number }
    | { readonly kind: "blend" }
    | { readonly kind: "premultiplied" }
    | { readonly kind: "add" }
    | { readonly kind: "multiply" }
    | { readonly kind: "alpha-to-coverage" };

/**
 * Per-material transparency behavior; Mask defaults to the 0.5 cutoff. A camera stacked over others
 * (Camera `order`) is blended over them by its scene alpha as coverage: Opaque and Mask cover fully,
 * Blend and Premultiplied by alpha, and AlphaToCoverage there covers round(alpha × 4) of the 4× MSAA
 * samples, each at full alpha, where a lone or bottom camera uses the GPU's alpha-to-coverage. On a
 * stacked camera, Add's light over pixels it left empty is dropped, and Multiply darkens the cameras
 * below toward black by its alpha instead of multiplying them.
 */
export const AlphaMode = {
    Opaque: { kind: "opaque" } as const,
    Mask: (cutoff = 0.5): Extract<AlphaMode, { kind: "mask" }> => ({ kind: "mask", cutoff }),
    Blend: { kind: "blend" } as const,
    Premultiplied: { kind: "premultiplied" } as const,
    Add: { kind: "add" } as const,
    Multiply: { kind: "multiply" } as const,
    AlphaToCoverage: { kind: "alpha-to-coverage" } as const,
};

/** Values carried in the mesh-instance table; Premultiplied and Add share one pipeline. */
export const AlphaModeCode = {
    Opaque: 0,
    Mask: 1,
    Blend: 2,
    Premultiplied: 3,
    Add: 4,
    Multiply: 5,
    AlphaToCoverage: 6,
} as const;

export const AlphaPipelineKey = {
    Opaque: 0,
    Mask: 1,
    Blend: 2,
    Premultiplied: 3,
    Multiply: 4,
    AlphaToCoverage: 5,
} as const;
export const ALPHA_PIPELINE_KEY_COUNT = 6;

/** Values carried in the mesh-instance table's `cullMode`, from a row's `GPUCullMode`. */
export const CullModeCode = { Back: 0, None: 1, Front: 2 } as const;
export type CullModeCode = (typeof CullModeCode)[keyof typeof CullModeCode];

export function cullModeCode(mode: GPUCullMode): CullModeCode {
    switch (mode) {
        case "back":
            return CullModeCode.Back;
        case "none":
            return CullModeCode.None;
        case "front":
            return CullModeCode.Front;
        default:
            throw new RangeError(`cullMode must be "back", "none" or "front"`);
    }
}

/** One pipeline variant: its alpha pipeline key within the band of its CullModeCode. */
export function pipelineKey(alphaPipelineKey: number, cullMode: number): number {
    return alphaPipelineKey + ALPHA_PIPELINE_KEY_COUNT * cullMode;
}

/** Keys per material type in pair indices: one alpha band per cull code up to the highest any
 * type's `pipelineKeys` holds, so worlds whose rows cull only back faces keep six. @internal */
export function pipelineKeyCount(sources: readonly (MaterialSource<any> | undefined)[]): number {
    let keys = 0;
    for (const source of sources) keys |= source?.pipelineKeys ?? 0;
    const highest = keys === 0 ? 0 : 31 - Math.clz32(keys);
    return ALPHA_PIPELINE_KEY_COUNT * (1 + Math.floor(highest / ALPHA_PIPELINE_KEY_COUNT));
}

export function alphaModeFields(mode: AlphaMode): { alphaMode: number; alphaCutoff: number } {
    switch (mode.kind) {
        case "opaque":
            return { alphaMode: AlphaModeCode.Opaque, alphaCutoff: 0.5 };
        case "mask":
            if (!Number.isFinite(mode.cutoff) || mode.cutoff < 0 || mode.cutoff > 1) {
                throw new RangeError(`AlphaMode.Mask cutoff must be between 0 and 1`);
            }
            return { alphaMode: AlphaModeCode.Mask, alphaCutoff: mode.cutoff };
        case "blend":
            return { alphaMode: AlphaModeCode.Blend, alphaCutoff: 0.5 };
        case "premultiplied":
            return { alphaMode: AlphaModeCode.Premultiplied, alphaCutoff: 0.5 };
        case "add":
            return { alphaMode: AlphaModeCode.Add, alphaCutoff: 0.5 };
        case "multiply":
            return { alphaMode: AlphaModeCode.Multiply, alphaCutoff: 0.5 };
        case "alpha-to-coverage":
            return { alphaMode: AlphaModeCode.AlphaToCoverage, alphaCutoff: 0.5 };
    }
}

export function alphaPipelineKey(alphaMode: number): number {
    switch (alphaMode) {
        case AlphaModeCode.Opaque:
            return AlphaPipelineKey.Opaque;
        case AlphaModeCode.Mask:
            return AlphaPipelineKey.Mask;
        case AlphaModeCode.Blend:
            return AlphaPipelineKey.Blend;
        case AlphaModeCode.Premultiplied:
        case AlphaModeCode.Add:
            return AlphaPipelineKey.Premultiplied;
        case AlphaModeCode.Multiply:
            return AlphaPipelineKey.Multiply;
        case AlphaModeCode.AlphaToCoverage:
            return AlphaPipelineKey.AlphaToCoverage;
        default:
            throw new RangeError(`unknown alpha mode ${alphaMode}`);
    }
}

export type AlphaPipelineKey = (typeof AlphaPipelineKey)[keyof typeof AlphaPipelineKey];

/** A material's externally owned parameter table and pipeline inputs.
 * The owner fills and uploads its table through World's table schedule, and publishes each consumer's
 * alphaMode/alphaCutoff/cullMode into meshInstanceTable after upload and before MeshPreprocessSystem.
 * alphaMode and cullMode read the selected row for CPU transparent sorting; cullMode omitted is
 * CullModeCode.Back. pipelineKeys is the growing bitset of `1 << pipelineKey(alphaPipelineKey,
 * cullMode)` values the type can draw. Rows remain valid for the source's stated lifetime.
 * cullBounds, when supplied, reads the selected row and writes a local-space [x, y, z, radius] sphere
 * into caller-owned scratch during transparent queuing. Shared consumers use the same row's geometry;
 * geometry-changing shaders must cover their expanded geometry here too. */
export interface MaterialSource<P extends AnyWgslStruct = AnyWgslStruct> {
    readonly table: GpuTable<P>;
    readonly pipelineKeys: number;
    alphaMode(row: number): number;
    cullMode?(row: number): number;
    cullBounds?(row: number, out: Float32Array): void;
}

/** The shader, parameter schema, and depth behavior shared by every instance of one mesh material type. */
export interface MaterialType<
    P extends AnyWgslStruct = AnyWgslStruct,
    B extends Record<string, MaterialBinding> = Record<string, MaterialBinding>,
    V extends Record<string, AnyWgslData> = Record<never, never>,
> extends Resource<MaterialAssets<P>> {
    readonly name: string;
    readonly parameters: P;
    readonly layout: MaterialLayout<B, P>;
    /** Omit for MaterialAssets; otherwise the renderer borrows this source. Accessing the type's asset
     * allocator refuses: resolve this resource to access its externally owned rows. */
    readonly source?: Resource<MaterialSource<P>>;
    /** Live resources owned by this material type; called at draw to observe replacements.
     * Mesh attributes and mesh.bindings take precedence. Built-ins resolve from their engine owners. */
    readonly bindings?: (world: World) => Record<string, import("./pipelines").BindResource>;
    /** Request this view's prepass this frame while this type has draws.
     * depth reads reverse-Z device depth; normal reads a world-space unit normal.
     * Undeclared prepassDepth/prepassNormal calls are refused during pipeline preparation.
     * A camera without the prepass this frame and depth passes read 0 / vec3f(0).
     * At 4x antialiasing both reads use sample 0. */
    readonly readsPrepass?: { depth?: true; normal?: true };
    readonly fragmentInputs?: { uv?: true; localPos?: true };
    /** Each varying takes one inter-stage variable after seven engine variables and uv/localPos
     * when declared. Pipelines exceeding the device's maxInterStageShaderVariables are refused
     * when prepared. */
    readonly varyings?: V;
    readonly vertex?: MaterialVertexFn<P, V>;
    readonly fragment: MaterialFragmentFn<P, V>;
    readonly depthPass?: { prepass?: boolean; shadows?: boolean };
    readonly defaults?: d.InferInput<P>;
}

/** A material handle can be added directly as `MeshMaterial`; alpha and cull fields mirror its live row. */
export interface MaterialHandle {
    readonly type: number;
    readonly material: number;
    readonly alphaMode: number;
    readonly alphaCutoff: number;
    readonly cullMode: number;
}

/** Pipeline options of one material row. cullMode is the faces the row does not draw in any pass,
 * shadows and prepasses included, as GPUCullMode names them; it defaults to "back". */
export interface MaterialRowOptions {
    alphaMode?: AlphaMode;
    cullMode?: GPUCullMode;
}

/** CPU-authored material assets with a lazily published GPU table using the type's parameter schema.
 * Externally sourced material types refuse construction; their rows belong to the declared source. */
export class MaterialAssets<P extends AnyWgslStruct = AnyWgslStruct> {
    private readonly _world: World;
    private readonly _type: MaterialType<P, any, any>;
    private readonly _rowBytes: number;
    private readonly _scratch: ArrayBuffer;
    private readonly _freeRows: number[] = [];
    private readonly _fields: readonly ParameterField[];
    private _bytes: Uint8Array;
    private _view: DataView;
    private _alphaModes = new Uint32Array([AlphaModeCode.Opaque]);
    private _alphaCutoffs = new Float32Array([0.5]);
    private _cullModes = new Uint32Array([CullModeCode.Back]);
    private readonly _handles: (MaterialHandle | undefined)[] = [undefined];
    private _highWater = 1;
    // row zero starts Opaque and culls back faces, as do rows reserved past the high-water mark
    private _pipelineKeys = 1 << AlphaPipelineKey.Opaque;
    private _table: GpuTable<P> | undefined;

    constructor(world: World, type: MaterialType<P, any, any>) {
        if (type.source)
            throw new Error(
                `MaterialAssets "${type.name}" cannot allocate assets: the material type uses an external parameter source`,
            );
        this._world = world;
        this._type = type;
        const id = registerMaterialType(world, type, type.name === "StandardMaterial");
        world.resource(MaterialTypes).assets[id] = this;
        world.resource(MaterialTypes).sources[id] = this;
        this._rowBytes = d.sizeOf(type.parameters);
        this._scratch = new ArrayBuffer(this._rowBytes);
        this._bytes = new Uint8Array(this._rowBytes);
        this._view = new DataView(this._bytes.buffer);
        this._fields = Object.entries(type.parameters.propTypes as Record<string, AnyWgslData>).map(
            ([name, field]) => {
                const layout = laneLayout(field);
                return {
                    name,
                    type: field,
                    offset: d.memoryLayoutOf(type.parameters, (row: any) => row[name]).offset,
                    size: d.sizeOf(field),
                    format: layout?.format,
                    lanes: layout?.lanes ?? 0,
                };
            },
        );
        writeToArrayBuffer(this._scratch, type.parameters, type.defaults ?? ({} as never));
        this._bytes.set(new Uint8Array(this._scratch), 0);
    }

    /** The renderer initializes this table at warm so its default row is uploaded before the first draw. */
    get table(): GpuTable<P> {
        if (!this._table) {
            const table = this._world.table(`material:${this._type.name}`, this._type.parameters);
            table.reserveSlots(this._highWater);
            table.bytes.set(this._bytes.subarray(0, this._highWater * this._rowBytes));
            table.markRange(0, this._highWater);
            this._table = table;
        }
        return this._table;
    }

    private reserveRows(rows: number): void {
        if (rows > this._bytes.length / this._rowBytes) {
            let capacity = Math.max(1, this._bytes.length / this._rowBytes);
            while (capacity < rows) capacity *= 2;
            const bytes = new Uint8Array(capacity * this._rowBytes);
            bytes.set(this._bytes);
            this._bytes = bytes;
            this._view = new DataView(bytes.buffer);
            const alphaModes = new Uint32Array(capacity);
            alphaModes.set(this._alphaModes);
            this._alphaModes = alphaModes;
            const alphaCutoffs = new Float32Array(capacity).fill(0.5);
            alphaCutoffs.set(this._alphaCutoffs);
            this._alphaCutoffs = alphaCutoffs;
            const cullModes = new Uint32Array(capacity);
            cullModes.set(this._cullModes);
            this._cullModes = cullModes;
        }
        this._highWater = Math.max(this._highWater, rows);
    }

    private publish(firstRow: number, count: number): void {
        if (!this._table || count === 0) return;
        const start = firstRow * this._rowBytes;
        const end = (firstRow + count) * this._rowBytes;
        this._table.reserveSlots(this._highWater);
        // A byte copy: a subarray view would allocate on every per-frame update.
        const target = this._table.bytes;
        const source = this._bytes;
        for (let i = start; i < end; i++) target[i] = source[i];
        this._table.markRange(firstRow, count);
    }

    /** Add a material value and return its type and type-local row for a MeshMaterial component.
     * A row's first use of an alpha mode and cull mode pair after warm compiles its pipelines at draw. */
    add(values: d.InferInput<P>, options: MaterialRowOptions = {}): MaterialHandle {
        const alphaMode = options.alphaMode ?? AlphaMode.Opaque;
        alphaModeFields(alphaMode);
        const cullMode = cullModeCode(options.cullMode ?? "back");
        const row = this._freeRows.pop() ?? this._highWater;
        this.reserveRows(row + 1);
        writeToArrayBuffer(this._scratch, this._type.parameters, values);
        this._bytes.set(new Uint8Array(this._scratch), row * this._rowBytes);
        this._writePipeline(row, alphaModeFields(alphaMode), cullMode);
        this.publish(row, 1);
        const handle = {
            type: materialTypeId(this._world, this._type),
            material: row,
            ...this.alpha(row),
            cullMode: this._cullModes[row],
        };
        this._handles[row] = handle;
        return handle;
    }

    /** Per-material alpha state consumed by mesh-instance preprocessing. */
    alpha(row: number): { alphaMode: number; alphaCutoff: number } {
        if (!Number.isSafeInteger(row) || row < 0 || row >= this._highWater) {
            throw new RangeError(`Materials.alpha: unknown ${this._type.name} row ${row}`);
        }
        return { alphaMode: this._alphaModes[row], alphaCutoff: this._alphaCutoffs[row] };
    }

    /** AlphaModeCode at one asset row, without allocating an alpha record. */
    alphaMode(row: number): number {
        return this._alphaModes[row];
    }

    /** CullModeCode at one asset row. */
    cullMode(row: number): number {
        return this._cullModes[row];
    }

    /** Bit `1 << pipelineKey(alphaPipelineKey, cullMode)` for each pair any of this type's rows has
     * held; it only grows. */
    get pipelineKeys(): number {
        return this._pipelineKeys;
    }

    private _writePipeline(
        row: number,
        fields: { alphaMode: number; alphaCutoff: number },
        cullMode: number,
    ): void {
        if (
            this._alphaModes[row] === fields.alphaMode &&
            this._alphaCutoffs[row] === fields.alphaCutoff &&
            this._cullModes[row] === cullMode
        )
            return;
        this._alphaModes[row] = fields.alphaMode;
        this._alphaCutoffs[row] = fields.alphaCutoff;
        this._cullModes[row] = cullMode;
        this._pipelineKeys |= 1 << pipelineKey(alphaPipelineKey(fields.alphaMode), cullMode);
        const handle = this._handles[row];
        if (handle) Object.assign(handle, fields, { cullMode });
        const declaresMeshMaterial = [...this._world.registry.entries()].some(
            ({ component }) => component === MeshMaterial,
        );
        if (declaresMeshMaterial) copyMeshAlpha(this._world, true);
    }

    /** Copy this type's row alpha and cull into the entity's MeshMaterial where they differ; an unknown row is left. @internal */
    copyAlpha(storage: MeshMaterialStorage, eid: number): void {
        const row = storage.material.get(eid);
        if (row >= this._highWater) return;
        const alphaMode = this._alphaModes[row];
        const alphaCutoff = this._alphaCutoffs[row];
        const cullMode = this._cullModes[row];
        if (storage.alphaMode.get(eid) !== alphaMode) storage.alphaMode.set(eid, alphaMode);
        if (storage.alphaCutoff.get(eid) !== alphaCutoff) storage.alphaCutoff.set(eid, alphaCutoff);
        if (storage.cullMode.get(eid) !== cullMode) storage.cullMode.set(eid, cullMode);
    }

    /**
     * Release an unused row so producers with entity lifetimes can reuse its storage. Row zero is reserved;
     * a handle whose row it does not currently hold, such as one already removed, is refused.
     */
    remove(handle: MaterialHandle): void {
        if (handle.type !== materialTypeId(this._world, this._type)) {
            throw new RangeError("Materials.remove: handle belongs to another material type");
        }
        if (
            !Number.isSafeInteger(handle.material) ||
            handle.material <= 0 ||
            handle.material >= this._highWater
        ) {
            throw new RangeError(
                `Materials.remove: unknown ${this._type.name} row ${handle.material}`,
            );
        }
        if (this._handles[handle.material] !== handle) {
            throw new RangeError(
                `Materials.remove: ${this._type.name} row ${handle.material} is not held by this handle`,
            );
        }
        this._handles[handle.material] = undefined;
        this._freeRows.push(handle.material);
    }

    /** Write per-entity data at a stable eid-indexed row, filling intervening rows with defaults. */
    setAt(row: number, values: d.InferInput<P>, options: MaterialRowOptions = {}): void {
        if (!Number.isSafeInteger(row) || row < 0) {
            throw new RangeError(`Materials.setAt: invalid ${this._type.name} row ${row}`);
        }
        if (options.alphaMode) alphaModeFields(options.alphaMode);
        const cullMode = options.cullMode && cullModeCode(options.cullMode);
        const previousHighWater = this._highWater;
        if (row >= previousHighWater) {
            this.reserveRows(row + 1);
            for (let index = previousHighWater; index <= row; index++) {
                writeToArrayBuffer(
                    this._scratch,
                    this._type.parameters,
                    this._type.defaults ?? ({} as never),
                );
                this._bytes.set(new Uint8Array(this._scratch), index * this._rowBytes);
            }
        }
        writeToArrayBuffer(this._scratch, this._type.parameters, values);
        this._bytes.set(new Uint8Array(this._scratch), row * this._rowBytes);
        this._writeOptions(row, options.alphaMode, cullMode);
        this.publish(Math.min(row, previousHighWater), Math.max(1, row - previousHighWater + 1));
    }

    /**
     * The values at one type-local row, as a fresh record that add and update accept; vectors keep
     * their authored length. Undefined for a row outside this type's rows, or a handle that no longer
     * holds its row; a handle of another material type is refused. Reads the values add, setAt and
     * update wrote, before or after upload. Allocates the record: not a per-frame read. A MeshMaterial
     * value read from an entity is a copy, not the handle: pass its `material` row.
     */
    get(handle: number | MaterialHandle): d.Infer<P> | undefined {
        let row: number;
        if (typeof handle === "number") {
            if (!Number.isSafeInteger(handle) || handle < 0 || handle >= this._highWater)
                return undefined;
            row = handle;
        } else {
            if (handle.type !== materialTypeId(this._world, this._type)) {
                throw new RangeError("Materials.get: handle belongs to another material type");
            }
            if (this._handles[handle.material] !== handle) return undefined;
            row = handle.material;
        }
        const offset = row * this._rowBytes;
        return readFromArrayBuffer(
            this._bytes.slice(offset, offset + this._rowBytes).buffer,
            this._type.parameters,
        ) as d.Infer<P>;
    }

    /**
     * Publish changed fields at one type-local row for the next frame upload, writing each given field in
     * place; fields left out or undefined keep their values. Scalar and vector f32, i32 and u32 fields are
     * written directly; other field types, such as f16, matrices, arrays and structs, are encoded by TypeGPU,
     * which allocates. A handle whose row it does not currently hold, such as one already removed, is
     * refused, as is a MeshMaterial value read from an entity, which is a copy: pass its `material` row.
     */
    update(
        handle: number | MaterialHandle,
        values: Partial<d.InferInput<P>>,
        options?: MaterialRowOptions,
    ): void {
        const row = typeof handle === "number" ? handle : handle.material;
        if (typeof handle !== "number" && handle.type !== materialTypeId(this._world, this._type)) {
            throw new RangeError(`Materials.update: handle belongs to another material type`);
        }
        if (!Number.isSafeInteger(row) || row < 0 || row >= this._highWater) {
            throw new RangeError(`Materials.update: unknown ${this._type.name} row ${row}`);
        }
        if (typeof handle !== "number" && this._handles[row] !== handle) {
            throw new RangeError(
                `Materials.update: ${this._type.name} row ${row} is not held by this handle`,
            );
        }
        const alphaMode = options?.alphaMode;
        if (alphaMode) alphaModeFields(alphaMode);
        const cullMode = options?.cullMode && cullModeCode(options.cullMode);
        const base = row * this._rowBytes;
        const given = values as Record<string, unknown>;
        for (let i = 0; i < this._fields.length; i++) {
            const field = this._fields[i];
            // Test presence first: loading every field name would make the load site megamorphic,
            // and a megamorphic load boxes a double value.
            if (!(field.name in given)) continue;
            const value = given[field.name];
            if (value === undefined) continue;
            const offset = base + field.offset;
            if (field.format === undefined) {
                writeToArrayBuffer(this._bytes.buffer as ArrayBuffer, field.type, value as never, {
                    startOffset: offset,
                    endOffset: offset + field.size,
                });
            } else if (field.lanes === 1) {
                writeLane(this._view, offset, field.format, value as number);
            } else {
                const lanes = value as ArrayLike<number>;
                for (let lane = 0; lane < field.lanes; lane++)
                    writeLane(this._view, offset + lane * 4, field.format, lanes[lane]);
            }
        }
        this._writeOptions(row, alphaMode, cullMode);
        this.publish(row, 1);
    }

    /** Write the given options at a row, keeping the ones left out. */
    private _writeOptions(row: number, alphaMode?: AlphaMode, cullMode?: number): void {
        if (!alphaMode && cullMode === undefined) return;
        this._writePipeline(
            row,
            alphaMode
                ? alphaModeFields(alphaMode)
                : { alphaMode: this._alphaModes[row], alphaCutoff: this._alphaCutoffs[row] },
            cullMode ?? this._cullModes[row],
        );
    }
}

/** A parameter field's place in a row; `format` is undefined where TypeGPU writes the field. */
interface ParameterField {
    readonly name: string;
    readonly type: AnyWgslData;
    readonly offset: number;
    readonly size: number;
    readonly format: LaneFormat | undefined;
    readonly lanes: number;
}

type ErasedMaterialType = MaterialType<any, any, any>;
type MeshMaterialStorage = ComponentStorage<typeof MeshMaterial>;

interface MaterialTypeState {
    readonly types: (ErasedMaterialType | undefined)[];
    /** each registered type's CPU store, by type id, once created */
    readonly assets: (MaterialAssets<any> | undefined)[];
    /** Renderer inputs by type id; an external source keeps ownership of its table. */
    readonly sources: (MaterialSource<any> | undefined)[];
    readonly ids: WeakMap<object, number>;
    readonly names: Map<string, ErasedMaterialType>;
}
const materialTypesKey = {
    create: (): MaterialTypeState => ({
        types: [undefined],
        assets: [],
        sources: [],
        ids: new WeakMap(),
        names: new Map(),
    }),
};

const MESH_MATERIALS = [MeshMaterial];
/** pipelineKeys bits of the blended alpha keys in every cull band. @internal */
export const TRANSPARENT_KEYS = [
    CullModeCode.Back,
    CullModeCode.None,
    CullModeCode.Front,
].reduce<number>(
    (keys, cull) =>
        keys |
        (1 << pipelineKey(AlphaPipelineKey.Blend, cull)) |
        (1 << pipelineKey(AlphaPipelineKey.Premultiplied, cull)) |
        (1 << pipelineKey(AlphaPipelineKey.Multiply, cull)),
    0,
);

/** Standard's upload-time classification; no ECS membership changes during table gathering. @internal */
export const TransparentMeshes = {
    create(world: World) {
        const candidates = new Set<number>();
        const external: Set<number>[] = [];
        const keys: number[] = [];
        const storage = world.storage(MeshMaterial);
        const { types, sources } = world.resource(MaterialTypes);
        const refresh = (eid: number) => {
            const type = storage.type.get(eid);
            const source = types[type]?.source ? sources[type] : undefined;
            const transparent = source
                ? (source.pipelineKeys & TRANSPARENT_KEYS) !== 0
                : ((1 << alphaPipelineKey(storage.alphaMode.get(eid))) & TRANSPARENT_KEYS) !== 0;
            if (
                transparent &&
                world.has(eid, MeshMaterial) &&
                world.has(eid, MeshInstance) &&
                world.has(eid, GlobalTransform)
            )
                candidates.add(eid);
            else candidates.delete(eid);
        };
        for (const component of [MeshInstance, GlobalTransform]) {
            world.observeMembership(component, (eid, present) => {
                if (present && world.has(eid, MeshMaterial)) refresh(eid);
                else candidates.delete(eid);
            });
        }
        world.observeMembership(MeshMaterial, (eid, present) => {
            if (present) return;
            candidates.delete(eid);
            for (const members of external) members?.delete(eid);
        });
        return {
            candidates,
            reconcile(eid: number) {
                for (const members of external) members?.delete(eid);
                const type = storage.type.get(eid);
                if (types[type]?.source) (external[type] ??= new Set()).add(eid);
                refresh(eid);
            },
            growSources() {
                for (let type = 0; type < sources.length; type++) {
                    const next = types[type]?.source ? (sources[type]?.pipelineKeys ?? 0) : 0;
                    if (keys[type] === next) continue;
                    keys[type] = next;
                    for (const eid of external[type] ?? []) refresh(eid);
                }
            },
        };
    },
};

/** Reconcile marked MeshMaterial alpha copies before the mesh table gathers fields. @internal */
export function copyMeshAlpha(world: World, all = false): void {
    const { assets, types } = world.resource(MaterialTypes);
    const storage = world.storage(MeshMaterial);
    const transparent = world.resource(TransparentMeshes);
    if (all) {
        for (const eid of world.query(MESH_MATERIALS)) {
            const type = storage.type.get(eid);
            if (!types[type]?.source) assets[type]?.copyAlpha(storage, eid);
            transparent.reconcile(eid);
        }
        return;
    }
    const type = world.fieldStorage(MeshMaterial, "type").dirty;
    const material = world.fieldStorage(MeshMaterial, "material").dirty;
    const mode = world.fieldStorage(MeshMaterial, "alphaMode").dirty;
    const cutoff = world.fieldStorage(MeshMaterial, "alphaCutoff").dirty;
    const cull = world.fieldStorage(MeshMaterial, "cullMode").dirty;
    for (let word = 0; word < type.length; word++) {
        let bits = type[word] | material[word] | mode[word] | cutoff[word] | cull[word];
        while (bits !== 0) {
            const bit = 31 - Math.clz32(bits & -bits);
            const eid = (word << 5) + bit;
            bits &= bits - 1;
            if (!world.has(eid, MeshMaterial)) continue;
            const id = storage.type.get(eid);
            if (!types[id]?.source) assets[id]?.copyAlpha(storage, eid);
            transparent.reconcile(eid);
        }
    }
}

/** Every material type registered in this World, with StandardMaterial reserved at type zero. */
export const MaterialTypes: Resource<MaterialTypeState> = {
    create: (world) => world.resource(materialTypesKey),
};

/** Register one material type. Type zero is reserved for the missing-material StandardMaterial default. */
export function registerMaterialType(
    world: World,
    type: ErasedMaterialType,
    standard = false,
): number {
    const state = world.resource(MaterialTypes);
    const prior = state.names.get(type.name);
    if (prior) {
        if (prior !== type) throw new Error(`material type "${type.name}" is already registered`);
        return state.ids.get(type)!;
    }
    const id = standard ? 0 : state.types.length;
    if (standard && state.types[0])
        throw new Error("StandardMaterial type slot is already registered");
    state.types[id] = type;
    state.names.set(type.name, type);
    state.ids.set(type, id);
    return id;
}

/** World-local numeric id; refuses unregistered types. */
export function materialTypeId(world: World, type: ErasedMaterialType): number {
    const id = world.resource(MaterialTypes).ids.get(type);
    if (id === undefined) throw new Error(`material type "${type.name}" is not registered`);
    return id;
}

/** @internal Resolve the renderer's parameter source without transferring its ownership. */
export function materialSource<P extends AnyWgslStruct>(
    world: World,
    type: MaterialType<P, any, any>,
): MaterialSource<P> {
    return type.source ? world.resource(type.source) : world.resource(type);
}

export function MaterialPlugin<
    P extends AnyWgslStruct,
    B extends Record<string, MaterialBinding>,
    V extends Record<string, AnyWgslData>,
>(type: MaterialType<P, B, V>): Plugin {
    return {
        name: `Material(${type.name})`,
        components: [MeshMaterial],
        initialize(world) {
            registerMaterialType(world, type as ErasedMaterialType);
            const source = materialSource(world, type);
            world.resource(MaterialTypes).sources[materialTypeId(world, type)] = source;
        },
    };
}

/**
 * Define a material resource with plain "use gpu" shaders whose arguments are inferred from its schemas.
 * Supply parameters alone, or a layout carrying them and owned bindings the shaders close over.
 */
export function materialType<
    P extends AnyWgslStruct,
    B extends Record<string, MaterialBinding>,
    V extends Record<string, AnyWgslData> = Record<never, never>,
>(
    spec: Omit<
        MaterialType<P, B, V>,
        "create" | "parameters" | "layout" | "vertex" | "fragment"
    > & {
        vertex?: ((
            input: d.InferGPU<ReturnType<typeof materialVertexInput<P>>>,
        ) => d.InferGPU<ReturnType<typeof materialVertexOutput<V>>>) & {
            readonly resourceType?: never;
        };
        fragment: ((
            context: d.InferGPU<ReturnType<typeof materialFragmentContext<P, V>>>,
        ) => d.v4f) & { readonly resourceType?: never };
    } & (
            | { layout: MaterialLayout<B, P>; parameters?: never }
            | { parameters: P; layout?: never }
        ) & { defaults: d.InferInput<P> },
): MaterialType<P, B, V> {
    const layout = spec.layout ?? materialLayout(spec.parameters!, {});
    const parameters = layout.parameters;
    if (!spec.name) throw new Error("material type name must not be empty");
    for (const stage of ["vertex", "fragment"] as const) {
        const shader = spec[stage];
        if (shader && "resourceType" in shader) {
            throw new Error(
                `material type "${spec.name}" ${stage} shader must be a plain "use gpu" function, not a prebuilt TypeGPU function`,
            );
        }
    }
    let definition!: MaterialType<P, B, V>;
    definition = {
        ...spec,
        layout,
        parameters,
        fragment: tgpu.fn(
            [materialFragmentContext<P, V>(parameters, spec.varyings)],
            d.vec4f,
        )(spec.fragment),
        vertex: spec.vertex
            ? tgpu.fn(
                  [materialVertexInput(parameters)],
                  materialVertexOutput<V>(spec.varyings),
              )(spec.vertex)
            : undefined,
        create: (world: World) => new MaterialAssets<P>(world, definition),
    } as MaterialType<P, B, V>;
    return definition;
}
