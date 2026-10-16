import type { TgpuBindGroupLayout, TgpuFn } from "typegpu";
import tgpu, { isTgpuFn, readFromArrayBuffer, writeToArrayBuffer } from "typegpu";
import type { AnyWgslData, AnyWgslStruct, WgslArray } from "typegpu/data";
import * as d from "typegpu/data";
import { Xform } from "../../core/rendering";
import type { Plugin, Resource, System, World } from "../../engine";
import type { ComponentStorage } from "../../engine/ecs";
import { MeshInstanceInput, MeshMaterial } from "./material-data";

type ShaderStage = "vertex" | "fragment";
const VS_FS: ShaderStage[] = ["vertex", "fragment"];

/** A material-owned resource or mesh attribute made available to its shaders. */
export type MaterialBinding =
    | { type: "uniform"; struct: AnyWgslStruct }
    | { type: "attribute"; element: AnyWgslData }
    | {
          type: "storage";
          element: AnyWgslData;
          access?: "read" | "read_write";
          visibility?: readonly ShaderStage[];
      }
    | { type: "texture-2d" }
    | { type: "texture-2d-array" }
    | { type: "texture-depth-2d" }
    | { type: "sampler" }
    | { type: "sampler-comparison" };

type EntryFor<B extends MaterialBinding> = B extends { type: "uniform" }
    ? { uniform: B["struct"]; visibility: ShaderStage[] }
    : B extends { type: "storage" | "attribute" }
      ? {
            storage: (count: number) => WgslArray<B["element"]>;
            access: "mutable" | "readonly";
            visibility: ShaderStage[];
        }
      : B extends { type: "texture-2d" }
        ? { texture: d.WgslTexture2d<d.F32>; visibility: ShaderStage[] }
        : B extends { type: "texture-2d-array" }
          ? { texture: d.WgslTexture2dArray<d.F32>; visibility: ShaderStage[] }
          : B extends { type: "texture-depth-2d" }
            ? { texture: d.WgslTextureDepth2d; visibility: ShaderStage[] }
            : B extends { type: "sampler" }
              ? { sampler: "filtering"; visibility: ShaderStage[] }
              : { sampler: "comparison"; visibility: ShaderStage[] };

function bindingEntry<B extends MaterialBinding>(binding: B): EntryFor<B> {
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
        case "texture-2d":
            return { texture: d.texture2d(), visibility: VS_FS } as EntryFor<B>;
        case "texture-2d-array":
            return { texture: d.texture2dArray(), visibility: VS_FS } as EntryFor<B>;
        case "texture-depth-2d":
            return { texture: d.textureDepth2d(), visibility: VS_FS } as EntryFor<B>;
        case "sampler":
            return { sampler: "filtering", visibility: VS_FS } as EntryFor<B>;
        case "sampler-comparison":
            return { sampler: "comparison", visibility: VS_FS } as EntryFor<B>;
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

/** Build the material type's bind group before its shader functions so they can close over its typed rows. */
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

/** Vertex input shared by standard material types. `material` is this type's parameter-table row. */
export const MaterialVertexInput = d
    .struct({
        localPos: d.vec3f,
        localNormal: d.vec3f,
        uv: d.vec2f,
        vidx: d.u32,
        eid: d.u32,
        iid: d.u32,
        xform: Xform,
        world: d.vec4f,
        worldNormal: d.vec3f,
        color: d.vec4f,
        material: d.u32,
    })
    .$name("MaterialVertexInput");

/** World-space fields a material vertex function may override, plus its declared varyings. */
export function materialVertexOutput<V extends Record<string, AnyWgslData> = Record<never, never>>(
    varyings: V = {} as V,
) {
    return d.struct({ world: d.vec4f, worldNormal: d.vec3f, ...varyings });
}

/** Context passed to a material's fragment function; `material` indexes its own parameter table. */
export function materialFragmentContext<
    V extends Record<string, AnyWgslData> = Record<never, never>,
>(varyings: V = {} as V) {
    return d.struct({
        eid: d.u32,
        world: d.vec3f,
        worldNormal: d.vec3f,
        uv: d.vec2f,
        localPos: d.vec3f,
        color: d.vec4f,
        material: d.u32,
        ...varyings,
    });
}

export type MaterialVertexFn<V extends Record<string, AnyWgslData> = Record<never, never>> = TgpuFn<
    (input: typeof MaterialVertexInput) => ReturnType<typeof materialVertexOutput<V>>
>;
export type MaterialFragmentFn<V extends Record<string, AnyWgslData> = Record<never, never>> =
    TgpuFn<(context: ReturnType<typeof materialFragmentContext<V>>) => d.Vec4f>;

export type AlphaMode =
    | { readonly kind: "opaque" }
    | { readonly kind: "mask"; readonly cutoff: number }
    | { readonly kind: "blend" }
    | { readonly kind: "premultiplied" }
    | { readonly kind: "add" }
    | { readonly kind: "multiply" }
    | { readonly kind: "alpha-to-coverage" };

/** Per-material transparency behavior; Mask defaults to the 0.5 cutoff. */
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

/** The shader, parameter schema, and depth behavior shared by every instance of one mesh material type. */
export interface MaterialType<
    P extends AnyWgslStruct = AnyWgslStruct,
    B extends Record<string, MaterialBinding> = Record<string, MaterialBinding>,
    V extends Record<string, AnyWgslData> = Record<never, never>,
> extends Resource<MaterialAssets<P>> {
    readonly name: string;
    readonly parameters: P;
    readonly layout: MaterialLayout<B, P>;
    readonly fragmentInputs?: { uv?: true; localPos?: true };
    readonly varyings?: V;
    readonly vertex?: MaterialVertexFn<V>;
    readonly fragment: MaterialFragmentFn<V>;
    readonly depthPass?: { prepass?: boolean; shadows?: boolean };
    readonly defaults?: d.InferInput<P>;
}

/** A material handle can be added directly as `MeshMaterial`; alpha fields mirror its live row. */
export interface MaterialHandle {
    readonly type: number;
    readonly material: number;
    readonly alphaMode: number;
    readonly alphaCutoff: number;
}

/** CPU-authored material rows with a lazily published GPU table using the type's parameter schema. */
export class MaterialAssets<P extends AnyWgslStruct = AnyWgslStruct> {
    private readonly _world: World;
    private readonly _type: MaterialType<P, any, any>;
    private readonly _rowBytes: number;
    private readonly _scratch: ArrayBuffer;
    private readonly _freeRows: number[] = [];
    private _bytes: Uint8Array;
    private _alphaModes = new Uint32Array([AlphaModeCode.Opaque]);
    private _alphaCutoffs = new Float32Array([0.5]);
    private readonly _handles: (MaterialHandle | undefined)[] = [undefined];
    private _highWater = 1;
    // row zero starts Opaque, and rows reserved past the high-water mark start at its zero code
    private _alphaKeys = 1 << AlphaPipelineKey.Opaque;
    private _table: ReturnType<World["table"]> | undefined;

    constructor(world: World, type: MaterialType<P, any, any>) {
        this._world = world;
        this._type = type;
        const id = registerMaterialType(world, type, type.name === "StandardMaterial");
        world.resource(MaterialTypes).assets[id] = this;
        this._rowBytes = d.sizeOf(type.parameters);
        this._scratch = new ArrayBuffer(this._rowBytes);
        this._bytes = new Uint8Array(this._rowBytes);
        writeToArrayBuffer(this._scratch, type.parameters, type.defaults ?? ({} as never));
        this._bytes.set(new Uint8Array(this._scratch), 0);
    }

    /** The renderer initializes this table at warm so its default row is uploaded before the first draw. */
    get table(): ReturnType<World["table"]> {
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
            const alphaModes = new Uint32Array(capacity);
            alphaModes.set(this._alphaModes);
            this._alphaModes = alphaModes;
            const alphaCutoffs = new Float32Array(capacity).fill(0.5);
            alphaCutoffs.set(this._alphaCutoffs);
            this._alphaCutoffs = alphaCutoffs;
        }
        this._highWater = Math.max(this._highWater, rows);
    }

    private publish(firstRow: number, count: number): void {
        if (!this._table || count === 0) return;
        const start = firstRow * this._rowBytes;
        const end = (firstRow + count) * this._rowBytes;
        this._table.reserveSlots(this._highWater);
        this._table.bytes.set(this._bytes.subarray(start, end), start);
        this._table.markRange(firstRow, count);
    }

    /** Add a material value and return its type and type-local row for a MeshMaterial component. */
    add(values: d.InferInput<P>, options: { alphaMode?: AlphaMode } = {}): MaterialHandle {
        const alphaMode = options.alphaMode ?? AlphaMode.Opaque;
        alphaModeFields(alphaMode);
        const row = this._freeRows.pop() ?? this._highWater;
        this.reserveRows(row + 1);
        writeToArrayBuffer(this._scratch, this._type.parameters, values);
        this._bytes.set(new Uint8Array(this._scratch), row * this._rowBytes);
        this._writeAlpha(row, alphaMode);
        this.publish(row, 1);
        const handle = {
            type: materialTypeId(this._world, this._type),
            material: row,
            ...this.alpha(row),
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

    /** Bit `1 << key` for each alpha pipeline key any of this type's rows has held; it only grows. */
    get alphaKeys(): number {
        return this._alphaKeys;
    }

    private _writeAlpha(row: number, mode: AlphaMode): void {
        const fields = alphaModeFields(mode);
        if (
            this._alphaModes[row] === fields.alphaMode &&
            this._alphaCutoffs[row] === fields.alphaCutoff
        )
            return;
        this._alphaModes[row] = fields.alphaMode;
        this._alphaCutoffs[row] = fields.alphaCutoff;
        this._alphaKeys |= 1 << alphaPipelineKey(fields.alphaMode);
        const handle = this._handles[row];
        if (handle) Object.assign(handle, fields);
        const declaresMeshMaterial = [...this._world.registry.entries()].some(
            ({ component }) => component === MeshMaterial,
        );
        if (declaresMeshMaterial) copyMeshAlpha(this._world);
    }

    /** Copy this type's row alpha into the entity's MeshMaterial where they differ; an unknown row is left. @internal */
    copyAlpha(storage: MeshMaterialStorage, eid: number): void {
        const row = storage.material.get(eid);
        if (row >= this._highWater) return;
        const alphaMode = this._alphaModes[row];
        const alphaCutoff = this._alphaCutoffs[row];
        if (storage.alphaMode.get(eid) !== alphaMode) storage.alphaMode.set(eid, alphaMode);
        if (storage.alphaCutoff.get(eid) !== alphaCutoff) storage.alphaCutoff.set(eid, alphaCutoff);
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
    setAt(row: number, values: d.InferInput<P>, options: { alphaMode?: AlphaMode } = {}): void {
        if (!Number.isSafeInteger(row) || row < 0) {
            throw new RangeError(`Materials.setAt: invalid ${this._type.name} row ${row}`);
        }
        if (options.alphaMode) alphaModeFields(options.alphaMode);
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
        if (options.alphaMode) this._writeAlpha(row, options.alphaMode);
        this.publish(Math.min(row, previousHighWater), Math.max(1, row - previousHighWater + 1));
    }

    /** Publish changed fields at one type-local row for the next frame upload. */
    update(
        handle: number | MaterialHandle,
        values: Partial<d.InferInput<P>>,
        options: { alphaMode?: AlphaMode } = {},
    ): void {
        const row = typeof handle === "number" ? handle : handle.material;
        if (typeof handle !== "number" && handle.type !== materialTypeId(this._world, this._type)) {
            throw new RangeError(`Materials.update: handle belongs to another material type`);
        }
        if (!Number.isSafeInteger(row) || row < 0 || row >= this._highWater) {
            throw new RangeError(`Materials.update: unknown ${this._type.name} row ${row}`);
        }
        if (options.alphaMode) alphaModeFields(options.alphaMode);
        const offset = row * this._rowBytes;
        const currentBytes = this._bytes.slice(offset, offset + this._rowBytes).buffer;
        const current = readFromArrayBuffer(currentBytes, this._type.parameters) as Record<
            string,
            unknown
        >;
        const merged = { ...current, ...values } as d.InferInput<P>;
        writeToArrayBuffer(this._scratch, this._type.parameters, merged);
        this._bytes.set(new Uint8Array(this._scratch), offset);
        if (options.alphaMode) this._writeAlpha(row, options.alphaMode);
        this.publish(row, 1);
    }
}

type ErasedMaterialType = MaterialType<any, any, any>;
type MeshMaterialStorage = ComponentStorage<typeof MeshMaterial>;

interface MaterialTypeState {
    readonly types: (ErasedMaterialType | undefined)[];
    /** each registered type's CPU store, by type id, once created */
    readonly assets: (MaterialAssets<any> | undefined)[];
    readonly ids: WeakMap<object, number>;
    readonly names: Map<string, ErasedMaterialType>;
}
const materialTypesKey = {
    create: (): MaterialTypeState => ({
        types: [undefined],
        assets: [],
        ids: new WeakMap(),
        names: new Map(),
    }),
};

const MESH_MATERIALS = [MeshMaterial];

// The row owns alpha; MeshMaterial's alphaMode and alphaCutoff are its copy for the mesh table's bulk fill.
// Field change marks belong to the frame upload, so this compares the copy with its row instead.
function copyMeshAlpha(world: World): void {
    const assets = world.resource(MaterialTypes).assets;
    const storage = world.storage(MeshMaterial);
    for (const eid of world.query(MESH_MATERIALS)) {
        assets[storage.type.get(eid)]?.copyAlpha(storage, eid);
    }
}

/**
 * Copies each MeshMaterial's row alpha into its `alphaMode` and `alphaCutoff` after simulation, so a
 * reassigned `type` or `material` takes its row's alpha mode in the same frame's upload.
 */
export const MeshMaterialAlphaSystem: System = {
    name: "mesh-material-alpha",
    group: "simulation",
    last: true,
    update: copyMeshAlpha,
};

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
            world.resource(type);
        },
    };
}

/** Define the world resource key and immutable shader contract for a material type. */
export function materialType<
    P extends AnyWgslStruct,
    B extends Record<string, MaterialBinding>,
    V extends Record<string, AnyWgslData> = Record<never, never>,
>(
    spec: Omit<MaterialType<P, B, V>, "create" | "parameters" | "layout"> &
        ({ layout: MaterialLayout<B, P>; parameters?: never } |
         { parameters: P; layout?: never }) & { defaults: d.InferInput<P> },
): MaterialType<P, B, V> {
    const layout = spec.layout ?? materialLayout(spec.parameters!, {});
    const parameters = layout.parameters;
    if (!spec.name) throw new Error("material type name must not be empty");
    if (!isTgpuFn(spec.fragment) || (spec.vertex && !isTgpuFn(spec.vertex))) {
        throw new Error(
            `material type "${spec.name}" shader is not a TGSL function from this engine's TypeGPU`,
        );
    }
    let definition!: MaterialType<P, B, V>;
    definition = {
        ...spec,
        layout,
        parameters,
        create: (world: World) => new MaterialAssets<P>(world, definition),
    } as MaterialType<P, B, V>;
    return definition;
}
