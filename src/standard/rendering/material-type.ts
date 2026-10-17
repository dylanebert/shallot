import type { TgpuBindGroupLayout, TgpuFn } from "typegpu";
import tgpu, { readFromArrayBuffer, writeToArrayBuffer } from "typegpu";
import type { AnyWgslData, AnyWgslStruct, WgslArray } from "typegpu/data";
import * as d from "typegpu/data";
import { Xform } from "../../core/rendering";
import type { GpuTable, Plugin, Resource, System, World } from "../../engine";
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

export const MeshVertexInput = d
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

/** Fragment context with this material's typed parameter row and declared varyings. */
function materialFragmentContext<
    P extends AnyWgslStruct,
    V extends Record<string, AnyWgslData> = Record<never, never>,
>(parameters: P, varyings: V = {} as V) {
    return d.struct({
        parameters,
        eid: d.u32,
        world: d.vec3f,
        worldNormal: d.vec3f,
        uv: d.vec2f,
        localPos: d.vec3f,
        color: d.vec4f,
        material: d.u32,
        alphaMode: d.u32,
        alphaCutoff: d.f32,
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

/** A material's externally owned parameter table and alpha pipeline inputs.
 * The owner fills and uploads its table through World's table schedule, and publishes each consumer's
 * alphaMode/alphaCutoff into meshInstanceTable after upload and before MeshPreprocessSystem.
 * alphaMode reads the selected row for CPU transparent sorting; alphaKeys is the growing bitset of
 * `1 << AlphaPipelineKey` values the type can draw. Rows remain valid for the source's stated lifetime.
 * cullBounds, when supplied, reads the selected row and writes a local-space [x, y, z, radius] sphere
 * into caller-owned scratch during transparent queuing. Shared consumers use the same row's geometry;
 * geometry-changing shaders must cover their expanded geometry here too. */
export interface MaterialSource<P extends AnyWgslStruct = AnyWgslStruct> {
    readonly table: GpuTable<P>;
    readonly alphaKeys: number;
    alphaMode(row: number): number;
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
    readonly fragmentInputs?: { uv?: true; localPos?: true };
    readonly varyings?: V;
    readonly vertex?: MaterialVertexFn<P, V>;
    readonly fragment: MaterialFragmentFn<P, V>;
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

/** CPU-authored material assets with a lazily published GPU table using the type's parameter schema.
 * Externally sourced material types refuse construction; their rows belong to the declared source. */
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

    /** AlphaModeCode at one asset row, without allocating an alpha record. */
    alphaMode(row: number): number {
        return this._alphaModes[row];
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
        if (declaresMeshMaterial) copyMeshAlpha(this._world, true);
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

// The row owns alpha; MeshMaterial's alphaMode and alphaCutoff are its copy for the mesh table's bulk fill.
function copyMeshAlpha(world: World, all = false): void {
    const { assets, types } = world.resource(MaterialTypes);
    const storage = world.storage(MeshMaterial);
    if (all) {
        for (const eid of world.query(MESH_MATERIALS)) {
            const type = storage.type.get(eid);
            if (!types[type]?.source) assets[type]?.copyAlpha(storage, eid);
        }
        return;
    }
    const type = world.fieldStorage(MeshMaterial, "type").dirty;
    const material = world.fieldStorage(MeshMaterial, "material").dirty;
    const mode = world.fieldStorage(MeshMaterial, "alphaMode").dirty;
    const cutoff = world.fieldStorage(MeshMaterial, "alphaCutoff").dirty;
    for (let word = 0; word < type.length; word++) {
        let bits = type[word] | material[word] | mode[word] | cutoff[word];
        while (bits !== 0) {
            const bit = 31 - Math.clz32(bits & -bits);
            const eid = (word << 5) + bit;
            bits &= bits - 1;
            if (!world.has(eid, MeshMaterial)) continue;
            const id = storage.type.get(eid);
            if (!types[id]?.source) assets[id]?.copyAlpha(storage, eid);
        }
    }
}

/**
 * Copies changed MeshMaterial members' row alpha into their `alphaMode` and `alphaCutoff` after
 * simulation, including frames without a fixed tick. A reassigned `type` or `material` takes its
 * row's alpha mode in the same frame's upload. Draw writes after upload reconcile next frame.
 * External table owners publish directly into the GPU mesh table after their table upload.
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
