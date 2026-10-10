import type { TgpuBindGroupLayout, TgpuFn } from "typegpu";
import tgpu, { isTgpuFn, readFromArrayBuffer, writeToArrayBuffer } from "typegpu";
import type { AnyWgslData, AnyWgslStruct, WgslArray } from "typegpu/data";
import * as d from "typegpu/data";
import type { Plugin, Resource, World } from "../../engine";
import { Xform } from "../../engine/utils";
import { MeshInstanceInput } from "./material-data";

/** A shader-stage name understood by TypeGPU. */
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
export function materialVertexOutput<V extends Record<string, AnyWgslData> = Record<string, never>>(
    varyings: V = {} as V,
) {
    return d.struct({ world: d.vec4f, worldNormal: d.vec3f, ...varyings });
}

/** Context passed to a material's fragment function; `material` indexes its own parameter table. */
export function materialFragmentContext<
    V extends Record<string, AnyWgslData> = Record<string, never>,
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

export type MaterialVertexFn<V extends Record<string, AnyWgslData> = Record<string, never>> =
    TgpuFn<(input: typeof MaterialVertexInput) => ReturnType<typeof materialVertexOutput<V>>>;
export type MaterialFragmentFn<V extends Record<string, AnyWgslData> = Record<string, never>> =
    TgpuFn<(context: ReturnType<typeof materialFragmentContext<V>>) => d.Vec4f>;

/** Current routing retained until AlphaMode lands: opaque writes depth, clip writes cutouts, alpha blends. */
export type MaterialBlend = "opaque" | "clip" | "alpha";

/** The shader, parameter schema, and depth behavior shared by every instance of one mesh material type. */
export interface MaterialType<
    P extends AnyWgslStruct = AnyWgslStruct,
    B extends Record<string, MaterialBinding> = Record<string, MaterialBinding>,
    V extends Record<string, AnyWgslData> = Record<string, never>,
> extends Resource<MaterialAssets<P>> {
    readonly name: string;
    readonly parameters: P;
    readonly layout: MaterialLayout<B, P>;
    readonly fragmentInputs?: { uv?: true; localPos?: true };
    readonly varyings?: V;
    readonly vertex?: MaterialVertexFn<V>;
    readonly fragment: MaterialFragmentFn<V>;
    readonly blend?: MaterialBlend;
    readonly depthPass?: { prepass?: boolean; shadows?: boolean };
    readonly defaults?: d.InferInput<P>;
}

/** A material handle stores the type id and its type-local row for `MeshMaterial`. */
export interface MaterialHandle {
    type: number;
    material: number;
}

/** CPU-authored material rows with a lazily published GPU table using the type's parameter schema. */
export class MaterialAssets<P extends AnyWgslStruct = AnyWgslStruct> {
    private readonly _world: World;
    private readonly _type: MaterialType<P, any, any>;
    private readonly _rowBytes: number;
    private readonly _scratch: ArrayBuffer;
    private readonly _freeRows: number[] = [];
    private _bytes: Uint8Array;
    private _highWater = 1;
    private _table: ReturnType<World["table"]> | undefined;

    constructor(world: World, type: MaterialType<P, any, any>) {
        this._world = world;
        this._type = type;
        registerMaterialType(world, type, type.name === "StandardMaterial");
        this._rowBytes = d.sizeOf(type.parameters);
        this._scratch = new ArrayBuffer(this._rowBytes);
        this._bytes = new Uint8Array(this._rowBytes);
        writeToArrayBuffer(this._scratch, type.parameters, type.defaults ?? ({} as never));
        this._bytes.set(new Uint8Array(this._scratch), 0);
    }

    /** The GPU table is created lazily when a renderer first consumes these CPU-authored rows. */
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
    add(values: d.InferInput<P>): MaterialHandle {
        const row = this._freeRows.pop() ?? this._highWater;
        this.reserveRows(row + 1);
        writeToArrayBuffer(this._scratch, this._type.parameters, values);
        this._bytes.set(new Uint8Array(this._scratch), row * this._rowBytes);
        this.publish(row, 1);
        return { type: materialTypeId(this._world, this._type), material: row };
    }

    /** Release an unused row so producers with entity lifetimes can reuse its storage. Row zero is reserved. */
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
        this._freeRows.push(handle.material);
    }

    /** Write per-entity data at a stable eid-indexed row, filling intervening rows with defaults. */
    setAt(row: number, values: d.InferInput<P>): void {
        if (!Number.isSafeInteger(row) || row < 0) {
            throw new RangeError(`Materials.setAt: invalid ${this._type.name} row ${row}`);
        }
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
        this.publish(Math.min(row, previousHighWater), Math.max(1, row - previousHighWater + 1));
    }

    /** Publish changed fields at one type-local row for the next frame upload. */
    update(handle: number | MaterialHandle, values: Partial<d.InferInput<P>>): void {
        const row = typeof handle === "number" ? handle : handle.material;
        if (typeof handle !== "number" && handle.type !== materialTypeId(this._world, this._type)) {
            throw new RangeError(`Materials.update: handle belongs to another material type`);
        }
        if (!Number.isSafeInteger(row) || row < 0 || row >= this._highWater) {
            throw new RangeError(`Materials.update: unknown ${this._type.name} row ${row}`);
        }
        const offset = row * this._rowBytes;
        const currentBytes = this._bytes.slice(offset, offset + this._rowBytes).buffer;
        const current = readFromArrayBuffer(currentBytes, this._type.parameters) as Record<
            string,
            unknown
        >;
        const merged = { ...current, ...values } as d.InferInput<P>;
        writeToArrayBuffer(this._scratch, this._type.parameters, merged);
        this._bytes.set(new Uint8Array(this._scratch), offset);
        this.publish(row, 1);
    }
}

type ErasedMaterialType = MaterialType<any, any, any>;

interface MaterialTypeState {
    readonly types: (ErasedMaterialType | undefined)[];
    readonly ids: WeakMap<object, number>;
    readonly names: Map<string, ErasedMaterialType>;
}
const materialTypesKey = {
    create: (): MaterialTypeState => ({ types: [undefined], ids: new WeakMap(), names: new Map() }),
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

/** Resolve a material type's world-local numeric id. */
export function materialTypeId(world: World, type: ErasedMaterialType): number {
    const id = world.resource(MaterialTypes).ids.get(type);
    if (id === undefined) throw new Error(`material type "${type.name}" is not registered`);
    return id;
}

/** The material types in type-id order, excluding the reserved but not-yet-filled standard slot. */
export function materialTypes(world: World): readonly (ErasedMaterialType | undefined)[] {
    return world.resource(MaterialTypes).types;
}

/** The CPU-authored store owned by one material type. */
export function materialAssets<
    P extends AnyWgslStruct,
    B extends Record<string, MaterialBinding>,
    V extends Record<string, AnyWgslData>,
>(world: World, type: MaterialType<P, B, V>) {
    return world.resource(type);
}

/** Create the CPU-authored store and register the material type in this World. */
export function MaterialPlugin<
    P extends AnyWgslStruct,
    B extends Record<string, MaterialBinding>,
    V extends Record<string, AnyWgslData>,
>(type: MaterialType<P, B, V>): Plugin {
    return {
        name: `Material(${type.name})`,
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
    V extends Record<string, AnyWgslData> = Record<string, never>,
>(
    spec: Omit<MaterialType<P, B, V>, "create"> & { defaults: d.InferInput<P> },
): MaterialType<P, B, V> {
    if (spec.layout.parameters !== spec.parameters) {
        throw new Error(
            `material type "${spec.name}" layout was built for a different parameter schema`,
        );
    }
    if (!spec.name) throw new Error("material type name must not be empty");
    if (!isTgpuFn(spec.fragment) || (spec.vertex && !isTgpuFn(spec.vertex))) {
        throw new Error(
            `material type "${spec.name}" shader is not a TGSL function from this engine's TypeGPU`,
        );
    }
    let definition!: MaterialType<P, B, V>;
    definition = {
        ...spec,
        create: (world: World) => new MaterialAssets<P>(world, definition),
    } as MaterialType<P, B, V>;
    return definition;
}
