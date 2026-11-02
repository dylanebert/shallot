import tgpu, { type TgpuBindGroupLayout } from "typegpu";
import type { AnyWgslStruct } from "typegpu/data";
import * as d from "typegpu/data";
import type { Component, GpuTable, World } from "../../engine";
import { laneLayout, writeLane } from "../../engine/ecs";

const Presence = d.struct({ present: d.u32 });

/** CPU component fields consumed by materials. Declare matching field names and GPU types;
 * list the declaration in a material's instanceData before warm, or in materialLayout's
 * options (extensionLayout for extensions) when supplying an explicit layout. All declarations in a world
 * share one holder-only record store and one storage binding. Mesh rows carry a u32 payload index;
 * zero means no declared component. read takes the material's own fallback;
 * present tests membership. GPU producer instances are absent, regardless of their row or eid.
 * Materials use these accessors, never the record's offsets or buffer directly.
 * Omit fields to declare presence alone. */
export function instanceData<P extends AnyWgslStruct = typeof Presence>(
    component: Component,
    fields?: P,
) {
    for (const [name, field] of Object.entries(fields?.propTypes ?? {})) {
        if (!laneLayout(field as d.BaseData))
            throw new Error(
                `Instance data field "${name}" requires an f32, i32 or u32 scalar or vector`,
            );
    }
    const schema = fields ?? (Presence as unknown as P);
    const readFn = tgpu.fn(
        [d.u32, schema],
        schema,
    )((_row, fallback) => {
        "use gpu";
        return fallback;
    });
    const presentFn = tgpu.fn(
        [d.u32],
        d.bool,
    )((_row) => {
        "use gpu";
        return false;
    });
    const read = tgpu.slot<typeof readFn>();
    const present = tgpu.slot<typeof presentFn>();
    return {
        component,
        fields,
        /** @internal World-specific shader substitutions. */
        readSlot: read,
        /** @internal */
        presentSlot: present,
        read: (ctx: { instanceRow: number }, fallback: d.InferGPU<P>): d.InferGPU<P> => {
            "use gpu";
            return read.$(ctx.instanceRow, fallback);
        },
        present: (ctx: { instanceRow: number }): boolean => {
            "use gpu";
            return present.$(ctx.instanceRow);
        },
    };
}

/** A component declaration shared by material definitions; storage belongs to each world. */
export type InstanceData = ReturnType<typeof instanceData<any>>;

/** @internal Fill declaration slots with world-local record offsets during pipeline preparation. */
export function instanceDataRoot(
    world: World,
    root: ReturnType<World["gpu"]["root"]["with"]>,
    layout: TgpuBindGroupLayout<any>,
    declarations: readonly InstanceData[] = [],
    rowSource: GpuTable,
) {
    if (!declarations.length) return root;
    const table = instanceDataTable(world, rowSource);
    const state = world.resource(stateKey);
    const record = table.record;
    const offset = (name: string) => d.memoryLayoutOf(record, (row: any) => row[name]).offset / 4;
    for (const declaration of declarations) {
        const index = state.declarations.indexOf(declaration);
        const presence = tgpu
            .fn(
                [d.u32],
                d.bool,
            )(`(row: u32) -> bool {
            if (row == 0xFFFFFFFFu) { return false; }
            let payload = bound.instanceData[row];
            if (payload == 0u) { return false; }
            return bound.instanceData[payload + ${offset(`p${index}`)}u] != 0u;
        }`)
            .$uses({ bound: layout.$ });
        root = root.with(declaration.presentSlot, presence);
        if (!declaration.fields) continue;
        const values = Object.entries(declaration.fields.propTypes).map(([name, field]) => {
            const { format, lanes } = laneLayout(field as d.BaseData)!;
            const word = offset(`c${index}_${name}`);
            const scalar = `${format}32`;
            const parts = Array.from({ length: lanes }, (_, lane) => {
                const value = `bound.instanceData[bound.instanceData[row] + ${word + lane}u]`;
                return scalar === "u32" ? value : `bitcast<${scalar}>(${value})`;
            });
            return lanes === 1 ? parts[0] : `vec${lanes}<${scalar}>(${parts.join(", ")})`;
        });
        const read = tgpu
            .fn(
                [d.u32, declaration.fields],
                declaration.fields,
            )(`(row: u32, fallback: Fields) -> Fields {
            if (!present(row)) { return fallback; }
            return Fields(${values.join(", ")});
        }`)
            .$uses({ bound: layout.$, Fields: declaration.fields, present: presence });
        root = root.with(declaration.readSlot, read);
    }
    return root;
}

const stateKey = {
    create: () => ({
        declarations: [] as InstanceData[],
        table: undefined as InstanceDataStore | undefined,
    }),
};

/** @internal */
export function registerInstanceData(world: World, declarations: readonly InstanceData[] = []) {
    const state = world.resource(stateKey);
    for (const declaration of declarations) {
        if (state.declarations.includes(declaration)) continue;
        if (state.table) throw new Error("Instance data declarations must register before warm");
        state.declarations.push(declaration);
    }
}

type Holder = { eid: number; row: number; slot: number };

// Bevy's MeshTag indirection: mesh rows spend one u32, while only holders have payloads.
// Index words and the holder records occupy the same storage binding.
class InstanceDataStore {
    readonly record: d.AnyWgslStruct;
    readonly rowBytes: number;
    buffer!: GPUBuffer;
    private _capacity = 0;
    private _payloadCapacity = 0;
    private _bytes = new Uint8Array(0);
    private _view = new DataView(this._bytes.buffer);
    private _indices = new Uint32Array(0);
    private _holders: Holder[] = [];
    private _byEid = new Map<number, Holder>();
    private _indexFirst = Infinity;
    private _indexLast = -1;
    private _payloadFirst = Infinity;
    private _payloadLast = -1;
    private readonly _bindings;
    private readonly _world: World;
    private readonly _source: GpuTable;

    constructor(world: World, source: GpuTable, declarations: readonly InstanceData[]) {
        this._world = world;
        this._source = source;
        const fields: Record<string, d.AnyWgslData> = {};
        for (const [index, declaration] of declarations.entries()) {
            fields[`p${index}`] = d.u32;
            for (const [name, field] of Object.entries(declaration.fields?.propTypes ?? {}))
                fields[`c${index}_${name}`] = field as d.AnyWgslData;
        }
        this.record = d.struct(fields);
        this.rowBytes = d.sizeOf(this.record);
        const offset = (name: string) =>
            d.memoryLayoutOf(this.record, (row: any) => row[name]).offset;
        this._bindings = declarations.map((declaration, index) => ({
            component: declaration.component,
            presence: offset(`p${index}`),
            fields: Object.entries(declaration.fields?.propTypes ?? {}).map(([name, type]) => {
                const source = world.fieldStorage(declaration.component, name);
                const layout = laneLayout(type as d.BaseData)!;
                const ctor =
                    layout.format === "f"
                        ? Float32Array
                        : layout.format === "i"
                          ? Int32Array
                          : Uint32Array;
                if (
                    source.type.ctor !== ctor ||
                    source.type.lanes < layout.lanes ||
                    source.type.encode ||
                    source.type.decode
                )
                    throw new Error(
                        `Instance data field "${name}" does not match its component field`,
                    );
                return {
                    offset: offset(`c${index}_${name}`),
                    ...layout,
                    source,
                    sourceLanes: source.type.lanes,
                };
            }),
        }));
        this.resize();
        for (const declaration of declarations) {
            world.onDispose(
                world.observeMembership(declaration.component, (eid) => this.reconcile(eid)),
            );
            for (const eid of world.query([declaration.component])) this.reconcile(eid);
        }
    }

    private reconcile(eid: number) {
        const present = this._bindings.some((binding) => this._world.has(eid, binding.component));
        const holder = this._byEid.get(eid);
        if (present && !holder) {
            const entry = { eid, row: -1, slot: this._holders.length };
            this._holders.push(entry);
            this._byEid.set(eid, entry);
        } else if (!present && holder) {
            if (holder.row >= 0) this.setIndex(holder.row, 0);
            const last = this._holders.pop()!;
            if (last !== holder) {
                last.slot = holder.slot;
                this._holders[last.slot] = last;
            }
            this._byEid.delete(eid);
        }
    }

    private setIndex(row: number, value: number) {
        if (this._indices[row] === value) return;
        this._indices[row] = value;
        this._indexFirst = Math.min(this._indexFirst, row);
        this._indexLast = Math.max(this._indexLast, row);
    }

    private resize() {
        let payloadCapacity = this._holders.length ? Math.max(1, this._payloadCapacity) : 0;
        while (payloadCapacity < this._holders.length) payloadCapacity *= 2;
        const capacity = Math.max(1, this._source.capacity);
        if (capacity === this._capacity && payloadCapacity === this._payloadCapacity) return;
        const size = capacity * 4 + payloadCapacity * this.rowBytes;
        const limit = Math.min(
            this._world.gpu.device.limits.maxStorageBufferBindingSize,
            this._world.gpu.device.limits.maxBufferSize,
        );
        if (size > limit)
            throw new Error(`Instance data needs ${size} bytes; storage binding limit is ${limit}`);
        const previous = this.buffer;
        this.buffer = this._world.gpu.device.createBuffer({
            label: "material-instance-data",
            size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        this._world.own(this.buffer);
        if (previous) this._world.retireGpuResource(previous);
        this._capacity = capacity;
        this._payloadCapacity = payloadCapacity;
        // A newly allocated GPU index is already zero. No CPU backing or upload is needed
        // until the world actually has a holder.
        this._bytes = new Uint8Array(payloadCapacity ? size : 0);
        this._view = new DataView(this._bytes.buffer);
        this._indices = new Uint32Array(this._bytes.buffer, 0, payloadCapacity ? capacity : 0);
        this._indexFirst = this._payloadFirst = Infinity;
        this._indexLast = this._payloadLast = -1;
    }

    upload() {
        this.resize();
        for (const holder of this._holders) {
            const row = this._source.rowIndex(holder.eid);
            if (row !== holder.row && holder.row >= 0) this.setIndex(holder.row, 0);
        }
        for (const holder of this._holders) {
            const row = this._source.rowIndex(holder.eid);
            holder.row = row;
            const base = this._capacity * 4 + holder.slot * this.rowBytes;
            if (row >= 0) this.setIndex(row, base / 4);
            let changed = false;
            for (const binding of this._bindings) {
                const present = this._world.has(holder.eid, binding.component);
                changed =
                    writeLane(this._view, base + binding.presence, "u", Number(present)) || changed;
                for (const field of binding.fields) {
                    const column = field.source.column;
                    for (let lane = 0; lane < field.lanes; lane++) {
                        const value = present ? column[holder.eid * field.sourceLanes + lane] : 0;
                        changed =
                            writeLane(
                                this._view,
                                base + field.offset + lane * 4,
                                field.format,
                                value,
                            ) || changed;
                    }
                }
            }
            if (changed) {
                this._payloadFirst = Math.min(this._payloadFirst, base);
                this._payloadLast = Math.max(this._payloadLast, base + this.rowBytes);
            }
        }
        if (this._indexLast >= this._indexFirst)
            this._world.uploadGpuTable(
                this.buffer,
                this._indexFirst * 4,
                this._bytes.buffer,
                (this._indexLast - this._indexFirst + 1) * 4,
            );
        if (this._payloadLast > this._payloadFirst)
            this._world.uploadGpuTable(
                this.buffer,
                this._payloadFirst,
                this._bytes.buffer,
                this._payloadLast - this._payloadFirst,
            );
        this._indexFirst = this._payloadFirst = Infinity;
        this._indexLast = this._payloadLast = -1;
    }
}

/** @internal */
export function instanceDataTable(world: World, rowSource: GpuTable) {
    const state = world.resource(stateKey);
    return (state.table ??= new InstanceDataStore(world, rowSource, state.declarations));
}

/** @internal Publish holder records after mesh row assignment and before material passes. */
export function uploadInstanceData(world: World) {
    world.resource(stateKey).table?.upload();
}
