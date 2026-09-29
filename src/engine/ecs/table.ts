import type { TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import type { Component, FieldSchema, TypedArray } from "./component";
import { idOf, isFieldSchema } from "./component";
import type { State } from "./state";

export type TableUploadPath = "none" | "writeBuffer";
export interface GpuTableOptions {
    /** No CPU backing or upload path; a GPU pass owns every row write. */
    gpuOnly?: boolean;
}

type Consumer = (buffer: GPUBuffer, generation: number) => void;
const ACTIVE_ROW = d.struct({ eid: d.u32, row: d.u32 });
type BoundField = {
    readonly recordName: string;
    readonly componentName: string;
    readonly offset: number;
    readonly lanes: number;
    readonly sourceLanes: number;
    readonly bytesPerLane: number;
};
type ComponentBinding = {
    readonly component: Component;
    readonly fields: BoundField[];
    readonly unsubscribes: (() => void)[];
    ownsRows: boolean;
};
type PresenceBinding = {
    readonly component: Component;
    readonly recordName: string;
    readonly offset: number;
    readonly mask: number;
};

/** Dense-slot storage with stable free-list allocation and one typed record layout. */
export class GpuTable<T extends d.AnyWgslData = d.AnyWgslData> {
    readonly name: string;
    readonly record: T;
    readonly rowBytes: number;
    readonly maxRows: number;
    readonly maxEntityRows: number;
    readonly gpuOnly: boolean;
    private readonly _state: State;
    private _capacity = 0;
    private _highWater = 1;
    private _generation = 0;
    private _bytes: Uint8Array | undefined;
    private _view: DataView | undefined;
    private _dirty = new Uint32Array(0);
    private _buffer!: GPUBuffer;
    private _typed!: TgpuBuffer<d.AnyWgslData>;
    private _disposed = false;
    private _lastUploadPath: TableUploadPath = "none";
    private _consumers: Consumer[] = [];

    private _eidToRow = new Uint32Array(0);
    private _mapCapacity = 0;
    private _mapHighWater = 0;
    private _mapEnabled = false;
    private _mapGeneration = 0;
    private _mapBuffer: GPUBuffer | undefined;
    private _mapTyped: TgpuBuffer<d.AnyWgslData> | undefined;
    private _mapDirty = false;
    private _lastMapUploadBytes = 0;
    private _mapConsumers: Consumer[] = [];
    private _freeRows = new Uint32Array(0);
    private _freeCount = 0;
    private _nextRow = 0;
    private _activeRows = new Uint32Array(0);
    private _activeIndex = new Int32Array(0);
    private _activeCount = 0;
    private _activeDirty = false;
    private _activeCapacity = 0;
    private _activeBuffer: GPUBuffer | undefined;
    private _activeTyped: TgpuBuffer<d.AnyWgslData> | undefined;
    private _activeGeneration = 0;
    private _activeConsumers: Consumer[] = [];
    private _componentBindings: ComponentBinding[] = [];
    private _membershipCounts = new Map<number, number>();
    private _presenceBindings: PresenceBinding[] = [];
    private _presenceUnsubscribes: (() => void)[] = [];
    private _rowObservers = new Set<(eid: number, row: number) => void>();
    private _rowReferences: { source: GpuTable; offset: number; unsubscribe: () => void }[] = [];

    constructor(state: State, name: string, record: T, options: GpuTableOptions = {}) {
        if (!name) throw new Error("GpuTable: name must not be empty");
        this._state = state;
        this.name = name;
        this.record = record;
        this.rowBytes = d.sizeOf(record);
        if (this.rowBytes <= 0 || (this.rowBytes & 3) !== 0) {
            throw new Error(
                `GpuTable "${name}": record size ${this.rowBytes} must be a positive multiple of 4 bytes`,
            );
        }
        this.gpuOnly = options.gpuOnly ?? false;
        const device = state.gpu.device;
        const bindingLimit = device.limits.maxStorageBufferBindingSize;
        const bufferLimit = device.limits.maxBufferSize;
        this.maxRows = Math.min(
            Math.floor(bindingLimit / this.rowBytes),
            Math.floor(bufferLimit / this.rowBytes),
        );
        this.maxEntityRows = Math.min(Math.floor(bindingLimit / 4), Math.floor(bufferLimit / 4));
        if (this.maxRows < 1) this.refuse(bindingLimit, 1);
        if (!this.gpuOnly) {
            this._bytes = new Uint8Array(0);
            this._view = new DataView(this._bytes.buffer);
        }
        this._freeRows = new Uint32Array(1);
        this.reserveSlots(1);
        this._activeRows = new Uint32Array(2);
        this._activeIndex = new Int32Array(1);
        this._activeIndex.fill(-1);
        this.replaceActiveBuffer(1);
    }

    /** Allocated GPU record slots. */
    get capacity(): number {
        return this._capacity;
    }

    /** Highest dense row touched by this table. */
    get highWater(): number {
        return this._highWater;
    }

    /** Number of active dense rows. */
    get count(): number {
        return this._activeCount;
    }

    /** Changes whenever the underlying record buffer is replaced. */
    get generation(): number {
        return this._generation;
    }

    /** Changes whenever the opt-in eid-to-row map buffer is replaced. */
    get mapGeneration(): number {
        return this._mapGeneration;
    }

    /** Bytes uploaded to the eid map on the most recent table upload. */
    get lastMapUploadBytes(): number {
        return this._lastMapUploadBytes;
    }

    /** Changes whenever the dense active-row list buffer is replaced. */
    get activeGeneration(): number {
        return this._activeGeneration;
    }

    get buffer(): GPUBuffer {
        return this._buffer;
    }

    get typed(): TgpuBuffer<d.AnyWgslData> {
        return this._typed;
    }

    /** Host record bytes; unavailable for a GPU-only table. */
    get bytes(): Uint8Array {
        if (!this._bytes) throw new Error(`GpuTable "${this.name}" is GPU-only`);
        return this._bytes;
    }

    /** Optional eid-to-row map, encoded as row + 1; zero means that the eid is not present. */
    get eidToRowBuffer(): GPUBuffer | undefined {
        return this._mapBuffer;
    }

    get eidToRowTyped(): TgpuBuffer<d.AnyWgslData> | undefined {
        return this._mapTyped;
    }

    /** Dense active row indices, compacted independently of stable row slots. */
    get activeRowsBuffer(): GPUBuffer | undefined {
        return this._activeBuffer;
    }

    get activeRowsTyped(): TgpuBuffer<d.AnyWgslData> | undefined {
        return this._activeTyped;
    }

    /** Chosen record upload path from the most recent draw-group upload point. */
    get lastUploadPath(): TableUploadPath {
        return this._lastUploadPath;
    }

    /** Ensure at least `rows` dense slots are allocated. */
    reserveSlots(rows: number): void {
        if (!Number.isSafeInteger(rows) || rows < 0) {
            throw new RangeError(
                `GpuTable "${this.name}": row count must be a non-negative safe integer, got ${rows}`,
            );
        }
        if (rows > this.maxRows)
            this.refuse(this._state.gpu.device.limits.maxStorageBufferBindingSize, rows);
        this._highWater = Math.max(this._highWater, rows);
        if (rows <= this._capacity) return;
        let capacity = Math.max(1, this._capacity);
        while (capacity < rows) capacity = Math.min(this.maxRows, capacity * 2);
        const oldCapacity = this._capacity;
        this._capacity = capacity;
        if (this._bytes) {
            const bytes = new Uint8Array(capacity * this.rowBytes);
            bytes.set(this._bytes);
            this._bytes = bytes;
            this._view = new DataView(bytes.buffer);
        }
        const dirty = new Uint32Array((capacity + 31) >>> 5);
        dirty.set(this._dirty);
        this._dirty = dirty;
        this.replaceRecordBuffer(oldCapacity);
        this.ensureSlotCapacity(capacity);
    }

    /** Mark a contiguous range of record rows changed without per-row callbacks. */
    markRange(firstRow: number, count: number): void {
        if (this.gpuOnly) throw new Error(`GpuTable "${this.name}" is GPU-only`);
        if (
            !Number.isSafeInteger(firstRow) ||
            !Number.isSafeInteger(count) ||
            firstRow < 0 ||
            count < 0
        ) {
            throw new RangeError(`GpuTable "${this.name}": invalid row range ${firstRow}+${count}`);
        }
        if (count === 0) return;
        this.reserveSlots(firstRow + count);
        const last = firstRow + count - 1;
        const firstWord = firstRow >>> 5;
        const lastWord = last >>> 5;
        for (let word = firstWord; word <= lastWord; word++) {
            const low = word === firstWord ? firstRow & 31 : 0;
            const high = word === lastWord ? last & 31 : 31;
            const upperMask = high === 31 ? 0xffffffff : (1 << (high + 1)) - 1;
            const lowerMask = low === 0 ? 0 : (1 << low) - 1;
            this._dirty[word] |= (upperMask & ~lowerMask) >>> 0;
        }
    }

    /** Bind component fields to this struct table and use component membership to allocate rows. */
    bindComponent(component: Component, fields: Record<string, string>): void {
        this.bindFields(component, fields);
        this.bindMembership(component);
    }

    /** Bind selected struct fields to component columns without making that component own rows. */
    bindFields(component: Component, fields: Record<string, string>): void {
        if (this.gpuOnly)
            throw new Error(`GpuTable "${this.name}" cannot bind CPU component columns`);
        const record = this.record as d.AnyWgslStruct;
        if (record.type !== "struct") {
            throw new Error(`GpuTable "${this.name}": component binding requires a struct record`);
        }
        const recordFields = record.propTypes as Record<string, d.BaseData>;
        const binding = this.binding(component);
        this._state.of(component);
        for (const [recordName, componentName] of Object.entries(fields)) {
            if (!recordFields[recordName]) {
                throw new Error(`GpuTable "${this.name}": unknown record field "${recordName}"`);
            }
            if (binding.fields.some((field) => field.recordName === recordName)) {
                throw new Error(
                    `GpuTable "${this.name}": record field "${recordName}" is already bound`,
                );
            }
            const descriptor = component[componentName] as FieldSchema;
            if (!isFieldSchema(descriptor)) {
                throw new Error(
                    `GpuTable "${this.name}": component field "${componentName}" is not a field schema`,
                );
            }
            const recordType = recordFields[recordName];
            const recordTypeName = (recordType as unknown as { type: string }).type;
            const format =
                descriptor.type.ctor === Float32Array
                    ? "f"
                    : descriptor.type.ctor === Int32Array
                      ? "i"
                      : descriptor.type.ctor === Uint32Array
                        ? "u"
                        : null;
            const expectedFormat =
                recordTypeName === "f32" || recordTypeName.endsWith("f")
                    ? "f"
                    : recordTypeName === "i32" || recordTypeName.endsWith("i")
                      ? "i"
                      : recordTypeName === "u32" || recordTypeName.endsWith("u")
                        ? "u"
                        : null;
            const lanes =
                (recordType as unknown as { componentCount?: number }).componentCount ?? 1;
            const bytesPerLane = d.sizeOf(recordType as d.AnyWgslData) / lanes;
            const sourceBytesPerLane = descriptor.type.ctor.BYTES_PER_ELEMENT;
            if (
                format === null ||
                format !== expectedFormat ||
                sourceBytesPerLane !== bytesPerLane ||
                descriptor.type.lanes < lanes ||
                descriptor.type.encode !== undefined ||
                descriptor.type.decode !== undefined
            ) {
                throw new Error(
                    `GpuTable "${this.name}": component field "${componentName}" does not match record field "${recordName}"`,
                );
            }
            const offset = d.memoryLayoutOf(this.record, (row: any) => row[recordName]).offset;
            const field: BoundField = {
                recordName,
                componentName,
                offset,
                lanes,
                sourceLanes: descriptor.type.lanes,
                bytesPerLane,
            };
            binding.fields.push(field);
            binding.unsubscribes.push(
                this._state.observeField(component, componentName, (eid) =>
                    this.writeBoundField(binding, field, eid),
                ),
            );
        }
        for (let i = 0; i < this._activeCount; i++) {
            const eid = this._activeRows[i * 2];
            for (const field of binding.fields) this.writeBoundField(binding, field, eid);
        }
    }

    /** Allocate rows for members of this component without binding component fields. */
    bindMembership(component: Component): void {
        const binding = this.binding(component);
        if (binding.ownsRows) return;
        binding.ownsRows = true;
        this._state.bindTableComponent(component, this);
    }

    /** Keep a u32 record field equal to another table's slot + 1, or zero when absent. */
    bindRowReference(source: GpuTable, recordName: string): void {
        if (this.gpuOnly || this._state !== source._state || source === this)
            throw new Error(`GpuTable "${this.name}": row references require distinct CPU tables in one State`);
        const record = this.record as d.AnyWgslStruct;
        if (record.type !== "struct" || record.propTypes[recordName]?.type !== "u32")
            throw new Error(`GpuTable "${this.name}": row reference "${recordName}" must be u32`);
        const offset = d.memoryLayoutOf(this.record, (row: any) => row[recordName]).offset;
        if (this._rowReferences.some(reference => reference.offset === offset))
            throw new Error(`GpuTable "${this.name}": row reference "${recordName}" is already bound`);
        const observer = (eid: number, row: number) => this.writeRowReference(eid, row, offset);
        source._rowObservers.add(observer);
        this._rowReferences.push({ source, offset, unsubscribe: () => source._rowObservers.delete(observer) });
        for (let i = 0; i < this._activeCount; i++) {
            const eid = this._activeRows[i * 2];
            this.writeRowReference(eid, source.rowIndex(eid), offset);
        }
    }

    private writeRowReference(eid: number, sourceRow: number, offset: number): void {
        const row = this.rowIndex(eid);
        if (row < 0) return;
        if (sourceRow < 0) {
            for (const binding of this._componentBindings) {
                if (!this._state.has(eid, binding.component)) continue;
                for (const field of binding.fields) {
                    if (field.offset === offset) {
                        this.writeBoundField(binding, field, eid);
                        return;
                    }
                }
            }
        }
        this._view!.setUint32(row * this.rowBytes + offset, sourceRow + 1, true);
        this.markRange(row, 1);
    }

    /** Mirror component presence into a u32 record mask without making that component own a row. */
    bindPresence(component: Component, recordName: string, mask = 1): void {
        if (this.gpuOnly)
            throw new Error(`GpuTable "${this.name}" cannot bind CPU component presence`);
        if (!Number.isSafeInteger(mask) || mask <= 0 || mask > 0xffffffff) {
            throw new RangeError(`GpuTable "${this.name}": presence mask must be a positive u32`);
        }
        const record = this.record as d.AnyWgslStruct;
        const recordType = record.type === "struct" ? record.propTypes[recordName] : undefined;
        if (!recordType || (recordType as unknown as { type: string }).type !== "u32") {
            throw new Error(`GpuTable "${this.name}": presence field "${recordName}" must be u32`);
        }
        const offset = d.memoryLayoutOf(this.record, (row: any) => row[recordName]).offset;
        const binding = { component, recordName, offset, mask };
        this._presenceBindings.push(binding);
        this._presenceUnsubscribes.push(
            this._state.observeMembership(component, (eid, present) =>
                this.writePresence(eid, binding, present),
            ),
        );
        const active = this._activeRows;
        for (let i = 0; i < this._activeCount; i++) {
            const eid = active[i * 2];
            this.writePresence(eid, binding, this._state.has(eid, component));
        }
    }

    private writePresence(eid: number, binding: PresenceBinding, present: boolean): void {
        const row = this.rowIndex(eid);
        if (row < 0) return;
        const byteOffset = row * this.rowBytes + binding.offset;
        const current = this._view!.getUint32(byteOffset, true);
        const next = present ? current | binding.mask : current & ~binding.mask;
        if (current === next) return;
        this._view!.setUint32(byteOffset, next, true);
        this.markRange(row, 1);
    }

    /** Attach an entity to a bound component table and seed its row from all bound columns. @internal */
    attachComponent(eid: number, component: Component): void {
        const binding = this._componentBindings.find(
            (item) => item.ownsRows && idOf(item.component) === idOf(component),
        );
        if (!binding) throw new Error(`GpuTable "${this.name}": component has no row binding`);
        this.acquire(eid);
        this._membershipCounts.set(eid, (this._membershipCounts.get(eid) ?? 0) + 1);
        for (const source of this._componentBindings) {
            for (const field of source.fields) this.writeBoundField(source, field, eid);
        }
        for (const presence of this._presenceBindings) {
            this.writePresence(eid, presence, this._state.has(eid, presence.component));
        }
    }

    /** Detach one component owner, releasing the row when its last owner leaves. @internal */
    detachComponent(eid: number, _component: Component): void {
        const count = this._membershipCounts.get(eid) ?? 0;
        if (count <= 1) this.release(eid);
        else this._membershipCounts.set(eid, count - 1);
    }

    private binding(component: Component): ComponentBinding {
        const existing = this._componentBindings.find(
            (item) => idOf(item.component) === idOf(component),
        );
        if (existing) return existing;
        const binding: ComponentBinding = {
            component,
            fields: [],
            unsubscribes: [],
            ownsRows: false,
        };
        this._componentBindings.push(binding);
        return binding;
    }

    /** Map an eid to a stable dense row, taking a freed slot before extending the table. */
    acquire(eid: number): number {
        this.ensureEntityRows(eid + 1);
        const existing = this._eidToRow[eid];
        if (existing !== 0) return existing - 1;
        const row = this._freeCount > 0 ? this._freeRows[--this._freeCount] : this._nextRow++;
        this.reserveSlots(row + 1);
        this._eidToRow[eid] = row + 1;
        this._mapDirty = true;
        if (this._bytes) {
            this._bytes.fill(0, row * this.rowBytes, (row + 1) * this.rowBytes);
            this.markRange(row, 1);
        }
        this.activateRow(eid, row);
        for (const reference of this._rowReferences)
            this.writeRowReference(eid, reference.source.rowIndex(eid), reference.offset);
        for (const observer of this._rowObservers) observer(eid, row);
        return row;
    }

    /** Release an eid's dense slot; the next acquire may reuse the slot. */
    release(eid: number): void {
        this._membershipCounts.delete(eid);
        if (!Number.isSafeInteger(eid) || eid < 0 || eid >= this._eidToRow.length) return;
        const encoded = this._eidToRow[eid];
        if (encoded === 0) return;
        const row = encoded - 1;
        this._eidToRow[eid] = 0;
        this._freeRows[this._freeCount++] = row;
        this._mapDirty = true;
        this.deactivateRow(row);
        for (const observer of this._rowObservers) observer(eid, -1);
    }

    /** Activate an eid row in the compact active list. */
    activate(eid: number): void {
        const row = this.rowIndex(eid);
        if (row < 0) throw new Error(`GpuTable "${this.name}": eid ${eid} has no row`);
        this.activateRow(eid, row);
    }

    /** Remove an eid row from the compact active list. */
    deactivate(eid: number): void {
        const row = this.rowIndex(eid);
        if (row >= 0) this.deactivateRow(row);
    }

    /** Row index shaders use for `eid`; -1 means absent from a dense table. */
    rowIndex(eid: number): number {
        if (!Number.isSafeInteger(eid) || eid < 0) return -1;
        return eid < this._eidToRow.length ? this._eidToRow[eid] - 1 : -1;
    }

    /** Subscribe to the record buffer; growth invokes consumers with its new generation. */
    subscribe(consumer: Consumer): () => void {
        this._consumers.push(consumer);
        consumer(this._buffer, this._generation);
        return () => removeConsumer(this._consumers, consumer);
    }

    /** Opt into an uploaded eid-to-row map for shaders that start from an eid. */
    enableEidLookup(): GPUBuffer {
        this._mapEnabled = true;
        if (this._mapCapacity === 0) this.ensureEntityRows(1);
        else if (!this._mapBuffer) this.replaceMapBuffer();
        this._mapDirty = true;
        return this._mapBuffer!;
    }

    /** Subscribe to the opt-in eid-to-row map buffer. */
    subscribeMap(consumer: Consumer): () => void {
        const buffer = this.enableEidLookup();
        this._mapConsumers.push(consumer);
        consumer(buffer, this._mapGeneration);
        return () => removeConsumer(this._mapConsumers, consumer);
    }

    /** Subscribe to the compact active-row list buffer. */
    subscribeActiveRows(consumer: Consumer): () => void {
        if (!this._activeBuffer) throw new Error(`GpuTable "${this.name}" has no active-row list`);
        this._activeConsumers.push(consumer);
        consumer(this._activeBuffer, this._activeGeneration);
        return () => removeConsumer(this._activeConsumers, consumer);
    }

    /** Upload once at the engine's draw-group upload point. */
    upload(): void {
        if (this._disposed) return;
        this._lastMapUploadBytes = 0;
        if (this.changedRows() === 0) {
            this._lastUploadPath = "none";
        } else {
            if (!this._bytes) throw new Error(`GpuTable "${this.name}" is GPU-only`);
            this._state.gpu.device.queue.writeBuffer(
                this._buffer,
                0,
                this._bytes.buffer,
                0,
                this._highWater * this.rowBytes,
            );
            this._lastUploadPath = "writeBuffer";
            this._dirty.fill(0);
        }
        this.uploadMap();
        this.uploadActiveRows();
    }

    /** Release this table's resources and registry entries. */
    dispose(): void {
        if (this._disposed) return;
        this._disposed = true;
        for (const binding of this._componentBindings) {
            for (const unsubscribe of binding.unsubscribes) unsubscribe();
            if (binding.ownsRows) this._state.unbindTableComponent(binding.component, this);
        }
        this._componentBindings.length = 0;
        for (const unsubscribe of this._presenceUnsubscribes) unsubscribe();
        this._presenceUnsubscribes.length = 0;
        this._presenceBindings.length = 0;
        this._membershipCounts.clear();
        for (const reference of this._rowReferences) reference.unsubscribe();
        this._rowReferences.length = 0;
        this._rowObservers.clear();
        const buffers = [this._buffer];
        if (this._mapBuffer) buffers.push(this._mapBuffer);
        if (this._activeBuffer) buffers.push(this._activeBuffer);
        for (const buffer of buffers) buffer.destroy();
        for (const name of [this.name, this.mapName, this.activeName]) {
            if (name && this._state.gpu.buffers.has(name)) this._state.gpu.buffers.delete(name);
            if (name && this._state.gpu.typed.has(name)) this._state.gpu.typed.delete(name);
        }
    }

    get mapName(): string | undefined {
        return this._mapBuffer ? `${this.name}:eid-to-row` : undefined;
    }

    get activeName(): string | undefined {
        return this._activeBuffer ? `${this.name}:active-rows` : undefined;
    }

    private refuse(limit: number, rows: number): never {
        throw new RangeError(
            `GpuTable "${this.name}" exceeds device limit maxStorageBufferBindingSize (${limit} bytes): ${rows} rows × ${this.rowBytes} bytes`,
        );
    }

    private changedRows(): number {
        let count = 0;
        for (let i = 0; i < this._dirty.length; i++) {
            let bits = this._dirty[i];
            while (bits !== 0) {
                bits &= bits - 1;
                count++;
            }
        }
        return count;
    }

    private writeBoundField(binding: ComponentBinding, field: BoundField, eid: number): void {
        if (!this._state.has(eid, binding.component)) return;
        for (const reference of this._rowReferences)
            if (reference.offset === field.offset && reference.source.rowIndex(eid) >= 0) return;
        const row = this.rowIndex(eid);
        if (row < 0) return;
        const bytes = this._bytes;
        const view = this._view;
        if (!bytes || !view) return;
        const source = this._state.of(binding.component)[field.componentName] as {
            column: TypedArray;
        };
        const column = source.column;
        const sourceBase = eid * field.sourceLanes;
        const targetBase = row * this.rowBytes + field.offset;
        for (let lane = 0; lane < field.lanes; lane++) {
            const offset = targetBase + lane * field.bytesPerLane;
            const value = column[sourceBase + lane];
            if (column instanceof Float32Array) view.setFloat32(offset, value, true);
            else if (column instanceof Int32Array) view.setInt32(offset, value, true);
            else if (column instanceof Uint32Array) view.setUint32(offset, value, true);
            else if (column instanceof Uint16Array) view.setUint16(offset, value, true);
            else if (column instanceof Uint8Array) view.setUint8(offset, value);
            else throw new Error(`GpuTable "${this.name}": unsupported component column`);
        }
        this.markRange(row, 1);
    }

    private ensureEntityRows(rows: number): void {
        if (!Number.isSafeInteger(rows) || rows < 1) {
            throw new RangeError(`GpuTable "${this.name}": invalid entity-map size ${rows}`);
        }
        if (this._mapEnabled && rows > this.maxEntityRows) {
            this.refuse(this._state.gpu.device.limits.maxStorageBufferBindingSize, rows);
        }
        this._mapHighWater = Math.max(this._mapHighWater, rows);
        if (rows <= this._mapCapacity) return;
        let capacity = Math.max(1, this._mapCapacity);
        while (capacity < rows) {
            capacity = this._mapEnabled ? Math.min(this.maxEntityRows, capacity * 2) : capacity * 2;
        }
        const map = new Uint32Array(capacity);
        map.set(this._eidToRow);
        this._eidToRow = map;
        this._mapCapacity = capacity;
        if (this._mapEnabled) this.replaceMapBuffer();
    }

    private replaceMapBuffer(): void {
        if (this._mapCapacity > this.maxEntityRows) {
            this.refuse(
                this._state.gpu.device.limits.maxStorageBufferBindingSize,
                this._mapCapacity,
            );
        }
        const registryName = `${this.name}:eid-to-row`;
        const registered = this._state.gpu.buffers.get(registryName);
        if (registered && registered !== this._mapBuffer) {
            throw new Error(
                `GpuTable "${this.name}": GPU registry name "${registryName}" is already in use`,
            );
        }
        const previous = this._mapBuffer;
        const buffer = this._state.gpu.device.createBuffer({
            label: `table-${this.name}-eid-to-row-g${this._mapGeneration + 1}`,
            size: Math.max(1, this._mapCapacity) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this._state.own(buffer);
        if (previous) {
            const encoder = this._state.gpu.device.createCommandEncoder({
                label: `table-${this.name}-grow-map`,
            });
            encoder.copyBufferToBuffer(previous, 0, buffer, 0, previous.size);
            this._state.gpu.device.queue.submit([encoder.finish()]);
            previous.destroy();
        }
        this._mapBuffer = buffer;
        this._mapTyped = this._state.gpu.root
            .createBuffer(d.arrayOf(d.u32, Math.max(1, this._mapCapacity)), buffer)
            .$usage("storage") as TgpuBuffer<d.AnyWgslData>;
        this._state.gpu.buffers.set(registryName, buffer);
        this._state.gpu.typed.set(registryName, this._mapTyped);
        this._mapGeneration++;
        for (let i = 0; i < this._mapConsumers.length; i++) {
            this._mapConsumers[i](buffer, this._mapGeneration);
        }
    }

    private ensureSlotCapacity(rows: number): void {
        const currentCapacity = this._activeRows.length >>> 1;
        if (rows <= currentCapacity) return;
        let capacity = Math.max(1, currentCapacity);
        while (capacity < rows) capacity = Math.min(this.maxRows, capacity * 2);
        const free = new Uint32Array(capacity);
        free.set(this._freeRows);
        this._freeRows = free;
        const active = new Uint32Array(capacity * 2);
        active.set(this._activeRows);
        this._activeRows = active;
        const indices = new Int32Array(capacity);
        indices.fill(-1);
        indices.set(this._activeIndex);
        this._activeIndex = indices;
    }

    private replaceRecordBuffer(oldCapacity: number): void {
        const device = this._state.gpu.device;
        const previous = oldCapacity > 0 ? this._buffer : undefined;
        const buffer = device.createBuffer({
            label: `table-${this.name}-g${this._generation + 1}`,
            size: this._capacity * this.rowBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this._state.own(buffer);
        if (previous) {
            const encoder = device.createCommandEncoder({ label: `table-${this.name}-grow` });
            encoder.copyBufferToBuffer(previous, 0, buffer, 0, previous.size);
            device.queue.submit([encoder.finish()]);
            previous.destroy();
        }
        this._buffer = buffer;
        this._typed = this._state.gpu.root
            .createBuffer(d.arrayOf(this.record, this._capacity), buffer)
            .$usage("storage") as TgpuBuffer<d.AnyWgslData>;
        this._state.gpu.buffers.set(this.name, buffer);
        this._state.gpu.typed.set(this.name, this._typed);
        this._generation++;
        for (let i = 0; i < this._consumers.length; i++) {
            this._consumers[i](buffer, this._generation);
        }
    }

    private uploadMap(): void {
        if (!this._mapDirty || !this._mapBuffer) return;
        this._lastMapUploadBytes = this._mapHighWater * 4;
        this._state.gpu.device.queue.writeBuffer(
            this._mapBuffer,
            0,
            this._eidToRow.buffer,
            0,
            this._lastMapUploadBytes,
        );
        this._mapDirty = false;
    }

    private replaceActiveBuffer(capacity: number): void {
        const device = this._state.gpu.device;
        const previous = this._activeBuffer;
        const buffer = device.createBuffer({
            label: `table-${this.name}-active-g${this._activeGeneration + 1}`,
            size: capacity * d.sizeOf(ACTIVE_ROW),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this._state.own(buffer);
        if (previous) {
            const encoder = device.createCommandEncoder({
                label: `table-${this.name}-grow-active`,
            });
            encoder.copyBufferToBuffer(previous, 0, buffer, 0, previous.size);
            device.queue.submit([encoder.finish()]);
            previous.destroy();
        }
        this._activeBuffer = buffer;
        this._activeCapacity = capacity;
        this._activeTyped = this._state.gpu.root
            .createBuffer(d.arrayOf(ACTIVE_ROW, capacity), buffer)
            .$usage("storage") as TgpuBuffer<d.AnyWgslData>;
        this._state.gpu.buffers.set(`${this.name}:active-rows`, buffer);
        this._state.gpu.typed.set(`${this.name}:active-rows`, this._activeTyped);
        this._activeGeneration++;
        for (let i = 0; i < this._activeConsumers.length; i++) {
            this._activeConsumers[i](buffer, this._activeGeneration);
        }
    }

    private activateRow(eid: number, row: number): void {
        this.ensureSlotCapacity(row + 1);
        if (this._activeIndex[row] >= 0) return;
        if (this._activeCount >= this._activeCapacity)
            this.replaceActiveBuffer(Math.min(this.maxRows, this._activeCapacity * 2));
        const activeIndex = this._activeCount++;
        this._activeIndex[row] = activeIndex;
        this._activeRows[activeIndex * 2] = eid;
        this._activeRows[activeIndex * 2 + 1] = row;
        this._activeDirty = true;
    }

    private deactivateRow(row: number): void {
        if (row < 0 || row >= this._activeIndex.length) return;
        const index = this._activeIndex[row];
        if (index < 0) return;
        const lastIndex = --this._activeCount;
        const lastEid = this._activeRows[lastIndex * 2];
        const lastRow = this._activeRows[lastIndex * 2 + 1];
        this._activeRows[index * 2] = lastEid;
        this._activeRows[index * 2 + 1] = lastRow;
        this._activeIndex[lastRow] = index;
        this._activeIndex[row] = -1;
        this._activeDirty = true;
    }

    private uploadActiveRows(): void {
        if (!this._activeDirty || !this._activeBuffer) return;
        if (this._activeCount > 0) {
            this._state.gpu.device.queue.writeBuffer(
                this._activeBuffer,
                0,
                this._activeRows.buffer,
                0,
                this._activeCount * d.sizeOf(ACTIVE_ROW),
            );
        }
        this._activeDirty = false;
    }
}

function removeConsumer(consumers: Consumer[], consumer: Consumer): void {
    const index = consumers.indexOf(consumer);
    if (index >= 0) consumers.splice(index, 1);
}
