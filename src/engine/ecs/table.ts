import type { TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import type { State } from "./state";

const SCATTER_STAGERS = 3;
const DEFAULT_UPLOAD_THRESHOLD: number | undefined = undefined;

export type TableUploadPath = "none" | "scatter" | "writeBuffer";
export interface GpuTableOptions {
    /** A mapped-staging scatter crossover, enabled only when measured to beat writeBuffer. */
    uploadThreshold?: number;
    /** No CPU backing or upload path; a GPU pass owns every row write. */
    gpuOnly?: boolean;
}

type Stager = {
    readonly buffer: GPUBuffer;
    readonly epoch: number;
    available: boolean;
    mapped(): void;
    rejected(error: unknown): void;
};

type Consumer = (buffer: GPUBuffer, generation: number) => void;

/** Dense-slot storage with stable free-list allocation and one typed record layout. */
export class GpuTable<T extends d.AnyWgslData = d.AnyWgslData> {
    readonly name: string;
    readonly record: T;
    readonly rowBytes: number;
    readonly maxRows: number;
    readonly maxEntityRows: number;
    readonly uploadThreshold: number | undefined;
    readonly gpuOnly: boolean;
    private readonly _state: State;
    private _capacity = 0;
    private _highWater = 1;
    private _generation = 0;
    private _bytes: Uint8Array | undefined;
    private _dirty = new Uint32Array(0);
    private _buffer!: GPUBuffer;
    private _typed!: TgpuBuffer<d.AnyWgslData>;
    private _pipeline!: GPUComputePipeline;
    private _layout!: GPUBindGroupLayout;
    private _source: GPUBuffer | undefined;
    private _dirtyGpu: GPUBuffer | undefined;
    private _bindGroup!: GPUBindGroup;
    private _stagers: Stager[] = [];
    private _epoch = 0;
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

    constructor(state: State, name: string, record: T, options: GpuTableOptions | number = {}) {
        if (!name) throw new Error("GpuTable: name must not be empty");
        const normalized = typeof options === "number" ? { uploadThreshold: options } : options;
        const threshold = normalized.uploadThreshold ?? DEFAULT_UPLOAD_THRESHOLD;
        if (threshold !== undefined && !(threshold > 0 && threshold <= 1)) {
            throw new Error(
                `GpuTable "${name}": upload threshold must be in (0, 1], got ${threshold}`,
            );
        }
        this._state = state;
        this.name = name;
        this.record = record;
        this.rowBytes = d.sizeOf(record);
        if (this.rowBytes <= 0 || (this.rowBytes & 3) !== 0) {
            throw new Error(
                `GpuTable "${name}": record size ${this.rowBytes} must be a positive multiple of 4 bytes`,
            );
        }
        this.gpuOnly = normalized.gpuOnly ?? false;
        this.uploadThreshold = threshold;
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
        }
        if (this.uploadThreshold !== undefined) this.createScatterPipeline();
        this._freeRows = new Uint32Array(1);
        this.reserveSlots(1);
        this._activeRows = new Uint32Array(1);
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

    /** Map an eid to a stable dense row, taking a freed slot before extending the table. */
    acquire(eid: number): number {
        this.ensureEntityRows(eid + 1);
        const existing = this._eidToRow[eid];
        if (existing !== 0) return existing - 1;
        const row = this._freeCount > 0 ? this._freeRows[--this._freeCount] : this._nextRow++;
        this.reserveSlots(row + 1);
        this._eidToRow[eid] = row + 1;
        this._mapDirty = true;
        this.activateRow(row);
        return row;
    }

    /** Release an eid's dense slot; the next acquire may reuse the slot. */
    release(eid: number): void {
        if (!Number.isSafeInteger(eid) || eid < 0 || eid >= this._eidToRow.length) return;
        const encoded = this._eidToRow[eid];
        if (encoded === 0) return;
        const row = encoded - 1;
        this._eidToRow[eid] = 0;
        this._freeRows[this._freeCount++] = row;
        this._mapDirty = true;
        this.deactivateRow(row);
    }

    /** Activate an eid row in the compact active list. */
    activate(eid: number): void {
        const row = this.rowIndex(eid);
        if (row < 0) throw new Error(`GpuTable "${this.name}": eid ${eid} has no row`);
        this.activateRow(row);
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
        const changed = this.changedRows();
        if (changed === 0) {
            this._lastUploadPath = "none";
        } else {
            const device = this._state.gpu.device;
            const highWater = this._addressedRows();
            if (
                this.uploadThreshold !== undefined &&
                changed / Math.max(1, highWater) <= this.uploadThreshold
            ) {
                const stager = this.availableStager();
                if (!stager)
                    throw new Error(
                        `GpuTable "${this.name}": mapped scatter staging is unavailable`,
                    );
                this.scatter(stager, highWater);
                this._lastUploadPath = "scatter";
            } else {
                if (!this._bytes) throw new Error(`GpuTable "${this.name}" is GPU-only`);
                device.queue.writeBuffer(
                    this._buffer,
                    0,
                    this._bytes.buffer,
                    0,
                    highWater * this.rowBytes,
                );
                this._lastUploadPath = "writeBuffer";
            }
            this._dirty.fill(0);
        }
        this.uploadMap();
        this.uploadActiveRows();
    }

    /** Release this table's resources and registry entries. */
    dispose(): void {
        if (this._disposed) return;
        this._disposed = true;
        this._epoch++;
        const buffers = [this._buffer, ...this._stagers.map((stager) => stager.buffer)];
        if (this._source) buffers.push(this._source);
        if (this._dirtyGpu) buffers.push(this._dirtyGpu);
        if (this._mapBuffer) buffers.push(this._mapBuffer);
        if (this._activeBuffer) buffers.push(this._activeBuffer);
        for (const buffer of buffers) buffer.destroy();
        this._stagers.length = 0;
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

    private _addressedRows(): number {
        return this._highWater;
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

    private createScatterPipeline(): void {
        const words = this.rowBytes >>> 2;
        const device = this._state.gpu.device;
        this._layout = device.createBindGroupLayout({
            label: `table-${this.name}-scatter-layout`,
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ],
        });
        const module = device.createShaderModule({
            label: `table-${this.name}-scatter`,
            code: `
@group(0) @binding(0) var<storage, read> changed: array<u32>;
@group(0) @binding(1) var<storage, read> source: array<u32>;
@group(0) @binding(2) var<storage, read_write> rows: array<u32>;
@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if ((changed[row >> 5u] & (1u << (row & 31u))) == 0u) { return; }
    let wordsPerRow = ${words}u;
    for (var word = 0u; word < wordsPerRow; word++) {
        rows[row * wordsPerRow + word] = source[row * wordsPerRow + word];
    }
}`,
        });
        const pipelineLayout = device.createPipelineLayout({
            label: `table-${this.name}-scatter-pipeline-layout`,
            bindGroupLayouts: [this._layout],
        });
        this._pipeline = device.createComputePipeline({
            label: `table-${this.name}-scatter-pipeline`,
            layout: pipelineLayout,
            compute: { module, entryPoint: "scatter" },
        });
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
        if (rows <= this._activeRows.length) return;
        let capacity = Math.max(1, this._activeRows.length);
        while (capacity < rows) capacity = Math.min(this.maxRows, capacity * 2);
        const free = new Uint32Array(capacity);
        free.set(this._freeRows);
        this._freeRows = free;
        const active = new Uint32Array(capacity);
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
        this.replaceScatterBuffers();
        this._generation++;
        for (let i = 0; i < this._consumers.length; i++) {
            this._consumers[i](buffer, this._generation);
        }
    }

    private replaceScatterBuffers(): void {
        if (this.uploadThreshold === undefined) return;
        const device = this._state.gpu.device;
        this._source?.destroy();
        this._dirtyGpu?.destroy();
        for (let i = 0; i < this._stagers.length; i++) this._stagers[i].buffer.destroy();
        const dirtyBytes = Math.max(4, this._dirty.byteLength);
        const bytes = Math.max(4, this._capacity * this.rowBytes);
        this._source = device.createBuffer({
            label: `table-${this.name}-scatter-source`,
            size: bytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this._dirtyGpu = device.createBuffer({
            label: `table-${this.name}-scatter-dirty`,
            size: dirtyBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this._state.own(this._source);
        this._state.own(this._dirtyGpu);
        this._bindGroup = device.createBindGroup({
            label: `table-${this.name}-scatter-bind-group`,
            layout: this._layout,
            entries: [
                { binding: 0, resource: { buffer: this._dirtyGpu } },
                { binding: 1, resource: { buffer: this._source } },
                { binding: 2, resource: { buffer: this._buffer } },
            ],
        });
        this._stagers = [];
        this._epoch++;
        if (this.uploadThreshold !== undefined) {
            const epoch = this._epoch;
            const stageBytes = dirtyBytes + bytes;
            for (let i = 0; i < SCATTER_STAGERS; i++) {
                const stager = this.createStager(device, stageBytes, epoch);
                this._stagers.push(stager);
                this._state.own(stager.buffer);
            }
        }
    }

    private createStager(device: GPUDevice, size: number, epoch: number): Stager {
        const stager: Stager = {
            buffer: device.createBuffer({
                label: `table-${this.name}-mapped-scatter`,
                size,
                usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
                mappedAtCreation: true,
            }),
            epoch,
            available: true,
            mapped: () => {
                if (this._disposed || this._epoch !== stager.epoch) stager.buffer.destroy();
                else stager.available = true;
            },
            rejected: () => {
                stager.buffer.destroy();
                if (!this._disposed && this._epoch === stager.epoch) {
                    this._stagingError = new Error(
                        `GpuTable "${this.name}" staging mapAsync failed`,
                    );
                }
            },
        };
        return stager;
    }

    private _stagingError: Error | undefined;

    private availableStager(): Stager | undefined {
        if (this._stagingError) throw this._stagingError;
        for (let i = 0; i < this._stagers.length; i++) {
            if (this._stagers[i].available) return this._stagers[i];
        }
        return undefined;
    }

    private scatter(stager: Stager, rows: number): void {
        const range = stager.buffer.getMappedRange();
        const dirtyBytes = this._dirty.byteLength;
        const dirty = new Uint8Array(range, 0, dirtyBytes);
        const source = new Uint8Array(range, dirtyBytes, this._capacity * this.rowBytes);
        dirty.set(new Uint8Array(this._dirty.buffer, this._dirty.byteOffset, dirtyBytes));
        if (this._bytes) source.set(this._bytes);
        stager.buffer.unmap();
        const device = this._state.gpu.device;
        const encoder = device.createCommandEncoder({ label: `table-${this.name}-scatter-upload` });
        encoder.copyBufferToBuffer(stager.buffer, 0, this._dirtyGpu!, 0, dirtyBytes);
        encoder.copyBufferToBuffer(
            stager.buffer,
            dirtyBytes,
            this._source!,
            0,
            this._capacity * this.rowBytes,
        );
        const pass = encoder.beginComputePass({ label: `table-${this.name}-scatter-pass` });
        pass.setPipeline(this._pipeline);
        pass.setBindGroup(0, this._bindGroup);
        pass.dispatchWorkgroups(Math.ceil(rows / 64));
        pass.end();
        device.queue.submit([encoder.finish()]);
        stager.available = false;
        stager.buffer.mapAsync(GPUMapMode.WRITE).then(stager.mapped, stager.rejected);
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
            size: capacity * 4,
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
            .createBuffer(d.arrayOf(d.u32, capacity), buffer)
            .$usage("storage") as TgpuBuffer<d.AnyWgslData>;
        this._state.gpu.buffers.set(`${this.name}:active-rows`, buffer);
        this._state.gpu.typed.set(`${this.name}:active-rows`, this._activeTyped);
        this._activeGeneration++;
        for (let i = 0; i < this._activeConsumers.length; i++) {
            this._activeConsumers[i](buffer, this._activeGeneration);
        }
    }

    private activateRow(row: number): void {
        this.ensureSlotCapacity(row + 1);
        if (this._activeIndex[row] >= 0) return;
        if (this._activeCount >= this._activeCapacity)
            this.replaceActiveBuffer(Math.min(this.maxRows, this._activeCapacity * 2));
        this._activeIndex[row] = this._activeCount;
        this._activeRows[this._activeCount++] = row;
        this._activeDirty = true;
    }

    private deactivateRow(row: number): void {
        if (row < 0 || row >= this._activeIndex.length) return;
        const index = this._activeIndex[row];
        if (index < 0) return;
        const lastIndex = --this._activeCount;
        const lastRow = this._activeRows[lastIndex];
        this._activeRows[index] = lastRow;
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
                this._activeCount * 4,
            );
        }
        this._activeDirty = false;
    }
}

function removeConsumer(consumers: Consumer[], consumer: Consumer): void {
    const index = consumers.indexOf(consumer);
    if (index >= 0) consumers.splice(index, 1);
}
