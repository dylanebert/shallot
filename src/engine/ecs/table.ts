import type { TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import type { State } from "./state";

const SCATTER_STAGERS = 3;
const UPLOAD_THRESHOLD = 0.15;

export type TableUploadPath = "none" | "scatter" | "writeBuffer";

type Stager = {
    readonly buffer: GPUBuffer;
    readonly epoch: number;
    available: boolean;
    mapped(): void;
    rejected(error: unknown): void;
};

/** An eid-addressed GPU table with one typed record layout. */
export class GpuTable<T extends d.AnyWgslData = d.AnyWgslData> {
    readonly name: string;
    readonly record: T;
    readonly rowBytes: number;
    readonly maxRows: number;
    readonly uploadThreshold: number;
    private readonly _state: State;
    private _capacity = 0;
    private _highWater = 1;
    private _generation = 0;
    private _bytes = new Uint8Array(0);
    private _view = new DataView(this._bytes.buffer);
    private _dirty = new Uint32Array(0);
    private _buffer!: GPUBuffer;
    private _typed!: TgpuBuffer<d.AnyWgslData>;
    private _slots!: GPUBuffer;
    private _values!: GPUBuffer;
    private _bindGroup!: GPUBindGroup;
    private _pipeline!: GPUComputePipeline;
    private _stagers: Stager[] = [];
    private _consumers: ((buffer: GPUBuffer, generation: number) => void)[] = [];
    private _epoch = 0;
    private _disposed = false;
    private _lastUploadPath: TableUploadPath = "none";

    constructor(state: State, name: string, record: T, uploadThreshold = UPLOAD_THRESHOLD) {
        if (!name) throw new Error("GpuTable: name must not be empty");
        if (!(uploadThreshold > 0 && uploadThreshold <= 1)) {
            throw new Error(
                `GpuTable "${name}": upload threshold must be in (0, 1], got ${uploadThreshold}`,
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
        this.uploadThreshold = uploadThreshold;
        const device = state.gpu.device;
        const limit = device.limits.maxStorageBufferBindingSize;
        this.maxRows = Math.min(
            Math.floor(limit / this.rowBytes),
            Math.floor(device.limits.maxBufferSize / this.rowBytes),
        );
        if (this.maxRows < 1) {
            throw new Error(
                `GpuTable "${name}" exceeds device limit maxStorageBufferBindingSize (${limit} bytes): one ${this.rowBytes}-byte record does not fit`,
            );
        }
        this.createPipeline();
        this.ensure(1);
    }

    /** Current row allocation, including eid zero, which remains available as a sentinel. */
    get capacity(): number {
        return this._capacity;
    }

    /** Highest eid range touched by this table. */
    get highWater(): number {
        return this._highWater;
    }

    /** Changes whenever the underlying GPU buffer is replaced. Consumers rebuild their bind groups then. */
    get generation(): number {
        return this._generation;
    }

    /** The raw storage buffer used by GPU consumers and the shared world registry. */
    get buffer(): GPUBuffer {
        return this._buffer;
    }

    /** TypeGPU wrapper for this table's runtime-sized array of records. */
    get typed(): TgpuBuffer<d.AnyWgslData> {
        return this._typed;
    }

    /** CPU rows in the same packed layout as {@link record}; the view may change after growth. */
    get bytes(): Uint8Array {
        return this._bytes;
    }

    /** Chosen path from the most recent upload point. */
    get lastUploadPath(): TableUploadPath {
        return this._lastUploadPath;
    }

    /** Register a GPU consumer; it is called initially and whenever growth replaces the buffer. */
    subscribe(consumer: (buffer: GPUBuffer, generation: number) => void): () => void {
        this._consumers.push(consumer);
        consumer(this._buffer, this._generation);
        return () => {
            const index = this._consumers.indexOf(consumer);
            if (index >= 0) this._consumers.splice(index, 1);
        };
    }

    /** Grow to include row `eid`, refusing before allocating beyond the named binding limit. */
    ensure(rows: number): void {
        if (!Number.isSafeInteger(rows) || rows < 0) {
            throw new RangeError(
                `GpuTable "${this.name}": row count must be a non-negative safe integer, got ${rows}`,
            );
        }
        if (rows > this.maxRows) {
            const limit = this._state.gpu.device.limits.maxStorageBufferBindingSize;
            throw new RangeError(
                `GpuTable "${this.name}" exceeds device limit maxStorageBufferBindingSize (${limit} bytes): ${rows} rows × ${this.rowBytes} bytes`,
            );
        }
        this._highWater = Math.max(this._highWater, rows);
        if (rows <= this._capacity) return;
        let capacity = Math.max(1, this._capacity);
        while (capacity < rows) capacity = Math.min(this.maxRows, capacity * 2);

        const bytes = new Uint8Array(capacity * this.rowBytes);
        bytes.set(this._bytes);
        this._bytes = bytes;
        this._view = new DataView(bytes.buffer);
        const dirty = new Uint32Array((capacity + 31) >>> 5);
        dirty.set(this._dirty);
        this._dirty = dirty;
        this._capacity = capacity;
        this.replaceBuffers();
    }

    /** Fill one typed record in place and mark its eid changed. */
    write(eid: number, fill: (view: DataView, byteOffset: number) => void): void {
        this.ensure(eid + 1);
        fill(this._view, eid * this.rowBytes);
        this._dirty[eid >>> 5] |= 1 << (eid & 31);
    }

    /** Copy one already-laid-out record and mark its eid changed. */
    writeBytes(eid: number, source: Uint8Array): void {
        if (source.byteLength !== this.rowBytes) {
            throw new RangeError(
                `GpuTable "${this.name}": row is ${source.byteLength} bytes, expected ${this.rowBytes}`,
            );
        }
        this.ensure(eid + 1);
        this._bytes.set(source, eid * this.rowBytes);
        this._dirty[eid >>> 5] |= 1 << (eid & 31);
    }

    /** Mark a row changed after writing directly through {@link bytes}. */
    mark(eid: number): void {
        this.ensure(eid + 1);
        this._dirty[eid >>> 5] |= 1 << (eid & 31);
    }

    /** Upload once at the engine's draw-group upload point. */
    upload(): void {
        if (this._disposed) return;
        const changed = this.changedRows();
        if (changed === 0) {
            this._lastUploadPath = "none";
            return;
        }
        const device = this._state.gpu.device;
        if (changed / this._highWater >= this.uploadThreshold) {
            device.queue.writeBuffer(
                this._buffer,
                0,
                this._bytes.buffer,
                0,
                this._highWater * this.rowBytes,
            );
            this._lastUploadPath = "writeBuffer";
        } else {
            const stager = this.availableStager();
            if (!stager) {
                throw new Error(`GpuTable "${this.name}": mapped scatter staging is unavailable`);
            }
            this.scatter(stager, changed);
            this._lastUploadPath = "scatter";
        }
        this._dirty.fill(0);
    }

    /** Release this table's resources and its shared registry entries. */
    dispose(): void {
        if (this._disposed) return;
        this._disposed = true;
        this._epoch++;
        if (this._state.gpu.buffers.get(this.name) === this._buffer) {
            this._state.gpu.buffers.delete(this.name);
            this._state.gpu.typed.delete(this.name);
        }
        this._buffer.destroy();
        this._slots.destroy();
        this._values.destroy();
        for (let i = 0; i < this._stagers.length; i++) this._stagers[i].buffer.destroy();
        this._stagers.length = 0;
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

    private createPipeline(): void {
        const words = this.rowBytes >>> 2;
        const device = this._state.gpu.device;
        const layout = device.createBindGroupLayout({
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
struct Slots { values: array<u32> };
@group(0) @binding(0) var<storage, read> slots: Slots;
@group(0) @binding(1) var<storage, read> values: array<u32>;
@group(0) @binding(2) var<storage, read_write> rows: array<u32>;
@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    let word = id.x;
    let count = slots.values[0];
    let wordsPerRecord = ${words}u;
    if (word >= count * wordsPerRecord) { return; }
    let record = word / wordsPerRecord;
    let lane = word % wordsPerRecord;
    rows[slots.values[record + 1u] * wordsPerRecord + lane] = values[word];
}`,
        });
        const pipelineLayout = device.createPipelineLayout({
            label: `table-${this.name}-scatter-pipeline-layout`,
            bindGroupLayouts: [layout],
        });
        this._pipeline = device.createComputePipeline({
            label: `table-${this.name}-scatter-pipeline`,
            layout: pipelineLayout,
            compute: { module, entryPoint: "scatter" },
        });
        this._layout = layout;
    }

    private _layout!: GPUBindGroupLayout;

    private replaceBuffers(): void {
        const device = this._state.gpu.device;
        const old = this._capacity > 0 ? this._buffer : undefined;
        const size = this._capacity * this.rowBytes;
        const buffer = device.createBuffer({
            label: `table-${this.name}-g${this._generation + 1}`,
            size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this._state.own(buffer);
        if (old) {
            const encoder = device.createCommandEncoder({ label: `table-${this.name}-grow` });
            encoder.copyBufferToBuffer(old, 0, buffer, 0, old.size);
            device.queue.submit([encoder.finish()]);
            old.destroy();
        }
        this._buffer = buffer;
        this._typed = this._state.gpu.root
            .createBuffer(d.arrayOf(this.record, this._capacity), buffer)
            .$usage("storage") as TgpuBuffer<d.AnyWgslData>;
        this._state.gpu.buffers.set(this.name, buffer);
        this._state.gpu.typed.set(this.name, this._typed);

        if (this._slots) this._slots.destroy();
        if (this._values) this._values.destroy();
        for (let i = 0; i < this._stagers.length; i++) this._stagers[i].buffer.destroy();
        this._slots = device.createBuffer({
            label: `table-${this.name}-scatter-slots`,
            size: (this._capacity + 1) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this._values = device.createBuffer({
            label: `table-${this.name}-scatter-values`,
            size: Math.max(4, size),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this._state.own(this._slots);
        this._state.own(this._values);
        this._bindGroup = device.createBindGroup({
            label: `table-${this.name}-scatter-bind-group`,
            layout: this._layout,
            entries: [
                { binding: 0, resource: { buffer: this._slots } },
                { binding: 1, resource: { buffer: this._values } },
                { binding: 2, resource: { buffer } },
            ],
        });
        this._stagers = [];
        this._epoch++;
        const epoch = this._epoch;
        const stagingSize = (this._capacity + 1) * 4 + size;
        for (let i = 0; i < SCATTER_STAGERS; i++) {
            const stager = this.createStager(device, stagingSize, epoch);
            this._stagers.push(stager);
            this._state.own(stager.buffer);
        }
        this._generation++;
        for (let i = 0; i < this._consumers.length; i++) {
            this._consumers[i](this._buffer, this._generation);
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
                if (this._disposed || this._epoch !== stager.epoch) {
                    stager.buffer.destroy();
                } else {
                    stager.available = true;
                }
            },
            rejected: (error) => {
                stager.buffer.destroy();
                if (!this._disposed && this._epoch === stager.epoch) {
                    throw new Error(`GpuTable "${this.name}" staging mapAsync failed`, {
                        cause: error,
                    });
                }
            },
        };
        return stager;
    }

    private availableStager(): Stager | undefined {
        for (let i = 0; i < this._stagers.length; i++) {
            if (this._stagers[i].available) return this._stagers[i];
        }
        return undefined;
    }

    private scatter(stager: Stager, changed: number): void {
        const range = stager.buffer.getMappedRange();
        const slots = new Uint32Array(range, 0, this._capacity + 1);
        const valueOffset = (this._capacity + 1) * 4;
        const payload = new Uint8Array(range, valueOffset, this._capacity * this.rowBytes);
        let packed = 0;
        for (let word = 0; word < this._dirty.length; word++) {
            let bits = this._dirty[word];
            const base = word << 5;
            while (bits !== 0) {
                const bit = 31 - Math.clz32(bits & -bits);
                const eid = base + bit;
                slots[packed + 1] = eid;
                const source = eid * this.rowBytes;
                const destination = packed * this.rowBytes;
                for (let byte = 0; byte < this.rowBytes; byte++) {
                    payload[destination + byte] = this._bytes[source + byte];
                }
                packed++;
                bits &= bits - 1;
            }
        }
        slots[0] = packed;
        stager.buffer.unmap();
        const encoder = this._state.gpu.device.createCommandEncoder({
            label: `table-${this.name}-scatter-upload`,
        });
        encoder.copyBufferToBuffer(stager.buffer, 0, this._slots, 0, (changed + 1) * 4);
        encoder.copyBufferToBuffer(
            stager.buffer,
            valueOffset,
            this._values,
            0,
            changed * this.rowBytes,
        );
        const pass = encoder.beginComputePass({ label: `table-${this.name}-scatter-pass` });
        pass.setPipeline(this._pipeline);
        pass.setBindGroup(0, this._bindGroup);
        pass.dispatchWorkgroups(Math.ceil((changed * (this.rowBytes >>> 2)) / 64));
        pass.end();
        this._state.gpu.device.queue.submit([encoder.finish()]);
        stager.available = false;
        stager.buffer.mapAsync(GPUMapMode.WRITE).then(stager.mapped, stager.rejected);
    }
}
