import type * as d from "typegpu/data";
import { ReadbackPool, type WorldGpu } from "../runtime";
import { FieldColumns } from "./columns";
import {
    type Component,
    Components,
    type ComponentValues,
    declaration,
    fields,
    freezeComponent,
    idOf,
    type ScalarField,
    sameComponentSchema,
    type Vector2Field,
    type Vector4Field,
} from "./component";
import { Entities, type EntityRef } from "./entity";
import { type Recovery, SnapshotComposition, type WorldSnapshot } from "./snapshot";

export type { WorldSnapshot } from "./snapshot";

import { Queries } from "./query";
import { ComponentRegistry } from "./registry";
import { Scheduler, type System, Time } from "./scheduler";
import { type ComponentStorage, WorldField } from "./storage";
import { GpuTable, type GpuTableOptions } from "./table";

const INITIAL_CAPACITY = 16;
const UPLOAD_CHUNK_BYTES = 64 * 1024;
interface UploadChunk {
    buffer: GPUBuffer;
    offset: number;
    fence: number;
    used: boolean;
}
interface FrameCopy {
    source: GPUBuffer | undefined;
    sourceOffset: number;
    destination: GPUBuffer | undefined;
    destinationOffset: number;
    size: number;
}

/** A world-owned value identified by its declaration, or an explicit reload-stable key. */
export type Resource<T> = {
    readonly create: (world: World) => T;
    /** Opt into carrying this value across re-evaluated declarations. The owner must refuse
     * a carried shape its new code cannot use. Keys must be unique to the resource. */
    readonly key?: symbol;
};

/**
 * Owns its entity storage, resources, GPU tables and allocations registered with {@link own};
 * {@link dispose} releases them, never the GPU device.
 */
export class World {
    /** this world's component registrations, with their defaults and requirements. @internal */
    readonly registry = new ComponentRegistry();
    private _scheduler = new Scheduler();
    private _frameEncoder: GPUCommandEncoder | undefined;
    private _frameFence: Promise<void> | undefined;
    private _inDrawGroup = false;
    private _retiredBuffers: GPUBuffer[] = [];
    private _frameCopies: FrameCopy[] = [];
    private _frameCopyCount = 0;
    private _uploadChunks: UploadChunk[] = [];
    private _frameUploadBytes = 0;
    private _stepping = false;
    private _columns = new FieldColumns();
    private _snapshots = new SnapshotComposition(
        () => this._stepping,
        () => `${this.registry.revision}/${this._storage.size}`,
        () => {
            for (const entry of this.registry.entries()) this.storage(entry.component);
        },
    );
    private _readback: ReadbackPool | undefined;

    /** One-shot buffer and texture staging owned by this world. */
    get readback(): ReadbackPool {
        if (this._disposed) throw new Error("readback world is disposed");
        return (this._readback ??= new ReadbackPool(this));
    }

    private readonly _stepInput = { deltaTime: Time.DEFAULT_DT };
    private readonly _runDrawGroup = (run: () => void) => this.runDrawGroup(run);
    private readonly _runStep = () =>
        this._scheduler.step(this, this._stepInput, this._runDrawGroup);
    private _entities = new Entities();
    private _components = new Components(
        () => this._queries.restore(this._components, this._entities),
        (id, eid, present) => {
            const component = this._storage.get(id)?.schema;
            if (!component) return;
            for (const table of this._tablesByComponent.get(id) ?? []) {
                if (present) table.attachComponent(eid, component);
                else table.detachComponent(eid, component);
            }
            this.notifyMembership(component, eid, present);
        },
        (id, eid) => this._queries.onComponentChanged(id, eid, this._components),
    );
    private _queries = new Queries();
    private readonly _held: number[] = [];
    private _storage = new Map<
        number,
        {
            schema: Component;
            schemas: WeakSet<Component>;
            fields: Map<string, WorldField>;
            storage: Record<string, unknown>;
        }
    >();
    private _resources = new Map<Resource<unknown> | symbol, unknown>();
    private _tables = new Map<string, GpuTable>();
    private _tablesByComponent = new Map<number, GpuTable[]>();
    private _membershipObservers = new Map<number, Set<(eid: number, present: boolean) => void>>();
    private get _highWater(): number {
        return this._columns.highWater;
    }
    private set _highWater(value: number) {
        this._columns.highWater = value;
    }
    private _pixelRatio: number | "auto";
    private _fieldUploadSeen = false;
    private _changesClearedAtUpload = false;
    private _disposals: (() => void)[] = [];
    private _controller: AbortController | undefined;
    private _disposed = false;
    private _gpu: WorldGpu | undefined;
    private _gpuResources = new Set<{ destroy(): void }>();

    constructor(opts?: { pixelRatio?: number | "auto" }) {
        this._pixelRatio = opts?.pixelRatio ?? "auto";
        this._columns.highWater = 1;
        this._snapshots.register(Symbol("entities"), this._entities);
        this._snapshots.register(Symbol("fields"), this._columns);
        this._snapshots.register(Symbol("clock"), this._scheduler);
        this._snapshots.register(Symbol("membership"), this._components);
    }

    /** Fence for the latest draw-group submission; a propagated draw error may set it to the engine-copy replay's fence.
     * Undefined when neither submitted. Reading it issues no fence. */
    get frameFence(): Promise<void> | undefined {
        return this._frameFence;
    }

    /** this world's GPU device, registries, typed handles and frame state; unavailable when the build acquired no device. */
    get gpu(): WorldGpu {
        if (!this._gpu)
            throw new Error(
                "World.gpu is unavailable: no enabled plugin declares a GPU requirement",
            );
        return this._gpu;
    }

    /** @internal GPU context for lifecycle code that must also run without a device. */
    get gpuIfAvailable(): WorldGpu | undefined {
        return this._gpu;
    }

    /** @internal attach this world's GPU context during build. */
    attachGpu(compute: WorldGpu): void {
        this._gpu = compute;
    }

    /**
     * Resolve once per declaration object or explicit key in this World. Reloaded declarations
     * create fresh values unless they share a key; carrying a value does not run the new creator.
     * Creators register cleanup with onDispose or own, which runs at world disposal.
     * Refuses after disposal; this does not invalidate caller-retained references.
     */
    resource<T>(declaration: Resource<T>): T {
        if (this._disposed) throw new Error("World.resource: world is disposed");
        const key = declaration.key ?? declaration;
        if (this._resources.has(key)) return this._resources.get(key) as T;
        const value = declaration.create(this);
        this._resources.set(key, value);
        return value;
    }

    /** own a raw GPU allocation until this world is disposed. */
    own(resource: { destroy(): void }): void {
        if (this._disposed) {
            resource.destroy();
            return;
        }
        if (this._gpuResources.has(resource)) return;
        this._gpuResources.add(resource);
        const destroy = resource.destroy;
        resource.destroy = () => {
            this._gpuResources.delete(resource);
            resource.destroy = destroy;
            destroy.call(resource);
        };
    }

    /** The current draw group's command encoder, opened on first use. Outside draw this refuses;
     * a CPU-only World returns undefined while a GPU-backed draw may record commands. */
    frameEncoder(): GPUCommandEncoder | undefined {
        if (!this._inDrawGroup)
            throw new Error("World.frameEncoder is available only during the draw group of step()");
        if (!this._gpu) return undefined;
        return (this._frameEncoder ??= this.gpu.device.createCommandEncoder({
            label: "shallot-frame",
        }));
    }

    private runDrawGroup(run: () => void): void {
        this.beginDrawGroup();
        try {
            run();
        } catch (error) {
            this.abortDrawGroup();
            throw error;
        }
        this.endDrawGroup();
    }

    private beginDrawGroup(): void {
        this._inDrawGroup = true;
    }

    /** Submit recorded work once, then issue the fence that owns its staging ranges. */
    private endDrawGroup(): void {
        this._inDrawGroup = false;
        const encoder = this._frameEncoder;
        this._frameEncoder = undefined;
        if (encoder && !this._disposed) this.submitDrawEncoder(encoder);
        else this.releaseRetiredBuffers();
        this.resetFrameCopies();
    }

    /** Drop system work and submit only the engine copies needed to preserve table state. */
    private abortDrawGroup(): void {
        this._inDrawGroup = false;
        this._frameEncoder = undefined;
        if (this._frameCopyCount > 0 && !this._disposed) {
            const encoder = this.gpu.device.createCommandEncoder({ label: "shallot-frame-replay" });
            for (let i = 0; i < this._frameCopyCount; i++) {
                const copy = this._frameCopies[i];
                encoder.copyBufferToBuffer(
                    copy.source!,
                    copy.sourceOffset,
                    copy.destination!,
                    copy.destinationOffset,
                    copy.size,
                );
            }
            this.submitDrawEncoder(encoder);
        } else {
            this.releaseRetiredBuffers();
        }
        this.resetFrameCopies();
    }

    private submitDrawEncoder(encoder: GPUCommandEncoder): void {
        this.gpu.device.queue.submit([encoder.finish()]);
        for (const chunk of this._uploadChunks) {
            if (!chunk.used) continue;
            chunk.used = false;
            chunk.fence = this.gpu.fences.issued + 1;
        }
        this._frameFence = this.gpu.sync();
        this.releaseRetiredBuffers();
    }

    private releaseRetiredBuffers(): void {
        for (const buffer of this._retiredBuffers) buffer.destroy();
        this._retiredBuffers.length = 0;
    }

    private recordFrameCopy(
        source: GPUBuffer,
        sourceOffset: number,
        destination: GPUBuffer,
        destinationOffset: number,
        size: number,
    ): void {
        let copy = this._frameCopies[this._frameCopyCount];
        if (copy) {
            copy.source = source;
            copy.sourceOffset = sourceOffset;
            copy.destination = destination;
            copy.destinationOffset = destinationOffset;
            copy.size = size;
        } else {
            copy = { source, sourceOffset, destination, destinationOffset, size };
            this._frameCopies.push(copy);
        }
        this._frameCopyCount++;
    }

    private resetFrameCopies(): void {
        for (let i = 0; i < this._frameCopyCount; i++) {
            const copy = this._frameCopies[i];
            copy.source = undefined;
            copy.destination = undefined;
            copy.size = 0;
            copy.sourceOffset = 0;
            copy.destinationOffset = 0;
        }
        this._frameCopyCount = 0;
    }

    /** @internal Growth records in draw; otherwise its immediate submission follows preceding queue writes.
     * Old buffers retire after the copy submission. */
    growGpuBuffer(previous: GPUBuffer, buffer: GPUBuffer): void {
        if (this._inDrawGroup) {
            this.frameEncoder()!.copyBufferToBuffer(previous, 0, buffer, 0, previous.size);
            this.recordFrameCopy(previous, 0, buffer, 0, previous.size);
            this._retiredBuffers.push(previous);
        } else {
            const encoder = this.gpu.device.createCommandEncoder();
            encoder.copyBufferToBuffer(previous, 0, buffer, 0, previous.size);
            this.gpu.device.queue.submit([encoder.finish()]);
            previous.destroy();
        }
    }

    /** @internal CPU metadata replaces old buffers without a GPU copy. */
    retireGpuBuffer(buffer: GPUBuffer): void {
        if (this._frameEncoder) this._retiredBuffers.push(buffer);
        else buffer.destroy();
    }

    /** @internal Outside draw, writes use queue order. Draw uploads share the encoder and recycle after its fence. */
    uploadGpuTable(buffer: GPUBuffer, offset: number, data: ArrayBufferLike, size: number): void {
        if (!this._inDrawGroup) {
            this.gpu.device.queue.writeBuffer(buffer, offset, data as ArrayBuffer, offset, size);
            return;
        }
        const limit = this.gpu.device.limits.maxBufferSize;
        if (size > limit - this._frameUploadBytes)
            throw new Error(
                `GPU table frame uploads exceed the ${limit}-byte frame budget: ${this._frameUploadBytes} bytes recorded, ${size} more requested`,
            );
        const encoder = this.frameEncoder()!;
        const chunk = this.uploadChunk(size);
        this._frameUploadBytes += size;
        const sourceOffset = chunk.offset;
        chunk.offset += size;
        chunk.used = true;
        this.gpu.device.queue.writeBuffer(
            chunk.buffer,
            sourceOffset,
            data as ArrayBuffer,
            offset,
            size,
        );
        encoder.copyBufferToBuffer(chunk.buffer, sourceOffset, buffer, offset, size);
        this.recordFrameCopy(chunk.buffer, sourceOffset, buffer, offset, size);
    }

    // Like wgpu's StagingBelt, append until submission completion makes an entire chunk reusable.
    // In-flight chunks may accept more bytes, but never overwrite a recorded copy's source.
    private uploadChunk(size: number): UploadChunk {
        for (const chunk of this._uploadChunks) {
            if (!chunk.used && chunk.fence <= this.gpu.fences.completed) chunk.offset = 0;
            if (chunk.buffer.size - chunk.offset >= size) return chunk;
        }
        const limit = this.gpu.device.limits.maxBufferSize;
        let capacity = Math.min(UPLOAD_CHUNK_BYTES, limit);
        while (capacity < size && capacity < limit) capacity = Math.min(capacity * 2, limit);
        const buffer = this.gpu.device.createBuffer({
            label: "table-upload-chunk",
            size: capacity,
            usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this.own(buffer);
        const chunk: UploadChunk = {
            buffer,
            offset: 0,
            fence: 0,
            used: false,
        };
        this._uploadChunks.push(chunk);
        return chunk;
    }

    /** Whether this world owns a registered resource. */
    owns(resource: { destroy(): void }): boolean {
        return this._gpuResources.has(resource);
    }

    /** Declare a world-owned dense-slot table with one TypeGPU record layout.
     * Names are unique within this world; disposal releases its buffers. */
    table<T extends d.AnyWgslData>(
        name: string,
        record: T,
        options?: GpuTableOptions,
    ): GpuTable<T> {
        if (this._tables.has(name)) throw new Error(`World.table: duplicate table "${name}"`);
        const registry = this._gpu?.buffers;
        if (
            registry?.has(name) ||
            registry?.has(`${name}:eid-to-row`) ||
            registry?.has(`${name}:active-rows`)
        ) {
            throw new Error(`World.table: GPU registry name "${name}" is already in use`);
        }
        const table = new GpuTable(this, name, record, options);
        this._tables.set(name, table);
        return table;
    }

    /** @internal Bind a table's dense rows to a component's membership lifecycle. */
    bindTableComponent(component: Component, table: GpuTable): void {
        const id = idOf(component);
        const tables = this._tablesByComponent.get(id);
        if (tables) {
            if (tables.includes(table)) return;
            tables.push(table);
        } else {
            this._tablesByComponent.set(id, [table]);
        }
        for (const eid of this.query([component])) table.attachComponent(eid, component);
    }

    /** @internal Observe component membership without putting state on the component schema. Observers must not change membership. */
    observeMembership(
        component: Component,
        observer: (eid: number, present: boolean) => void,
    ): () => void {
        const id = idOf(component);
        let observers = this._membershipObservers.get(id);
        if (!observers) this._membershipObservers.set(id, (observers = new Set()));
        observers.add(observer);
        return () => {
            observers!.delete(observer);
            if (observers!.size === 0) this._membershipObservers.delete(id);
        };
    }

    private notifyMembership(component: Component, eid: number, present: boolean): void {
        this._membershipObservers.get(idOf(component))?.forEach((observer) => {
            observer(eid, present);
        });
    }

    /** @internal Resolve the world-owned column for engine change consumers. */
    fieldStorage(component: Component, name: string): WorldField {
        this.storage(component);
        const field = this._storage.get(idOf(component))?.fields.get(name);
        if (!field) throw new Error(`World.fieldStorage: unknown field "${name}"`);
        return field;
    }

    /** @internal Unbind a table from a component's membership lifecycle. */
    unbindTableComponent(component: Component, table: GpuTable): void {
        const id = idOf(component);
        const tables = this._tablesByComponent.get(id);
        if (!tables) return;
        const index = tables.indexOf(table);
        if (index >= 0) tables.splice(index, 1);
        if (tables.length === 0) this._tablesByComponent.delete(id);
    }

    /** @internal Find the table that owns a current record, active-list or eid-map buffer. */
    tableForBuffer(buffer: GPUBuffer): GpuTable | undefined {
        for (const table of this._tables.values()) {
            if (
                table.buffer === buffer ||
                table.activeRowsBuffer === buffer ||
                table.eidToRowBuffer === buffer
            )
                return table;
        }
        return undefined;
    }

    /** @internal Upload all declared tables at the head of draw. */
    uploadTables(): void {
        for (const table of this._tables.values()) table.upload();
    }

    /** Highest allocated entity id plus one; CPU storage grows with this value. */
    get entityHighWater(): number {
        return this._highWater;
    }

    /** render device-pixel ratio fixed to this world's build config. */
    get pixelRatio(): number | "auto" {
        return this._pixelRatio;
    }

    /** current component schemas and their world-owned field columns. @internal */
    storageEntries(): IterableIterator<{
        schema: Component;
        fields: Map<string, WorldField>;
    }> {
        return this._storage.values();
    }

    /** clear field marks after this world's frame upload point. @internal */
    clearChanges(): void {
        for (const entry of this._storage.values()) {
            for (const field of entry.fields.values()) field.dirty.fill(0);
        }
        this._changesClearedAtUpload = true;
    }

    /** record that this world's field upload point is running. @internal */
    markFieldUploadPoint(): void {
        this._fieldUploadSeen = true;
    }

    /** clear marks only when no earlier upload system did so this frame. @internal */
    clearChangesIfNeeded(): void {
        if (!this._fieldUploadSeen && !this._changesClearedAtUpload) this.clearChanges();
    }

    /** Resolve a schema to this world's storage. Retain these accessors at system setup,
     * not their `column` arrays: growth replaces arrays, while accessors remain valid.
     * Setters and bulk `writeEncoded` publish frame-scoped field marks for table upload. */
    storage<T extends Component>(component: T): ComponentStorage<T> {
        const id = idOf(component);
        const existing = this._storage.get(id);
        if (existing) {
            if (existing.schemas.has(component)) return existing.storage as ComponentStorage<T>;
            if (!sameComponentSchema(existing.schema, component)) {
                throw new Error(
                    `world.storage: component "${declaration(component, "world.storage").key}" schema changed: this world already stores another record under that key with different fields`,
                );
            }
            freezeComponent(component);
            existing.schemas.add(component);
            return existing.storage as ComponentStorage<T>;
        }
        freezeComponent(component);
        const columns = new Map<string, WorldField>();
        const storage: Record<string, unknown> = {};
        for (const { name, field } of fields(component)) {
            const column = new WorldField(field, INITIAL_CAPACITY, this);
            column.ensure(this._highWater);
            this._columns.register(column);
            columns.set(name, column);
            storage[name] = column.bind();
        }
        this._storage.set(id, {
            schema: component,
            schemas: new WeakSet([component]),
            fields: columns,
            storage,
        });
        return storage as ComponentStorage<T>;
    }

    get time(): Readonly<Time> {
        return this._scheduler.time;
    }

    /** Advance a virtual frame by `deltaTime` seconds, with paced fixed work, simulation and draw.
     * Pause, scale and the catch-up cap apply only here. Refuses inside a step or tick,
     * or a negative or non-finite delta.
     * A propagated setup/update throw ends the step with a named Error and the thrown value as cause.
     * A propagated draw throw drops the unfinished system encoder without submitting its commands.
     * Only recorded table growth and staged upload copies replay through a fresh submission; if there are
     * none, no submission or fence issues. The GPU frame does not advance. With
     * {@link logAndPauseSystemErrors}, the failed system is paused, later draw systems run, and the frame submits. */
    step(deltaTime = Time.DEFAULT_DT): void {
        if (this._stepping) throw new Error("World.step: refuses inside a step or tick");
        this._frameFence = undefined;
        this._frameUploadBytes = 0;
        this._fieldUploadSeen = false;
        this._changesClearedAtUpload = false;
        this._stepInput.deltaTime = deltaTime;
        let stepped = false;
        this._stepping = true;
        try {
            this._runStep();
            stepped = true;
        } finally {
            this._stepping = false;
            // A failed scheduler has no CPU-only fallback upload point; preserve its pending marks.
            if (!this._gpu && stepped) this.clearChangesIfNeeded();
            if (this._gpu && stepped) {
                this._gpu.frame++;
                this._readback?.advance(this._gpu.frame);
            }
        }
    }

    /** Advance exactly one fixed tick, ignoring pause, scale and catch-up limits. Runs only the fixed
     * group (including its lazy system setup), not setup, simulation or draw groups. Does not advance
     * the GPU frame or readback; queued table writes remain ordered before the next frame submission.
     * Refuses inside a step or tick. Errors follow {@link step}; the tick count is not rolled back. */
    tick(): void {
        if (this._stepping) throw new Error("World.tick: refuses inside a step or tick");
        this._stepping = true;
        try {
            this._scheduler.tick(this);
        } finally {
            this._stepping = false;
        }
    }

    /** Capture entity identity and allocation, component membership, all stored and registered
     * fields and their pending change marks through the entity high-water mark, and fixedTick.
     * Local to this world and registry;
     * includes declared participants' hidden simulation state; excludes pacing, GPU and host state. Refuses during step/tick.
     * Query order is derived from restored membership. The image is reusable, opaque and independent
     * of writes, not a save format. */
    snapshot(): WorldSnapshot {
        return this._snapshots.snapshot();
    }

    /** Restore a local image between ticks. Refuses during step/tick, another world's image or
     * a changed component registry. Retained accessors and references resolve the restored state;
     * columns never shrink, queries and membership consumers reconcile, and fields publish changes.
     * Participants restore hidden simulation state after ECS and clock recovery; pacing, GPU and host state stay current. */
    restore(snapshot: WorldSnapshot): void {
        this._snapshots.restore(snapshot);
    }

    /** @internal Register an ordered simulation owner. */
    registerRecovery<S>(
        name: string | symbol,
        recovery: Recovery<S> | "stateless" | undefined,
    ): void {
        this._snapshots.register(name, recovery);
    }

    /** Freeze the virtual frame clock and step's fixed work, not {@link tick}. The real frame clock
     * keeps running. Takes effect next frame; resume retains the prior scale. */
    pause(): void {
        this._scheduler.pause();
    }

    /** unfreeze the virtual clock. */
    resume(): void {
        this._scheduler.resume();
    }

    /** set the virtual timescale: 1 real time, <1 slow-mo, >1 fast-forward, 0 freeze (negative clamps to 0).
     * Applies to step's virtual clock and tick frequency, never to explicit {@link tick}. Read via `time.scale`. */
    setTimeScale(scale: number): void {
        this._scheduler.setScale(scale);
    }

    create(): number {
        const eid = this._entities.add();
        if (eid + 1 > this._highWater) this._highWater = eid + 1;
        this._columns.ensure(eid + 1);
        this._queries.onEntityCreated(eid);
        return eid;
    }

    /** Remove every component, notify membership observers, clear its fields, and free the eid; no-op when it is not alive. */
    destroy(eid: number): void {
        if (!this._entities.exists(eid)) return;
        const held = this._held;
        const count = this._components.held(eid, held);
        this._components.clear(eid);
        if (count !== 0 && this._membershipObservers.size !== 0) {
            if (count > 1)
                for (let i = 1; i < count; i++) {
                    const id = held[i];
                    let j = i;
                    while (j > 0 && held[j - 1] > id) {
                        held[j] = held[j - 1];
                        j--;
                    }
                    held[j] = id;
                }
            for (let i = 0; i < count; i++) {
                const observers = this._membershipObservers.get(held[i]);
                if (observers) for (const observer of observers) observer(eid, false);
            }
        }
        this._queries.onEntityRemoved(eid);
        for (const tables of this._tablesByComponent.values()) {
            for (const table of tables) table.release(eid);
        }
        for (let i = 0; i < count; i++) {
            const entry = this._storage.get(held[i]);
            if (entry) for (const field of entry.fields.values()) field.clear(eid);
        }
        this._entities.remove(eid);
    }

    exists(eid: number): boolean {
        return this._entities.exists(eid);
    }

    /**
     * Keep an entity across frames and storage growth, not destruction or eid reuse.
     * Returns 0 for a dead eid. Local snapshot recovery restores identity, so a reference
     * captured with its entity resolves again after restore. Valid only in this World and run,
     * never across saves.
     * The 21-bit generation wraps after 2^21 reuses of one eid and warns once.
     */
    ref(eid: number): EntityRef {
        return (
            this._entities.exists(eid) ? this._entities.generation(eid) * 2 ** 32 + eid : 0
        ) as EntityRef;
    }

    /**
     * Resolve a reference kept across frames or storage growth to its live eid, or 0
     * after destruction or reuse. 0 means missing. Use only references from this World
     * and run; after the 21-bit generation wraps, an old reference can alias a live eid.
     */
    resolve(ref: EntityRef): number {
        const eid = ref >>> 0;
        return this._entities.exists(eid) &&
            this._entities.generation(eid) === Math.floor(ref / 2 ** 32)
            ? eid
            : 0;
    }

    /** snapshot of every alive entity id */
    entities(): readonly number[] {
        return this._entities.all();
    }

    /**
     * Attach a component with optional field values (vectors are arrays).
     * Missing fields keep the declared defaults when this world registers the component, and
     * required companions are added only then. Defaults and starting values go through
     * field setters, publishing changes. An already attached component is unchanged.
     */
    add<T>(eid: number, component: T, values?: ComponentValues<NoInfer<T>>): void {
        const storage = this.storage(component as Component);
        if (this._components.add(eid, component)) {
            const id = idOf(component as Component);
            this.notifyMembership(component as Component, eid, true);
            const tables = this._tablesByComponent.get(id);
            const attached: GpuTable[] = [];
            try {
                if (tables) {
                    for (const table of tables) {
                        table.attachComponent(eid, component as Component);
                        attached.push(table);
                    }
                }
            } catch (error) {
                for (const table of attached) table.detachComponent(eid, component as Component);
                this._components.remove(eid, component);
                this.notifyMembership(component as Component, eid, false);
                throw error;
            }
            this._queries.onComponentChanged(id, eid, this._components);
            for (const required of this.registry.getRequirements(component as Component)) {
                if (!this.has(eid, required)) this.add(eid, required);
            }
            this.registry.applyDefaults(this, component as Component, eid);
            if (values) {
                for (const name in values) {
                    const value = values[name];
                    if (value === undefined) continue;
                    const field = storage[name] as ScalarField | Vector2Field | Vector4Field;
                    if (typeof value === "number") (field as ScalarField).set(eid, value);
                    else {
                        const lanes = value as readonly number[];
                        field.set(eid, lanes[0], lanes[1], lanes[2], lanes[3]);
                    }
                }
            }
        } else {
            console.warn("world.add: component already attached to entity", eid);
        }
    }

    /** Remove a component and clear its fields after membership observers, table detachment and query updates. */
    remove(eid: number, component: any): void {
        if (this._components.remove(eid, component)) {
            const id = idOf(component as Component);
            this.notifyMembership(component as Component, eid, false);
            const tables = this._tablesByComponent.get(id);
            if (tables)
                for (const table of tables) table.detachComponent(eid, component as Component);
            this._queries.onComponentChanged(id, eid, this._components);
            const entry = this._storage.get(id);
            if (entry) for (const field of entry.fields.values()) field.clear(eid);
        }
    }

    has<T>(eid: number, component: T): boolean {
        return this._components.has(eid, component);
    }

    /** Find matching entities. An idle query starts in ascending eid order; nested iterations keep
     * the current array order. Membership changes follow RegisteredQuery's iteration contract;
     * unchanged iterations allocate nothing after iterator-pool warmup. */
    query(terms: any[]): Iterable<number> {
        return this._queries.find(terms, this._components, this._entities);
    }

    /**
     * find exactly one entity, warns if multiple match, returns -1 when nothing matches
     */
    only(terms: any[]): number {
        let result = -1;
        let count = 0;
        for (const eid of this.query(terms)) {
            if (count === 0) result = eid;
            count++;
            if (count > 1) break;
        }
        if (count > 1) {
            console.warn("world.only: expected 1 match, found multiple");
        }
        return result;
    }

    /** Fixed systems belong to a plugin's declared recovery. Without plugin attribution,
     * snapshot refuses by system name until the system is removed. */
    addSystem(system: System, pluginName?: string): void {
        if (system.group === "fixed") {
            if (pluginName) this._snapshots.require(pluginName);
            else this._snapshots.requireSystem(system);
        }
        if (system.boundary) this._scheduler.registerBoundary(system, system.boundary, pluginName);
        else this._scheduler.register(system, pluginName);
    }

    removeSystem(system: System): void {
        this._scheduler.unregister(system);
        this._snapshots.removeSystem(system);
    }

    /**
     * hot-swap a live system's behavior in place. the reloaded module's
     * `update`/`setup`/`dispose` replace the old ones on the same registered
     * object, preserving its identity, ordering, and setup state. The engine
     * `swapPlugins` (plugin-level) drives this per system; not a per-frame call.
     */
    swapSystem(old: System, next: System): void {
        this._scheduler.swap(old, next);
    }

    /** true if the system is live in the scheduler; `swapPlugins` validates its pairing against this */
    hasSystem(system: System): boolean {
        return this._scheduler.has(system);
    }

    /**
     * the CPU timing sink, or `undefined` when profiling is off. Hot-path
     * callers can read this once and skip timed work entirely when absent.
     */
    get recordSink(): ((name: string, ms: number) => void) | undefined {
        return this._scheduler.record;
    }

    set recordSink(fn: ((name: string, ms: number) => void) | undefined) {
        this._scheduler.record = fn;
    }

    /** @internal The running host chooses log-and-pause so a hot-reloaded bug cannot wedge it.
     * Failed systems pause until swapped or rebuilt; later draw systems run, and the frame submits when it recorded work. */
    logAndPauseSystemErrors(): void {
        this._scheduler.logAndPauseErrors = true;
    }

    /** the GPU fence-wait telemetry sink, or `undefined` when profiling is off */
    get fenceWaitSink(): ((ms: number) => void) | undefined {
        return this._scheduler.fenceWait;
    }

    set fenceWaitSink(fn: ((ms: number) => void) | undefined) {
        this._scheduler.fenceWait = fn;
    }

    /**
     * true once {@link dispose} has run. An async plugin step that awaits across a teardown (a glTF decode
     * resolving after a scene switch) checks this before touching the World, so a late result no-ops instead
     * of mutating a dead world.
     */
    get disposed(): boolean {
        return this._disposed;
    }

    /**
     * register a teardown callback tied to this World's lifetime. Callbacks run in LIFO order (last
     * registered, first run) at {@link dispose}, so a DOM mount, listener, or rAF loop keeps its cleanup
     * beside its creation site. A callback registered after dispose has already run fires immediately
     * (paired with {@link signal}, already aborted), so a late async step never leaks silently.
     */
    onDispose(fn: () => void): void {
        if (this._disposed) {
            fn();
            return;
        }
        this._disposals.push(fn);
    }

    /**
     * an {@link AbortSignal} tied to this World's lifetime, aborted when {@link dispose} runs (already
     * aborted if read afterward). Pass it as `{ signal }` to `addEventListener`, `fetch`, or any
     * abortable API to detach on teardown with zero removal code. Lazily created on first read.
     */
    get signal(): AbortSignal {
        if (!this._controller) this._controller = new AbortController();
        if (this._disposed && !this._controller.signal.aborted) this._controller.abort();
        return this._controller.signal;
    }

    /** tear down the world; disposes every registered system */
    dispose(): void {
        if (this._disposed) return;
        this._disposed = true;
        this._controller?.abort();
        this._readback?.dispose();
        this._readback = undefined;
        // the list carries user cleanups (a Svelte unmount, an app rAF stop) that throw more readily
        // than engine hooks, and LIFO runs them first — a throw must not skip the remaining callbacks or
        // the scheduler/query teardown below, or their resources leak. Report, never mask.
        for (let i = this._disposals.length - 1; i >= 0; i--) {
            try {
                this._disposals[i]();
            } catch (err) {
                console.error("World.dispose: a teardown callback threw:", err);
            }
        }
        this._disposals.length = 0;
        this._scheduler.dispose(this);
        this._queries.clear();
        this._storage.clear();
        this._columns.clear();
        this._snapshots.clear();
        for (const table of this._tables.values()) table.dispose();
        this._tables.clear();
        this._frameEncoder = undefined;
        this._inDrawGroup = false;
        for (const buffer of this._retiredBuffers) buffer.destroy();
        this._retiredBuffers.length = 0;
        this.resetFrameCopies();
        this._uploadChunks.length = 0;
        this._frameUploadBytes = 0;
        this._resources.clear();

        this.registry.clear();
        for (const resource of this._gpuResources) {
            try {
                resource.destroy();
            } catch (err) {
                console.error("World.dispose: GPU resource release threw:", err);
            }
        }
        this._gpuResources.clear();
        this._gpu?.buffers.clear();
        this._gpu?.textures.clear();
        this._gpu?.samplers.clear();
        this._gpu?.typed.clear();
    }
}
