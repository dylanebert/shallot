import type * as d from "typegpu/data";
import { ReadbackPool } from "../runtime";
import {
    bindFields,
    type Component,
    Components,
    type FieldSchema,
    fields,
    idOf,
    type Membership,
    useState,
} from "./component";
import { Entities } from "./entity";
import {
    forgetGlobalTransformEntity,
    type GlobalTransformRuntime,
    globalTransformProducerChanged,
    prepareGlobalTransformFrame,
    retainsGlobalTransform,
} from "./global-transform";
import { Identity } from "./identity";
import { Queries } from "./query";
import { Scheduler, type System, Time } from "./scheduler";
import { type ComponentStorage, sameSchema, WorldField } from "./storage";
import { GpuTable, type GpuTableOptions } from "./table";
import { ComponentRegistry } from "./traits";

const INITIAL_CAPACITY = 16;

/**
 * render device-pixel ratio for canvas-bound views, fixed at app construction. `"auto"`
 * (default) clamps the display's `devicePixelRatio` to `[1, 2]` (react-three-fiber's default):
 * crisp on HiDPI, never below logical resolution, capped so a DPR-3 phone doesn't pay 9× the fill.
 * A fixed number overrides: `1` renders at CSS resolution (cheapest, the three.js literal default),
 * `2` forces 2×, a value below 1 downscales for a pixel-art look (the upscale switches to
 * nearest-neighbor). Read at every resize (see {@link attachCanvas}), so dragging a window between
 * monitors re-sizes the backing. Set via `build({ pixelRatio })`.
 */
export const pixelRatio: number | "auto" = "auto";

export interface WorldGpu {
    readonly device: GPUDevice;
    readonly adapter: { class: string; identity: string; reason?: string };
    readonly root: any;
    frame: number;
    pending(): number;
    sync(): Promise<void>;
    readonly buffers: Map<string, GPUBuffer>;
    readonly textures: Map<string, GPUTexture>;
    readonly samplers: Map<string, GPUSampler>;
    readonly typed: Map<string, any>;
    span?: (
        name: string,
    ) => GPUComputePassTimestampWrites | GPURenderPassTimestampWrites | undefined;
    indirect?: (name: string, count: number) => void;
    precompiled?: (label: string, start: number, end: number) => void;
}

/**
 * ecs state passed to every system
 * @expand
 * @example
 * const MySystem: System = {
 *     update(state) {
 *         // state passed in every frame
 *     },
 * };
 */
export class State {
    /** this world's component registrations, defaults, exclusions, and reflection data. @internal */
    readonly registry = new ComponentRegistry();
    private _scheduler = new Scheduler();
    /** @internal Fixed world placement and renderer-only GPU history, owned by this world. */
    globalTransformRuntime: GlobalTransformRuntime | undefined;
    private _frameEncoder: GPUCommandEncoder | undefined;
    private _retiredBuffers: GPUBuffer[] = [];
    private _pendingCopies: { source: GPUBuffer; target: GPUBuffer }[] = [];
    private _uploadStages = new Map<GPUBuffer, GPUBuffer>();
    private _stepping = false;
    private _readback: ReadbackPool | undefined;

    /** One-shot buffer and texture staging owned by this world. */
    get readback(): ReadbackPool {
        if (this._disposed) throw new Error("readback world is disposed");
        return (this._readback ??= new ReadbackPool(this));
    }

    private readonly _stepInput = { deltaTime: Time.DEFAULT_DT };
    private readonly _runStep = () => this._scheduler.step(this, this._stepInput);
    private _entities = new Entities();
    private _components = new Components();
    private _queries = new Queries();
    private _storage = new Map<
        number,
        {
            schema: Component;
            schemas: WeakSet<Component>;
            fields: Map<string, WorldField>;
            storage: Record<string, unknown>;
        }
    >();
    private _resources = new Map<PropertyKey, unknown>();
    private _tables = new Map<string, GpuTable>();
    private _tablesByComponent = new Map<number, GpuTable[]>();
    private _membershipObservers = new Map<number, Set<(eid: number, present: boolean) => void>>();
    private _highWater = 1;
    private _pixelRatio: number | "auto";
    private _fieldUploadSeen = false;
    private _changesClearedAtUpload = false;
    private _identity = new Identity();
    private _disposals: (() => void)[] = [];
    private _controller: AbortController | undefined;
    private _disposed = false;
    private _gpu: WorldGpu | undefined;
    private _withCompute: ((callback: () => void) => void) | undefined;
    private _gpuResources = new Set<{ destroy(): void }>();

    constructor(opts?: { pixelRatio?: number | "auto" }) {
        this._pixelRatio = opts?.pixelRatio ?? "auto";
    }

    /** this world's GPU device, registries, typed handles and frame state. */
    get gpu(): WorldGpu {
        if (!this._gpu) throw new Error("State.gpu is unavailable before build acquires a device");
        return this._gpu;
    }

    /** @internal attach this world's GPU context during build. */
    attachGpu(compute: WorldGpu, withCompute: (callback: () => void) => void): void {
        this._gpu = compute;
        this._withCompute = withCompute;
    }

    /** resolve a typed world resource once for this State; the entry dies with its world. */
    resource<T>(key: PropertyKey, create: (state: State) => T): T {
        if (this._resources.has(key)) return this._resources.get(key) as T;
        const value = create(this);
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

    /** @internal The renderer opens one encoder; engine work records into it. */
    beginGpuFrame(encoder: GPUCommandEncoder): void {
        this._frameEncoder = encoder;
        for (const copy of this._pendingCopies)
            encoder.copyBufferToBuffer(copy.source, 0, copy.target, 0, copy.source.size);
        this._pendingCopies.length = 0;
        prepareGlobalTransformFrame(this, encoder);
    }

    /** @internal Release buffers retired by growth only after the frame was submitted. */
    endGpuFrame(): void {
        this._frameEncoder = undefined;
        for (const buffer of this._retiredBuffers) buffer.destroy();
        this._retiredBuffers.length = 0;
    }

    /** @internal Growth during a frame shares its encoder and retains referenced old buffers. */
    growGpuBuffer(previous: GPUBuffer, buffer: GPUBuffer): void {
        const oldStage = this._uploadStages.get(previous);
        if (oldStage) {
            this._uploadStages.delete(previous);
            this.retireGpuBuffer(oldStage);
        }
        if (this._frameEncoder) {
            this._frameEncoder.copyBufferToBuffer(previous, 0, buffer, 0, previous.size);
            this._retiredBuffers.push(previous);
        } else if (this._stepping && this.globalTransformRuntime?.enabled) {
            this._pendingCopies.push({ source: previous, target: buffer });
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
        if (this._frameEncoder || (this._stepping && this.globalTransformRuntime?.enabled))
            this._retiredBuffers.push(buffer);
        else buffer.destroy();
    }

    /** @internal A table upload during draw must follow recorded growth copies, not precede them. */
    uploadGpuTable(buffer: GPUBuffer, offset: number, data: ArrayBufferLike, size: number): void {
        const encoder = this._frameEncoder;
        if (!encoder) {
            this.gpu.device.queue.writeBuffer(buffer, offset, data as ArrayBuffer, offset, size);
            return;
        }
        let staging = this._uploadStages.get(buffer);
        if (!staging) {
            staging = this.gpu.device.createBuffer({
                size: buffer.size,
                usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            });
            this.own(staging);
            this._uploadStages.set(buffer, staging);
        }
        this.gpu.device.queue.writeBuffer(staging, offset, data as ArrayBuffer, offset, size);
        encoder.copyBufferToBuffer(staging, offset, buffer, offset, size);
    }

    /** Whether this world owns a registered resource. */
    owns(resource: { destroy(): void }): boolean {
        return this._gpuResources.has(resource);
    }

    /** Declare one dense-slot GPU table with a single record layout. */
    table<T extends d.AnyWgslData>(
        name: string,
        record: T,
        options?: GpuTableOptions,
    ): GpuTable<T> {
        if (this._tables.has(name)) throw new Error(`State.table: duplicate table "${name}"`);
        const registry = this._gpu?.buffers;
        if (
            registry?.has(name) ||
            registry?.has(`${name}:eid-to-row`) ||
            registry?.has(`${name}:active-rows`)
        ) {
            throw new Error(`State.table: GPU registry name "${name}" is already in use`);
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

    /** @internal Observe component membership without putting state on the component schema. */
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
        globalTransformProducerChanged(this, component, eid, present);
    }

    /** @internal Observe a field setter without putting state on the component schema. */
    observeField(component: Component, name: string, observer: (eid: number) => void): () => void {
        const field = this._storage.get(idOf(component))?.fields.get(name);
        if (!field) throw new Error(`State.observeField: unknown field "${name}"`);
        return field.observe(observer);
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

    /** resolve a component schema to its world-owned columns. Call once at system setup, then retain the result. */
    of<T extends Component>(component: T): ComponentStorage<T> {
        useState(this);
        const id = idOf(component);
        const existing = this._storage.get(id);
        if (existing) {
            if (existing.schemas.has(component)) return existing.storage as ComponentStorage<T>;
            if (!sameSchema(existing.schema, component)) {
                throw new Error(
                    `state.of: component schema changed for "${String(id)}"; rebuild this world`,
                );
            }
            bindFields(component);
            existing.schemas.add(component);
            return existing.storage as ComponentStorage<T>;
        }
        bindFields(component);
        const columns = new Map<string, WorldField>();
        const storage: Record<string, unknown> = {};
        for (const { name, field } of fields(component)) {
            const column = new WorldField(field as FieldSchema, INITIAL_CAPACITY);
            column.ensure(this._highWater);
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

    /** current frame time and delta */
    get time(): Readonly<Time> {
        return this._scheduler.time;
    }

    /** advance one frame */
    step(deltaTime = Time.DEFAULT_DT): void {
        useState(this);
        this._fieldUploadSeen = false;
        this._changesClearedAtUpload = false;
        this._stepInput.deltaTime = deltaTime;
        let stepped = false;
        this._stepping = true;
        try {
            if (this._withCompute) this._withCompute(this._runStep);
            else this._runStep();
            stepped = true;
        } finally {
            this._stepping = false;
            useState(this);
            if (!this._gpu) this.clearChangesIfNeeded();
            if (this._gpu && stepped) {
                this._gpu.frame++;
                this._readback?.advance(this._gpu.frame);
            }
        }
    }

    /** freeze the virtual clock: gameplay (`time.deltaTime`/`elapsed`) and physics hold; the real clock keeps
     * running for camera/UI/input. takes effect next frame. {@link resume} restores the prior {@link timescale}. */
    pause(): void {
        this._scheduler.pause();
    }

    /** unfreeze the virtual clock. */
    resume(): void {
        this._scheduler.resume();
    }

    /** set the virtual timescale: 1 real time, <1 slow-mo, >1 fast-forward, 0 freeze (negative clamps to 0).
     * read via `time.scale`. */
    timescale(scale: number): void {
        this._scheduler.setScale(scale);
    }

    /** create a new entity, returns its ID */
    create(): number {
        const eid = this._entities.add();
        if (eid + 1 > this._highWater) this._highWater = eid + 1;
        for (const entry of this._storage.values()) {
            for (const field of entry.fields.values()) field.ensure(eid + 1);
        }
        return eid;
    }

    /** destroy an entity */
    destroy(eid: number): void {
        if (!this._entities.exists(eid)) return;
        forgetGlobalTransformEntity(this, eid);
        this._queries.onEntityRemoved(eid);
        for (const tables of this._tablesByComponent.values()) {
            for (const table of tables) table.release(eid);
        }
        this._components.clear(eid);
        for (const entry of this._storage.values()) {
            for (const field of entry.fields.values()) field.clear(eid);
        }
        this._entities.remove(eid);
        this._identity.forget(eid);
    }

    /** true if entity ID is alive */
    exists(eid: number): boolean {
        return this._entities.exists(eid);
    }

    /** snapshot of every alive entity id */
    entities(): readonly number[] {
        return this._entities.all();
    }

    /**
     * create-stamp for an entity id, bumped on every allocation (`create`), fresh or recycled. A
     * consumer holding an eid across frames caches the stamp beside it and compares alongside a
     * membership check: `has(eid, Component)` catches a plain despawn (destroy leaves the stamp
     * unchanged), the stamp catches a same-update destroy+create realias that membership misses.
     * Neither alone suffices. `0` for an eid never created.
     * @example
     * const stamp = state.stamp(eid); // cache beside the held eid
     * if (!state.has(eid, Body) || state.stamp(eid) !== stamp) evict(); // despawn or realias
     */
    stamp(eid: number): number {
        return this._entities.stamp(eid);
    }

    /**
     * entity identity recorded by `load`: the authored set + each entity's
     * scene `id`. `serialize` reads it to round-trip refs by name and to skip
     * warm-derived entities. See {@link Identity}
     */
    get identity(): Identity {
        return this._identity;
    }

    /**
     * read access to the component-membership bitset. A GPU producer that
     * scans a buffer by index gates on `state.membership.bit(C)` rather than a
     * per-field sentinel; the standard membership mirror flushes the bitset to
     * the `"membership"` buffer each frame. See {@link Membership}
     */
    get membership(): Membership {
        return this._components;
    }

    /**
     * attach a component to an entity. Default values declared via the component's
     * `Traits.defaults` are routed through each field's `.set` (for fields
     * implementing the `Single` contract): dirty tracking falls out automatically.
     * @example
     * state.add(eid, Health);
     * Health.current.set(eid, 100);
     */
    add<T>(eid: number, component: T): void {
        useState(this);
        const excluded = this.registry.getExclusions(component as Component);
        if (excluded) {
            for (const other of excluded) {
                if (this._components.has(eid, other)) {
                    const a = this.registry.getName(component as Component) ?? "?";
                    const b = this.registry.getName(other) ?? "?";
                    throw new Error(
                        `state.add: cannot attach "${a}" to entity ${eid} — excluded by "${b}"`,
                    );
                }
            }
        }
        this.of(component as Component);
        if (this._components.add(eid, component)) {
            this.notifyMembership(component as Component, eid, true);
            const tables = this._tablesByComponent.get(idOf(component as Component));
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
            this._queries.onComponentChanged(eid, component, this._components);
            this.registry.applyDefaults(this, component as Component, eid);
        } else {
            console.warn("state.add: component already attached to entity", eid);
        }
    }

    /** detach a component from an entity */
    remove(eid: number, component: any): void {
        if (retainsGlobalTransform(this, eid, component)) return;
        if (this._components.remove(eid, component)) {
            this.notifyMembership(component as Component, eid, false);
            const tables = this._tablesByComponent.get(idOf(component as Component));
            if (tables)
                for (const table of tables) table.detachComponent(eid, component as Component);
            this._queries.onComponentChanged(eid, component, this._components);
        }
    }

    /** true if entity has the component */
    has<T>(eid: number, component: T): boolean {
        return this._components.has(eid, component);
    }

    /**
     * find entities matching component terms
     * @example
     * for (const eid of state.query([Health, not(Dead)])) {
     *     Health.current[eid] -= 1;
     * }
     */
    query(terms: any[]): Iterable<number> {
        return this._queries.find(terms, this._components, this._entities);
    }

    /**
     * find exactly one entity, warns if multiple match, returns -1 when nothing matches
     * @example
     * const player = state.only([Player]);
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
            console.warn("state.only: expected 1 match, found multiple");
        }
        return result;
    }

    /** wire a system into the scheduler */
    addSystem(system: System, pluginName?: string): void {
        this._scheduler.register(system, pluginName);
    }

    /** remove a previously-added system */
    removeSystem(system: System): void {
        this._scheduler.unregister(system);
    }

    /**
     * hot-swap a live system's behavior in place. the reloaded module's
     * `update`/`setup`/`dispose` replace the old ones on the same registered
     * object, preserving its identity, ordering, and setup state. The engine
     * `swap` (plugin-level) drives this per system; not a per-frame call.
     */
    swap(old: System, next: System): void {
        this._scheduler.swap(old, next);
    }

    /** true if the system is live in the scheduler; `swap` validates its pairing against this */
    hasSystem(system: System): boolean {
        return this._scheduler.has(system);
    }

    /** record a CPU timing entry; no-op when no sink is installed */
    record(name: string, ms: number): void {
        this._scheduler.record?.(name, ms);
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

    /** report a GPU fence-wait duration; no-op when no sink is installed */
    fenceWait(ms: number): void {
        this._scheduler.fenceWait?.(ms);
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
     * resolving after a scene switch) checks this before touching the State, so a late result no-ops instead
     * of mutating a dead world.
     */
    get disposed(): boolean {
        return this._disposed;
    }

    /**
     * register a teardown callback tied to this State's lifetime. Callbacks run in LIFO order (last
     * registered, first run) at {@link dispose}, so a DOM mount, listener, or rAF loop keeps its cleanup
     * beside its creation site. A callback registered after dispose has already run fires immediately
     * (paired with {@link signal}, already aborted), so a late async step never leaks silently.
     * @example
     * const el = document.createElement("div");
     * container.appendChild(el);
     * state.onDispose(() => el.remove());
     */
    onDispose(fn: () => void): void {
        if (this._disposed) {
            fn();
            return;
        }
        this._disposals.push(fn);
    }

    /**
     * an {@link AbortSignal} tied to this State's lifetime, aborted when {@link dispose} runs (already
     * aborted if read afterward). Pass it as `{ signal }` to `addEventListener`, `fetch`, or any
     * abortable API to detach on teardown with zero removal code. Lazily created on first read.
     * @example
     * window.addEventListener("resize", onResize, { signal: state.signal });
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
        // the list now carries user cleanups (a Svelte unmount, an app rAF stop) that throw more readily
        // than engine hooks, and LIFO runs them first — a throw must not skip the remaining callbacks or
        // the scheduler/query teardown below, or it re-opens the leak this list closes. Report, never mask.
        for (let i = this._disposals.length - 1; i >= 0; i--) {
            try {
                this._disposals[i]();
            } catch (err) {
                console.error("State.dispose: a teardown callback threw:", err);
            }
        }
        this._disposals.length = 0;
        if (this._withCompute) this._withCompute(() => this._scheduler.dispose(this));
        else this._scheduler.dispose(this);
        this._queries.clear();
        this._storage.clear();
        for (const table of this._tables.values()) table.dispose();
        this._tables.clear();
        this.globalTransformRuntime = undefined;
        this.endGpuFrame();
        this._pendingCopies.length = 0;
        this._uploadStages.clear();
        this._resources.clear();
        this.registry.clear();
        for (const resource of this._gpuResources) {
            try {
                resource.destroy();
            } catch (err) {
                console.error("State.dispose: GPU resource release threw:", err);
            }
        }
        this._gpuResources.clear();
        this._gpu?.buffers.clear();
        this._gpu?.textures.clear();
        this._gpu?.samplers.clear();
        this._gpu?.typed.clear();
    }
}
