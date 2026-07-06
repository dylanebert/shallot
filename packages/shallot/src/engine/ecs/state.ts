import { Components, type Membership } from "./component";
import { Entities } from "./entity";
import { Identity } from "./identity";
import { Queries } from "./query";
import { Scheduler, type System, Time } from "./scheduler";
import { applyDefaults, getExclusions, getName } from "./traits";

/**
 * entity capacity, fixed at app construction. defaults to 65536. override via
 * `build({ capacity })` before any allocation; shared across every {@link State}
 * in the process. multi-state scenarios (networking server+client, rollback)
 * all share one capacity by construction.
 *
 * footgun: don't bind at module top level (`const SIZE = capacity * 4`) — captures
 * the default before `build` runs. read inside functions or factories instead.
 */
export let capacity = 65536;

/**
 * render device-pixel ratio for canvas-bound views, fixed at app construction. `"auto"`
 * (default) clamps the display's `devicePixelRatio` to `[1, 2]` (react-three-fiber's default):
 * crisp on HiDPI, never below logical resolution, capped so a DPR-3 phone doesn't pay 9× the fill.
 * A fixed number overrides: `1` renders at CSS resolution (cheapest, the three.js literal default),
 * `2` forces 2×, a value below 1 downscales for a pixel-art look (the upscale switches to
 * nearest-neighbor). Read at every resize (see {@link attachCanvas}), so dragging a window between
 * monitors re-sizes the backing. Set via `build({ pixelRatio })`.
 */
export let pixelRatio: number | "auto" = "auto";

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
    /**
     * when set, scheduler skips systems whose `annotations.mode` differs.
     * `"always"`-annotated systems run in both modes. fixed at construction.
     * rebuild the app to switch modes. leave undefined to run every system
     * regardless of annotation.
     */
    readonly mode: "edit" | "play" | undefined;

    private _scheduler = new Scheduler();
    private _entities = new Entities();
    private _components = new Components();
    private _queries = new Queries();
    private _identity = new Identity();
    private _disposed = false;

    constructor(opts?: {
        capacity?: number;
        pixelRatio?: number | "auto";
        mode?: "edit" | "play";
    }) {
        if (opts?.capacity !== undefined) capacity = opts.capacity;
        if (opts?.pixelRatio !== undefined) pixelRatio = opts.pixelRatio;
        this.mode = opts?.mode;
    }

    /** current frame time and delta */
    get time(): Readonly<Time> {
        return this._scheduler.time;
    }

    /** advance one frame */
    step(deltaTime = Time.DEFAULT_DT): void {
        this._scheduler.step(this, deltaTime);
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
        if (eid + 1 > capacity) {
            throw new Error(
                `Entity count ${eid + 1} exceeds configured capacity ${capacity}. ` +
                    `Increase via app build config: { capacity: ${Math.max(eid + 1, capacity * 2)} }.`,
            );
        }
        return eid;
    }

    /** destroy an entity */
    destroy(eid: number): void {
        if (!this._entities.exists(eid)) return;
        this._queries.onEntityRemoved(eid);
        this._components.clear(eid);
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
        const excluded = getExclusions(component as Record<string, unknown>);
        if (excluded) {
            for (const other of excluded) {
                if (this._components.has(eid, other)) {
                    const a = getName(component as Record<string, unknown>) ?? "?";
                    const b = getName(other) ?? "?";
                    throw new Error(
                        `state.add: cannot attach "${a}" to entity ${eid} — excluded by "${b}"`,
                    );
                }
            }
        }
        if (this._components.add(eid, component)) {
            this._queries.onComponentChanged(eid, component, this._components);
            applyDefaults(component as Record<string, unknown>, eid);
        } else {
            console.warn("state.add: component already attached to entity", eid);
        }
    }

    /** detach a component from an entity */
    remove(eid: number, component: any): void {
        if (this._components.remove(eid, component)) {
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
     * find exactly one entity, warns if multiple match
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

    /** tear down the world; disposes every registered system */
    dispose(): void {
        if (this._disposed) return;
        this._scheduler.dispose(this);
        this._queries.clear();
        this._disposed = true;
    }
}
