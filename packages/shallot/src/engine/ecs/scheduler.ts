import type { State } from "./state";

/**
 * frame timing constants and per-frame data
 * @expand
 */
export const Time = {
    FIXED_DT: 1 / 60,
    DEFAULT_DT: 1 / 60,
    MAX_FIXED_STEPS: 4,
} as const;

export interface Time {
    /** virtual seconds since last frame: clamped, then scaled by {@link Time.scale} (0 while paused). the
     * default gameplay clock; sim systems read this, so pause and slow-mo reach them for free */
    deltaTime: number;
    /** seconds since last frame, raw rAF interval before clamping or scaling */
    rawDeltaTime: number;
    /** real seconds since last frame: clamped, never scaled. the escape hatch for presentation that must run
     * through a pause/slow-mo (camera juice, UI, input) */
    realDeltaTime: number;
    /** fixed timestep interval (1/60), constant; slow-mo reduces tick frequency, not per-tick dt */
    fixedDeltaTime: number;
    /** total elapsed virtual time in seconds (advances with {@link Time.deltaTime}, frozen while paused) */
    elapsed: number;
    /** total elapsed real time in seconds (advances with {@link Time.realDeltaTime}, runs through a pause) */
    realElapsed: number;
    /** virtual timescale multiplier (1 = real time, <1 slow-mo, >1 fast-forward). set via `state.timescale` */
    scale: number;
    /** when true the virtual clock is frozen: `deltaTime`/`elapsed` hold and no fixed steps run. set via
     * `state.pause`/`state.resume`. separate from `scale = 0` so resume restores the prior speed */
    paused: boolean;
    /** fixed steps taken this frame (0–4) */
    fixedSteps: number;
    /** cumulative fixed tick count */
    fixedTick: number;
    /** fraction of a fixed step past the last tick, in [0, 1). use for render interpolation */
    fixedAlpha: number;
    /** true when fixed steps were clamped this frame */
    throttled: boolean;
}

export type SystemGroup = "setup" | "fixed" | "simulation" | "draw";

/** unit of behavior: update, setup, dispose, scheduling */
export interface System {
    readonly update?: (state: State) => void;
    readonly setup?: (state: State) => void;
    readonly dispose?: (state: State) => void;
    /** profiler/debug label; falls back to `pluginName/index` when omitted */
    readonly name?: string;
    readonly group?: SystemGroup;
    readonly annotations?: Record<string, unknown>;
    readonly first?: boolean;
    readonly last?: boolean;
    readonly before?: readonly System[];
    readonly after?: readonly System[];
}

export class Scheduler {
    private readonly _systems = new Set<System>();
    private _systemsVersion = 0;
    private _accumulator = 0;
    private readonly _initialized = new WeakSet<System>();
    private readonly _errored = new Set<System>();
    private _cache = new Map<SystemGroup, System[]>();
    private _cacheVersion = -1;
    private _time: Time = {
        deltaTime: 0,
        rawDeltaTime: 0,
        realDeltaTime: 0,
        fixedDeltaTime: Time.FIXED_DT,
        elapsed: 0,
        realElapsed: 0,
        scale: 1,
        paused: false,
        fixedSteps: 0,
        fixedTick: 0,
        fixedAlpha: 0,
        throttled: false,
    };
    private readonly _names = new Map<System, string>();
    private readonly _nameCounters = new Map<string, number>();
    /** optional CPU timing sink — installed by the profile plugin */
    record?: (name: string, ms: number) => void;
    /** optional fence-wait telemetry sink — installed by the profile plugin */
    fenceWait?: (ms: number) => void;

    get time(): Readonly<Time> {
        return this._time;
    }

    pause(): void {
        this._time.paused = true;
    }

    resume(): void {
        this._time.paused = false;
    }

    setScale(scale: number): void {
        this._time.scale = Math.max(0, scale);
    }

    dispose(state: State): void {
        for (const system of this._systems) {
            system.dispose?.(state);
        }
    }

    register(system: System, pluginName?: string): void {
        this._systems.add(system);
        this._systemsVersion++;
        // a system's own `name` labels its profiler row legibly (`Sear/forward`); without
        // one, fall back to the registration index (`Sear/1`)
        if (system.name !== undefined) {
            this._names.set(system, pluginName ? `${pluginName}/${system.name}` : system.name);
        } else if (pluginName !== undefined) {
            const prefix = pluginName || "?";
            const idx = this._nameCounters.get(prefix) ?? 0;
            this._nameCounters.set(prefix, idx + 1);
            this._names.set(system, `${prefix}/${idx}`);
        }
    }

    unregister(system: System): void {
        if (this._systems.delete(system)) {
            this._errored.delete(system);
            this._systemsVersion++;
        }
    }

    has(system: System): boolean {
        return this._systems.has(system);
    }

    /**
     * hot-swap a live system's behavior in place. Copies the new system's
     * `update`/`setup`/`dispose` onto the registered object, so its identity
     * (ordering edges, `_initialized` setup state, profiler label) is preserved
     * while the code that runs becomes the reloaded module's. No version bump —
     * ordering is unchanged, so the sort cache stays valid. PlayCanvas `swap`.
     */
    swap(old: System, next: System): void {
        if (!this._systems.has(old)) return;
        const m = old as {
            update?: System["update"];
            setup?: System["setup"];
            dispose?: System["dispose"];
        };
        m.update = next.update;
        m.setup = next.setup;
        m.dispose = next.dispose;
        // the swapped-in code is the fix a paused (thrown) system was waiting for — let it run
        this._errored.delete(old);
    }

    step(state: State, deltaTime = Time.DEFAULT_DT): void {
        const fixedDt = Time.FIXED_DT;
        const maxDt = fixedDt * Time.MAX_FIXED_STEPS;

        this._time.rawDeltaTime = deltaTime;
        this._time.throttled = deltaTime > maxDt;
        // clamp on the real dt — the spiral-of-death gate, before scaling (a large scale must not reintroduce it)
        const real = Math.min(deltaTime, maxDt);
        this._time.realDeltaTime = real;
        this._time.realElapsed += real;

        // the virtual clock the sim reads — paused freezes it, scale slows/speeds it. the fixed accumulator
        // follows it, so pause/slow-mo reach physics for free (fewer ticks, never a shorter per-tick dt).
        const scaled = this._time.paused ? 0 : real * this._time.scale;
        this._time.deltaTime = scaled;
        this._time.elapsed += scaled;
        this._accumulator += scaled;

        this.runGroup(state, "setup");

        let steps = 0;
        while (this._accumulator >= fixedDt) {
            this._time.deltaTime = fixedDt;
            this._time.fixedTick++;
            this.runGroup(state, "fixed");
            this._accumulator -= fixedDt;
            steps++;
        }
        this._time.fixedSteps = steps;
        this._time.fixedAlpha = this._accumulator / fixedDt;

        this._time.deltaTime = scaled;
        this.runGroup(state, "simulation");
        this.runGroup(state, "draw");
    }

    private runGroup(state: State, group: SystemGroup): void {
        const record = this.record;
        const mode = state.mode;
        for (const system of this.getSorted(group)) {
            if (this._errored.has(system)) continue;
            if (mode !== undefined) {
                const systemMode = (system.annotations?.mode as string | undefined) ?? "play";
                if (systemMode !== "always" && systemMode !== mode) continue;
            }
            // quarantine, not crash: a throwing system must not kill the frame loop (a hot-reloaded
            // bug would wedge a live host). It pauses after the first throw — a failed setup stays
            // uninitialized so the fix retries it — and resumes on its next swap or a rebuild.
            try {
                if (!this._initialized.has(system)) {
                    system.setup?.(state);
                    this._initialized.add(system);
                }
                if (system.update) {
                    if (record) {
                        const t0 = performance.now();
                        system.update(state);
                        record(this._names.get(system) ?? "?", performance.now() - t0);
                    } else {
                        system.update(state);
                    }
                }
            } catch (e) {
                this._errored.add(system);
                console.error(
                    `System "${this._names.get(system) ?? system.name ?? "?"}" threw and is paused until its next reload:`,
                    e,
                );
            }
        }
    }

    private getSorted(group: SystemGroup): System[] {
        if (this._systemsVersion !== this._cacheVersion) {
            this._cache.clear();
            this._cacheVersion = this._systemsVersion;
        }

        const cached = this._cache.get(group);
        if (cached) return cached;

        const all = Array.from(this._systems);
        const filtered = all.filter((s) => (s.group ?? "simulation") === group);
        const sorted = sortSystems(filtered, all);
        this._cache.set(group, sorted);
        return sorted;
    }
}

function sortSystems(systems: System[], all: System[]): System[] {
    validate(systems, all);

    const first = systems.filter((s) => s.first);
    const last = systems.filter((s) => s.last);
    const normal = systems.filter((s) => !s.first && !s.last);

    return [
        ...kahnSort(first, edgesOf(first)),
        ...kahnSort(normal, edgesOf(normal)),
        ...kahnSort(last, edgesOf(last)),
    ];
}

function kahnSort(nodes: System[], edges: [System, System][]): System[] {
    if (nodes.length === 0) return [];
    const adj = new Map<System, System[]>();
    const inDegree = new Map<System, number>();
    for (const node of nodes) {
        adj.set(node, []);
        inDegree.set(node, 0);
    }
    for (const [from, to] of edges) {
        if (!adj.has(from) || !inDegree.has(to)) continue;
        adj.get(from)!.push(to);
        inDegree.set(to, inDegree.get(to)! + 1);
    }
    const sorted: System[] = [];
    const queue: System[] = [];
    for (const node of nodes) if (inDegree.get(node) === 0) queue.push(node);
    while (queue.length) {
        const node = queue.shift()!;
        sorted.push(node);
        for (const next of adj.get(node)!) {
            const d = inDegree.get(next)! - 1;
            inDegree.set(next, d);
            if (d === 0) queue.push(next);
        }
    }
    if (sorted.length !== nodes.length) {
        throw new Error("Circular dependency between systems");
    }
    return sorted;
}

function edgesOf(systems: System[]): [System, System][] {
    const edges: [System, System][] = [];
    for (const system of systems) {
        for (const target of system.before ?? []) {
            if (systems.includes(target)) edges.push([system, target]);
        }
        for (const target of system.after ?? []) {
            if (systems.includes(target)) edges.push([target, system]);
        }
    }
    return edges;
}

function validate(systems: System[], all: System[]): void {
    for (const s of systems) {
        if (s.first && s.last) {
            throw new Error("System cannot have both first and last constraints");
        }
        const group = s.group ?? "simulation";
        for (const ref of [...(s.before ?? []), ...(s.after ?? [])]) {
            if (!all.includes(ref)) continue;
            const refGroup = ref.group ?? "simulation";
            if (refGroup !== group) {
                throw new Error(`Cross-group constraint: ${group} references ${refGroup}`);
            }
        }
    }
}
