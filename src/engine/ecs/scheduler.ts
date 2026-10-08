import type { World } from "./world";

function invalidDelta(): never {
    throw new Error("step deltaTime must be a finite, non-negative number");
}

/**
 * frame timing constants and per-frame data
 */
export const Time = {
    FIXED_DT: 1 / 60,
    DEFAULT_DT: 1 / 60,
    MAX_FIXED_STEPS: 4,
} as const;

export interface Time {
    /** In fixed, FIXED_DT seconds; outside fixed, clamped and scaled virtual frame seconds (0 while paused). */
    deltaTime: number;
    /** seconds since last frame, raw rAF interval before clamping or scaling */
    rawDeltaTime: number;
    /** real seconds since last frame: clamped, never scaled. the escape hatch for presentation that must run
     * through a pause/slow-mo (camera juice, UI, input) */
    realDeltaTime: number;
    /** fixed timestep interval (1/60), constant; slow-mo reduces tick frequency, not per-tick dt */
    fixedDeltaTime: number;
    /** In fixed, fixedTick * FIXED_DT seconds; outside fixed, total virtual frame seconds, frozen while paused. */
    elapsed: number;
    /** total elapsed real time in seconds (advances with {@link Time.realDeltaTime}, runs through a pause) */
    realElapsed: number;
    /** virtual timescale multiplier (1 = real time, <1 slow-mo, >1 fast-forward). set via `world.setTimeScale` */
    scale: number;
    /** Freezes the virtual frame clock and step's fixed work, not explicit ticks. Resume retains scale. */
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
    readonly update?: (world: World) => void;
    readonly setup?: (world: World) => void;
    readonly dispose?: (world: World) => void;
    /** profiler/debug label; falls back to `pluginName/index` when omitted */
    readonly name?: string;
    readonly group?: SystemGroup;
    readonly annotations?: Record<string, unknown>;
    readonly first?: boolean;
    readonly last?: boolean;
    /** runs after every other system in its group; at most one terminal system is allowed */
    readonly terminal?: boolean;
    /** @internal Module-owned slot outside ordinary ordering, including terminal systems.
     * At most one boundary occupies each end of a group. */
    readonly boundary?: "before" | "after";
    readonly before?: readonly System[];
    readonly after?: readonly System[];
}

export class Scheduler {
    logAndPauseErrors = false;
    private readonly _systems = new Set<System>();
    private readonly _boundaries = new Map<System, "before" | "after">();
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

    /** @internal Capture the integer simulation clock, not frame pacing. */
    snapshot(): number {
        return this._time.fixedTick;
    }

    /** @internal Recovery changes only the integer simulation clock, never frame pacing. */
    restore(tick: number): void {
        this._time.fixedTick = tick;
    }

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
        if (!Number.isFinite(scale))
            throw new Error(`timescale received ${scale} (must be a finite number)`);
        this._time.scale = Math.max(0, scale);
    }

    dispose(world: World): void {
        for (const system of this._systems) {
            try {
                system.dispose?.(world);
            } catch (err) {
                console.error(
                    `System "${this._names.get(system) ?? system.name ?? "?"}" threw during dispose:`,
                    err,
                );
            }
        }
    }

    /** @internal Module-owned slots bracket every ordinary system, including terminal systems. */
    registerBoundary(system: System, position: "before" | "after", pluginName?: string): void {
        for (const [registered, slot] of this._boundaries) {
            if (registered !== system && slot === position && registered.group === system.group)
                throw new Error(`System group ${system.group} already has a ${position} boundary`);
        }
        this._boundaries.set(system, position);
        this.register(system, pluginName);
    }

    register(system: System, pluginName?: string): void {
        this._systems.add(system);
        this._systemsVersion++;
        // a system's own `name` labels its profiler row legibly (`StandardRenderer/forward`); without
        // one, fall back to the registration index (`StandardRenderer/1`)
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
            this._boundaries.delete(system);
            this._names.delete(system);
            this._initialized.delete(system);
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

    step(world: World, input: Readonly<{ deltaTime: number }>): void {
        const deltaTime = input.deltaTime;
        if (!Number.isFinite(deltaTime) || deltaTime < 0) {
            invalidDelta();
        }
        const fixedDt = Time.FIXED_DT;
        const maxDt = fixedDt * Time.MAX_FIXED_STEPS;

        this._time.rawDeltaTime = deltaTime;
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

        this.runGroup(world, "setup");

        // the cap runs on the post-scale accumulator — timescale must not reintroduce the spiral it clamps
        // out of `real` above. Past the cap, drop the backlog rather than carry debt into future frames.
        let steps = 0;
        while (this._accumulator >= fixedDt && steps < Time.MAX_FIXED_STEPS) {
            this.tick(world);
            this._accumulator -= fixedDt;
            steps++;
        }
        // throttled if either clock overran its budget: the pre-scale clamp discarded raw frame time
        // (moot while paused — no virtual work was owed), or the scaled accumulator outran the step cap.
        const postScaleOverrun = this._accumulator >= fixedDt;
        if (postScaleOverrun) this._accumulator %= fixedDt;
        this._time.throttled = (!this._time.paused && deltaTime > maxDt) || postScaleOverrun;
        this._time.fixedSteps = steps;
        this._time.fixedAlpha = this._accumulator / fixedDt;

        this._time.deltaTime = scaled;
        this.runGroup(world, "simulation");
        this.runGroup(world, "draw");
    }

    tick(world: World): void {
        const deltaTime = this._time.deltaTime;
        const elapsed = this._time.elapsed;
        this._time.fixedTick++;
        this._time.deltaTime = Time.FIXED_DT;
        this._time.elapsed = this._time.fixedTick * Time.FIXED_DT;
        try {
            this.runGroup(world, "fixed");
        } finally {
            this._time.deltaTime = deltaTime;
            this._time.elapsed = elapsed;
        }
    }

    private runGroup(world: World, group: SystemGroup): void {
        const record = this.record;
        const systems = this.getSorted(group);
        for (let i = 0; i < systems.length; i++) {
            const system = systems[i];
            if (this._errored.has(system)) continue;
            try {
                if (!this._initialized.has(system)) {
                    system.setup?.(world);
                    this._initialized.add(system);
                }
                if (system.update) {
                    if (record) {
                        const t0 = performance.now();
                        system.update(world);
                        record(this._names.get(system) ?? "?", performance.now() - t0);
                    } else {
                        system.update(world);
                    }
                }
            } catch (e) {
                if (!this.logAndPauseErrors) {
                    const name = this._names.get(system) ?? system.name ?? "?";
                    throw new Error(
                        `System "${name}" threw: ${e instanceof Error ? e.message : String(e)}`,
                        { cause: e },
                    );
                }
                // A hot-reloaded bug must not wedge a live host. Pause until a swap supplies the fix;
                // failed setup stays uninitialized so the replacement retries it.
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
            if (this._cache.size !== 0) this._cache.clear();
            this._cacheVersion = this._systemsVersion;
        }

        const cached = this._cache.get(group);
        if (cached) return cached;

        const all = Array.from(this._systems);
        const filtered = all.filter((s) => (s.group ?? "simulation") === group);
        const ordinary = filtered.filter((s) => !this._boundaries.has(s));
        const sorted = [
            ...filtered.filter((s) => this._boundaries.get(s) === "before"),
            ...sortSystems(ordinary, all),
            ...filtered.filter((s) => this._boundaries.get(s) === "after"),
        ];
        this._cache.set(group, sorted);
        return sorted;
    }
}

function sortSystems(systems: System[], all: System[]): System[] {
    validate(systems, all);

    const first = systems.filter((s) => s.first);
    const last = systems.filter((s) => s.last);
    const terminal = systems.filter((s) => s.terminal);
    const normal = systems.filter((s) => !s.first && !s.last && !s.terminal);

    return [
        ...kahnSort(first, edgesOf(first)),
        ...kahnSort(normal, edgesOf(normal)),
        ...kahnSort(last, edgesOf(last)),
        ...terminal,
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
    const terminals = systems.filter((s) => s.terminal);
    if (terminals.length > 1)
        throw new Error("System group cannot have more than one terminal system");

    for (const s of systems) {
        if (s.first && s.last)
            throw new Error("System cannot have both first and last constraints");
        if (s.terminal && (s.first || s.last)) {
            throw new Error("System cannot combine terminal with first or last constraints");
        }
        const group = s.group ?? "simulation";
        const sRank = s.first ? 0 : s.last ? 2 : s.terminal ? 3 : 1;
        for (const ref of [...(s.before ?? []), ...(s.after ?? [])]) {
            if (!all.includes(ref)) continue;
            const refGroup = ref.group ?? "simulation";
            if (refGroup !== group) {
                throw new Error(`Cross-group constraint: ${group} references ${refGroup}`);
            }
            // satisfiable cross-partition constraints hold by construction (first < normal < last)
            // and stay silent; only the unsatisfiable direction errors
            const refRank = ref.first ? 0 : ref.last ? 2 : ref.terminal ? 3 : 1;
            if (s.before?.includes(ref) && sRank > refRank) {
                throw new Error(
                    `Unsatisfiable ordering: a ${partitionName(sRank)} system cannot run before a ${partitionName(refRank)} system`,
                );
            }
            if (s.after?.includes(ref) && sRank < refRank) {
                throw new Error(
                    `Unsatisfiable ordering: a ${partitionName(sRank)} system cannot run after a ${partitionName(refRank)} system`,
                );
            }
        }
    }
}

function partitionName(rank: number): string {
    return rank === 0 ? "first" : rank === 2 ? "last" : rank === 3 ? "terminal" : "normal";
}
