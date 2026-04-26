import type { State } from "./state";

export class CycleError extends Error {
    constructor(message = "Circular dependency detected") {
        super(message);
        this.name = "CycleError";
    }
}

export function toposort<T>(nodes: T[], edges: [T, T][]): T[] {
    if (nodes.length === 0) return [];

    const graph = new Map<T, Set<T>>();
    const inDegree = new Map<T, number>();

    for (const node of nodes) {
        graph.set(node, new Set());
        inDegree.set(node, 0);
    }

    for (const [from, to] of edges) {
        if (!graph.has(from) || !graph.has(to)) continue;
        graph.get(from)!.add(to);
        inDegree.set(to, inDegree.get(to)! + 1);
    }

    detectCycle(nodes, graph);

    const queue: T[] = [];
    const sorted: T[] = [];

    for (const node of nodes) {
        if (inDegree.get(node) === 0) queue.push(node);
    }

    while (queue.length > 0) {
        const node = queue.shift()!;
        sorted.push(node);

        for (const dep of graph.get(node)!) {
            const newDegree = inDegree.get(dep)! - 1;
            inDegree.set(dep, newDegree);
            if (newDegree === 0) queue.push(dep);
        }
    }

    return sorted;
}

function detectCycle<T>(nodes: T[], graph: Map<T, Set<T>>): void {
    const visited = new Set<T>();
    const stack = new Set<T>();

    function hasCycle(node: T): boolean {
        if (stack.has(node)) return true;
        if (visited.has(node)) return false;

        visited.add(node);
        stack.add(node);

        for (const dep of graph.get(node)!) {
            if (hasCycle(dep)) return true;
        }

        stack.delete(node);
        return false;
    }

    for (const node of nodes) {
        if (hasCycle(node)) throw new CycleError();
    }
}

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
    /** seconds since last frame, clamped to fixedDeltaTime * MAX_FIXED_STEPS */
    deltaTime: number;
    /** seconds since last frame, raw rAF interval before clamping */
    rawDeltaTime: number;
    /** fixed timestep interval (1/60) */
    fixedDeltaTime: number;
    /** total elapsed time in seconds */
    elapsed: number;
    /** fixed steps taken this frame (0–4) */
    fixedSteps: number;
    /** cumulative fixed tick count */
    fixedTick: number;
    /** true when fixed steps were clamped this frame */
    throttled: boolean;
    /** ms spent awaiting the prior frame's GPU fence before this frame began */
    fenceWaitMs: number;
}

export type SystemGroup = "setup" | "fixed" | "simulation" | "draw";

/** unit of behavior — update, setup, dispose, scheduling */
export interface System {
    readonly update?: (state: State) => void;
    readonly setup?: (state: State) => void;
    readonly dispose?: (state: State) => void;
    readonly group?: SystemGroup;
    readonly annotations?: Record<string, unknown>;
    readonly first?: boolean;
    readonly last?: boolean;
    readonly before?: readonly System[];
    readonly after?: readonly System[];
}

export class OrderingError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "OrderingError";
    }
}

export class Scheduler {
    private readonly _systems = new Set<System>();
    private _systemsVersion = 0;
    private _accumulator = 0;
    private readonly _initialized = new WeakSet<System>();
    private _cache = new Map<SystemGroup, System[]>();
    private _cacheVersion = -1;
    private _time: Time = {
        deltaTime: 0,
        rawDeltaTime: 0,
        fixedDeltaTime: Time.FIXED_DT,
        elapsed: 0,
        fixedSteps: 0,
        fixedTick: 0,
        throttled: false,
        fenceWaitMs: 0,
    };
    private readonly _names = new Map<System, string>();
    private readonly _nameCounters = new Map<string, number>();
    private _cpu = new Map<string, number>();
    mode: "edit" | "play" | undefined = undefined;

    get systems(): ReadonlySet<System> {
        return this._systems;
    }

    get systemsVersion(): number {
        return this._systemsVersion;
    }

    get accumulator(): number {
        return this._accumulator;
    }

    set accumulator(value: number) {
        this._accumulator = value;
    }

    get time(): Readonly<Time> {
        return this._time;
    }

    get cpu(): ReadonlyMap<string, number> {
        return this._cpu;
    }

    reportCpu(name: string, ms: number): void {
        this._cpu.set(name, (this._cpu.get(name) ?? 0) + ms);
    }

    reportFenceWait(ms: number): void {
        this._time.fenceWaitMs = ms;
    }

    register(system: System, pluginName?: string): void {
        this._systems.add(system);
        this._systemsVersion++;
        if (pluginName !== undefined) {
            const prefix = pluginName || "?";
            const idx = this._nameCounters.get(prefix) ?? 0;
            this._nameCounters.set(prefix, idx + 1);
            this._names.set(system, `${prefix}/${idx}`);
        }
    }

    unregister(system: System): void {
        if (this._systems.delete(system)) {
            this._systemsVersion++;
        }
    }

    step(state: State, deltaTime = Time.DEFAULT_DT): void {
        const fixedDt = Time.FIXED_DT;
        const maxDt = fixedDt * Time.MAX_FIXED_STEPS;

        this._cpu.clear();

        this._time.rawDeltaTime = deltaTime;
        this._time.throttled = deltaTime > maxDt;
        deltaTime = Math.min(deltaTime, maxDt);
        this._time.deltaTime = deltaTime;
        this._time.elapsed += deltaTime;
        this._accumulator += deltaTime;

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

        this._time.deltaTime = deltaTime;
        this.runGroup(state, "simulation");
        this.runGroup(state, "draw");
    }

    private runGroup(state: State, group: SystemGroup): void {
        for (const system of this.getSorted(group)) {
            if (this.mode !== undefined) {
                const systemMode = (system.annotations?.mode as string | undefined) ?? "play";
                if (systemMode !== "always" && systemMode !== this.mode) continue;
            }
            if (!this._initialized.has(system)) {
                system.setup?.(state);
                this._initialized.add(system);
            }
            if (system.update) {
                const t0 = performance.now();
                system.update(state);
                const dt = performance.now() - t0;
                const name = this._names.get(system) ?? "?";
                this._cpu.set(name, (this._cpu.get(name) ?? 0) + dt);
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

function sortSystems(systems: System[], allSystems?: System[]): System[] {
    const all = allSystems ?? systems;
    validateSystems(systems, all);

    const first = systems.filter((s) => s.first);
    const last = systems.filter((s) => s.last);
    const normal = systems.filter((s) => !s.first && !s.last);

    return [
        ...toposort(first, buildEdges(first)),
        ...toposort(normal, buildEdges(normal)),
        ...toposort(last, buildEdges(last)),
    ];
}

function buildEdges(systems: System[]): [System, System][] {
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

function validateSystems(systems: System[], all: System[]): void {
    for (const s of systems) {
        if (s.first && s.last) {
            throw new OrderingError("System cannot have both first and last constraints");
        }

        const group = s.group ?? "simulation";
        for (const ref of s.before ?? []) checkGroup(ref, group, all);
        for (const ref of s.after ?? []) checkGroup(ref, group, all);
    }
}

function checkGroup(ref: System, group: string, all: System[]): void {
    if (!all.includes(ref)) return;
    const refGroup = ref.group ?? "simulation";
    if (refGroup !== group) {
        throw new OrderingError(`Cross-group constraint: ${group} references ${refGroup}`);
    }
}
