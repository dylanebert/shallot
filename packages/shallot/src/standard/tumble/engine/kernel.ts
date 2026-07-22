// Loader for the wasm-simd128 physics kernel (kernel/, inlined by scripts/build-kernel.ts).
//
// The kernel is ~tens of KB, too large for a synchronous main-thread compile, so instantiation is
// async: call `init()` once before the first `step()`. `step()` itself stays synchronous. Every
// consumer shares one kernel instance over one `WebAssembly.Memory` — the SoA solver columns live
// in its linear memory and the TS side views them as `Float32Array`s.
//
// Two artifacts (scripts/build-kernel.ts). `init()` resolves threading itself: standalone (bun/node) and a
// cross-origin-isolated browser get the multithreaded artifact, which needs a shared `WebAssembly.Memory`;
// a browser without that isolation runs single-thread after one plain log naming the COOP/COEP headers the
// host is missing. `init({ threads })` is the advanced escape — 0 forces single-thread, n overrides the
// auto count. The MT artifact loads behind a dynamic `import()`, so a single-thread consumer never parses it.

import { KERNEL_WASM_BASE64 } from "./kernel.wasm";
import { createPool, maxWorkers, type Pool } from "./pool";

/** The kernel's exported surface. Grows as stage 3 ports each solver phase. */
export type Kernel = {
    memory: WebAssembly.Memory;
    /** Toolchain smoke buffer offset + scale, the standing wasm-simd128 cliff gate (kernel.test.ts). */
    scratchPtr(): number;
    smokeScale(len: number, k: number): void;

    // Shared-column arena (kernel/src/arena.rs). `reserve` lays out the columns for one step's counts
    // and may grow memory; `layoutPtr` returns the byte-offset header the TS views derive from (see
    // columns.ts). The phase shims drive the solve over those columns, one phase at a time.
    reserve(
        body: number,
        contact: number,
        manifold: number,
        point: number,
        wide: number,
        color: number,
        joint: number,
    ): void;
    layoutPtr(): number;

    // Persistent body columns (kernel/src/bodies.rs) — the awake body state held resident across
    // steps (4a), first in linear memory. `reserveBodies` sizes the region to the total-body
    // high-water (grow-only), relocating the manifold + geometry regions above it in place on a grow;
    // `bodyLayoutPtr` returns the byte-offset header TS derives its column views from (bodycolumns.ts).
    reserveBodies(cap: number): number;
    bodyLayoutPtr(): number;
    /** The record capacity the resident body region is sized to — the single source of truth for the
     * TS body-store's column-view lengths (bodycolumns.ts). Zero before the first `reserveBodies`. */
    bodyCap(): number;

    // Persistent fat-AABB column (kernel/src/fataabb.rs) — one enlarged broad-phase AABB per shape,
    // held resident so the in-kernel recycle loop (4b) tests contact overlap without a marshal. A second
    // low persistent region above the body region; `reserveFatAabb` sizes it to the shape high-water
    // (grow-only), relocating the manifold + geometry regions above it on a grow. `fatAabbLayoutPtr`
    // returns the byte-offset header TS derives its view from (fataabbcolumns.ts); `fatAabbCap` is the
    // authoritative capacity.
    reserveFatAabb(cap: number): number;
    fatAabbLayoutPtr(): number;
    fatAabbCap(): number;

    // Persistent shape column (kernel/src/shapes.rs) — one record per shapeId (type code, local
    // geometry, nextShapeId), held resident so the in-kernel finalize refit walks a body's shape list
    // without a marshal. A third low persistent region above the fat-AABB region; `reserveShapes` sizes
    // it to the shape high-water (grow-only), relocating the manifold + geometry regions above it on a
    // grow. `shapeLayoutPtr` returns the byte-offset header TS derives its views from
    // (shapecolumns.ts); `shapeCap` is the authoritative capacity.
    reserveShapes(cap: number): number;
    shapeLayoutPtr(): number;
    shapeCap(): number;

    // Persistent broad-phase columns (kernel/src/broad.rs) — the three dynamic-tree node pools plus the
    // pair-set membership arrays, held resident so the in-kernel pair query + tree rebuild (3d) run over
    // them without a marshal. A persistent region between the manifold and geometry regions;
    // `reserveBroad` sizes each of the six sub-columns (grow-only per column: pass 0 to hold a column at
    // its current size), relocating the geometry region above it on a grow. `broadLayoutPtr` returns the
    // byte-offset header TS derives its views from (broadcolumns.ts); `broadTreeCap`/`broadSetCap` are
    // the authoritative capacities.
    reserveBroad(capS: number, capK: number, capD: number, setCap: number): number;
    broadLayoutPtr(): number;
    broadTreeCap(i: number): number;
    broadSetCap(): number;
    broadGen(): number;

    // Broad-phase pair query + tree rebuild (kernel/src/pairwork.rs, 3d). `reservePairs` lays out the
    // per-step slab at the solver base (tree-state header + move buffer + dynamic moved-bitset + the
    // candidate output + rebuild scratch); TS writes the inputs through the `pairs*Ptr` headers,
    // `queryPairs` finds the surviving pairs (dedup + pair-set membership) into the candidate slab and
    // returns the entry count (grow + re-run if it exceeds `candCap`), and `rebuildTrees` median-rebuilds
    // the dynamic then kinematic trees, writing each new `[root, nodeCount, freeList]` to the rebuild-out
    // header. Both run over the resident broad-phase region (kernel/src/broad.rs); TS applies the
    // remaining filters + creates contacts over the returned slab (src/pairs.ts).
    reservePairs(moveCount: number, movedWords: number, candCap: number, maxProxy: number): void;
    pairsStatePtr(): number;
    pairsMovePtr(): number;
    pairsMovedPtr(): number;
    pairsCandEndPtr(): number;
    pairsCandPtr(): number;
    pairsRebuildOutPtr(): number;
    queryPairs(setCap: number): number;
    rebuildTrees(): void;

    // Static geometry columns (kernel/src/geo.rs) — convex-hull pools uploaded once per interned hull,
    // read by the convex narrowphase. `reserveGeometry` lays out the pools (before the solver columns)
    // for the given totals; `geoLayoutPtr` returns the byte-offset header TS writes the hulls through
    // (geocolumns.ts). `collideHullsGeo` runs the hull-hull narrowphase over two column-backed hulls,
    // writing the manifold to the buffer at `geoOutPtr` — the 3c.2b geometry-read verification.
    reserveGeometry(hulls: number, verts: number, edges: number, faces: number): void;
    geoLayoutPtr(): number;

    // Persistent contact-manifold columns (kernel/src/manifolds.rs) — the warm-start state that
    // survives across steps, keyed by contactId. `reserveManifolds` lays out the directory + pool for
    // the given capacities (before the geometry + solver columns), preserving live pool data across a
    // grow; `manifoldLayoutPtr` returns the byte-offset header TS derives its views from
    // (manifoldstore.ts).
    reserveManifolds(contactCap: number, manifoldCap: number): void;
    manifoldLayoutPtr(): number;
    collideHullsGeo(
        a: number,
        b: number,
        px: number,
        py: number,
        pz: number,
        qx: number,
        qy: number,
        qz: number,
        qs: number,
    ): number;
    geoOutPtr(): number;

    // Convex narrowphase batched dispatch (kernel/src/arena.rs, 3c.3). `reserveDispatch` lays out the
    // per-record input + output columns (at the solver base, consumed within collide); the collect pass
    // writes the input through `dispatchPtr`, `dispatchConvex` runs `compute_convex_manifold` per record
    // over the geometry + manifold columns, and the finish pass reads the touching flags at `dispatchOutPtr`.
    reserveDispatch(count: number): void;
    dispatchPtr(): number;
    dispatchOutPtr(): number;
    dispatchConvex(count: number): void;

    // Contact-recycle batched pass (kernel/src/arena.rs, 4b.3c). `reserveRecycle` lays out the per-record
    // input + output columns (at the solver base, consumed within collide, before the convex dispatch);
    // the collide walk writes the input through `recyclePtr`, `dispatchRecycle` runs box3d's recycle branch
    // per record over the resident body + fat-AABB + manifold columns, and the finish pass reads each
    // contact's result (0 recycled / 1 needs-narrowphase / 2 disjoint) at `recycleOutPtr`.
    reserveRecycle(count: number): void;
    recyclePtr(): number;
    recycleOutPtr(): number;
    dispatchRecycle(count: number, recycleDist: number, recycleDistNonTouching: number): void;

    integrateVelocities(gx: number, gy: number, gz: number, h: number): void;
    integratePositions(h: number, maxLinearVelocity: number, invDt: number): void;

    // Scalar (mesh / overflow) contact phases over a `[start, count)` contact-record range.
    prepareContacts(
        start: number,
        count: number,
        csBias: number,
        csMass: number,
        csImpulse: number,
        ssBias: number,
        ssMass: number,
        ssImpulse: number,
        warmStartScale: number,
    ): void;
    warmStartContacts(start: number, count: number): void;
    solveContacts(
        start: number,
        count: number,
        useBias: number,
        invH: number,
        contactSpeed: number,
    ): void;
    restitution(start: number, count: number, threshold: number): void;
    storeImpulses(start: number, count: number, hitEventThreshold: number): void;

    // Wide (convex) contact phases over a `[start, count)` wide-record range.
    prepareWideContacts(
        start: number,
        count: number,
        csBias: number,
        csMass: number,
        csImpulse: number,
        ssBias: number,
        ssMass: number,
        ssImpulse: number,
        warmStartScale: number,
    ): void;
    warmStartWideContacts(start: number, count: number): void;
    solveWideContacts(
        start: number,
        count: number,
        useBias: number,
        invH: number,
        contactSpeed: number,
    ): void;
    restitutionWide(start: number, count: number, threshold: number): void;
    storeWideImpulses(start: number, count: number, hitEventThreshold: number): void;

    // Batched per-color loop (jointless scenes) — one FFI crossing runs every active color's
    // wide-then-mesh block over the COLOR_SPAN column instead of a TS crossing per color.
    warmStartColors(): void;
    solveColors(useBias: number, invH: number, contactSpeed: number): void;
    restitutionColors(threshold: number): void;

    // The staged multithreaded solve (kernel/src/solve.rs), for a scene on a live pool. The whole
    // prepare → substep-loop → restitution → store → finalize region becomes ONE crossing: `solveBuild`
    // lays out the stage list on the main thread (before the pool wakes), then the orchestrator runs
    // `runMt` while each pooled worker runs `workerMain(index)` exactly once, stealing blocks over the
    // shared columns. The pose finalize rides the same stage list as its terminal stage (`dt` is the
    // full-step h it sweeps with; `enableContinuous` its fast-candidate gate), so solve + finalize cost
    // one wake round. Bit-identical to the serial path at any thread count (kernel/src/stages.rs).
    solveBuild(
        threadCount: number,
        subStepCount: number,
        wideTotal: number,
        meshStart: number,
        meshTotal: number,
        overflowStart: number,
        overflowCount: number,
        jointTotal: number,
        overflowJointStart: number,
        overflowJointCount: number,
        gx: number,
        gy: number,
        gz: number,
        h: number,
        invH: number,
        dt: number,
        invDt: number,
        maxLinearVelocity: number,
        contactSpeed: number,
        csBias: number,
        csMass: number,
        csImpulse: number,
        ssBias: number,
        ssMass: number,
        ssImpulse: number,
        warmStartScale: number,
        restitutionThreshold: number,
        hitEventThreshold: number,
        enableContinuous: number,
    ): void;
    // The outer collide phases on the same pool (kernel/src/parfor.rs): one flat block-claim sweep per
    // phase, each with its own build + run pair, each forked after its own `reserve*` so nothing grows
    // memory between the fork and the join. `parBuild` partitions `count` records over `threadCount`
    // threads and names the job the next `runMt` round drives; `a`/`b` carry the phase's scalars
    // (recycle's two tolerances; the convex dispatch has none). It **returns 1 to fork, 0 to run the
    // serial shim** — the fork floor (a sweep too small to beat its own wake) is priced in the kernel,
    // next to the machinery (kernel/src/parfor.rs).
    parBuild(kind: ParKind, count: number, threadCount: number, a: number, b: number): number;
    /** Run the built job (staged solve or parallel-for) on the calling thread — the orchestrator. */
    runMt(): void;
    /** Run the built job as pooled worker `index` (1-based). Exactly once per round. */
    workerMain(index: number): void;
    /** Abandon the running solve — a worker trapped. Breaks the orchestrator's wasm-side spins, which
     * no JS event can reach; the worker's round body calls it before it acks (pool.ts). */
    workerFault(): void;

    /** The serial pose-finalize shim, for the single-thread / no-pool path. On a live pool the staged
     * solve runs finalize as its terminal stage instead (`solveBuild`), so this never runs there. */
    finalize(h: number, invDt: number, enableContinuous: number): void;
};

/** Which outer phase a {@link KernelExports.parBuild} names (kernel/src/solve.rs `Job`). */
export const ParKind = {
    Recycle: 1,
    Convex: 2,
} as const;
export type ParKind = (typeof ParKind)[keyof typeof ParKind];

/**
 * Run one outer phase (`kind`, over `count` records) across the pool, or `serial` on the calling thread
 * when there is no pool or `parBuild` prices the fork above the work. `a`/`b` are the phase's scalars.
 *
 * The caller must have run the phase's `reserve*` already: `parBuild` fixes the columns the blocks read,
 * and nothing may grow memory between the fork and the join (the MT concurrency invariant). The build is what
 * names the job, so it always immediately precedes the run — a `pool.run` with no build ahead of it would
 * drive whatever the last one left behind.
 */
export function runPar(
    kind: ParKind,
    count: number,
    a: number,
    b: number,
    serial: () => void,
): void {
    const pool = workers();
    if (pool === null) {
        serial();
        return;
    }
    const k = kernel();
    if (k.parBuild(kind, count, pool.size + 1, a, b) === 0) {
        serial();
        return;
    }
    runPool(pool, () => k.runMt());
}

/** Options for {@link init}. */
export type InitOptions = {
    /**
     * the advanced escape from the default threading. {@link init} multithreads on its own wherever the
     * host allows it, so this is rarely needed: pass `0` to force the single-thread kernel, or `n` to
     * override the auto count (counting the calling thread, clamped to the ceiling the shadow stack
     * affords). `n` still needs a host that can hold shared memory — a browser without cross-origin
     * isolation runs single-thread whatever you ask. read {@link threads} for what you got. the
     * multithreaded kernel caps linear memory at 1 GiB (the single-thread one is unbounded), which bounds
     * the scene it can hold.
     */
    threads?: number;
};

let instance: Kernel | null = null;
let sharedMemory: WebAssembly.Memory | null = null;
let pool: Pool | null = null;
let resolved = 1;
let booting: Promise<void> | null = null;
/** Latched when a worker trapped inside a pool round — see `runPool`. The kernel never steps again. */
let dead = false;

function decode(base64: string): Uint8Array<ArrayBuffer> {
    const bin = atob(base64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; ++i) bytes[i] = bin.charCodeAt(i);
    return bytes;
}

/** Threads {@link init} runs by default when the host allows it. Flat, not
 * scale-aware: 4 is optimal or within noise at every scene size and never regresses a small one, where
 * more threads would — the wake cost outweighs the split. */
export const AUTO_THREADS = 4;

/** The one-time log a browser gets when it blocks multithreading for want of cross-origin isolation.
 * Loud and host-actionable — the fix is the host's headers, not the caller's code. */
export const COOP_COEP_HINT =
    "physics: running single-threaded. Multithreading needs a cross-origin-isolated page — serve it with COOP and COEP headers (Cross-Origin-Opener-Policy: same-origin, Cross-Origin-Embedder-Policy: require-corp).";

/** The resolved threading plan: `want` threads to attempt (counting the caller; 0 is single-thread), and
 * whether a browser blocked multithreading (the one case {@link announce} logs). */
export type Threading = { want: number; warn: boolean };

/** The host signals {@link resolve} branches on. `shared` — can this host hold a shared
 * `WebAssembly.Memory`: cross-origin isolation in a browser, `SharedArrayBuffer` existing standalone.
 * `browser` picks whether a blocked host earns the COOP/COEP hint (headers only fix a page). */
export type Host = { browser: boolean; shared: boolean };

/**
 * Resolve the threading plan from the caller's request and the host — the pure decision table (the seam a
 * unit test fakes). Standalone always multithreads (`SharedArrayBuffer` is unconditional in bun/node); a
 * browser multithreads when cross-origin isolated and otherwise runs single-thread with a warning; an
 * explicit `0` forces single-thread with no warning; `n` overrides the auto count (the pool clamps it to
 * the link bound). The warning fires only when a browser wanted threads and couldn't have them.
 */
export function resolve(threads: number | undefined, host: Host): Threading {
    if (threads === 0) return { want: 0, warn: false };
    const want = threads ?? AUTO_THREADS;
    if (host.shared) return { want, warn: false };
    return { want: 0, warn: host.browser };
}

/** Emit the one-time COOP/COEP log for a browser that blocked threads. Split from {@link resolve} so the
 * decision table stays pure. */
export function announce(plan: Threading): void {
    if (plan.warn) console.log(COOP_COEP_HINT);
}

/** This host's threading signals. Standalone is detected the way the pool spawns its workers — a real
 * `process.versions.node` (node, bun, deno 2's node compat; pool.ts's spawn branch) — so the resolver and
 * the spawn path cannot disagree. Keying "browser" on `crossOriginIsolated` being a boolean would misroute
 * deno, which exposes that global too, into the COOP/COEP warning. */
function host(): Host {
    const p = (globalThis as { process?: { versions?: { node?: unknown } } }).process;
    if (p?.versions?.node != null) {
        return { browser: false, shared: typeof SharedArrayBuffer !== "undefined" };
    }
    const g = globalThis as { crossOriginIsolated?: boolean };
    return { browser: true, shared: g.crossOriginIsolated === true };
}

async function single(): Promise<void> {
    const result = await WebAssembly.instantiate(decode(KERNEL_WASM_BASE64), {});
    // Don't clobber an instance a lazy `kernel()` created while this was in flight — every consumer must
    // share one instance (one linear memory), so first writer wins.
    instance ??= result.instance.exports as unknown as Kernel;
}

async function multi(want: number): Promise<void> {
    const { KERNEL_SHARED_WASM_BASE64, SHARED_INITIAL_PAGES, SHARED_MAX_PAGES, SHARED_STACK_SIZE } =
        await import("./kernel.shared.wasm");

    // The module declares its own memory floor (data + shadow stack), so `initial` may not go below it;
    // `maximum` is the link-time ceiling and a JS memory may only lower it.
    const memory = new WebAssembly.Memory({
        initial: SHARED_INITIAL_PAGES,
        maximum: SHARED_MAX_PAGES,
        shared: true,
    });
    const module = await WebAssembly.compile(decode(KERNEL_SHARED_WASM_BASE64));
    const exports = (await WebAssembly.instantiate(module, { env: { memory } }))
        .exports as unknown as Kernel & {
        // biome-ignore lint/style/useNamingConvention: LLD's global, exported under its own name.
        __stack_pointer: WebAssembly.Global;
    };
    // A lazy `kernel()` can have run during those awaits; it wins (a live world already holds views into
    // its memory), and the threads are declined rather than swapping the memory out from under it.
    if (instance) return;

    const count = Math.min(want, 1 + maxWorkers(SHARED_STACK_SIZE)) - 1;
    // Instantiating ran the start function to completion on THIS thread, which is the ordering the pool
    // depends on: it CAS-guards data init, and a thread that loses that CAS blocks on
    // `memory.atomic.wait32` — which traps on a browser main thread. Workers may only spawn after it, and
    // they take the main instance's exports as their argument, so they cannot spawn before it exists.
    const spawned =
        count > 0
            ? await createPool(
                  module,
                  memory,
                  count,
                  exports.__stack_pointer.value as number,
                  SHARED_STACK_SIZE,
              )
            : null;

    instance = { ...exports, memory } as Kernel;
    sharedMemory = memory;
    pool = spawned;
    resolved = count + 1;
}

async function boot(threads: number | undefined): Promise<void> {
    // A lazy `kernel()` may already have instantiated the single-thread module (bun/node skip the await),
    // and every consumer must share one instance — one linear memory. Swapping it out from under a live
    // world would strand its columns, so the first instance wins and the threads are declined.
    if (instance) return;
    const plan = resolve(threads, host());
    announce(plan);
    if (plan.want >= 1) {
        try {
            await multi(plan.want);
            return;
        } catch (e) {
            // A host with shared memory can still refuse the workers themselves (a CSP that blocks blob:
            // URLs, a worker-count limit). Fall back to single-thread rather than leaving the caller with
            // no physics at all — but never silently: an invisible perf cliff between deployments is what
            // the default-on design exists to avoid.
            console.log(
                `physics: running single-threaded. The host blocked the worker pool: ${e instanceof Error ? e.message : String(e)}`,
            );
            await pool?.terminate();
            pool = null;
            sharedMemory = null;
            instance = null;
            resolved = 1;
        }
    }
    await single();
}

/**
 * instantiate the physics kernel. idempotent — subsequent calls resolve immediately, and the first call
 * decides threading. await once before the first `step()`. required in a browser, where the main thread
 * refuses to compile a wasm module this size synchronously; outside a browser (bun/node/deno) `step()`
 * also instantiates lazily, so the await is optional there — but a lazy instance is single-threaded, so
 * `init()` before you touch a `World` if you want threads.
 *
 * threading resolves itself: standalone and a cross-origin-isolated browser multithread; a browser without
 * that isolation logs the missing COOP/COEP headers once and runs single-thread. Pass
 * {@link InitOptions.threads} only to force single-thread (`0`) or override the count. the worker pool
 * never holds the process open — a standalone script exits when its own work is done, no `shutdown()`
 * needed.
 *
 * @example
 * await init(); // multithreaded wherever the host allows it
 * console.log(threads()); // what the host actually gave
 * @example
 * await init({ threads: 0 }); // force single-thread
 */
export function init(options?: InitOptions): Promise<void> {
    booting ??= boot(normalizeThreads(options?.threads));
    return booting;
}

/** Normalize the caller's `threads`: absent stays absent (auto-resolve), a non-finite value becomes 0
 * (single-thread), everything else floors to a non-negative integer. */
function normalizeThreads(v: number | undefined): number | undefined {
    if (v === undefined) return undefined;
    if (!Number.isFinite(v)) return 0;
    return Math.max(0, Math.floor(v));
}

/** threads the kernel resolved to — 1 when it is running single-threaded. */
export function threads(): number {
    return resolved;
}

/** Stop the worker pool; the kernel keeps stepping, single-threaded. Optional: the pooled workers are
 * `unref`'d at boot, so a script that inits, steps, and ends exits on its own without this (pool.ts). Call
 * it to release the worker threads deterministically — at a test suite's teardown, say. */
export async function shutdown(): Promise<void> {
    await pool?.terminate();
    pool = null;
    resolved = 1;
}

/**
 * The kernel instance. Lazily instantiates synchronously if `init()` hasn't run — fine in bun/node/
 * deno; a browser main thread throws on a synchronous compile this large, so browser callers must
 * `await init()` first.
 */
export function kernel(): Kernel {
    if (dead) {
        throw new Error(
            "physics kernel is dead: a worker trapped mid-step, so the shared columns hold a partial one",
        );
    }
    if (!instance) {
        if (booting) throw new Error("await init() before stepping");
        const mod = new WebAssembly.Module(decode(KERNEL_WASM_BASE64));
        instance = new WebAssembly.Instance(mod, {}).exports as unknown as Kernel;
    }
    return instance;
}

/**
 * The shared memory's current byte length, or 0 single-threaded — the staleness key for views over the
 * kernel's columns.
 *
 * A `memory.grow` detaches every view over an unshared memory (`length === 0`, the single-thread
 * path's guard). A shared memory never detaches: grow hands back a *new* `SharedArrayBuffer` object
 * aliasing the same backing store, so an old view still reads and writes the correct physical bytes and
 * only misses the new tail. Views over the shared path therefore key staleness on this length changing.
 */
export function sharedBytes(): number {
    return sharedMemory === null ? 0 : sharedMemory.buffer.byteLength;
}

/** The worker pool the solve may run on, or null when the kernel is single-threaded — or when a worker
 * has faulted, which kills the kernel (`runPool`). */
export function workers(): Pool | null {
    return pool?.alive ? pool : null;
}

/**
 * Wake the pool for one round, and kill the kernel if a worker trapped inside it.
 *
 * A trapped worker stops claiming blocks *mid-sweep*, and every phase the pool drives writes state that
 * outlives the step: the staged solve's impulses, and — since the outer phases moved onto the pool — the
 * persistent manifold pool and contact directory. So the survivors are not the casualty; the columns are.
 * Retiring the pool and stepping on single-threaded would run the next step off a half-written manifold
 * store, which is silent corruption. The kernel is poisoned instead: the throw reaches the caller, and
 * every later call throws too.
 *
 * A trap here is a kernel bug (an out-of-bounds column access), not a condition a caller can handle —
 * there is nothing to recover to.
 */
export function runPool(pool: Pool, orchestrate: () => void): void {
    try {
        pool.run(orchestrate);
    } catch (e) {
        dead = true;
        // The dead worker is gone; terminate the survivors to reclaim the threads (they are `unref`'d, so
        // they would not block exit, but they are live and now useless). Not awaited — this path is
        // already unwinding.
        void pool.terminate();
        throw e;
    }
}
