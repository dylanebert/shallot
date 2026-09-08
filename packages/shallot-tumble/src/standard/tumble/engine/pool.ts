// The worker pool behind `init({ threads })` (src/kernel.ts). Persistent workers, parked in
// `Atomics.wait` between steps and woken once per solve — the staged solver's own stage transitions are
// wasm-side spins over shared memory, not JS wakes.
//
// Bootstrap. A shared-memory module is instantiated once per thread against ONE `WebAssembly.Memory`,
// which means one shadow stack and one TLS block unless each instance re-points its own. LLD's start
// function CAS-guards data init and only the CAS winner assigns `__tls_base`, so every later instance
// comes up with `__stack_pointer` at the module default (all threads on the main stack) and
// `__tls_base == 0`. Each worker therefore sets both itself, immediately after instantiating:
//
//     ex.__stack_pointer.value = stackTop;   // its own slice
//     ex.__wasm_init_tls(tlsBase);           // global.set __tls_base + memory.init — the LLD-blessed path
//
// Stack slices. The shadow stack (`-zstack-size`, kernel.shared.wasm.ts) is the lowest thing in linear
// memory, below the grow-only column arena, so a region grow never relocates it. It is partitioned into
// `SLICE`-sized slices: the main thread keeps the top `MAIN_STACK`, workers take one slice each below it,
// and slice 0 is left unused — a worker based there would get `__tls_base == 0`, which works but turns
// any null deref into silent TLS corruption. Each worker's TLS block sits at the base of its slice and
// its stack grows down from the top of the same slice (deepest kernel frame ≈ 4–8 KB by static analysis:
// `compute_convex_manifold`'s LocalManifold plus two `[ClipVertex; 64]` buffers, nothing recursive — a
// 256 KiB slice clears it ~30×).

/** One thread's stack + TLS slice. Also the thread ceiling: the link-time stack size divides by it. */
const SLICE = 1 << 18;
/** Shadow stack reserved for the main thread, at the top of the stack region. */
const MAIN_STACK = 1 << 20;
/** Threads the kernel can serve: it reserves one null-lane identity record per thread, and that count
 * is fixed at build (kernel/src/bodies.rs `IDENT_RECORDS`). Asking for more is a hard error there, so
 * the clamp lives here — a future stack-size bump must not silently spawn a worker the kernel can't
 * give a record to. */
const MAX_THREADS = 8;

/** Workers the kernel can run: every stack slice below the main thread's (less the null guard), capped
 * by the kernel's thread ceiling. */
export function maxWorkers(stackSize: number): number {
    const slices = Math.floor((stackSize - MAIN_STACK) / SLICE) - 1;
    return Math.max(0, Math.min(slices, MAX_THREADS - 1));
}

const CTL_SEQ = 0;
const CTL_OP = 1;
const CTL_DONE = 2;
const CTL_FAULT = 3;
const CTL_WORDS = 4;

const OP_EXIT = 0;
const OP_SOLVE = 1;

/** Runs in every worker. One source string, two hosts, zero imports — the `WebAssembly.Module` and the
 * shared `Memory` structure-clone into the worker, so there is no script or wasm asset to resolve. */
const WORKER_SRC = `
const boot = (d, post) => {
    const ex = new WebAssembly.Instance(d.module, { env: { memory: d.memory } }).exports;
    ex.__stack_pointer.value = d.stackTop;
    ex.__wasm_init_tls(d.tlsBase);
    const ctl = new Int32Array(d.ctl);
    post({ index: d.index, stackPointer: ex.__stack_pointer.value, tlsBase: ex.__tls_base.value, scratchPtr: ex.scratchPtr() });
    let seen = 0;
    for (;;) {
        while (Atomics.load(ctl, ${CTL_SEQ}) === seen) Atomics.wait(ctl, ${CTL_SEQ}, seen);
        seen = Atomics.load(ctl, ${CTL_SEQ});
        if (Atomics.load(ctl, ${CTL_OP}) === ${OP_EXIT}) return;
        try {
            // Test-only fault injection (pool.test.ts's poison-path test): a worker booted with
            // \`testFault\` traps synthetically instead of calling into the module, exercising the same
            // catch/fault/rethrow path a real wasm trap takes below. \`testFault\` is false at every
            // production boot site (createPool's \`faultWorker\` is undefined unless a caller opts in).
            if (d.testFault) throw new Error("test fault injection");
            // The round: run the staged solve the orchestrator built, as this worker (index 0 is the
            // orchestrator, so the pool's worker i is stage-worker i+1). Exactly once per round —
            // stages::run's join contract.
            ex.workerMain(d.index + 1);
        } catch (e) {
            // A trap in the solve. Two joins have to be released, in this order: the orchestrator is
            // spinning INSIDE wasm (a stage barrier or the exit join) for a worker that will now never
            // claim a block or ack the sentinel, and only the kernel's own fault flag can break that
            // spin — a JS error event can't reach a thread that never yields. Then the JS ack, which the
            // orchestrator only reaches after its wasm spin returns.
            ex.workerFault();
            Atomics.store(ctl, ${CTL_FAULT}, 1);
            Atomics.add(ctl, ${CTL_DONE}, 1);
            throw e;
        }
        Atomics.add(ctl, ${CTL_DONE}, 1);
    }
};
if (typeof process !== "undefined" && process.versions != null && process.versions.node != null) {
    import("node:worker_threads").then((wt) => boot(wt.workerData, (m) => wt.parentPort.postMessage(m)));
} else {
    self.onmessage = (e) => { self.onmessage = null; boot(e.data, (m) => self.postMessage(m)); };
}
`;

type Spawned = {
    /** Resolves once the worker has instantiated and bootstrapped; rejects if it failed to. */
    ready: Promise<WorkerReady>;
    /** Drop the worker's hold on the host event loop (node/bun), so a script that never calls
     * {@link Pool.terminate} still exits on its own. A no-op in the browser, where a Worker never pins
     * page teardown. */
    unref(): void;
    terminate(): Promise<unknown>;
};

/** Boot data for one worker. `module` + `memory` structure-clone; `ctl` is the park/wake control block. */
type Boot = {
    module: WebAssembly.Module;
    memory: WebAssembly.Memory;
    ctl: SharedArrayBuffer;
    index: number;
    stackTop: number;
    tlsBase: number;
    /** Test-only: this worker traps synthetically on its first round instead of calling `workerMain`
     * (`createPool`'s `faultWorker`). Always false in production. */
    testFault: boolean;
};

/** What a worker reports once it has instantiated and bootstrapped — the two globals it set plus a real
 * call into the module, so a broken instantiation surfaces at `init()` instead of mid-solve. */
export type WorkerReady = {
    index: number;
    stackPointer: number;
    tlsBase: number;
    scratchPtr: number;
};

export type Pool = {
    /** Worker count. The solve runs on `size + 1` threads — the caller is the orchestrator. */
    readonly size: number;
    /** What each worker reported at boot, in index order. */
    readonly ready: WorkerReady[];
    /** False once a worker has faulted: that thread is gone, so a later solve built for `size + 1`
     * threads would wait forever for its ack. A dead pool is never run again (`workers()`, kernel.ts) —
     * the kernel keeps stepping single-threaded. */
    readonly alive: boolean;
    /**
     * Wake every worker, run `orchestrate` on the calling thread, then join. The orchestrator's work
     * happens *between* the wake and the join by construction: the workers spin on the sync bits it
     * publishes, so a pool that woke them and then blocked on the join would deadlock.
     */
    run(orchestrate: () => void): void;
    /** Stop the workers and await their exit. The workers are `unref`'d at boot so the process can exit
     * without this; call it to tear the pool down deterministically. */
    terminate(): Promise<void>;
};

async function spawn(boot: Boot): Promise<Spawned> {
    const node =
        typeof process !== "undefined" && process.versions != null && process.versions.node != null;
    if (node) {
        // Kept off the static import graph: a browser bundler must never try to resolve it, and the
        // browser branch below never reaches it.
        const spec = "node:worker_threads";
        const { Worker } = (await import(
            /* @vite-ignore */ spec
        )) as typeof import("node:worker_threads");
        const w = new Worker(WORKER_SRC, { eval: true, workerData: boot });
        const ready = new Promise<WorkerReady>((resolve, reject) => {
            w.on("message", resolve);
            w.on("error", reject);
            w.on("exit", (code) => reject(new Error(`worker exited during boot (code ${code})`)));
        });
        return { ready, unref: () => w.unref(), terminate: () => w.terminate() };
    }
    const url = URL.createObjectURL(new Blob([WORKER_SRC], { type: "text/javascript" }));
    const w = new Worker(url);
    const ready = new Promise<WorkerReady>((resolve, reject) => {
        w.onmessage = (e: MessageEvent<WorkerReady>) => {
            URL.revokeObjectURL(url); // the worker is up; revoking before that has raced in WebKit
            resolve(e.data);
        };
        w.onerror = () => reject(new Error(`worker ${boot.index} failed to boot`));
    });
    w.postMessage(boot);
    return {
        ready,
        unref: () => {}, // a browser Worker never pins page teardown — nothing to release.
        terminate: async () => {
            URL.revokeObjectURL(url);
            w.terminate();
        },
    };
}

/**
 * Spawn `count` workers against the already-instantiated module + shared memory, and resolve once every
 * one of them has instantiated and bootstrapped its stack + TLS.
 *
 * `stackTop` is the main instance's `__stack_pointer` after instantiation — i.e. the top of the shadow
 * stack region, which the slices partition downward from.
 *
 * The main thread MUST have instantiated (and its start function run to completion) before this is
 * called: a thread that loses the start function's data-init CAS executes `memory.atomic.wait32`, which
 * traps on a browser main thread. Instantiation order is the only thing preventing that, and no bun/node
 * test can catch its regression — so it is structural here (the main instance is an argument).
 */
export async function createPool(
    module: WebAssembly.Module,
    memory: WebAssembly.Memory,
    count: number,
    stackTop: number,
    stackSize: number,
    /** Test-only: the (0-based) worker index that traps synthetically on its first round, exercising
     * the fault path without a real wasm trap (pool.test.ts's poison-path test). */
    faultWorker?: number,
): Promise<Pool> {
    const base = stackTop - stackSize;
    const ctl = new SharedArrayBuffer(CTL_WORDS * 4);
    const ctlView = new Int32Array(ctl);

    // `allSettled`, not `all`: one worker the host refuses to spawn must not strand the ones that came
    // up — they are never `unref`'d, so a leaked worker hangs the process on exit.
    const spawns = await Promise.allSettled(
        Array.from({ length: count }, (_, i) => {
            const slice = i + 1; // slice 0 is the null guard
            return spawn({
                module,
                memory,
                ctl,
                index: i,
                stackTop: base + (slice + 1) * SLICE,
                tlsBase: base + slice * SLICE,
                testFault: i === faultWorker,
            });
        }),
    );
    const workers = spawns.filter((s) => s.status === "fulfilled").map((s) => s.value);
    const kill = async (e: unknown) => {
        await Promise.all(workers.map((w) => w.terminate()));
        throw e;
    };
    if (workers.length < count) {
        await kill((spawns.find((s) => s.status === "rejected") as PromiseRejectedResult).reason);
    }
    let ready: WorkerReady[];
    try {
        ready = await Promise.all(workers.map((w) => w.ready));
    } catch (e) {
        await kill(e);
        throw e; // unreachable — `kill` rethrows; keeps `ready` definitely assigned
    }

    // Parked workers pin the host event loop, so a script that inits and steps but never calls
    // `terminate` would hang at exit. `unref` drops that hold — the main thread's own busy-wait keeps the
    // process alive across a solve round, so this can never let it exit mid-step (measured to exit
    // cleanly on bun and node; the exit-test gates it). The browser branch is a no-op.
    for (const w of workers) w.unref();

    let seq = 0;
    let alive = true;
    return {
        size: count,
        ready,
        get alive(): boolean {
            return alive;
        },
        run(orchestrate: () => void): void {
            // A dead pool is missing a worker for good — a round built for `count` acks would spin
            // forever waiting for one that can never come (pool.test.ts's poison-path test pins this).
            // Every current call site already gates on `alive` (kernel.ts `workers()`), so this guard is
            // a second line of defense, not the only one — but the primitive should refuse on its own.
            if (!alive) {
                throw new Error("pool.run called on a dead pool — a worker already faulted");
            }
            // No `memory.grow`/region relocation while workers are active (tumble.md concurrency
            // invariants): every `reserve*` runs pre-fork on the main thread, so parallel phases touch
            // pre-reserved columns only. Held by construction — snapshot the byte length at wake and
            // compare after the join to catch a future violator (a reserve that slipped inside a round)
            // as a loud throw instead of a silent bit-exactness/safety break.
            const bytesAtWake = memory.buffer.byteLength;
            Atomics.store(ctlView, CTL_DONE, 0);
            Atomics.store(ctlView, CTL_OP, OP_SOLVE);
            seq += 1;
            Atomics.store(ctlView, CTL_SEQ, seq);
            Atomics.notify(ctlView, CTL_SEQ);

            orchestrate();

            // Spin, never `Atomics.wait`: a browser main thread is not allowed to block on it, and the
            // orchestrator arrives here with the workers already nearly done.
            while (Atomics.load(ctlView, CTL_DONE) < count) {}
            if (memory.buffer.byteLength !== bytesAtWake) {
                throw new Error(
                    "shared memory grew while workers were active — violates the no-grow-while-workers-active invariant (every reserve must run pre-fork)",
                );
            }
            if (Atomics.load(ctlView, CTL_FAULT) !== 0) {
                // The faulted worker is gone; the survivors left their spins on the kernel's fault flag.
                // Retire the pool — a later solve built for `count + 1` threads would join on an ack
                // that can never come.
                alive = false;
                throw new Error("a physics worker faulted");
            }
        },
        async terminate(): Promise<void> {
            Atomics.store(ctlView, CTL_OP, OP_EXIT);
            seq += 1;
            Atomics.store(ctlView, CTL_SEQ, seq);
            Atomics.notify(ctlView, CTL_SEQ);
            await Promise.all(workers.map((w) => w.terminate()));
        },
    };
}
