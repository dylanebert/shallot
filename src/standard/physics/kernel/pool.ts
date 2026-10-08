// LLD's shared-memory start initializes data once, but each instance needs its own shadow stack and
// TLS. Workers partition the link-time stack below the main thread; slice zero guards null pointers.
const SLICE = 1 << 18;
const MAIN_STACK = 1 << 20;
const MAX_THREADS = 8;

export function maxWorkers(stackSize: number): number {
    return Math.max(0, Math.min(Math.floor((stackSize - MAIN_STACK) / SLICE) - 1, MAX_THREADS - 1));
}

function clockPause(): boolean {
    return performance.now() >= 0;
}
export const solverPause = typeof Atomics.pause === "function" ? Atomics.pause : clockPause;

const WORKER_SRC = `
const boot = (d, post) => {
    let clock = new Float64Array(d.memory.buffer);
    const ex = new WebAssembly.Instance(d.module, { env: {
        memory: d.memory, solverPause: typeof Atomics.pause === "function" ? Atomics.pause : (${clockPause.toString()}),
        now(p) { if (clock.buffer !== d.memory.buffer) clock = new Float64Array(d.memory.buffer); clock[p >>> 3] = performance.now(); },
        queryCallback() { throw new Error("physics: worker invoked a user query callback"); },
        materialCallback() { throw new Error("physics: worker invoked a user material callback"); },
        collisionCallback() { throw new Error("physics: worker invoked a user collision callback"); },
        kernelPanic(p, n) { console.error("physics kernel " + new TextDecoder().decode(new Uint8Array(d.memory.buffer, p, n).slice())); }
    } }).exports;
    ex.__stack_pointer.value = d.stackTop;
    ex.__wasm_init_tls(d.tlsBase);
    ex.workerRegister();
    post({ index: d.index, stackPointer: ex.__stack_pointer.value, tlsBase: ex.__tls_base.value });
    try { ex.workerMain(d.testFault ? 1 : 0); }
    catch (e) { ex.workerFault(); throw e; }
};
if (typeof process !== "undefined" && process.versions?.node != null) {
    import("node:worker_threads").then((wt) => boot(wt.workerData, (m) => wt.parentPort.postMessage(m)));
} else {
    self.onmessage = (e) => { self.onmessage = null; boot(e.data, (m) => self.postMessage(m)); };
}
`;

type Boot = {
    module: WebAssembly.Module;
    memory: WebAssembly.Memory;
    index: number;
    stackTop: number;
    tlsBase: number;
    testFault: boolean;
};
export type WorkerReady = { index: number; stackPointer: number; tlsBase: number };
type Spawned = {
    ready: Promise<WorkerReady>;
    unref(): void;
    terminate(): Promise<unknown>;
};
export type Pool = {
    readonly size: number;
    readonly ready: WorkerReady[];
    readonly alive: boolean;
    /** Observe the wasm fault flag after a kernel continuation. */
    checkFault(): void;
    /** Stop the wasm scheduler and release its host workers. */
    terminate(): Promise<void>;
};

async function spawn(boot: Boot): Promise<Spawned> {
    if (typeof process !== "undefined" && process.versions?.node != null) {
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
            URL.revokeObjectURL(url);
            resolve(e.data);
        };
        w.onerror = () => reject(new Error(`worker ${boot.index} failed to boot`));
    });
    w.postMessage(boot);
    return {
        ready,
        unref: () => {},
        terminate: async () => {
            URL.revokeObjectURL(url);
            w.terminate();
        },
    };
}

type SchedulerExports = {
    schedulerStart(): void;
    schedulerStop(): void;
    schedulerFaultPtr(): number;
};
/** The main instance must have completed LLD's data initialization before workers instantiate. */
export async function createPool(
    module: WebAssembly.Module,
    memory: WebAssembly.Memory,
    count: number,
    stackTop: number,
    stackSize: number,
    scheduler: SchedulerExports,
    faultWorker?: number,
): Promise<Pool> {
    scheduler.schedulerStart();
    // The flag lives in the module's static data; this view remains valid when shared memory grows.
    const fault = new Int32Array(memory.buffer, scheduler.schedulerFaultPtr(), 1);
    const base = stackTop - stackSize;
    const spawns = await Promise.allSettled(
        Array.from({ length: count }, (_, i) =>
            spawn({
                module,
                memory,
                index: i,
                stackTop: base + (i + 2) * SLICE,
                tlsBase: base + (i + 1) * SLICE,
                testFault: i === faultWorker,
            }),
        ),
    );
    const workers = spawns.filter((s) => s.status === "fulfilled").map((s) => s.value);
    let stopped = false;
    const terminate = async () => {
        stopped = true;
        scheduler.schedulerStop();
        await Promise.all(workers.map((w) => w.terminate()));
    };
    let ready: WorkerReady[];
    try {
        if (workers.length !== count)
            throw (spawns.find((s) => s.status === "rejected") as PromiseRejectedResult).reason;
        ready = await Promise.all(workers.map((w) => w.ready));
    } catch (e) {
        await terminate();
        throw e;
    }
    for (const worker of workers) worker.unref();
    return {
        size: count,
        ready,
        get alive() {
            return !stopped;
        },
        checkFault() {
            if (Atomics.load(fault, 0) !== 0) {
                stopped = true;
                throw new Error("a physics worker faulted");
            }
        },
        terminate,
    };
}
