import { Compute } from "../../engine";
import { type DecodedGltf, decode } from "./assets";
import { Scheduler } from "./scheduler";
import { pickTargets, type Targets } from "./target";
import type { DecodeReply, DecodeRequest } from "./worker";

// The decode pool — the process-scoped worker service the asset cache decodes through. A pool of N module workers (decode.worker.ts) fed by a priority queue (scheduler.ts):
// a request submits, the scheduler dispatches it to an idle worker, the worker runs the SAME deviceless
// `decode` and transfers the {@link DecodedGltf} back zero-copy. So a multi-asset first load decodes in
// parallel, off the main thread — no hitch. Module-level + lazy like the codec singletons (loadDraco /
// loadBasis): the workers spawn once and survive every State rebuild, never torn down in a lifecycle hook.
// One decode implementation, two call sites — where Worker is absent (bun-webgpu / tests) the pool runs
// `decode` inline, never a second decode.

// leave a core for the main thread; cap low — each worker instantiates the Draco + Basis wasm, so more workers
// cost memory (three.js' DRACOLoader defaults a workerLimit of 4). A tuning constant.
const MAX_WORKERS = 4;

function poolSize(): number {
    const cores = (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 4;
    return Math.min(MAX_WORKERS, Math.max(1, cores - 1));
}

// per slot, the settle of its one in-flight decode. The scheduler runs one task per slot at a time, so a
// worker's next message IS that task's reply — reply correlation by worker identity, no request id (the
// three.js WorkerPool per-slot resolver).
interface Pending {
    resolve: (d: DecodedGltf) => void;
    reject: (e: unknown) => void;
}

let _workers: Worker[] = [];
let _pending: (Pending | null)[] = [];
let _scheduler: Scheduler<DecodeRequest, DecodedGltf> | null = null;

function ensurePool(): Scheduler<DecodeRequest, DecodedGltf> {
    if (_scheduler) return _scheduler;
    const n = poolSize();
    _pending = new Array(n).fill(null);
    _workers = Array.from({ length: n }, (_, slot) => spawn(slot));
    _scheduler = new Scheduler<DecodeRequest, DecodedGltf>({ slots: n, run: dispatch });
    return _scheduler;
}

function spawn(slot: number): Worker {
    const w = new Worker(new URL("./decode.worker.ts", import.meta.url), { type: "module" });
    w.onmessage = (e: MessageEvent<DecodeReply>) => {
        const reply = e.data;
        if (reply.ok) settle(slot, (p) => p.resolve(reply.decoded));
        else fail(slot, reply.error);
    };
    w.onerror = (e) => fail(slot, `[gltf] decode worker error: ${e.message}`);
    w.onmessageerror = () => fail(slot, "[gltf] decode worker message deserialization failed");
    return w;
}

function settle(slot: number, f: (p: Pending) => void): void {
    const p = _pending[slot];
    _pending[slot] = null;
    if (p) f(p);
}

function fail(slot: number, msg: string): void {
    settle(slot, (p) => p.reject(new Error(msg)));
}

function dispatch(slot: number, req: DecodeRequest): Promise<DecodedGltf> {
    return new Promise<DecodedGltf>((resolve, reject) => {
        _pending[slot] = { resolve, reject };
        _workers[slot].postMessage(req);
    });
}

/**
 * drop the pool's queued decodes: the cancel-on-dispose seam (`GltfPlugin.dispose` on a scene switch / State
 * teardown). Queued requests reject (their awaiters guard on `state.disposed`, so a drop is silent teardown,
 * not a load failure); an in-flight decode finishes on its worker and lands in the asset cache, where a late
 * result is safe (idempotent, no LRU eviction race). A no-op before any pool spins up (the bun-webgpu inline
 * path never builds one).
 */
export function abortDecodes(): void {
    _scheduler?.abort(new Error("[gltf] decode aborted (State disposed)"));
}

/**
 * decode a glTF through the worker pool, resolving the transferred {@link DecodedGltf}. The url is absolutized
 * against the document (a module worker's base url is the bundle chunk, not the page, so a relative .bin /
 * image / KTX2 fetch would resolve wrong); `targets` are the device's per-slot compressed formats, resolved
 * main-thread-side (the deviceless contract, so undefined is fine for an untextured / PNG asset). Where Worker
 * is unavailable (bun-webgpu / tests) it decodes inline, the same `decode`, no second implementation.
 * `priority` orders the queue (higher dispatches first). The asset cache ({@link ensureDecoded}) routes through
 * this; it caches on the original src, so this returns a `DecodedGltf` whose `url` is the absolutized form,
 * the caller normalizes it back.
 */
export function poolDecode(
    url: string,
    opts: { clip?: number; targets?: Targets; priority?: number; live?: boolean } = {},
): Promise<DecodedGltf> {
    const clip = opts.clip ?? 0;
    const live = opts.live ?? false;
    // inline where there's no browser worker context (bun-webgpu / tests): no `location` to absolutize against
    // and no module-worker bundling. bun defines `Worker` but not `location`, so gate on the document url.
    if (typeof location === "undefined" || typeof Worker === "undefined")
        return decode(url, { clip, targets: opts.targets, live });
    const req: DecodeRequest = {
        url: new URL(url, location.href).href,
        clip,
        live,
        targets: opts.targets,
    };
    return ensurePool().submit(req, opts.priority ?? 0);
}

/**
 * decode a glTF off the main thread: submit it to the worker pool and resolve the {@link DecodedGltf} a worker
 * transfers back zero-copy. The transcode target is resolved from the device (the deviceless contract), so this
 * needs a built State. Feed the result to {@link register} exactly like {@link decode}'s output; for the cached
 * load path use {@link loadGltf} / {@link ensureDecoded} instead, which route through the same pool.
 *
 * @example
 * const decoded = await decodeInWorker("Fox.glb", { clip: 2 });
 * const { eids } = await register(state, decoded);
 */
export function decodeInWorker(url: string, opts: { clip?: number } = {}): Promise<DecodedGltf> {
    const device = Compute.device;
    if (!device) throw new Error("[gltf] no GPU device — call decodeInWorker after build()");
    return poolDecode(url, { clip: opts.clip ?? 0, targets: pickTargets(device) });
}
