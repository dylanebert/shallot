import { afterEach, describe, expect, test } from "bun:test";
import type { DecodedGltf } from "./assets";
import { _resetPool, poolDecode } from "./pool";
import type { DecodeReply } from "./worker";

// pool.ts's transport wiring (spawn/settle/fail) — the seam scheduler.test.ts leaves to the real Chrome gym
// run. Here we fake the Worker to exercise the pool's own error handling: a worker that errors outside a
// pending request. No pool.test.ts existed; this is the new arm, placed beside the module it tests.

// a fake Worker the pool's `spawn` wires onto. `succeed` makes postMessage reply with ok:true; otherwise
// ok:false (the default — the test only needs to observe that the dispatch settles). `dieOnMessage` makes
// postMessage fire onerror instead of replying (a worker that dies mid-dispatch). `alive` models a real
// worker's lifecycle: once dead, postMessage goes nowhere — the defect's hang.
class FakeWorker {
    onmessage: ((e: MessageEvent<DecodeReply>) => void) | null = null;
    onerror: ((e: ErrorEvent) => void) | null = null;
    onmessageerror: (() => void) | null = null;
    terminated = false;
    alive = true;
    succeed = false;
    dieOnMessage = false;

    postMessage(_msg: unknown): void {
        if (!this.alive) return;
        if (this.dieOnMessage) {
            this.alive = false;
            queueMicrotask(() => {
                if (!this.terminated)
                    this.onerror?.(new ErrorEvent("error", { message: "died on dispatch" }));
            });
            return;
        }
        queueMicrotask(() => {
            if (this.succeed)
                this.onmessage?.({
                    data: { ok: true, decoded: {} as DecodedGltf },
                } as MessageEvent<DecodeReply>);
            else
                this.onmessage?.({
                    data: { ok: false, error: "fake" },
                } as MessageEvent<DecodeReply>);
        });
    }

    terminate(): void {
        this.terminated = true;
        this.alive = false;
    }
}

// flush the microtask queue (bounded — the scheduler's .then/.finally chain is a fixed depth)
async function flush(): Promise<void> {
    for (let i = 0; i < 20; i++) await Promise.resolve();
}

describe("decode pool", () => {
    const realWorker = globalThis.Worker;
    const realLocation = globalThis.location;
    const realHwConcurrency = navigator.hardwareConcurrency;
    let workers: FakeWorker[];

    afterEach(() => {
        globalThis.Worker = realWorker;
        globalThis.location = realLocation;
        Object.defineProperty(navigator, "hardwareConcurrency", {
            value: realHwConcurrency,
            configurable: true,
            writable: true,
        });
        _resetPool();
    });

    function installFakes(): void {
        workers = [];
        globalThis.Worker = function (this: unknown, _url: URL, _opts?: unknown) {
            const w = new FakeWorker();
            workers.push(w);
            return w;
        } as unknown as typeof Worker;
        globalThis.location = { href: "http://localhost/" } as unknown as Location;
        Object.defineProperty(navigator, "hardwareConcurrency", {
            value: 2,
            configurable: true,
            writable: true,
        });
    }

    test("a worker that errors outside a pending request does not leave its slot hung", async () => {
        installFakes();

        // first dispatch creates the pool (1 slot) and goes to worker 0; it replies (rejects) — catch and move on
        await poolDecode("http://localhost/box.glb").catch(() => {});
        await flush();

        expect(workers).toHaveLength(1);
        expect(workers[0].terminated).toBe(false);

        // the worker dies — postMessage will go nowhere, like a real dead worker
        workers[0].alive = false;
        // the worker errors with no pending request — the defect: error is dropped, slot stays broken
        workers[0].onerror?.(new ErrorEvent("error", { message: "died at load" }));

        // second dispatch: pre-fix it hangs (the dead worker never replies); post-fix the respawned worker
        // handles it and the promise settles. Detect a hang without a sleep: flush microtasks (bounded depth)
        // and check the promise settled — a never-settling promise is the defect, made observable.
        let settled = false;
        poolDecode("http://localhost/box.glb").then(
            () => {
                settled = true;
            },
            () => {
                settled = true;
            },
        );
        await flush();
        expect(settled).toBe(true);
    });

    test("a worker that fires onmessageerror outside a pending request does not leave its slot hung", async () => {
        installFakes();

        // first dispatch creates the pool and settles (worker replies with failure)
        await poolDecode("http://localhost/box.glb").catch(() => {});
        await flush();

        expect(workers).toHaveLength(1);

        // the worker dies — postMessage will go nowhere
        workers[0].alive = false;
        // onmessageerror with no pending request — pre-fix `fail`→`settle` drops it silently (same defect as
        // onerror); post-fix routes through `workerError` and respawns the slot.
        workers[0].onmessageerror?.();

        // second dispatch must settle — the respawned worker handles it
        let settled = false;
        poolDecode("http://localhost/box.glb").then(
            () => {
                settled = true;
            },
            () => {
                settled = true;
            },
        );
        await flush();
        expect(settled).toBe(true);
    });

    test("a worker that fails on every spawn has bounded respawns and rejects after budget", async () => {
        workers = [];
        let spawnCount = 0;
        globalThis.Worker = function (this: unknown, _url: URL, _opts?: unknown) {
            const w = new FakeWorker();
            w.alive = false; // dies at load — postMessage goes nowhere
            workers.push(w);
            spawnCount++;
            // fire onerror on the next microtask (dies at load)
            queueMicrotask(() => {
                if (!w.terminated)
                    w.onerror?.(new ErrorEvent("error", { message: "died at load" }));
            });
            return w;
        } as unknown as typeof Worker;
        globalThis.location = { href: "http://localhost/" } as unknown as Location;
        Object.defineProperty(navigator, "hardwareConcurrency", {
            value: 2,
            configurable: true,
            writable: true,
        });

        // first dispatch: creates pool, dispatches to worker 0. Worker 0 fires onerror (microtask); at that
        // point _pending[0] exists → fail + respawn. Subsequent respawned workers fire onerror with no pending
        // → respawn until budget exhausted → mark dead.
        await poolDecode("http://localhost/box.glb").catch(() => {});
        await flush();

        // 1 initial spawn + 3 respawns (MAX_RESPAWN) = 4 total — the respawn is bounded
        expect(spawnCount).toBe(4);

        // dispatch after budget exhausted: rejects (does not hang)
        let settled = false;
        let rejection: unknown = null;
        poolDecode("http://localhost/box.glb").then(
            () => {
                settled = true;
            },
            (e) => {
                settled = true;
                rejection = e;
            },
        );
        await flush();
        expect(settled).toBe(true);
        expect(rejection).toBeInstanceOf(Error);
        expect((rejection as Error).message).toContain("dead");
    });

    test("a dead slot does not poison the pool — healthy slots handle all decodes", async () => {
        workers = [];
        let spawnIdx = 0;
        globalThis.Worker = function (this: unknown, _url: URL, _opts?: unknown) {
            const w = new FakeWorker();
            const idx = spawnIdx++;
            workers.push(w);
            if (idx === 0) {
                // slot 0 — healthy, replies with success
                w.succeed = true;
            } else {
                // slot 1 and its respawns — die at load
                w.alive = false;
                queueMicrotask(() => {
                    if (!w.terminated)
                        w.onerror?.(new ErrorEvent("error", { message: "died at load" }));
                });
            }
            return w;
        } as unknown as typeof Worker;
        globalThis.location = { href: "http://localhost/" } as unknown as Location;
        Object.defineProperty(navigator, "hardwareConcurrency", {
            value: 3, // poolSize = min(4, max(1, 3-1)) = 2
            configurable: true,
            writable: true,
        });

        // trigger pool creation — slot 1's worker dies and respawns until dead; the first decode goes to
        // slot 1 (LIFO top) and rejects, so catch and flush to let the respawn cycle complete
        await poolDecode("http://localhost/box.glb").catch(() => {});
        await flush();

        // slot 1 is now dead; N successive decodes should all succeed via slot 0
        for (let i = 0; i < 5; i++) {
            const result = await poolDecode("http://localhost/box.glb");
            expect(result).toBeDefined();
        }
    });

    test("all slots dead rejects queued waiters with a named error (not a hang)", async () => {
        workers = [];
        globalThis.Worker = function (this: unknown, _url: URL, _opts?: unknown) {
            const w = new FakeWorker();
            w.alive = false; // dies at load
            workers.push(w);
            queueMicrotask(() => {
                if (!w.terminated)
                    w.onerror?.(new ErrorEvent("error", { message: "died at load" }));
            });
            return w;
        } as unknown as typeof Worker;
        globalThis.location = { href: "http://localhost/" } as unknown as Location;
        Object.defineProperty(navigator, "hardwareConcurrency", {
            value: 2, // poolSize = 1
            configurable: true,
            writable: true,
        });

        // trigger pool creation — the single slot dies and exhausts respawns
        await poolDecode("http://localhost/box.glb").catch(() => {});
        await flush();

        // all slots dead — a new decode must reject with the named error, not hang
        let settled = false;
        let rejection: unknown = null;
        poolDecode("http://localhost/box.glb").then(
            () => {
                settled = true;
            },
            (e) => {
                settled = true;
                rejection = e;
            },
        );
        await flush();
        // settles within the flush window — distinguishing a named terminal error from a timeout
        expect(settled).toBe(true);
        expect(rejection).toBeInstanceOf(Error);
        expect((rejection as Error).message).toContain("no healthy decode slot available");
    });

    test("respawn count resets on a successful decode — budget is non-cumulative", async () => {
        workers = [];
        let spawnCount = 0;
        let dispatchCount = 0;
        globalThis.Worker = function (this: unknown, _url: URL, _opts?: unknown) {
            const w = new FakeWorker();
            spawnCount++;
            workers.push(w);
            // override postMessage: 2nd dispatch succeeds, all others die on dispatch
            w.postMessage = (_msg: unknown): void => {
                if (!w.alive) return;
                dispatchCount++;
                if (dispatchCount === 2) {
                    queueMicrotask(() => {
                        w.onmessage?.({
                            data: { ok: true, decoded: {} as DecodedGltf },
                        } as MessageEvent<DecodeReply>);
                    });
                } else {
                    w.alive = false;
                    queueMicrotask(() => {
                        if (!w.terminated)
                            w.onerror?.(new ErrorEvent("error", { message: "died on dispatch" }));
                    });
                }
            };
            return w;
        } as unknown as typeof Worker;
        globalThis.location = { href: "http://localhost/" } as unknown as Location;
        Object.defineProperty(navigator, "hardwareConcurrency", {
            value: 2, // poolSize = 1
            configurable: true,
            writable: true,
        });

        // dispatch 1: worker dies → fail + respawn (count 1). dispatch 2: respawned worker succeeds
        // (count resets to 0). dispatches 3-5: die → respawn (count 1, 2, 3). dispatch 6: die → dead.
        // total spawns: 1 initial + 4 respawns = 5 (without reset: 1 initial + 3 respawns = 4)
        await poolDecode("http://localhost/box.glb").catch(() => {}); // dispatch 1 — dies
        await flush();
        await poolDecode("http://localhost/box.glb").catch(() => {}); // dispatch 2 — succeeds
        await flush();
        await poolDecode("http://localhost/box.glb").catch(() => {}); // dispatch 3 — dies
        await flush();
        await poolDecode("http://localhost/box.glb").catch(() => {}); // dispatch 4 — dies
        await flush();
        await poolDecode("http://localhost/box.glb").catch(() => {}); // dispatch 5 — dies
        await flush();
        // dispatch 6 — dies, budget exhausted, slot dead
        let settled = false;
        let rejection: unknown = null;
        poolDecode("http://localhost/box.glb").then(
            () => {
                settled = true;
            },
            (e) => {
                settled = true;
                rejection = e;
            },
        );
        await flush();
        expect(settled).toBe(true);
        expect(rejection).toBeInstanceOf(Error);
        // 5 spawns with reset (1 initial + 4 respawns); without reset it would be 4
        expect(spawnCount).toBe(5);
    });
});
