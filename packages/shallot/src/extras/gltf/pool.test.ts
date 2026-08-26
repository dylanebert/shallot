import { afterEach, describe, expect, test } from "bun:test";
import { _resetPool, poolDecode } from "./pool";
import type { DecodeReply } from "./worker";

// pool.ts's transport wiring (spawn/settle/fail) — the seam scheduler.test.ts leaves to the real Chrome gym
// run. Here we fake the Worker to exercise the pool's own error handling: a worker that errors outside a
// pending request. No pool.test.ts existed; this is the new arm, placed beside the module it tests.

// a fake Worker the pool's `spawn` wires onto. postMessage replies on the next microtask with a failure (the
// test only needs to observe that the dispatch settles, not that decode succeeds). `alive` models a real
// worker's lifecycle: once dead, postMessage goes nowhere — the defect's hang.
class FakeWorker {
    onmessage: ((e: MessageEvent<DecodeReply>) => void) | null = null;
    onerror: ((e: ErrorEvent) => void) | null = null;
    onmessageerror: (() => void) | null = null;
    terminated = false;
    alive = true;

    postMessage(_msg: unknown): void {
        if (!this.alive) return;
        queueMicrotask(() => {
            this.onmessage?.({ data: { ok: false, error: "fake" } } as MessageEvent<DecodeReply>);
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
});
