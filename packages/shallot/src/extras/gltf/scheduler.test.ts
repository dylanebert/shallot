import { describe, expect, test } from "bun:test";
import { Scheduler } from "./scheduler";

// the pure decode scheduler — priority order, slot saturation, backpressure, result/reject propagation. No
// Worker (pool.ts wires the real transport); `run` is a controllable deferred so dispatch is observed
// step-by-step. The real worker round-trip is the gym `render` `gltf-worker` mode (bun-webgpu has no Worker).

// a `run` that records each dispatch and hands back a manual resolve/reject, so a test drives completion order
function runner() {
    const calls: {
        slot: number;
        task: string;
        resolve: () => void;
        reject: (e: unknown) => void;
    }[] = [];
    const run = (slot: number, task: string) =>
        new Promise<string>((res, rej) => {
            calls.push({ slot, task, resolve: () => res(task), reject: rej });
        });
    return { calls, run };
}

// flush the microtask queue so the `.then(...).finally(drain)` chain after a resolve settles — bounded depth,
// so a fixed number of turns drains it (awaiting the deterministic condition, not sleeping on the clock)
async function settle(): Promise<void> {
    for (let i = 0; i < 10; i++) await Promise.resolve();
}

describe("Scheduler", () => {
    test("dispatches highest priority first, FIFO within a priority", async () => {
        const { calls, run } = runner();
        const s = new Scheduler<string, string>({ slots: 1, run });
        // a takes the only slot immediately (queue was empty); b/c/d wait
        s.submit("a", 0);
        s.submit("b", 10);
        s.submit("c", 5);
        s.submit("d", 10); // same priority as b but later — FIFO breaks the tie
        expect(calls.map((c) => c.task)).toEqual(["a"]);

        calls[0].resolve();
        await settle();
        expect(calls.map((c) => c.task)).toEqual(["a", "b"]); // 10 beats 5

        calls[1].resolve();
        await settle();
        expect(calls.map((c) => c.task)).toEqual(["a", "b", "d"]); // d (10) before c (5)

        calls[2].resolve();
        await settle();
        expect(calls.map((c) => c.task)).toEqual(["a", "b", "d", "c"]);
    });

    test("never runs more than the slot count concurrently (saturation + drain)", async () => {
        const { calls, run } = runner();
        const s = new Scheduler<string, string>({ slots: 2, run });
        for (const t of ["a", "b", "c", "d"]) s.submit(t, 0);
        await settle();
        // only 2 in flight; c/d are queued behind the saturated pool
        expect(calls.map((c) => c.task)).toEqual(["a", "b"]);
        // distinct slots — the two concurrent runs don't share one
        expect(new Set(calls.map((c) => c.slot)).size).toBe(2);

        calls[0].resolve();
        await settle();
        expect(calls.map((c) => c.task)).toEqual(["a", "b", "c"]); // freed slot → next in FIFO

        calls[1].resolve();
        await settle();
        expect(calls.map((c) => c.task)).toEqual(["a", "b", "c", "d"]);
    });

    test("submit resolves with the run result", async () => {
        const s = new Scheduler<string, string>({ slots: 1, run: async (_slot, t) => `ok:${t}` });
        expect(await s.submit("x")).toBe("ok:x");
    });

    test("abort drops queued tasks, leaves in-flight running, stays reusable", async () => {
        const { calls, run } = runner();
        const s = new Scheduler<string, string>({ slots: 1, run });
        const pa = s.submit("a", 0); // takes the only slot — in flight
        const pb = s.submit("b", 0); // queued
        const pc = s.submit("c", 0); // queued
        expect(calls.map((c) => c.task)).toEqual(["a"]);

        // the cancel-on-dispose drop: queued b/c reject, the in-flight a is untouched
        s.abort(new Error("aborted"));
        await expect(pb).rejects.toThrow("aborted");
        await expect(pc).rejects.toThrow("aborted");

        // a was dispatched before the abort — it completes normally and lands its result (the "in-flight decode
        // finishes into the cache" guarantee at the scheduler layer)
        calls[0].resolve();
        await settle();
        expect(await pa).toBe("a");

        // the freed slot accepts new work after the abort — submitting again is unaffected
        const pd = s.submit("d", 0);
        await settle();
        expect(calls.map((c) => c.task)).toEqual(["a", "d"]);
        calls[1].resolve();
        expect(await pd).toBe("d");
    });

    test("a rejected task propagates and frees its slot (no deadlock)", async () => {
        let runs = 0;
        const s = new Scheduler<string, string>({
            slots: 1,
            run: async (_slot, t) => {
                runs++;
                if (t === "bad") throw new Error("boom");
                return t;
            },
        });
        await expect(s.submit("bad")).rejects.toThrow("boom");
        // the slot freed despite the throw, so the next task still runs
        expect(await s.submit("good")).toBe("good");
        expect(runs).toBe(2);
    });
});
