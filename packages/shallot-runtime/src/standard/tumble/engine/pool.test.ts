import { expect, test } from "bun:test";
import {
    KERNEL_SHARED_WASM_BASE64,
    SHARED_INITIAL_PAGES,
    SHARED_MAX_PAGES,
    SHARED_STACK_SIZE,
} from "./kernel.shared.wasm";
import { createPool, maxWorkers } from "./pool";

// The multithreaded bootstrap, end to end: the shared artifact instantiates against a JS-created shared
// memory, and every worker comes up on its own shadow-stack slice with its own TLS block — which LLD's
// start function does NOT give it (only the CAS winner is assigned `__tls_base`; every later instance
// would otherwise run on the main thread's stack with `__tls_base == 0`). Then the park/wake round trip
// the solve is driven by, and the shared memory's no-detach grow behaviour the view discipline rests on.
//
// Driven over its own module + memory rather than through `init()`: the kernel is a process-wide
// singleton and bun runs the test files of one run in one process, so flipping it to the shared artifact
// here would hand it to every other test file too. `init({ threads })` itself is gated by the fixture
// suite on the shared artifact (`bun run test:fixture:mt`).

const SLICE = 1 << 18;
const MAIN_STACK = 1 << 20;

function decode(base64: string): Uint8Array<ArrayBuffer> {
    const bin = atob(base64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; ++i) bytes[i] = bin.charCodeAt(i);
    return bytes;
}

type SharedExports = {
    // biome-ignore lint/style/useNamingConvention: LLD's global, exported under its own name.
    __stack_pointer: WebAssembly.Global;
    scratchPtr(): number;
    reserveBodies(cap: number): number;
};

async function bootShared(): Promise<{
    module: WebAssembly.Module;
    memory: WebAssembly.Memory;
    exports: SharedExports;
}> {
    const memory = new WebAssembly.Memory({
        initial: SHARED_INITIAL_PAGES,
        maximum: SHARED_MAX_PAGES,
        shared: true,
    });
    const module = await WebAssembly.compile(decode(KERNEL_SHARED_WASM_BASE64));
    const inst = await WebAssembly.instantiate(module, { env: { memory } });
    return { module, memory, exports: inst.exports as unknown as SharedExports };
}

// The link-time stack size is the thread ceiling: the main thread keeps the top 1 MiB and each worker
// takes a 256 KiB slice below it, less slice 0 (the null guard). 3 MiB ⇒ 7 workers ⇒ 8 threads.
test("the shadow stack bounds the thread count", () => {
    expect(maxWorkers(SHARED_STACK_SIZE)).toBe(7);
    expect(maxWorkers(1 << 20)).toBe(0); // a stack with no room below the main thread's reserve
});

test("every worker boots on its own stack slice and TLS block", async () => {
    const { module, memory, exports } = await bootShared();
    const stackTop = exports.__stack_pointer.value as number;
    expect(stackTop).toBe(SHARED_STACK_SIZE); // the stack is the lowest region, below the column arena

    const pool = await createPool(module, memory, 3, stackTop, SHARED_STACK_SIZE);
    try {
        expect(pool.size).toBe(3);
        for (const [i, r] of pool.ready.entries()) {
            expect(r.index).toBe(i);
            // A real call into the module from the worker: same module ⇒ same static.
            expect(r.scratchPtr).toBe(exports.scratchPtr());
            // Worker i takes slice i+1 — TLS at its base, stack top at the next boundary. Slice 0 is left
            // to the null guard, so no worker's TLS is at address 0.
            expect(r.tlsBase).toBe((i + 1) * SLICE);
            expect(r.stackPointer).toBe((i + 2) * SLICE);
            expect(r.tlsBase).toBeGreaterThan(0);
            // and none of them is on the main thread's stack.
            expect(r.stackPointer).toBeLessThanOrEqual(stackTop - MAIN_STACK);
        }

        // Park/wake: the workers are blocked in `Atomics.wait` between rounds, and each round must wake
        // all of them, run the orchestrator's own share, and join. Repeated, because a lost wakeup (a
        // worker parking on a sequence number the orchestrator has already moved past) shows up as a hang
        // on the round after the one that raced, not on the round that raced.
        let orchestrated = 0;
        for (let i = 0; i < 256; ++i) {
            pool.run(() => {
                orchestrated += 1;
            });
        }
        expect(orchestrated).toBe(256);
    } finally {
        await pool.terminate();
    }
});

// A worker fault poisons the pool, not just that one thread (tumble.md "A worker fault poisons the
// kernel, it doesn't just terminate survivors"): the round it died in is unrecoverable (every phase
// writes state that outlives the step), so the pool retires itself rather than pretend the survivors
// alone can carry on. No real trap is available in this own-module harness (no solve job is ever built
// here, so `workerMain` just no-ops instead of touching memory) — `createPool`'s `faultWorker` param
// injects a synthetic one on the worker's first round, taking the exact same catch/workerFault/rethrow
// path pool.ts's real trap handler does.
test("a worker fault poisons the pool: run throws, alive flips false, every later run throws too", async () => {
    const { module, memory, exports } = await bootShared();
    const stackTop = exports.__stack_pointer.value as number;

    const pool = await createPool(module, memory, 3, stackTop, SHARED_STACK_SIZE, 1);
    try {
        expect(pool.alive).toBe(true);
        expect(() => pool.run(() => {})).toThrow();
        expect(pool.alive).toBe(false);
        // The dead worker can never ack again — a round built for `size` acks would spin forever if the
        // pool didn't refuse outright.
        expect(() => pool.run(() => {})).toThrow();
    } finally {
        // The survivors (workers 0 and 2) are still parked and responsive; termination must not hang.
        await pool.terminate();
    }
});

// The no-grow-while-workers-active invariant (tumble.md concurrency invariants), asserted at runtime:
// `pool.run` snapshots the shared memory's byte length at wake and compares after the join. Every
// `reserve*` is supposed to run pre-fork; a reserve that slipped inside a round would relocate the
// column arena under the workers, a silent bit-exactness/safety break. The orchestrate callback runs on
// the main thread mid-round (between wake and join), so a `reserveBodies` there is the real violation
// shape — it grows the shared memory while the workers are live. Red-verify: delete the byteLength
// compare in `pool.run` and this stops throwing.
test("a memory grow mid-round trips the no-grow assertion", async () => {
    const { module, memory, exports } = await bootShared();
    const stackTop = exports.__stack_pointer.value as number;

    const pool = await createPool(module, memory, 3, stackTop, SHARED_STACK_SIZE);
    try {
        const bytes = memory.buffer.byteLength;
        expect(() =>
            pool.run(() => {
                exports.reserveBodies(65536); // a real `memory.grow` while the workers are active
            }),
        ).toThrow(/no-grow-while-workers-active/);
        expect(memory.buffer.byteLength).toBeGreaterThan(bytes); // the grow really happened
        // A clean round after the violation still passes — the assertion is per-round, not a latch.
        let ran = false;
        pool.run(() => {
            ran = true;
        });
        expect(ran).toBe(true);
    } finally {
        await pool.terminate();
    }
});

test("a shared memory grow keeps old views aliasing the same store", async () => {
    const { memory, exports } = await bootShared();
    const before = new Float32Array(memory.buffer, exports.scratchPtr(), 4);
    before.set([1, 2, 3, 4]);

    // Reserve a body region far past the current capacity — a real `memory.grow`. Unlike an unshared
    // memory, a shared one hands back a *new* SharedArrayBuffer object over the *same* backing store: old
    // views never detach, they just can't see the new tail. That is what lets a worker hold views across
    // a grow, and it is why the view discipline keys staleness on the byte length (kernel.ts
    // `sharedBytes`) rather than on detachment.
    const bytes = memory.buffer.byteLength;
    exports.reserveBodies(65536);
    expect(memory.buffer.byteLength).toBeGreaterThan(bytes);
    expect(before.length).toBe(4); // still attached

    const after = new Float32Array(memory.buffer, exports.scratchPtr(), 4);
    expect(Array.from(after)).toEqual([1, 2, 3, 4]);
    after[0] = 9;
    expect(before[0]).toBe(9); // the same physical bytes
});
