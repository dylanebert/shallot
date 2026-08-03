import { describe, expect, test } from "bun:test";
import { createGenerationWarm, withTemporaryOwner } from "./accel";

function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
    let resolve = (_value: T) => {};
    const promise = new Promise<T>((done) => {
        resolve = done;
    });
    return { promise, resolve };
}

interface TestState {
    disposed: boolean;
    cleanup: (() => void)[];
}

interface TestOwner {
    name: string;
}

test("latest warm wins out of order and stale State teardown preserves it", async () => {
    const drafts = new Map<string, ReturnType<typeof deferred<TestOwner>>>();
    const activated: string[] = [];
    const cleaned: string[] = [];
    const active: { current: TestOwner | null } = { current: null };
    const warm = createGenerationWarm<TestState, string, TestOwner>({
        prepare(_state, name) {
            const draft = deferred<TestOwner>();
            drafts.set(name, draft);
            return draft.promise;
        },
        current: () => active.current,
        activate(owner) {
            active.current = owner;
            activated.push(owner.name);
        },
        clear(owner) {
            if (active.current === owner) active.current = null;
        },
        initialize() {},
        onDispose: (state, cleanup) => state.cleanup.push(cleanup),
        cleanup: (owner) => cleaned.push(owner.name),
        valid: (state) => !state.disposed,
    });
    const stateA: TestState = { disposed: false, cleanup: [] };
    const stateB: TestState = { disposed: false, cleanup: [] };
    const stateC: TestState = { disposed: false, cleanup: [] };

    const warmingA = warm(stateA, "a");
    const warmingB = warm(stateB, "b");
    const ownerB = { name: "b" };
    drafts.get("b")!.resolve(ownerB);
    expect(await warmingB).toBe(ownerB);
    drafts.get("a")!.resolve({ name: "a" });
    expect(await warmingA).toBeNull();

    const warmingC = warm(stateC, "c");
    const ownerC = { name: "c" };
    drafts.get("c")!.resolve(ownerC);
    expect(await warmingC).toBe(ownerC);
    for (const cleanup of stateB.cleanup) cleanup();

    expect(active.current).toBe(ownerC);
    expect(activated).toEqual(["b", "c"]);
    expect(cleaned).toEqual(["a", "b"]);
});

describe("temporary accel arm rollback", () => {
    test("partial mirror creation rolls back and disposes exact resources", async () => {
        const events: string[] = [];
        let current = "live";
        await expect(
            withTemporaryOwner("live", "other", {
                activate(owner, own) {
                    current = owner;
                    own("nodes");
                    throw new Error("hits mirror failed");
                },
                async run() {
                    events.push("run");
                    return true;
                },
                disposeResource: (resource) => events.push(`dispose:${resource}`),
                restore(snapshot) {
                    current = snapshot;
                    events.push("restore");
                },
                cleanup: (owner) => events.push(`cleanup:${owner}`),
            }),
        ).rejects.toThrow("hits mirror failed");

        expect(current).toBe("live");
        expect(events).toEqual(["dispose:nodes", "restore", "cleanup:other"]);
    });

    test("an asynchronous gate failure disposes mirrors in reverse then restores before owner cleanup", async () => {
        const events: string[] = [];
        let current = "live";
        await expect(
            withTemporaryOwner("live", "other", {
                activate(owner, own) {
                    current = owner;
                    own("nodes");
                    own("hits");
                },
                async run() {
                    throw new Error("gate failed");
                },
                disposeResource: (resource) => events.push(`dispose:${resource}`),
                restore(snapshot) {
                    current = snapshot;
                    events.push("restore");
                },
                cleanup: (owner) => events.push(`cleanup:${owner}`),
            }),
        ).rejects.toThrow("gate failed");

        expect(current).toBe("live");
        expect(events).toEqual(["dispose:hits", "dispose:nodes", "restore", "cleanup:other"]);
    });
});
