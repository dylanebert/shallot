import { expect, test } from "bun:test";
import { controlledReadback } from "./readback.fixture";

const copy = () => {};

test("out-of-order one-shot maps keep each request's bytes and copy stamps independent", async () => {
    await controlledReadback(async (state, slots) => {
        state.step(1 / 60);
        state.gpu.frame = 31;
        const first = state.readback.request(4, "older result", copy);
        state.step(1 / 60);
        state.gpu.frame = 32;
        const second = state.readback.request(4, "newer result", copy);
        expect(slots).toHaveLength(2);
        slots[1].resolve([2, 2, 2, 2]);
        const newer = await second;
        slots[0].resolve([1, 1, 1, 1]);
        const older = await first;
        expect([newer.frame, newer.fixedTick]).toEqual([32, 2]);
        expect([older.frame, older.fixedTick]).toEqual([31, 1]);
        expect([...new Uint8Array(newer.bytes)]).toEqual([2, 2, 2, 2]);
        expect([...new Uint8Array(older.bytes)]).toEqual([1, 1, 1, 1]);
        const third = state.readback.request(4, "reused staging", copy);
        slots[0].resolve([3, 3, 3, 3]);
        await third;
        expect([...new Uint8Array(older.bytes)]).toEqual([1, 1, 1, 1]);
    });
});

test("disposing a world rejects a pending readback and late map completion cannot publish a result", async () => {
    await controlledReadback(async (state, slots) => {
        const pool = state.readback;
        const pending = pool.request(4, "disposal result", copy);
        state.dispose();
        await expect(pending).rejects.toThrow("disposed");
        slots[0].resolve([4, 3, 2, 1]);
        await Promise.resolve();
        expect(pool.allocated).toBe(0);
        expect(slots[0].destroyed).toBe(true);
    });
});

test("zero unused frames releases returned staging immediately without invalidating owned bytes", async () => {
    await controlledReadback(async (state, slots) => {
        state.readback.maxUnusedFrames = 0;
        const pending = state.readback.request(4, "no idle staging", copy);
        slots[0].resolve([7, 0, 0, 0]);
        const result = await pending;
        expect(state.readback.allocated).toBe(0);
        expect(slots[0].destroyed).toBe(true);
        expect(new Uint8Array(result.bytes)[0]).toBe(7);
    });
});

test("a refused step does not count as an unused frame or release staging", async () => {
    await controlledReadback(async (state, slots) => {
        const pending = state.readback.request(4, "idle frame witness", copy);
        slots[0].resolve([0, 0, 0, 0]);
        await pending;
        state.readback.maxUnusedFrames = 1;
        expect(() => state.step(-1)).toThrow("deltaTime");
        expect(state.gpu.frame).toBe(0);
        expect(slots[0].destroyed).toBe(false);
        state.step(0);
        expect(slots[0].destroyed).toBe(true);
    });
});

test("a rejected mapping releases staging and a later request recovers", async () => {
    await controlledReadback(async (state, slots) => {
        const pending = state.readback.request(4, "rejected map", copy);
        slots[0].reject(new Error("deliberate map rejection"));
        await expect(pending).rejects.toThrow("deliberate map rejection");
        expect(state.readback.allocated).toBe(0);
        expect(slots[0].destroyed).toBe(true);
        const retry = state.readback.request(4, "recovery result", copy);
        slots[1].resolve([9, 0, 0, 0]);
        expect(new Uint8Array((await retry).bytes)[0]).toBe(9);
    });
});

test("the first GPU validation error immediately rejects pending readback without awaiting mapping", async () => {
    await controlledReadback(async (state, slots, errors) => {
        const pending = state.readback.request(4, "validation result", copy);
        const event = new Event("uncapturederror");
        Object.defineProperty(event, "error", { value: new Error("first validation error") });
        errors.dispatchEvent(event);
        await expect(pending).rejects.toThrow("first validation error");
        expect(state.readback.allocated).toBe(0);
        expect(slots[0].destroyed).toBe(true);
    });
});
