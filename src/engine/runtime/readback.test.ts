import { expect, test } from "bun:test";
import { controlledReadback } from "./readback.fixture";

const copy = () => {};

test("out-of-order one-shot maps keep each request's bytes and copy stamps independent", async () => {
    await controlledReadback(async (world, slots) => {
        world.step(1 / 60);
        world.gpu.frame = 31;
        const first = world.readback.request(4, "older result", copy);
        world.step(1 / 60);
        world.gpu.frame = 32;
        const second = world.readback.request(4, "newer result", copy);
        expect(slots).toHaveLength(2);
        slots[1].resolve([2, 2, 2, 2]);
        const newer = await second;
        slots[0].resolve([1, 1, 1, 1]);
        const older = await first;
        expect([newer.frame, newer.fixedTick]).toEqual([32, 2]);
        expect([older.frame, older.fixedTick]).toEqual([31, 1]);
        expect([...new Uint8Array(newer.bytes)]).toEqual([2, 2, 2, 2]);
        expect([...new Uint8Array(older.bytes)]).toEqual([1, 1, 1, 1]);
        const third = world.readback.request(4, "reused staging", copy);
        slots[0].resolve([3, 3, 3, 3]);
        await third;
        expect([...new Uint8Array(older.bytes)]).toEqual([1, 1, 1, 1]);
    });
});

test("disposing a world rejects a pending readback and late map completion cannot publish a result", async () => {
    await controlledReadback(async (world, slots) => {
        const pool = world.readback;
        const pending = pool.request(4, "disposal result", copy);
        world.dispose();
        await expect(pending).rejects.toThrow(
            "disposal result: frame 0 tick 0 readback failed: readback world disposed",
        );
        slots[0].resolve([4, 3, 2, 1]);
        await Promise.resolve();
        expect(pool.allocated).toBe(0);
        expect(slots[0].destroyed).toBe(true);
    });
});

test("zero unused frames releases returned staging immediately without invalidating owned bytes", async () => {
    await controlledReadback(async (world, slots) => {
        world.readback.maxUnusedFrames = 0;
        const pending = world.readback.request(4, "no idle staging", copy);
        slots[0].resolve([7, 0, 0, 0]);
        const result = await pending;
        expect(world.readback.allocated).toBe(0);
        expect(slots[0].destroyed).toBe(true);
        expect(new Uint8Array(result.bytes)[0]).toBe(7);
    });
});

test("a refused step does not count as an unused frame or release staging", async () => {
    await controlledReadback(async (world, slots) => {
        const pending = world.readback.request(4, "idle frame witness", copy);
        slots[0].resolve([0, 0, 0, 0]);
        await pending;
        world.readback.maxUnusedFrames = 1;
        expect(() => world.step(-1)).toThrow("deltaTime");
        expect(world.gpu.frame).toBe(0);
        expect(slots[0].destroyed).toBe(false);
        world.step(0);
        expect(slots[0].destroyed).toBe(true);
    });
});

test("a rejected mapping releases staging and a later request recovers", async () => {
    await controlledReadback(async (world, slots) => {
        const pending = world.readback.request(4, "rejected map", copy);
        slots[0].reject(new Error("deliberate map rejection"));
        await expect(pending).rejects.toThrow(
            "rejected map: frame 0 tick 0 readback failed: deliberate map rejection",
        );
        expect(world.readback.allocated).toBe(0);
        expect(slots[0].destroyed).toBe(true);
        const retry = world.readback.request(4, "recovery result", copy);
        slots[1].resolve([9, 0, 0, 0]);
        expect(new Uint8Array((await retry).bytes)[0]).toBe(9);
    });
});

test("the first GPU validation error immediately rejects pending readback without awaiting mapping", async () => {
    await controlledReadback(async (world, slots, errors) => {
        const pending = world.readback.request(4, "validation result", copy);
        const event = new Event("uncapturederror");
        Object.defineProperty(event, "error", { value: new Error("first validation error") });
        errors.dispatchEvent(event);
        await expect(pending).rejects.toThrow(
            "validation result: frame 0 tick 0 readback failed: first validation error",
        );
        expect(world.readback.allocated).toBe(0);
        expect(slots[0].destroyed).toBe(true);
    });
});
