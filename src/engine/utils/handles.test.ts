import { expect, test } from "bun:test";
import { HandleAllocator } from "./handles";

test("recycling dense indices changes generation and rejects stale or malformed handles", () => {
    const allocator = new HandleAllocator();
    const a = allocator.allocate();
    const b = allocator.allocate();
    const index = allocator.index(a);
    expect(allocator.release(a)).toBe(true);
    expect(allocator.release(a)).toBe(false);
    const next = allocator.allocate();
    expect(next).not.toBe(a);
    expect(allocator.index(next)).toBe(index);
    for (const invalid of [a, -1, 0.5, NaN, Infinity, 2 ** 31, 2 ** 32 + next]) {
        expect(allocator.index(invalid)).toBeUndefined();
        expect(allocator.release(invalid)).toBe(false);
    }
    expect(allocator.index(b)).toBe(1);
    expect(allocator.index(next)).toBe(index);
});

test("clear invalidates handles, and reusable snapshots preserve generations and free indices", () => {
    const allocator = new HandleAllocator();
    const beforeClear = allocator.allocate();
    allocator.clear();
    const saved = allocator.snapshot();
    const live = allocator.allocate();
    expect(live).not.toBe(beforeClear);
    expect(allocator.index(beforeClear)).toBeUndefined();
    allocator.release(live);
    allocator.allocate();
    for (let i = 0; i < 2; i++) {
        allocator.restore(saved);
        expect(allocator.index(beforeClear)).toBeUndefined();
        expect(allocator.index(live)).toBeUndefined();
        expect(allocator.allocate()).toBe(live);
    }
});

test("generation exhaustion retires an index rather than aliasing any earlier handle", () => {
    const allocator = new HandleAllocator();
    const first = allocator.allocate();
    let handle = first;
    for (let generation = 0; generation < 2 ** 11; generation++) {
        expect(allocator.index(handle)).toBe(0);
        allocator.release(handle);
        handle = allocator.allocate();
    }
    expect(allocator.index(first)).toBeUndefined();
    expect(allocator.index(handle)).toBe(1);
});

test("index exhaustion refuses allocation and the largest packed handle fits signed i32", () => {
    const allocator = new HandleAllocator();
    let last = 0;
    for (let index = 0; index < 2 ** 20; index++) last = allocator.allocate();
    expect(() => allocator.allocate()).toThrow("handle index space exhausted");
    const state = allocator.snapshot();
    state.generations[last] = 2 ** 11 - 1;
    allocator.restore(state);
    const largest = 2 ** 31 - 1;
    expect(allocator.index(largest)).toBe(last);
    expect(new Int32Array([largest])[0]).toBe(largest);
    allocator.release(largest);
    expect(allocator.index(largest)).toBeUndefined();
    expect(() => allocator.allocate()).toThrow("handle index space exhausted");
});
