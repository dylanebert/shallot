import { expect, test } from "bun:test";
import { PhysicsAssets } from "./assets";

test("duplicate labels retain independent handles and removal cannot redirect a stale handle", () => {
    const assets = new PhysicsAssets<{ name: string; size: number }>();
    const first = { name: "box", size: 1 };
    const second = { name: "box", size: 2 };
    const a = assets.add(first);
    const b = assets.add(second);
    expect(a).not.toBe(b);
    expect(assets.get(a)).toBe(first);
    expect(assets.get(b)).toBe(second);
    expect(assets.size).toBe(2);
    expect([...assets.entries()]).toEqual([
        [a, first],
        [b, second],
    ]);
    expect([...assets]).toEqual([first, second]);
    expect(assets.remove(a)).toBe(first);
    expect(assets.remove(a)).toBeUndefined();
    const successor = assets.add(second);
    expect(successor).not.toBe(a);
    expect(assets.has(a)).toBe(false);
    expect(assets.get(a)).toBeUndefined();
    expect(assets.has(b)).toBe(true);
    expect([...assets.values()]).toEqual([second, second]);
    const saved = assets.snapshot();
    assets.clear();
    const afterClear = assets.add(first);
    for (const stale of [a, b, successor]) {
        expect(assets.has(stale)).toBe(false);
        expect(assets.remove(stale)).toBeUndefined();
    }
    assets.restore(saved);
    expect(assets.has(a)).toBe(false);
    expect(assets.has(afterClear)).toBe(false);
    expect(assets.get(b)).toBe(second);
    expect(assets.get(successor)).toBe(second);
});
