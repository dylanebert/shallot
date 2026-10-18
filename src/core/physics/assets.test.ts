import { expect, test } from "bun:test";
import { PhysicsAssets } from "./assets";

test("assets keep duplicate labels, remove by handle and reuse the lowest free handle", () => {
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
    expect(assets.has(a)).toBe(false);
    expect(assets.get(a)).toBeUndefined();
    expect(assets.has(b)).toBe(true);
    expect(assets.add(first)).toBe(a);
    expect([...assets.values()]).toEqual([second, first]);
    assets.clear();
    expect(assets.size).toBe(0);
    expect([...assets.entries()]).toEqual([]);
    expect(assets.add(second)).toBe(0);
});
