import { expect, test } from "bun:test";
import { Transform, World } from "@dylanebert/shallot";
import { Coin, decode, encode, migrate, placement, readSave } from "./save";

function world() {
    const w = new World();
    w.registry.register(Coin);
    w.registry.register(Transform);
    return w;
}
test("fields encode and decode each coin's chosen components", () => {
    const w = world();
    const eid = w.create();
    w.add(eid, Coin, { height: 0.25, speed: 2 });
    w.add(eid, Transform, {
        translation: [1, 2, 3],
        rotation: [0, 0.6, 0, 0.8],
        scale: [0.28, 0.08, 0.28],
    });
    const saved = encode(w);
    decode(w, saved);
    expect(encode(w)).toEqual(saved);
    expect([...w.query([Coin])]).toHaveLength(1);
    w.dispose();
});
test("the previous version's bare count migrates to coins on the platform", () => {
    const w = world();
    decode(w, { version: 1, count: 30 });
    const coins = [...w.query([Coin])];
    expect(coins).toHaveLength(30);
    coins.forEach((eid, index) => {
        expect(w.storage(Transform).translation.x.get(eid)).toBeCloseTo(placement(index)[0]);
        expect(w.storage(Transform).translation.y.get(eid)).toBeCloseTo(placement(index)[1]);
    });
    expect(migrate(encode(w))).toEqual(encode(w));
    w.dispose();
});
test("scattered coins keep their spots after save and load", () => {
    const w = world();
    decode(w, { version: 1, count: 70 });
    for (let index = 0; index < 70; index++) {
        const [x, , z] = placement(index);
        expect(Math.abs(x) + 0.14).toBeLessThan(2);
        expect(Math.abs(z) + 0.14).toBeLessThan(2);
    }
    const saved = encode(w);
    decode(w, saved);
    expect(encode(w).entities).toHaveLength(70);
    expect(encode(w).entities).toEqual(expect.arrayContaining(saved.entities));
    w.dispose();
});
test("a missing localStorage record has no save to load", () => {
    expect(readSave({ getItem: () => null })).toBeNull();
});
