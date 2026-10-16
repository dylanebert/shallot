import { expect, test } from "bun:test";
import { Transform, World } from "@dylanebert/shallot";
import { COIN_CAP, Coin, decode, encode, migrate, placement, readSave, sampleSpots } from "./save";

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
test("the seeded Poisson-disc sampler repeats its seed and changes with another seed", () => {
    expect(sampleSpots(7)).toEqual(sampleSpots(7));
    expect(sampleSpots(8)).not.toEqual(sampleSpots(7));
});
test("the seeded Poisson-disc sample fills the cap with separated spots in the lit region", () => {
    const spots = sampleSpots(0x5a1107);
    expect(spots.length).toBeGreaterThanOrEqual(COIN_CAP);
    for (const [i, [x, z]] of spots.entries()) {
        expect(x).toBeGreaterThanOrEqual(-1.8);
        expect(x).toBeLessThanOrEqual(1.8);
        expect(z).toBeGreaterThanOrEqual(0.65);
        expect(z).toBeLessThanOrEqual(1.8);
        for (const [otherX, otherZ] of spots.slice(i + 1))
            expect(Math.hypot(x - otherX, z - otherZ)).toBeGreaterThanOrEqual(0.15);
    }
});
test("the first ten shuffled spots span the scatter rather than its growth frontier", () => {
    const spots = Array.from({ length: 10 }, (_, i) => placement(i));
    expect(
        Math.max(...spots.map((p) => p[0])) - Math.min(...spots.map((p) => p[0])),
    ).toBeGreaterThan(2);
    expect(
        Math.max(...spots.map((p) => p[2])) - Math.min(...spots.map((p) => p[2])),
    ).toBeGreaterThan(0.6);
});
test("a missing localStorage record has no save to load", () => {
    expect(readSave({ getItem: () => null })).toBeNull();
});
