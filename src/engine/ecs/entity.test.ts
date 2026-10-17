import { expect, test } from "bun:test";
import { Entities } from "./entity";

test("entity allocation reserves 0xFFFFFFFF and reuses freed ids after exhaustion", () => {
    const entities = new Entities();
    const image = entities.snapshot();
    image.nextId = 0xfffffffe;
    entities.restore(image);
    const last = entities.add();
    expect(last).toBe(0xfffffffe);
    expect(() => entities.add()).toThrow("id range is exhausted; 0xFFFFFFFF is the invalid id");
    entities.remove(last);
    expect(entities.add()).toBe(last);
    expect(() => entities.add()).toThrow("id range is exhausted");
});

test("entity restore refuses an out-of-range next id or a live invalid id before replacing state", () => {
    const entities = new Entities();
    const live = entities.add();
    const image = entities.snapshot();
    image.nextId = 0x100000000;
    expect(() => entities.restore(image)).toThrow("0xFFFFFFFF");
    expect(entities.exists(live)).toBe(true);
    image.nextId = 0xffffffff;
    image.dense = [0xffffffff];
    image.count = 1;
    expect(() => entities.restore(image)).toThrow("0xFFFFFFFF");
    expect(entities.exists(live)).toBe(true);
});
