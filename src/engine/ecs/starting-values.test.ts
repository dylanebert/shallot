import { expect, test } from "bun:test";
import { component } from "./component";
import { f16, f32, vec2, vec3, vec4, World } from "./index";

const Component = { scalar: f32, pair: vec2, quad: vec4, half: f16, omitted: f32 };

test("add writes starting values through world storage and keeps omitted defaults", () => {
    const Component = { scalar: f32, pair: vec2, quad: vec4, half: f16, omitted: f32 };
    const world = new World();
    world.registry.register(
        component("starting-values", Component, {
            defaults: () => ({ scalar: 10, omitted: 42 }),
        }),
    );
    const eid = world.create();
    world.add(eid, Component, { scalar: 3, pair: [4, 5], quad: [6, 7, 8, 9], half: 1.5 });
    const storage = world.storage(Component);
    expect(storage.scalar.get(eid)).toBe(3);
    expect(Array.from(storage.pair.read(eid, new Float32Array(2)))).toEqual([4, 5]);
    expect(Array.from(storage.quad.read(eid, new Float32Array(4)))).toEqual([6, 7, 8, 9]);
    expect(storage.half.get(eid)).toBe(1.5);
    expect(storage.omitted.get(eid)).toBe(42);
    const defaultEid = world.create();
    world.add(defaultEid, Component);
    expect(storage.scalar.get(defaultEid)).toBe(10);
    expect(storage.omitted.get(defaultEid)).toBe(42);
    expect(storage.pair.x.get(defaultEid)).toBe(0);
    world.dispose();
});

test("add treats undefined starting values as omitted and keeps their defaults", () => {
    const Component = { scalar: f32, pair: vec2, quad: vec4, half: f16, omitted: f32 };
    const world = new World();
    world.registry.register(
        component("undefined-values", Component, {
            defaults: () => ({ scalar: 10, pair: [2, 3], quad: [4, 5, 6, 7] }),
        }),
    );
    const eid = world.create();
    try {
        world.add(eid, Component, {
            scalar: undefined,
            pair: undefined,
            quad: undefined,
            half: 1.5,
        });
        const storage = world.storage(Component);
        expect(world.has(eid, Component)).toBe(true);
        expect(storage.scalar.get(eid)).toBe(10);
        expect(Array.from(storage.pair.read(eid, new Float32Array(2)))).toEqual([2, 3]);
        expect(Array.from(storage.quad.read(eid, new Float32Array(4)))).toEqual([4, 5, 6, 7]);
        expect(storage.half.get(eid)).toBe(1.5);
    } finally {
        world.dispose();
    }
});

function startingValueTypeControls(world: World, eid: number): void {
    world.add(eid, Component, { pair: [1, 2], quad: [1, 2, 3, 4] });
    // @ts-expect-error Only declared fields accept starting values.
    world.add(eid, Component, { unknown: 1 });
    // @ts-expect-error A vec2 requires exactly two lanes.
    world.add(eid, Component, { pair: [1, 2, 3] });
    // @ts-expect-error A vec4 requires exactly four lanes.
    world.add(eid, Component, { quad: [1, 2, 3] });
    // @ts-expect-error Scalar fields accept numbers, not arrays.
    world.add(eid, Component, { scalar: [1] });
}
void startingValueTypeControls;

// A renamed vec4 keeps four-value meaning; the name is a label, not the vec3 discriminant.
const Lanes = { xyz: vec3, quad: vec4, renamed: { ...vec4, name: "vec3" as const } };

function setTypeControls(world: World, eid: number): void {
    const storage = world.storage(Lanes);
    storage.xyz.set(eid, 1, 2, 3);
    storage.xyz.set(eid, 1, 2, 3, 4);
    storage.quad.set(eid, 1, 2, 3, 4);
    // @ts-expect-error A vec4 set requires all four lanes.
    storage.quad.set(eid, 1, 2, 3);
    // @ts-expect-error A renamed vec4 set still requires all four lanes.
    storage.renamed.set(eid, 1, 2, 3);
    world.add(eid, Lanes, { xyz: [1, 2, 3] });
    // @ts-expect-error A renamed vec4 still requires four starting lanes.
    world.add(eid, Lanes, { renamed: [1, 2, 3] });
}
void setTypeControls;
