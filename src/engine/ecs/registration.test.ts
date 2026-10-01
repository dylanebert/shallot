import { expect, test } from "bun:test";
import { f32, vec2, vec4 } from "./component";
import { dump, inspect, readFields, snapshot } from "./reflection";
import { World } from "./state";
import { registration } from "./traits";

const Component = { scalarValue: f32, pair: vec2, vectorValue: vec4 };

test("registrations are plain data with flat options and schema-typed defaults", () => {
    const defaults = () => ({ scalarValue: 3 });
    expect(
        registration("ExactKey", Component, {
            defaults,
            excludes: [Component],
            provides: [Component],
        }),
    ).toEqual({
        key: "ExactKey",
        component: Component,
        defaults,
        excludes: [Component],
        provides: [Component],
    });
    registration("TypedDefaults", Component, {
        // @ts-expect-error defaults cannot name an undeclared field, even alongside a declared one
        defaults: () => ({ scalarValue: 1, unknownField: 2 }),
    });
    registration("TypedDefaults", Component, {
        // @ts-expect-error vec4 defaults must have four lanes
        defaults: () => ({ vectorValue: [1, 2, 3] }),
    });
});

test("snapshot reports exact registration keys and declared scalar and vector fields", () => {
    const world = new World();
    world.registry.register(registration("ExactKey", Component));
    const eid = world.create();
    world.add(eid, Component, { scalarValue: 7, pair: [8, 9], vectorValue: [1, 2, 3, 4] });
    const fields = { scalarValue: 7, pair: [8, 9], vectorValue: [1, 2, 3, 4] };
    const data = { eid, components: { ExactKey: fields } };
    expect(snapshot(world)).toEqual([data]);
    expect(inspect(world, eid)).toEqual(data);
    expect(readFields(world, Component, eid)).toEqual(fields);
    expect(dump(world, eid)).toBe(
        `Entity ${eid}:\n  ExactKey: scalarValue: 7, pair: [8,9], vectorValue: [1,2,3,4]`,
    );
    world.dispose();
});
