import { expect, test } from "bun:test";
import { resolvePlugins } from "../app/compose";
import { component, declaration, f32, vec2, vec4 } from "./component";
import { GlobalTransform } from "./global-transform";
import { World } from "./world";

const fields = () => ({ scalarValue: f32, pair: vec2, vectorValue: vec4 });

test("components return their field record with schema-typed defaults off the fields", () => {
    const record = fields();
    const defaults = () => ({ scalarValue: 3 });
    const Component = component("ExactKey", record, { defaults, requires: [record] });
    expect(Component).toBe(record);
    expect(Object.keys(Component)).toEqual(["scalarValue", "pair", "vectorValue"]);
    expect(declaration(Component, "Probe")).toEqual({
        key: "ExactKey",
        component: record,
        defaults,
        requires: [record],
    });
    component("TypedDefaults", fields(), {
        // @ts-expect-error defaults cannot name an undeclared field, even alongside a declared one
        defaults: () => ({ scalarValue: 1, unknownField: 2 }),
    });
    component("TypedDefaults", fields(), {
        // @ts-expect-error vec4 defaults must have four lanes
        defaults: () => ({ vectorValue: [1, 2, 3] }),
    });
});

test("composition refuses an undeclared record naming its plugin and fields", () => {
    expect(() => resolvePlugins([{ name: "UndeclaredProbe", components: [fields()] }])).toThrow(
        'plugin "UndeclaredProbe" contains an undeclared component with fields [scalarValue, pair, vectorValue]',
    );
});

test("required GlobalTransform is inserted when missing and remains after its requirer is removed", () => {
    const Component = component("Producer", fields(), { requires: [GlobalTransform] });
    const world = new World();
    world.registry.register(Component);
    const eid = world.create();
    world.add(eid, Component);
    expect(world.has(eid, GlobalTransform)).toBe(true);
    world.storage(GlobalTransform).translation.set(eid, 7, 8, 9, 0);
    world.remove(eid, Component);
    world.add(eid, Component);
    expect(world.storage(GlobalTransform).translation.x.get(eid)).toBe(7);
    world.remove(eid, Component);
    world.step(0);
    expect(world.has(eid, GlobalTransform)).toBe(true);
    world.dispose();
});
