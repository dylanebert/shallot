import { expect, test } from "bun:test";
import { resolvePlugins } from "../app/compose";
import { component, declaration, f32, vec2, vec4 } from "./component";

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
