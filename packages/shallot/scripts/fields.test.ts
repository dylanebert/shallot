import { beforeEach, describe, expect, test } from "bun:test";
import { eulerAlias, f32, formatHex, sparse, u32, vec2, vec4 } from "@dylanebert/shallot";
import { clear, register } from "@dylanebert/shallot/ecs/core";
import type { Node } from "@dylanebert/shallot/editor";
import { sections } from "../editor/src/lib/sections";
import { fieldRows, fieldTable } from "./fields";

// synthetic components, one per schema FieldKind — the same set sections.test.ts uses, so the FIELDS
// generator and the inspector are validated against identical inputs. A FIELDS marker is the inspector's
// rows at the default pose, so the generator must agree with `sections()` field-for-field.
const Light = { intensity: sparse(f32) };
const LightTraits = { defaults: () => ({ intensity: 1 }) };

const Tint = { color: sparse(f32) };
const TintTraits = { defaults: () => ({ color: 0xff8080 }), format: { color: formatHex } };

const Body = { pos: sparse(vec4), extent: sparse(vec2) };
const BodyTraits = { defaults: () => ({ pos: [0, 0, 0, 0], extent: [2, 3] }) };

const Shape = { kind: sparse(u32) };
const ShapeTraits = { defaults: () => ({ kind: 1 }), enums: { kind: { Box: 0, Sphere: 1 } } };

const Spin = { rot: sparse(vec4) };
const SpinTraits = { defaults: () => ({ rot: [0, 0, 0, 1] }), aliases: { rot: eulerAlias("rot") } };

function node(name: string): Node {
    return { attrs: [{ name, value: "" }], children: [] };
}

beforeEach(() => {
    clear();
});

describe("fieldRows — one row per schema kind", () => {
    test("scalar → float with its default", () => {
        register("light", Light, LightTraits);
        expect(fieldRows("light")).toEqual([{ field: "intensity", type: "float", default: "1" }]);
    });

    test("color → hex default via the component's format", () => {
        register("tint", Tint, TintTraits);
        expect(fieldRows("tint")).toEqual([{ field: "color", type: "color", default: "0xff8080" }]);
    });

    test("vec4 shows xyz, vec2 shows xy", () => {
        register("body", Body, BodyTraits);
        expect(fieldRows("body")).toEqual([
            { field: "pos", type: "vec3", default: "0 0 0" },
            { field: "extent", type: "vec2", default: "2 3" },
        ]);
    });

    test("enum → option labels and the default's label (clean data, no markdown escaping)", () => {
        register("shape", Shape, ShapeTraits);
        expect(fieldRows("shape")).toEqual([
            { field: "kind", type: "box | sphere", default: "sphere" },
        ]);
    });

    test("aliased quaternion → euler axes at the identity default", () => {
        register("spin", Spin, SpinTraits);
        expect(fieldRows("spin")).toEqual([{ field: "rot", type: "vec3", default: "0 0 0" }]);
    });

    test("unregistered component → null", () => {
        expect(fieldRows("nope")).toBeNull();
    });
});

describe("fieldTable — markdown + drift", () => {
    test("renders a header and a row per field", () => {
        register("body", Body, BodyTraits);
        const { table, error } = fieldTable("body");
        expect(error).toBeUndefined();
        expect(table).toBe(
            "| Field | Type | Default |\n| --- | --- | --- |\n" +
                "| `pos` | vec3 | 0 0 0 |\n| `extent` | vec2 | 2 3 |",
        );
    });

    test("an unregistered component is an error, not an empty table", () => {
        const { table, error } = fieldTable("nope");
        expect(table).toBe("");
        expect(error).toContain("nope");
    });

    test("enum pipes are escaped in the markdown cell (not in the row data)", () => {
        register("shape", Shape, ShapeTraits);
        // fieldRows is clean (`box | sphere`); the table cell escapes the pipe so it doesn't split columns
        expect(fieldTable("shape").table).toContain("| `kind` | box \\| sphere | sphere |");
    });

    test("renaming a field changes the table", () => {
        register("body", Body, BodyTraits);
        const before = fieldTable("body").table;
        clear();
        // a different field name under the same component — the table follows the live schema
        register("body", { offset: sparse(vec4), extent: sparse(vec2) }, BodyTraits);
        const after = fieldTable("body").table;
        expect(after).not.toBe(before);
        expect(after).toContain("| `offset` | vec3 |");
        expect(after).not.toContain("`pos`");
    });
});

// the can't-drift guarantee: the generated rows are the inspector's rows. Both walk schema() + getTraits();
// this pins them field-for-field so a divergence in either fails here.
describe("the table is the inspector's table", () => {
    const fixtures: [string, object, object][] = [
        ["light", Light, LightTraits],
        ["tint", Tint, TintTraits],
        ["body", Body, BodyTraits],
        ["shape", Shape, ShapeTraits],
        ["spin", Spin, SpinTraits],
    ];

    for (const [name, component, traits] of fixtures) {
        test(`${name}: fieldRows labels match sections()`, () => {
            register(name, component as never, traits as never);
            const inspectorLabels = sections(node(name), undefined, null, [])[0].fields.map(
                (f) => f.label,
            );
            const docLabels = fieldRows(name)?.map((r) => r.field);
            expect(docLabels).toEqual(inspectorLabels);
        });
    }
});
