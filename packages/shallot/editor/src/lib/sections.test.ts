import { beforeEach, describe, expect, test } from "bun:test";
import {
    angle,
    degrees,
    eulerAlias,
    f32,
    formatHex,
    radians,
    State,
    sparse,
    u32,
    vec2,
    vec4,
} from "@dylanebert/shallot";
import { clear, register } from "@dylanebert/shallot/ecs/core";
import type { Node } from "@dylanebert/shallot/editor";
import { formatFields } from "@dylanebert/shallot/scene/core";
import { dotted, multiSections, sections, wide } from "./sections";

// synthetic components exercise each schema FieldKind path without coupling the test to an engine
// plugin — the editor's reflection→UI derivation is component-agnostic. One per kind so a single
// section's fields are unambiguous.
const Light = { intensity: sparse(f32) };
const LightTraits = { defaults: () => ({ intensity: 1 }) };

const Tint = { color: sparse(f32) };
const TintTraits = { defaults: () => ({ color: 0xff8080 }), format: { color: formatHex } };

const Body = { pos: sparse(vec4), extent: sparse(vec2) };
const BodyTraits = { defaults: () => ({ pos: [0, 0, 0, 0], extent: [2, 3] }) };

const Shape = { kind: sparse(u32) };
const ShapeTraits = { defaults: () => ({ kind: 1 }), enums: { kind: { Box: 0, Sphere: 1 } } };

// a component whose stored quaternion is authored as euler via an alias — the rot class. buildFields
// must read `traits.aliases` to render the alias's axes and decode quat→euler. Reading the wrong
// location (or treating rot as a raw quat) crashes / mis-renders on select.
const Spin = { rot: sparse(vec4) };
const SpinTraits = { defaults: () => ({ rot: [0, 0, 0, 1] }), aliases: { rot: eulerAlias("rot") } };

// a radians scalar shown in degrees via the `angle` input — the unit widget path.
const Lens = { fov: sparse(f32) };
const LensTraits = { defaults: () => ({ fov: 1 }), inputs: { fov: angle } };

function node(attrs: { name: string; value: string }[] = []): Node {
    return { attrs, children: [] };
}

beforeEach(() => {
    clear();
});

describe("sections — ECS path", () => {
    test("an aliased quaternion renders as editable euler axes (rot class)", () => {
        register("spin", Spin, SpinTraits);
        const state = new State();
        const eid = state.create();
        state.add(eid, Spin as never);

        const [spin] = sections(node([{ name: "spin", value: "" }]), eid, state, []);

        expect(spin.registered).toBe(true);
        // identity quaternion → euler (0, 0, 0); edits route through the alias, not raw lanes
        expect(spin.fields).toEqual([
            {
                type: "vec",
                base: "rot",
                label: "rot",
                aliased: true,
                axes: [
                    { label: "x", key: "alias:rot:0", value: 0 },
                    { label: "y", key: "alias:rot:1", value: 0 },
                    { label: "z", key: "alias:rot:2", value: 0 },
                ],
            },
        ]);
    });

    test("scalar field becomes an editable float", () => {
        register("light", Light, LightTraits);
        const state = new State();
        const eid = state.create();
        state.add(eid, Light as never);
        Light.intensity.set(eid, 4.5);

        const [light] = sections(node([{ name: "light", value: "" }]), eid, state, []);

        expect(light.fields).toEqual([
            { type: "float", key: "intensity", label: "intensity", value: 4.5 },
        ]);
    });

    test("a derived-trait component on the entity never becomes a section (glTF route decorations)", () => {
        register("light", Light, LightTraits);
        const Deco = { id: sparse(u32) };
        register("deco", Deco, { defaults: () => ({ id: 0 }), derived: true });
        const state = new State();
        const eid = state.create();
        state.add(eid, Light as never);
        state.add(eid, Deco as never); // a system's runtime add — the node doesn't author it

        const result = sections(node([{ name: "light", value: "" }]), eid, state, []);

        expect(result.map((s) => s.name)).toEqual(["light"]);
    });

    test("a color-tagged field becomes a color swatch", () => {
        register("tint", Tint, TintTraits);
        const state = new State();
        const eid = state.create();
        state.add(eid, Tint as never);

        const [tint] = sections(node([{ name: "tint", value: "" }]), eid, state, []);

        expect(tint.fields).toEqual([
            { type: "color", key: "color", label: "color", value: 0xff8080 },
        ]);
    });

    test("a Quad becomes editable xyz (w dropped); a Pair becomes editable xy", () => {
        register("body", Body, BodyTraits);
        const state = new State();
        const eid = state.create();
        state.add(eid, Body as never);

        const [body] = sections(node([{ name: "body", value: "" }]), eid, state, []);

        expect(body.fields).toContainEqual({
            type: "vec",
            base: "pos",
            label: "pos",
            aliased: false,
            axes: [
                { label: "x", key: "pos.x", value: 0 },
                { label: "y", key: "pos.y", value: 0 },
                { label: "z", key: "pos.z", value: 0 },
            ],
        });
        expect(body.fields).toContainEqual({
            type: "vec",
            base: "extent",
            label: "extent",
            aliased: false,
            axes: [
                { label: "x", key: "extent.x", value: 2 },
                { label: "y", key: "extent.y", value: 3 },
            ],
        });
    });

    test("an enum field carries its options", () => {
        register("shape", Shape, ShapeTraits);
        const state = new State();
        const eid = state.create();
        state.add(eid, Shape as never);

        const [shape] = sections(node([{ name: "shape", value: "" }]), eid, state, []);

        expect(shape.fields).toEqual([
            { type: "enum", key: "kind", label: "kind", value: 1, options: { Box: 0, Sphere: 1 } },
        ]);
    });

    test("a unit-annotated field carries its unit menu and the raw stored value", () => {
        register("lens", Lens, LensTraits);
        const state = new State();
        const eid = state.create();
        state.add(eid, Lens as never);
        Lens.fov.set(eid, 0.123456);

        const [lens] = sections(node([{ name: "lens", value: "" }]), eid, state, []);
        const field = lens.fields[0];

        expect(field).toMatchObject({
            type: "unit",
            key: "fov",
            label: "fov",
            units: [degrees, radians],
        });
        // raw stored radians — a unit field is NOT pre-rounded the way a float is (it rounds in the
        // shown unit, after converting), so full precision survives to the editor boundary
        expect(field.type === "unit" && field.value).toBeCloseTo(0.123456, 5);
    });

    test("wide() claims the full column for unit and vec, not float", () => {
        register("lens", Lens, LensTraits);
        register("light", Light, LightTraits);
        register("body", Body, BodyTraits);
        const state = new State();
        const eid = state.create();
        state.add(eid, Lens as never);
        state.add(eid, Light as never);
        state.add(eid, Body as never);

        const secs = sections(node(), eid, state, []);
        const first = (name: string) => secs.find((s) => s.name === name)!.fields[0];

        expect(wide(first("lens"))).toBe(true);
        expect(wide(first("body"))).toBe(true);
        expect(wide(first("light"))).toBe(false);
    });

    test("appends a component present on the entity but absent from the node attrs", () => {
        register("light", Light, LightTraits);
        register("shape", Shape, ShapeTraits);
        const state = new State();
        const eid = state.create();
        state.add(eid, Light as never);
        state.add(eid, Shape as never);

        const result = sections(node([{ name: "light", value: "" }]), eid, state, []);

        expect(result.map((s) => s.name)).toEqual(["light", "shape"]);
    });
});

describe("sections — scene-attr path (no live ECS)", () => {
    test("reads field values from the attr value", () => {
        register("light", Light, LightTraits);
        const value = formatFields("light", { intensity: 5 });

        const [light] = sections(node([{ name: "light", value }]), undefined, null, []);

        expect(light.registered).toBe(true);
        expect(light.fields).toEqual([
            { type: "float", key: "intensity", label: "intensity", value: 5 },
        ]);
    });

    test("a malformed attr value yields a registered section with no fields", () => {
        register("light", Light, LightTraits);

        const [light] = sections(
            node([{ name: "light", value: "intensity: abc" }]),
            undefined,
            null,
            [],
        );

        expect(light).toMatchObject({ name: "light", registered: true, parsed: null, fields: [] });
    });

    test("an unregistered attr is an unregistered section carrying its diagnostic", () => {
        const n = node([{ name: "wobble", value: "x: 1" }]);
        const diag = {
            node: n,
            attr: "wobble",
            kind: "unregistered",
            message: '"wobble" is not registered',
        };

        const [section] = sections(n, undefined, null, [diag]);

        expect(section).toMatchObject({
            name: "wobble",
            registered: false,
            fields: [],
            diagnosticMessage: '"wobble" is not registered',
        });
    });
});

describe("multiSections — shared components across a selection", () => {
    test("keeps only components present on every selected node", () => {
        register("light", Light, LightTraits);
        register("shape", Shape, ShapeTraits);
        const state = new State();
        const a = state.create();
        state.add(a, Light as never);
        state.add(a, Shape as never);
        const b = state.create();
        state.add(b, Light as never);

        const secs = multiSections([node(), node()], [a, b], state, []);

        // shape is on a but not b — the intersection drops it
        expect(secs.map((s) => s.name)).toEqual(["light"]);
    });

    test("an agreeing field shows its value and is not mixed", () => {
        register("light", Light, LightTraits);
        const state = new State();
        const a = state.create();
        state.add(a, Light as never);
        Light.intensity.set(a, 2);
        const b = state.create();
        state.add(b, Light as never);
        Light.intensity.set(b, 2);

        const [light] = multiSections([node(), node()], [a, b], state, []);

        expect(light.fields[0]).toEqual({
            type: "float",
            key: "intensity",
            label: "intensity",
            value: 2,
            mixed: false,
        });
    });

    test("a differing scalar field is flagged mixed (showing the active = last node's value)", () => {
        register("light", Light, LightTraits);
        const state = new State();
        const a = state.create();
        state.add(a, Light as never);
        Light.intensity.set(a, 2);
        const b = state.create();
        state.add(b, Light as never);
        Light.intensity.set(b, 5);

        // b is last → the active node, so its value (5) is the one shown
        const [light] = multiSections([node(), node()], [a, b], state, []);

        expect(light.fields[0]).toMatchObject({ type: "float", value: 5, mixed: true });
    });

    test("a vec flags only the axes that differ", () => {
        register("body", Body, BodyTraits);
        const state = new State();
        const a = state.create();
        state.add(a, Body as never);
        Body.pos.set(a, 1, 2, 3, 0);
        const b = state.create();
        state.add(b, Body as never);
        Body.pos.set(b, 1, 9, 3, 0);

        const [body] = multiSections([node(), node()], [a, b], state, []);
        const pos = body.fields.find((f) => f.type === "vec" && f.base === "pos");

        expect(pos?.type === "vec" && pos.axes.map((ax) => ax.mixed)).toEqual([false, true, false]);
    });

    test("a single-node selection routes through unchanged (never flags mixed)", () => {
        register("light", Light, LightTraits);
        const state = new State();
        const a = state.create();
        state.add(a, Light as never);
        Light.intensity.set(a, 3);

        const [light] = multiSections([node()], [a], state, []);

        expect(light.fields[0]).toEqual({
            type: "float",
            key: "intensity",
            label: "intensity",
            value: 3,
        });
    });
});

describe("dotted", () => {
    // load-bearing for the alias scrub: `live` carries the old array form plus the accumulating dotted
    // lanes, so a dotted lane must override the stale array (else each scrub frame reads stale values).
    test("an explicit dotted lane overrides the array form", () => {
        expect(dotted({ rot: [1, 2, 3, 4], "rot.x": 9 })).toEqual({
            "rot.x": 9,
            "rot.y": 2,
            "rot.z": 3,
            "rot.w": 4,
        });
    });
});
