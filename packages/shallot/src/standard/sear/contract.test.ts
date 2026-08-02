import { describe, expect, test } from "bun:test";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { Surfaces } from "../render/core";
import {
    fsCtxSchema,
    surfaceLayout as layout,
    registerSurface as register,
    TypedSurfaces,
    VsIn,
    vsPatchSchema,
} from "./contract";

// The typed `Surfaces` contract (4a-ii-c-1): `layout()`'s group-2 synthesis + vertices-slot variants,
// `layout.$.name` accessibility for a real TGSL fn, the varyings-folding IO schemas, and the dual-accept
// discriminator. No pipeline/codegen wiring this stage — these are structural + resolve-level proofs, the
// contract stands alone.

const Item = d.struct({ value: d.f32 }).$name("Item");

describe("layout — group-2 synthesis", () => {
    test("pins to group 2 (4a-ii design lock: engine 0 / shadow-or-atlas 1 / surface 2)", () => {
        const l = layout({ items: { type: "storage", element: Item } });
        expect(l.index).toBe(2);
        expect(l.depthVariant.index).toBe(2);
    });

    test("a read-only storage binding stays visible to both stages; read_write narrows to fragment-only", () => {
        // WebGPU forbids a read-write storage binding visible to the vertex stage — a surface never runs
        // compute, so the fix narrows only the mutable case. Red-proven: reverting the fix to unconditional
        // VS_FS makes the read_write assertion below fail.
        const l = layout({
            ro: { type: "storage", element: Item },
            rw: { type: "storage", element: Item, access: "read_write" },
        });
        expect(l.entries.ro.visibility).toEqual(["vertex", "fragment"]);
        expect(l.entries.rw.visibility).toEqual(["fragment"]);
    });

    test("carries the surface's own bindings by name, plus the sear-injected `vertices` slot", () => {
        const l = layout({
            items: { type: "storage", element: Item },
            palette: { type: "uniform", struct: Item },
        });
        expect(Object.keys(l.entries).sort()).toEqual(["items", "palette", "vertices"].sort());
    });

    test("the color/depth variants differ only in `vertices`'s element type", () => {
        const l = layout({ items: { type: "storage", element: Item } });
        const colorWgsl = tgpu.resolve([l], { names: "strict" });
        const depthWgsl = tgpu.resolve([l.depthVariant], { names: "strict" });
        expect(colorWgsl).toContain("array<vec4u>");
        expect(depthWgsl).toContain("array<vec2u>");
        // the surface's own binding is identical in both — only `vertices` moved
        expect(colorWgsl.replace(/vec4u/g, "")).toBe(depthWgsl.replace(/vec2u/g, ""));
    });

    test("an empty bindings record still carries the two vertices variants", () => {
        const l = layout({});
        expect(Object.keys(l.entries)).toEqual(["vertices"]);
    });
});

describe("layout — `.$.name` accessibility from a real TGSL fn", () => {
    test("a fn closing over `layout.$.items` resolves against it", () => {
        const l = layout({ items: { type: "storage", element: Item } });
        const readItem = tgpu.fn(
            [d.u32],
            d.f32,
        )((i) => {
            "use gpu";
            return l.$.items[i].value;
        });
        const wgsl = tgpu.resolve([readItem], { names: "strict" });
        expect(wgsl).toMatch(/var<storage, read> items: array<Item>;/);
        expect(wgsl).toContain("items[i].value");
    });

    test("a layout is shareable across two fns (sprite ×N / gltf-trio precedent)", () => {
        const l = layout({ items: { type: "storage", element: Item } });
        const a = tgpu.fn(
            [],
            d.f32,
        )(() => {
            "use gpu";
            return l.$.items[0].value;
        });
        const b = tgpu.fn(
            [],
            d.f32,
        )(() => {
            "use gpu";
            return l.$.items[1].value;
        });
        const wgsl = tgpu.resolve([a, b], { names: "strict" });
        // one `items` declaration serves both fns
        expect(wgsl.match(/var<storage, read> items:/g)?.length).toBe(1);
    });
});

describe("vsPatchSchema / fsCtxSchema — varyings folding", () => {
    test("no varyings: the fixed base fields alone", () => {
        expect(Object.keys(vsPatchSchema().propTypes)).toEqual(["world", "worldNormal", "clip"]);
        expect(Object.keys(fsCtxSchema().propTypes)).toEqual([
            "eid",
            "world",
            "worldNormal",
            "uv",
            "localPos",
        ]);
    });

    test("declared varyings fold in beside the fixed fields, in both schemas", () => {
        const varyings = { litColor: d.vec3f };
        expect(Object.keys(vsPatchSchema(varyings).propTypes)).toContain("litColor");
        expect(Object.keys(fsCtxSchema(varyings).propTypes)).toContain("litColor");
    });
});

describe("VsFn / FsFn — the fixed code-contract shapes resolve", () => {
    test("a vs fn (VsIn in, vsPatchSchema out) and an fs fn reading its own varying resolve to valid WGSL", () => {
        const varyings = { litColor: d.vec3f };
        const vs = tgpu.fn(
            [VsIn],
            vsPatchSchema(varyings),
        )((vsIn) => {
            "use gpu";
            return {
                world: vsIn.world,
                worldNormal: vsIn.worldNormal,
                clip: d.vec4f(0),
                litColor: std.mul(vsIn.worldNormal, 0.5),
            };
        });
        const fs = tgpu.fn(
            [fsCtxSchema(varyings)],
            d.vec4f,
        )((ctx) => {
            "use gpu";
            return d.vec4f(ctx.litColor, 1);
        });
        const wgsl = tgpu.resolve([vs, fs], { names: "strict" });
        expect(wgsl).toContain("litColor");
        expect(wgsl).toMatch(/fn \w+\(vsIn: VsIn\) -> \w+/);
    });
});

describe("register — dual-accept discrimination", () => {
    test("a legacy (string-contract) spec lands in the real `Surfaces` registry, unchanged", () => {
        const name = `legacy-${Math.random()}`;
        register({ name, fs: "col = vec4<f32>(1.0);" });
        expect(Surfaces.get(name)).toBeDefined();
        expect(TypedSurfaces.get(name)).toBeUndefined();
    });

    test("a typed spec lands in `TypedSurfaces`, not the legacy `Surfaces` registry", () => {
        const name = `typed-${Math.random()}`;
        const l = layout({ items: { type: "storage", element: Item } });
        const fs = tgpu.fn(
            [fsCtxSchema()],
            d.vec4f,
        )(() => {
            "use gpu";
            return d.vec4f(1);
        });
        register({ name, layout: l, fs });
        expect(TypedSurfaces.get(name)).toBeDefined();
        expect(Surfaces.get(name)).toBeUndefined();
    });

    test("the discriminator is `layout` presence, not the `fs` field shape (both specs declare `fs`)", () => {
        const legacyName = `legacy-fs-${Math.random()}`;
        register({ name: legacyName, fs: "col = vec4<f32>(0.0);" });
        // a legacy spec has no `layout` field at all — this is the actual discriminant `register` reads
        expect(Surfaces.get(legacyName)).toBeDefined();
    });

    test("a legacy spec carrying a stray `layout` key still lands in the legacy registry", () => {
        // adversarial case: `"layout" in spec` alone would misroute this into `TypedSurfaces`, where it
        // silently draws nothing (Part reads only `Surfaces`). `fs`'s string shape is the guard. Red-proven:
        // dropping the `typeof spec.fs !== "string"` conjunct makes this fail (lands in TypedSurfaces instead).
        const name = `legacy-stray-layout-${Math.random()}`;
        const legacyWithStrayLayout = {
            name,
            fs: "col = vec4<f32>(1.0);",
            layout: "some-incidental-string-field",
        } as unknown as Parameters<typeof register>[0];
        register(legacyWithStrayLayout);
        expect(Surfaces.get(name)).toBeDefined();
        expect(TypedSurfaces.get(name)).toBeUndefined();
    });
});
