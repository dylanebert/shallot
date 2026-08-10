import { describe, expect, test } from "bun:test";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { State } from "../../engine";
import {
    Backgrounds,
    BgCtx,
    backgroundLayout,
    fsCtxSchema,
    surfaceLayout as layout,
    registerSurface as register,
    registerBackground,
    Surfaces,
    VsIn,
    vsPatchSchema,
} from "./contract";

// The `Surfaces` contract: group-2 synthesis + vertices-slot variants, `layout.$.name` accessibility,
// varyings-folding IO schemas, and State-owned registry cleanup.

const Item = d.struct({ value: d.f32 }).$name("Item");

describe("layout — group-2 synthesis", () => {
    test("the public surface registry rejects name-only records at compile time", () => {
        const registerBroken = () => {
            // @ts-expect-error a registered surface requires its concrete layout and fragment function
            Surfaces.register({ name: "broken" });
        };
        expect(registerBroken).toBeFunction();
        expect(Surfaces.get("broken")).toBeUndefined();
    });

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

describe("register — State ownership", () => {
    const fs = tgpu.fn(
        [fsCtxSchema()],
        d.vec4f,
    )(() => {
        "use gpu";
        return d.vec4f(1);
    });
    const bgFs = tgpu.fn(
        [BgCtx],
        d.vec3f,
    )(() => {
        "use gpu";
        return d.vec3f(1);
    });

    test("one disposer per registry owns every registration made by one State", () => {
        Surfaces.clear();
        Backgrounds.clear();
        const state = new State();
        const surface = (name: string) => ({ name, layout: layout({}), fs });
        register(state, surface("owned-a"));
        register(state, surface("owned-b"));
        register(state, surface("owned-a"));
        registerBackground(state, {
            name: "owned-bg",
            layout: backgroundLayout({}),
            fs: bgFs,
        });

        expect((state as unknown as { _disposals: (() => void)[] })._disposals.length).toBe(2);
        state.dispose();
        expect(Surfaces.get("owned-a")).toBeUndefined();
        expect(Surfaces.get("owned-b")).toBeUndefined();
        expect(Backgrounds.get("owned-bg")).toBeUndefined();
    });

    test("an old State cannot delete a same-name replacement owned by a newer State", () => {
        Surfaces.clear();
        const first = new State();
        const second = new State();
        const firstSpec = { name: "replacement", layout: layout({}), fs };
        const secondSpec = { name: "replacement", layout: layout({}), fs };
        register(first, firstSpec);
        register(second, secondSpec);
        first.dispose();
        expect(Surfaces.get("replacement")).toBe(secondSpec);
        second.dispose();
        expect(Surfaces.get("replacement")).toBeUndefined();
    });
});

describe("register — foreign-copy brand check", () => {
    const fs = tgpu.fn(
        [fsCtxSchema()],
        d.vec4f,
    )(() => {
        "use gpu";
        return d.vec4f(1);
    });
    const bgFs = tgpu.fn(
        [BgCtx],
        d.vec3f,
    )(() => {
        "use gpu";
        return d.vec3f(1);
    });

    // a real second copy of typegpu stamps its own per-copy `Symbol()` internal marker
    // (`typegpu/shared/symbols.js`, never `Symbol.for`) — this reproduces exactly that shape without
    // installing a second copy: right `resourceType`, a foreign `$internal` symbol the engine's own
    // `isTgpuFn` can't recognize (de-risked 2026-08-10, spec `shallot-peer-identity` stage 1).
    const foreignInternal = Symbol("typegpu:0.11.9:$internal");
    const foreignFs = { resourceType: "function", [foreignInternal]: true } as unknown as typeof fs;

    // the name is deliberately not "foreign": the diagnostic prefixes the spec's own name, so matching
    // a word the caller supplied would pass on an empty message body. Match the diagnostic's own
    // sentence, the seam that named it, and the remedy — the three parts the message promises.
    const diagnostic = /registerSurface "tinted" fs:.*foreign copy of typegpu.*Dedupe typegpu/s;

    test("a foreign-copy fs throws a named diagnostic at registerSurface", () => {
        Surfaces.clear();
        const state = new State();
        expect(() =>
            register(state, { name: "tinted", layout: layout({}), fs: foreignFs }),
        ).toThrow(diagnostic);
    });

    test("a foreign-copy fs throws a named diagnostic at registerBackground", () => {
        Backgrounds.clear();
        const state = new State();
        expect(() =>
            registerBackground(state, {
                name: "sky",
                layout: backgroundLayout({}),
                fs: foreignFs as unknown as typeof bgFs,
            }),
        ).toThrow(/registerBackground "sky" fs:.*foreign copy of typegpu.*Dedupe typegpu/s);
    });

    test("a genuine engine fs registers clean", () => {
        Surfaces.clear();
        const state = new State();
        expect(() => register(state, { name: "genuine", layout: layout({}), fs })).not.toThrow();
        expect(Surfaces.get("genuine")).toBeDefined();
        state.dispose();
    });
});
