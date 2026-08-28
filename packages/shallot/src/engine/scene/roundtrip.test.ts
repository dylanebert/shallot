import { beforeEach, describe, expect, test } from "bun:test";
import { attach } from "../../../tests/helpers";
import {
    AudioPlugin,
    build,
    entity,
    f32,
    GlazePlugin,
    InputPlugin,
    i32,
    LinesPlugin,
    laneAlias,
    load,
    type Node,
    OrbitPlugin,
    PartPlugin,
    type Plugin,
    parse,
    RenderPlugin,
    SearPlugin,
    State,
    serialize,
    sparse,
    stringify,
    TextPlugin,
    TransformsPlugin,
    TweenPlugin,
    u8,
    u32,
    vec2,
    vec4,
} from "../..";
import { clear, getComponent, register, snapshot } from "../ecs/core";
import { formatFields, normalizeAttr, parseFields } from "./core";

const SCENE_FILES = [
    "recipes/custom-material/public/scenes/custom-material.scene",
    "recipes/gpu-particles/public/scenes/gpu-particles.scene",
    "showcase/visualization/public/scenes/tween.scene",
];

const PLUGINS: Plugin[] = [
    TransformsPlugin,
    InputPlugin,
    RenderPlugin,
    PartPlugin,
    OrbitPlugin,
    TweenPlugin,
    LinesPlugin,
    TextPlugin,
    SearPlugin,
    AudioPlugin,
    GlazePlugin,
];

function registerPlugins() {
    const state = new State();
    for (const plugin of PLUGINS) {
        for (const [n, c] of Object.entries(plugin.components ?? {}))
            register(n, c, plugin.traits?.[n]);
        attach(state, plugin);
    }
    return state;
}

async function readScene(name: string): Promise<string> {
    const path = `${import.meta.dir}/../../../../../examples/${name}`;
    return Bun.file(path).text();
}

function compareNodes(a: Node[], b: Node[]) {
    expect(b.length).toBe(a.length);
    for (let i = 0; i < a.length; i++) {
        const na = a[i];
        const nb = b[i];
        expect(nb.id).toBe(na.id);
        expect(nb.attrs.length).toBe(na.attrs.length);
        for (let j = 0; j < na.attrs.length; j++) {
            expect(nb.attrs[j].name).toBe(na.attrs[j].name);
            expect(nb.attrs[j].value).toBe(na.attrs[j].value);
        }
        compareNodes(na.children, nb.children);
    }
}

function countNodes(nodes: Node[]): number {
    let n = nodes.length;
    for (const node of nodes) n += countNodes(node.children);
    return n;
}

describe("Scene Roundtrip", () => {
    beforeEach(() => {
        clear();
        registerPlugins();
    });

    test.each(SCENE_FILES)("node-level roundtrip: %s", async (file) => {
        const xml = await readScene(file);
        const original = parse(xml);
        const serialized = stringify(original);
        const reparsed = parse(serialized);

        expect(countNodes(reparsed)).toBe(countNodes(original));
        compareNodes(original, reparsed);
    });

    test.each(SCENE_FILES)("idempotent serialization: %s", async (file) => {
        const xml = await readScene(file);
        const once = stringify(parse(xml));
        const twice = stringify(parse(once));
        expect(twice).toBe(once);
    });

    describe("field-level roundtrip", () => {
        // RED witnessed: The continues let the round-trip pass asserting nothing.
        // Made formatFields return "" → exit 1 (6 fail); today the assertion floor reds a body that
        // reaches no expect call.
        test.each(SCENE_FILES)("component fields survive roundtrip: %s", async (file) => {
            const xml = await readScene(file);
            const nodes = parse(xml);
            let assertionsExecuted = 0;

            function checkNode(node: Node) {
                for (const attr of node.attrs) {
                    if (!attr.value) continue;
                    if (!getComponent(attr.name)) continue;

                    let fields: Record<string, number | string>;
                    try {
                        fields = parseFields(attr.name, attr.value);
                    } catch {
                        continue;
                    }
                    const formatted = formatFields(attr.name, fields);
                    if (!formatted) continue;
                    const fields2 = parseFields(attr.name, formatted);

                    for (const key of Object.keys(fields2)) {
                        if (!(key in fields)) continue;
                        assertionsExecuted++;
                        const a = fields[key];
                        const b = fields2[key];
                        if (typeof a === "number" && typeof b === "number") {
                            // formatNumber serializes non-integers via toPrecision(7);
                            // round-trip error is bounded by a half-ULP at the 7th
                            // significant figure
                            expect(Math.abs(b - a)).toBeLessThanOrEqual(
                                5e-7 * Math.max(1, Math.abs(a)),
                            );
                        } else {
                            expect(b).toBe(a);
                        }
                    }
                }
                for (const child of node.children) checkNode(child);
            }

            for (const node of nodes) checkNode(node);

            // floor: the continues (empty value, unregistered component, parse
            // failure, empty format, absent key) let the test pass asserting nothing
            expect(assertionsExecuted).toBeGreaterThan(0);
        });
    });

    describe("normalization idempotence", () => {
        // RED witnessed: The continues let the idempotence check pass asserting
        // nothing. Made formatFields return "" → exit 1 (6 fail); today the assertion floor reds a
        // body that reaches no expect call.
        test.each(SCENE_FILES)("normalizeAttr is idempotent: %s", async (file) => {
            const xml = await readScene(file);
            const nodes = parse(xml);
            let assertionsExecuted = 0;

            function checkNode(node: Node) {
                for (const attr of node.attrs) {
                    if (!attr.value) continue;
                    const first = normalizeAttr(attr.name, attr.value);
                    // null = unregistered; "" = all fields at default (the bare form), a terminal that
                    // re-normalizes to null. Idempotence is the fixed-point claim on a non-empty result.
                    if (first === null || first === "") continue;
                    assertionsExecuted++;
                    const second = normalizeAttr(attr.name, first);
                    expect(second).toBe(first);
                }
                for (const child of node.children) checkNode(child);
            }

            for (const node of nodes) checkNode(node);

            // floor: the continues (empty value, null/empty normalize) let the test
            // pass asserting nothing
            expect(assertionsExecuted).toBeGreaterThan(0);
        });
    });
});

const formatHex = Object.assign((n: number) => "0x" + (n >>> 0).toString(16).padStart(6, "0"), {
    kind: "color" as const,
});

// synthetic components spanning the storage-type matrix serialize must cover: every scalar type,
// both vector widths, plus a color (format-trait) and an enum (format + parse trait) field. sparse
// keeps the round-trip pure CPU (no device, no build()) and exercises the exact Single/Pair/Quad
// surface a slab-backed component presents to readFields/formatFields.
const Spatial = { pos: sparse(vec4), vel: sparse(vec2) };
const Stats = { hp: sparse(f32), level: sparse(u32), flags: sparse(u8), charge: sparse(i32) };
const Style = { tint: sparse(f32), mode: sparse(u8) };

function registerSynthetic() {
    clear();
    register("spatial", Spatial, { defaults: () => ({ pos: [0, 0, 0, 0], vel: [0, 0] }) });
    register("stats", Stats, { defaults: () => ({ hp: 100, level: 1, flags: 0, charge: 0 }) });
    register("style", Style, {
        defaults: () => ({ tint: 0xffffff, mode: 0 }),
        format: { tint: formatHex },
        parse: { tint: (v: string) => Number.parseInt(v.replace("0x", ""), 16) },
        enums: { mode: { Idle: 0, Run: 1, Jump: 2 } },
    });
}

function expectSnapshotsMatch(
    expected: ReturnType<typeof snapshot>,
    actual: ReturnType<typeof snapshot>,
) {
    expect(actual.length).toBe(expected.length);
    for (let i = 0; i < expected.length; i++) {
        // both states allocate eids 1..N in creation order, and load recreates in serialize order,
        // so the i-th entity is the same logical one in each
        expect(actual[i].eid).toBe(expected[i].eid);
        const ec = expected[i].components;
        const ac = actual[i].components;
        expect(Object.keys(ac).sort()).toEqual(Object.keys(ec).sort());
        for (const name of Object.keys(ec)) {
            for (const [key, a] of Object.entries(ec[name])) {
                const b = ac[name][key];
                if (typeof a === "number" && typeof b === "number") {
                    // f32 fields round-trip through formatNumber's toPrecision(7); the error is bounded
                    // by a half-ULP at the 7th significant figure. integer-typed fields hit 0 exactly.
                    expect(Math.abs(b - a)).toBeLessThanOrEqual(5e-7 * Math.max(1, Math.abs(a)));
                } else {
                    expect(b).toEqual(a);
                }
            }
        }
    }
}

describe("serialize(state)", () => {
    test("known component field values survive serialize → stringify → parse → load", () => {
        registerSynthetic();
        const state = new State();

        const a = state.create();
        state.add(a, Spatial);
        state.add(a, Stats);
        Spatial.pos.set(a, 1.5, -2.25, 3.125, 0);
        Spatial.vel.set(a, 0.5, 1.5);
        Stats.hp.set(a, 42.5);
        Stats.level.set(a, 7);
        Stats.flags.set(a, 3);
        Stats.charge.set(a, -12);

        const b = state.create();
        state.add(b, Style);
        Style.tint.set(b, 0x3366cc);
        Style.mode.set(b, 2);

        // an entity whose only component sits entirely at defaults: serialize emits a bare `<a spatial />`,
        // and the round-trip must re-default it rather than drop the component
        const c = state.create();
        state.add(c, Spatial);

        const expected = snapshot(state);

        // these entities were created outside `load`, so they aren't in the authored set the no-arg
        // serialize captures — pass them explicitly (the "spawned outside load" override)
        const xml = stringify(serialize(state, [a, b, c]));
        const reloaded = new State();
        load(parse(xml), reloaded);

        expectSnapshotsMatch(expected, snapshot(reloaded));
    });

    test("load applies non-trivial defaults through state.add (codec double-defaults)", () => {
        // a component whose defaults are non-zero (white, not the zero-init slab default) — so a missing
        // defaults path reads as the zero-init value, not the trait's declared default
        registerSynthetic();
        const state = new State();
        load(parse(`<scene><a style /></scene>`), state);
        const eid = state.only([Style as never]);
        // Style defaults: tint = 0xffffff, mode = 0
        expect(Style.tint.get(eid)).toBe(0xffffff);
        expect(Style.mode.get(eid)).toBe(0);
    });

    test("a derived-trait component never serializes (a system owns it, scenes don't)", () => {
        clear();
        const Deco = { id: sparse(u32) };
        register("mark", Mark);
        register("deco", Deco, { defaults: () => ({ id: 0 }), derived: true });

        const state = new State();
        load(parse(`<scene><a id="thing" mark="v: 3" /></scene>`), state);
        const eid = state.only([Mark as never]);
        state.add(eid, Deco as never);
        Deco.id.set(eid, 7); // a non-default value — elision alone wouldn't hide it

        const xml = stringify(serialize(state));
        expect(xml).toContain("mark");
        expect(xml).not.toContain("deco");
    });
});

describe("NaN-sentinel default elision", () => {
    // RED witnessed: before b9dafde, formatFields emitted "roughness: NaN" on the alias-lane path
    // because `value === defaults[dotKey]` was false for NaN. Witnessed red: expected
    // not.toContain("NaN"), received "metallic: 0; roughness: NaN; emissive: 0; occlusion: 0".
    // b9dafde routed the comparison through atDefault (codec.ts), which treats a field at its NaN
    // default as default, so today a NaN-sentinel lane elides on the alias-lane path.
    test("NaN-sentinel default elides on the alias-lane path", () => {
        clear();
        const Mat = { params: sparse(vec4) };
        register("mat", Mat, {
            defaults: () => ({ params: [0, Number.NaN, 0, 0] }),
            aliases: {
                params: laneAlias("params", ["metallic", "roughness", "emissive", "occlusion"]),
            },
        });
        // params.y (roughness) sits at its NaN default — it should elide, not emit "roughness: NaN"
        const formatted = formatFields("mat", {
            "params.x": 0,
            "params.y": Number.NaN,
            "params.z": 0,
            "params.w": 0,
        });
        expect(formatted).not.toContain("NaN");
        // value half of the round trip: parse the emitted text back, merge with defaults
        // (elided lanes restore from defaults), and confirm the NaN-sentinel lane comes back
        // as NaN and the non-sentinel lanes at their input values
        const defaults = {
            "params.x": 0,
            "params.y": Number.NaN,
            "params.z": 0,
            "params.w": 0,
        };
        const restored: Record<string, number | string> = {
            ...defaults,
            ...parseFields("mat", formatted),
        };
        expect(Number.isNaN(restored["params.y"] as number)).toBe(true);
        expect(restored["params.x"]).toBe(0);
        expect(restored["params.z"]).toBe(0);
        expect(restored["params.w"]).toBe(0);
    });

    // RED witnessed: before b9dafde, formatFields emitted "pos: 0 0 0 NaN" on the positional
    // Pair/Quad path because `v === defaultValues[i]` was false for NaN. Witnessed red: expected
    // not.toContain("NaN"), received "pos: 0 0 0 NaN". b9dafde routed the comparison through
    // atDefault (codec.ts), which treats a field at its NaN default as default, so today a
    // NaN-sentinel lane elides on the positional Pair/Quad path.
    test("NaN-sentinel default elides on the positional Pair/Quad path", () => {
        clear();
        const Vec = { pos: sparse(vec4) };
        register("vec", Vec, {
            defaults: () => ({ pos: [0, 0, 0, Number.NaN] }),
        });
        // pos.w sits at its NaN default — it should elide, not emit "pos: 0 0 0 NaN"
        const formatted = formatFields("vec", {
            "pos.x": 0,
            "pos.y": 0,
            "pos.z": 0,
            "pos.w": Number.NaN,
        });
        expect(formatted).not.toContain("NaN");
        // value half of the round trip: parse back, merge with defaults, and confirm the
        // NaN-sentinel lane comes back as NaN and the non-sentinel lanes at their input values
        const defaults = {
            "pos.x": 0,
            "pos.y": 0,
            "pos.z": 0,
            "pos.w": Number.NaN,
        };
        const restored: Record<string, number | string> = {
            ...defaults,
            ...parseFields("vec", formatted),
        };
        expect(Number.isNaN(restored["pos.w"] as number)).toBe(true);
        expect(restored["pos.x"]).toBe(0);
        expect(restored["pos.y"]).toBe(0);
        expect(restored["pos.z"]).toBe(0);
    });

    test("NaN-sentinel default trims a trailing NaN lane on the positional path", () => {
        clear();
        const Vec = { pos: sparse(vec4) };
        register("vec", Vec, {
            defaults: () => ({ pos: [0, 0, 0, Number.NaN] }),
        });
        // pos.w is NaN (default), pos.x is non-default — the trailing NaN lane should trim
        const formatted = formatFields("vec", {
            "pos.x": 5,
            "pos.y": 0,
            "pos.z": 0,
            "pos.w": Number.NaN,
        });
        expect(formatted).not.toContain("NaN");
        expect(formatted).toBe("pos: 5 0 0");
        // value half: the non-sentinel lane (pos.x = 5) survives, and the elided NaN-sentinel
        // lane (pos.w) restores to NaN from defaults
        const defaults = {
            "pos.x": 0,
            "pos.y": 0,
            "pos.z": 0,
            "pos.w": Number.NaN,
        };
        const restored: Record<string, number | string> = {
            ...defaults,
            ...parseFields("vec", formatted),
        };
        expect(restored["pos.x"]).toBe(5);
        expect(Number.isNaN(restored["pos.w"] as number)).toBe(true);
    });

    // positive control: a non-NaN value on a NaN-default field does NOT elide — proves the
    // NaN-aware comparison doesn't over-elide valid values beside the sentinel
    test("a non-NaN value on a NaN-default field does not elide (positive control)", () => {
        clear();
        const Vec = { pos: sparse(vec4) };
        register("vec", Vec, {
            defaults: () => ({ pos: [0, 0, 0, Number.NaN] }),
        });
        // pos.w holds 3 (not the NaN default) — it should emit, not elide
        const formatted = formatFields("vec", {
            "pos.x": 0,
            "pos.y": 0,
            "pos.z": 0,
            "pos.w": 3,
        });
        expect(formatted).toContain("3");
        expect(formatted).not.toContain("NaN");
    });
});

const Link = { target: sparse(entity) };
const Mark = { v: sparse(u32) };

describe("serialize identity + refs", () => {
    test("an entity-ref field round-trips by scene id, not raw eid", () => {
        clear();
        register("link", Link, { defaults: () => ({ target: 0 }) });

        const state = new State();
        load(parse(`<scene><a id="anchor" /><a id="bob" link="target: @anchor" /></scene>`), state);

        const xml = stringify(serialize(state));
        // symbolic, not a literal creation-order eid
        expect(xml).toContain("@anchor");
        expect(xml).not.toContain("target: 1");

        const reloaded = new State();
        load(parse(xml), reloaded);

        // bob's target resolves to whatever eid "anchor" landed on this build
        const bob = reloaded.only([Link as never]);
        expect(reloaded.identity.id(Link.target.get(bob))).toBe("anchor");
    });

    test("an unset ref field (default eid 0) elides, with no spurious @-ref", () => {
        clear();
        register("link", Link, { defaults: () => ({ target: 0 }) });

        const state = new State();
        load(parse(`<scene><a id="lonely" link /></scene>`), state);

        const out = stringify(serialize(state));
        expect(out).not.toContain("@"); // nothing to resolve — the null-ref sentinel stays put
        expect(out).not.toContain("target"); // at its default, so it elides like any default field
    });

    test("a ref to an un-named target mints a scene id, surviving the eid shift a reload causes", () => {
        clear();
        register("link", Link, { defaults: () => ({ target: 0 }) });
        register("mark", Mark, { defaults: () => ({ v: 0 }) });

        const state = new State();
        state.create(); // a dummy left OUT of the serialized set, so reload compacts eids
        const anchor = state.create(); // no scene id — referenced only programmatically
        state.add(anchor, Mark);
        Mark.v.set(anchor, 77);
        const bob = state.create();
        state.add(bob, Link);
        Link.target.set(bob, anchor);

        // dropping the dummy shifts every eid on reload, so a raw-eid ref would now point at the
        // wrong entity — only the minted @-ref keyed on the target's id resolves correctly
        const xml = stringify(serialize(state, [anchor, bob]));
        expect(xml).toContain("@"); // a minted @-ref, not a literal eid

        const reloaded = new State();
        load(parse(xml), reloaded);

        const bob2 = reloaded.only([Link as never]);
        const target2 = Link.target.get(bob2);
        expect(target2).not.toBe(bob2);
        expect(Mark.v.get(target2)).toBe(77);
    });

    // RED witnessed: before e200a63, serialize(state, [a]) emitted "to: 2" (raw eid) instead of
    // "to: @b" because resolveRef returned undefined for a ref target outside the serialized
    // subset. Witnessed red: expected to contain "to: @b", received `arrow="to: 2"` — the raw eid
    // points at the wrong (recycled) entity on reload, breaking the round-trip-by-name contract.
    // e200a63 made resolveRef resolve an out-of-set target to its scene id (codec.ts), so today a
    // subset serialize emits `@name` for a target outside the serialized set; b9dafde later made
    // it throw when that target has no scene id (the sibling arm below).
    test("subset serialize emits @-ref for a target outside the serialized set", () => {
        clear();
        const Arrow = { to: sparse(entity) };
        register("arrow", Arrow, { defaults: () => ({ to: 0 }) });
        const state = new State();
        load(parse(`<scene><a id="a" arrow="to: @b" /><a id="b" /></scene>`), state);
        const a = state.only([Arrow as never]);
        expect(stringify(serialize(state, [a]))).toContain("to: @b");
    });

    // RED witnessed: before b9dafde, serialize(state, [a]) did not throw — resolveRef returned
    // undefined for a ref target outside the serialized set with no scene id, leaving a raw eid
    // in the output. Witnessed red: expected toThrow, received `arrow="to: 2"` — the raw eid
    // pointed at the wrong (recycled) entity on reload, silently breaking the round-trip-by-name
    // contract. b9dafde made resolveRef throw for a destroyed or unnamed out-of-set target
    // (codec.ts), so today serialize fails loud instead of emitting a raw eid.
    test("serialize throws on a ref to a destroyed entity (not a raw eid)", () => {
        clear();
        const Arrow = { to: sparse(entity) };
        register("arrow", Arrow, { defaults: () => ({ to: 0 }) });
        const state = new State();
        const a = state.create();
        state.add(a, Arrow);
        const b = state.create();
        Arrow.to.set(a, b);
        state.destroy(b);
        // b is destroyed — the ref target is genuinely unresolvable, not merely absent
        expect(() => serialize(state, [a])).toThrow(/destroyed/);
    });

    test("serialize throws on a ref to an unnamed entity outside the serialized set", () => {
        clear();
        const Arrow = { to: sparse(entity) };
        register("arrow", Arrow, { defaults: () => ({ to: 0 }) });
        const state = new State();
        const a = state.create();
        state.add(a, Arrow);
        const b = state.create(); // alive, no scene id, outside the serialized set
        Arrow.to.set(a, b);
        // b is alive but outside the set and unnamed — the ref cannot be expressed as @name
        expect(() => serialize(state, [a])).toThrow(/outside the serialized set/);
    });

    test("captures authored entities, excludes warm-derived ones — a restore never doubles them", async () => {
        clear();
        const Tree = { kind: sparse(u32) };
        const Plot = { count: sparse(u32) };
        const Grove: Plugin = {
            name: "grove",
            components: { Tree, Plot },
            // warm re-derives a Tree per authored Plot every build
            warm(state) {
                for (const plot of state.query([Plot])) {
                    const t = state.create();
                    state.add(t, Tree);
                    Tree.kind.set(t, Plot.count.get(plot));
                }
            },
        };

        const scene = `<scene><a id="meadow" plot="count: 3" /></scene>`;
        const { state } = await build({ plugins: [Grove], defaults: false, scene });

        // 1 authored Plot + 1 warm-derived Tree
        expect(state.entities().length).toBe(2);
        expect([...state.query([Tree])].length).toBe(1);

        const xml = stringify(serialize(state));
        expect(xml).toContain("plot");
        expect(xml).not.toContain("tree"); // the derived entity is absent by construction

        // restore: load(Plot) re-creates the authored side, warm(Tree) re-derives — 2 again, not 3
        const restored = await build({ plugins: [Grove], defaults: false, scene: xml });
        expect(restored.state.entities().length).toBe(2);
        expect([...restored.state.query([Tree])].length).toBe(1);
        expect(Plot.count.get(restored.state.only([Plot]))).toBe(3);
    });
});
