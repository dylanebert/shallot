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
    "showcase/fountain/public/scenes/fountain.scene",
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
        test.each(SCENE_FILES)("component fields survive roundtrip: %s", async (file) => {
            const xml = await readScene(file);
            const nodes = parse(xml);

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
        });
    });

    describe("normalization idempotence", () => {
        test.each(SCENE_FILES)("normalizeAttr is idempotent: %s", async (file) => {
            const xml = await readScene(file);
            const nodes = parse(xml);

            function checkNode(node: Node) {
                for (const attr of node.attrs) {
                    if (!attr.value) continue;
                    const first = normalizeAttr(attr.name, attr.value);
                    // null = unregistered; "" = all fields at default (the bare form), a terminal that
                    // re-normalizes to null. Idempotence is the fixed-point claim on a non-empty result.
                    if (first === null || first === "") continue;
                    const second = normalizeAttr(attr.name, first);
                    expect(second).toBe(first);
                }
                for (const child of node.children) checkNode(child);
            }

            for (const node of nodes) checkNode(node);
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

const Link = { target: sparse(entity) };
const Mark = { v: sparse(u32) };

describe("serialize identity + refs (Stage 6)", () => {
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
