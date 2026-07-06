import { beforeEach, describe, expect, test } from "bun:test";
import { attach } from "../../../tests/helpers";
import {
    diagnose,
    f32,
    laneAlias,
    load,
    type Plugin,
    parse,
    State,
    sparse,
    stringify,
    u32,
    vec2,
    vec4,
} from "../..";
import { clear, readFields, register, schema } from "../ecs/core";
import { formatFields, normalizeAttr, parseFields } from "./core";

describe("XML", () => {
    beforeEach(() => {
        clear();
    });

    describe("parse", () => {
        test("parses entities flat", () => {
            const Position = { x: [] as number[], y: [] as number[], z: [] as number[] };
            register("position", Position);

            const nodes = parse(`
                <scene>
                    <a id="first" />
                    <a id="second" />
                    <a id="third" />
                </scene>
            `);
            expect(nodes).toHaveLength(3);
            expect(nodes[0].id).toBe("first");
            expect(nodes[1].id).toBe("second");
            expect(nodes[2].id).toBe("third");
        });

        test("rejects nested entities", () => {
            expect(() => parse(`<scene><a id="parent"><a id="child"></a></a></scene>`)).toThrow();
        });

        test("parses component attributes", () => {
            const nodes = parse(`
                <scene>
                    <a position="x: 10; y: 20" />
                    <a position />
                </scene>
            `);
            expect(nodes[0].attrs[0].name).toBe("position");
            expect(nodes[0].attrs[0].value).toBe("x: 10; y: 20");
            expect(nodes[1].attrs[0].name).toBe("position");
            expect(nodes[1].attrs[0].value).toBe("");
        });

        test("rejects invalid tags", () => {
            expect(() => parse(`<Scene></Scene>`)).toThrow("lowercase");
            expect(() => parse(`<scene><A /></scene>`)).toThrow("lowercase");
            expect(() => parse(`<scene><unclosed>`)).toThrow("xml parse error");
        });

        test("stores raw attributes", () => {
            const nodes = parse(`<scene><a foo="bar" transform /></scene>`);
            expect(nodes[0].attrs[0].name).toBe("foo");
            expect(nodes[0].attrs[0].value).toBe("bar");
            expect(nodes[0].attrs[1].name).toBe("transform");
        });
    });

    describe("load", () => {
        const Position = { x: [] as number[], y: [] as number[], z: [] as number[] };

        const TestPlugin: Plugin = {
            name: "Test",
            components: { Position },
            traits: { Position: { defaults: () => ({ x: 0, y: 0, z: 0 }) } },
        };

        let state: State;

        beforeEach(() => {
            clear();
            state = new State();
            for (const [n, c] of Object.entries(TestPlugin.components ?? {}))
                register(n, c, TestPlugin.traits?.[n]);
            attach(state, TestPlugin);
        });

        test("creates entities with components", () => {
            const nodes = parse(`<scene><a id="player" position="x: 10; y: 20; z: 30" /></scene>`);
            const nodeToEntity = load(nodes, state);

            const eid = nodeToEntity.get(nodes[0])!;
            expect(state.exists(eid)).toBe(true);
            expect(Position.x[eid]).toBe(10);
            expect(Position.y[eid]).toBe(20);
            expect(Position.z[eid]).toBe(30);
        });

        test("rejects nested scene children", () => {
            expect(() => parse(`<scene><a id="parent"><a id="child"></a></a></scene>`)).toThrow();
        });

        test("suggests correct field name for typos", () => {
            const nodes = parse(`<scene><a position="xx: 10" /></scene>`);
            expect(() => load(nodes, state)).toThrow('did you mean "x"');
        });

        test("skips unknown component", () => {
            const nodes = parse(`<scene><a id="ball" unknown="value: 1" /></scene>`);
            const map = load(nodes, state);
            expect(map.size).toBe(1);
        });

        test("skips unknown component with empty value", () => {
            const nodes = parse(`<scene><a unknown-tag /></scene>`);
            const map = load(nodes, state);
            expect(map.size).toBe(1);
        });

        test("parse trait resolves string value to numeric ID", () => {
            const Comp = { mode: [] as number[] };
            const lookup = new Map([
                ["fast", 1],
                ["slow", 2],
            ]);
            register("Comp", Comp, { parse: { mode: (v: string) => lookup.get(v) } });

            const nodes = parse(`<scene><a comp="mode: fast" /></scene>`);
            const nodeToEntity = load(nodes, state);
            const eid = nodeToEntity.get(nodes[0])!;
            expect(Comp.mode[eid]).toBe(1);
        });

        test("numeric values still work with parse present", () => {
            const Comp = { mode: [] as number[] };
            register("Comp", Comp, { parse: { mode: () => undefined } });

            const nodes = parse(`<scene><a comp="mode: 42" /></scene>`);
            const nodeToEntity = load(nodes, state);
            const eid = nodeToEntity.get(nodes[0])!;
            expect(Comp.mode[eid]).toBe(42);
        });

        test("unknown parse value errors", () => {
            const Comp = { mode: [] as number[] };
            register("Comp", Comp, { parse: { mode: () => undefined } });

            const nodes = parse(`<scene><a comp="mode: unknown" /></scene>`);
            expect(() => load(nodes, state)).toThrow("Invalid number");
        });

        test("traits.parse handles multi-token non-numeric values", () => {
            const labels: string[] = [];
            const Comp = { label: [] as number[], size: [] as number[] };
            register("Comp", Comp, {
                defaults: () => ({ size: 1 }),
                parse: {
                    label: (raw: string) => {
                        const existing = labels.indexOf(raw);
                        if (existing >= 0) return existing;
                        labels.push(raw);
                        return labels.length - 1;
                    },
                },
            });

            const nodes = parse(`<scene><a comp="label: hello world; size: 5" /></scene>`);
            const nodeToEntity = load(nodes, state);
            const eid = nodeToEntity.get(nodes[0])!;
            expect(labels[Comp.label[eid]]).toBe("hello world");
            expect(Comp.size[eid]).toBe(5);
        });
    });

    describe("entity-ref fields", () => {
        let state: State;
        const Link = { target: [] as number[] };

        beforeEach(() => {
            clear();
            state = new State();
            register("Link", Link, { defaults: () => ({ target: 0 }) });
        });

        test("resolves component-field ref to target entity", () => {
            const nodes = parse(
                `<scene><a id="cube" /><a id="tween" link="target: @cube" /></scene>`,
            );
            const nodeToEntity = load(nodes, state);

            const cube = nodeToEntity.get(nodes[0])!;
            const tween = nodeToEntity.get(nodes[1])!;
            expect(Link.target[tween]).toBe(cube);
        });

        test("resolves forward references", () => {
            const nodes = parse(
                `<scene><a id="tween" link="target: @cube" /><a id="cube" /></scene>`,
            );
            const nodeToEntity = load(nodes, state);

            const tween = nodeToEntity.get(nodes[0])!;
            const cube = nodeToEntity.get(nodes[1])!;
            expect(Link.target[tween]).toBe(cube);
        });

        test("errors on unknown entity reference", () => {
            const nodes = parse(`<scene><a link="target: @missing" /></scene>`);
            expect(() => load(nodes, state)).toThrow("@missing");
        });
    });

    describe("stringify", () => {
        test("serializes simple nodes", () => {
            const Position = { x: [] as number[], y: [] as number[], z: [] as number[] };
            register("position", Position);

            const nodes = parse(`<scene><a id="player" position="x: 10" /></scene>`);
            const xml = stringify(nodes);

            expect(xml).toContain("<scene>");
            expect(xml).toContain("</scene>");
            expect(xml).toContain('id="player"');
            expect(xml).toContain("position");
        });

        test("serializes flat entities", () => {
            const Position = { x: [] as number[], y: [] as number[], z: [] as number[] };
            register("position", Position);

            const nodes = parse(`<scene><a id="first" /><a id="second" /></scene>`);
            const xml = stringify(nodes);

            expect(xml).toContain('id="first"');
            expect(xml).toContain('id="second"');
        });

        test("round-trip preserves structure", () => {
            const Position = { x: [] as number[], y: [] as number[], z: [] as number[] };
            register("position", Position);

            const original = parse(
                `<scene><a id="a" position="x: 1" /><a id="b" /><a id="c" /></scene>`,
            );
            const xml = stringify(original);
            const reparsed = parse(xml);

            expect(reparsed.length).toBe(original.length);
            expect(reparsed[0].id).toBe(original[0].id);
            expect(reparsed[1].id).toBe(original[1].id);
            expect(reparsed[2].id).toBe(original[2].id);
        });

        test("serializes entity-ref field values as @-refs", () => {
            const nodes = parse(
                `<scene><a id="cube" /><a id="tween" link="target: @cube" /></scene>`,
            );
            const xml = stringify(nodes);

            expect(xml).toContain("@cube");
        });
    });

    describe("formatFields", () => {
        beforeEach(() => {
            clear();
        });

        test("round-trips with parseFields", () => {
            const Comp = { x: [] as number[], y: [] as number[], z: [] as number[] };
            register("comp", Comp, { defaults: () => ({ x: 0, y: 0, z: 0 }) });

            const fields = { x: 10, y: 20, z: 30 };
            const str = formatFields("comp", fields);
            const parsed = parseFields("comp", str);

            expect(parsed.x).toBe(10);
            expect(parsed.y).toBe(20);
            expect(parsed.z).toBe(30);
        });

        test("omits default values", () => {
            const Comp = { x: [] as number[], y: [] as number[] };
            register("comp", Comp, { defaults: () => ({ x: 0, y: 1 }) });

            const str = formatFields("comp", { x: 0, y: 1 });
            expect(str).toBe("");
        });

        // a NaN sentinel default (Tween.from = "capture at runtime") must elide like any other default.
        // NaN !== NaN, so a plain `value === default` check missed it and emitted `from: NaN` — a scene
        // that throws on re-parse. The field at its NaN default drops out and re-defaults on load.
        test("omits a field sitting at its NaN sentinel default", () => {
            const Comp = { from: [] as number[], to: [] as number[] };
            register("comp", Comp, { defaults: () => ({ from: Number.NaN, to: 0 }) });

            const str = formatFields("comp", { from: Number.NaN, to: 5 });
            expect(str).toBe("to: 5");
            expect(parseFields("comp", str)).toEqual({ to: 5 });
        });

        test("uses format trait for named lookups", () => {
            const Comp = { mode: [] as number[] };
            const names = new Map([
                [1, "fast"],
                [2, "slow"],
            ]);
            register("comp", Comp, {
                defaults: () => ({ mode: 0 }),
                parse: { mode: (v: string) => (v === "fast" ? 1 : v === "slow" ? 2 : undefined) },
                format: { mode: (v: number) => names.get(v) },
            });

            const str = formatFields("comp", { mode: 1 });
            expect(str).toBe("mode: fast");
        });

        test("format trait hex round-trip", () => {
            const Comp = { color: [] as number[] };
            const formatHex = (n: number) => "0x" + (n >>> 0).toString(16).padStart(6, "0");
            register("comp", Comp, {
                defaults: () => ({ color: 0xffffff }),
                format: { color: formatHex },
            });

            const str = formatFields("comp", { color: 0xff8800 });
            expect(str).toBe("color: 0xff8800");

            const parsed = parseFields("comp", str);
            expect(parsed.color).toBe(0xff8800);
        });

        test("errors on unknown component", () => {
            expect(() => formatFields("nonexistent", { x: 1 })).toThrow("Unknown component");
        });

        test("stripDefaults: false preserves default-valued scalars", () => {
            const Comp = { x: [] as number[], y: [] as number[] };
            register("comp", Comp, { defaults: () => ({ x: 0, y: 1 }) });

            const str = formatFields("comp", { x: 0, y: 1 }, { stripDefaults: false });
            expect(str).toBe("x: 0; y: 1");
        });
    });

    describe("normalizeAttr", () => {
        beforeEach(() => {
            clear();
        });

        test("returns null for unknown components", () => {
            expect(normalizeAttr("nonexistent", "x: 1")).toBeNull();
        });

        test("returns null for empty value", () => {
            register("comp", { x: [] as number[] });
            expect(normalizeAttr("comp", "")).toBeNull();
        });

        test("preserves authored field set without injecting defaults", () => {
            const Comp = { x: [] as number[], y: [] as number[] };
            register("comp", Comp, { defaults: () => ({ x: 0, y: 1 }) });

            const result = normalizeAttr("comp", "x: 5");
            expect(result).toBe("x: 5");
            expect(result).not.toContain("y:");
        });

        test("normalizes precision", () => {
            const Comp = { x: [] as number[] };
            register("comp", Comp, { defaults: () => ({ x: 0 }) });

            expect(normalizeAttr("comp", "x: 0.50000000")).toBe("x: 0.5");
        });

        test("normalizes kebab-case", () => {
            const Comp = { myField: [] as number[] };
            register("comp", Comp, { defaults: () => ({ myField: 0 }) });

            expect(normalizeAttr("comp", "myField: 5")).toBe("my-field: 5");
        });

        test("is idempotent", () => {
            const Transform = {
                posX: [] as number[],
                posY: [] as number[],
                posZ: [] as number[],
            };
            register("transform", Transform, { defaults: () => ({ posX: 0, posY: 0, posZ: 0 }) });

            const first = normalizeAttr("transform", "pos: 1 2 3");
            const second = normalizeAttr("transform", first!);
            expect(second).toBe(first);
        });

        test("strips fields sitting at their default — the minimal form serialize emits", () => {
            const Comp = { x: [] as number[], y: [] as number[] };
            register("comp", Comp, { defaults: () => ({ x: 0, y: 1 }) });

            // both fields at their default elide to the empty (bare-component) form
            expect(normalizeAttr("comp", "x: 0; y: 1")).toBe("");
            // a non-default field is kept; a sibling sitting at its default still elides
            expect(normalizeAttr("comp", "x: 5; y: 1")).toBe("x: 5");
        });
    });

    describe("direct Pair/Quad component shape — scene parsing", () => {
        const Pose = {
            pos: sparse(vec4),
            rot: sparse(vec4),
            velocity: sparse(vec2),
            health: sparse(f32),
            flags: sparse(u32),
        };

        const PosePlugin: Plugin = {
            name: "Pose",
            components: { Pose },
            traits: {
                Pose: {
                    defaults: () => ({
                        pos: [0, 0, 0, 0],
                        rot: [0, 0, 0, 1],
                        velocity: [0, 0],
                        health: 100,
                        flags: 0,
                    }),
                },
            },
        };

        let state: State;

        beforeEach(() => {
            state = new State();
            register("Pose", Pose, PosePlugin.traits!.Pose);
            attach(state, PosePlugin);
        });

        test("Quad accepts 3-, 4-, or 1-value (broadcast) input; rejects 2", () => {
            const nodes = parse(
                `<scene>
                    <a id="three" pose="pos: 1 2 3" />
                    <a id="four" pose="rot: 0.5 0.5 0.5 0.5" />
                    <a id="bcast" pose="pos: 5" />
                </scene>`,
            );
            const map = load(nodes, state);
            const three = map.get(nodes[0])!;
            const four = map.get(nodes[1])!;
            const bcast = map.get(nodes[2])!;

            expect(Pose.pos.x.get(three)).toBe(1);
            expect(Pose.pos.y.get(three)).toBe(2);
            expect(Pose.pos.z.get(three)).toBe(3);

            expect(Pose.rot.x.get(four)).toBe(0.5);
            expect(Pose.rot.w.get(four)).toBe(0.5);

            expect(Pose.pos.x.get(bcast)).toBe(5);
            expect(Pose.pos.w.get(bcast)).toBe(5);

            expect(() => load(parse(`<scene><a pose="pos: 1 2" /></scene>`), state)).toThrow();
        });

        test("Pair accepts 2 values", () => {
            const nodes = parse(`<scene><a pose="velocity: 3 4" /></scene>`);
            const eid = load(nodes, state).get(nodes[0])!;
            expect(Pose.velocity.x.get(eid)).toBe(3);
            expect(Pose.velocity.y.get(eid)).toBe(4);
        });

        test("dotted lane attribute writes one lane, leaves siblings at default", () => {
            const nodes = parse(`<scene><a pose="pos.x: 7; pos.z: 9" /></scene>`);
            const eid = load(nodes, state).get(nodes[0])!;
            expect(Pose.pos.x.get(eid)).toBe(7);
            expect(Pose.pos.y.get(eid)).toBe(0);
            expect(Pose.pos.z.get(eid)).toBe(9);
        });

        test("array-form defaults apply on state.add", () => {
            const eid = state.create();
            state.add(eid, Pose);
            expect(Pose.pos.x.get(eid)).toBe(0);
            expect(Pose.rot.w.get(eid)).toBe(1);
            expect(Pose.velocity.x.get(eid)).toBe(0);
            expect(Pose.health.get(eid)).toBe(100);
            expect(Pose.flags.get(eid)).toBe(0);
        });
    });

    describe("named-lane authoring via an identity alias — scene parsing", () => {
        const Mat = { params: sparse(vec4) };
        const MatPlugin: Plugin = {
            name: "Mat",
            components: { Mat },
            traits: {
                Mat: {
                    defaults: () => ({ params: [0, 1, 0, 1] }),
                    aliases: {
                        params: laneAlias("params", [
                            "metallic",
                            "roughness",
                            "emissive",
                            "occlusion",
                        ]),
                    },
                },
            },
        };

        let state: State;

        beforeEach(() => {
            state = new State();
            register("Mat", Mat, MatPlugin.traits!.Mat);
            attach(state, MatPlugin);
        });

        test("named axes write their lanes; unset lanes stay at default", () => {
            const nodes = parse(`<scene><a mat="metallic: 1; roughness: 0.25" /></scene>`);
            const eid = load(nodes, state).get(nodes[0])!;
            expect(Mat.params.x.get(eid)).toBe(1); // metallic
            expect(Mat.params.y.get(eid)).toBe(0.25); // roughness
            expect(Mat.params.z.get(eid)).toBe(0); // emissive (default)
            expect(Mat.params.w.get(eid)).toBe(1); // occlusion (default)
        });

        test("positional authoring still works alongside the named form", () => {
            const nodes = parse(`<scene><a mat="params: 1 0.25 0.5 0.75" /></scene>`);
            const eid = load(nodes, state).get(nodes[0])!;
            expect(Mat.params.x.get(eid)).toBe(1);
            expect(Mat.params.z.get(eid)).toBe(0.5);
            expect(Mat.params.w.get(eid)).toBe(0.75);
        });

        test("serialize emits named keys (non-defaults only) and round-trips", () => {
            const nodes = parse(`<scene><a mat="metallic: 1; roughness: 0.25" /></scene>`);
            const eid = load(nodes, state).get(nodes[0])!;
            const attr = formatFields("mat", readFields(Mat, eid));
            expect(attr).toContain("metallic: 1");
            expect(attr).toContain("roughness: 0.25");
            expect(attr).not.toContain("params");
            expect(attr).not.toContain("emissive"); // at default → elided
            expect(attr).not.toContain("occlusion");
            const parsed = parseFields("mat", attr);
            expect(parsed["params.x"]).toBe(1);
            expect(parsed["params.y"]).toBe(0.25);
        });

        test("a flat material serializes to empty (every lane at its default)", () => {
            const eid = state.create();
            state.add(eid, Mat);
            expect(formatFields("mat", readFields(Mat, eid))).toBe("");
        });
    });

    describe("direct Pair/Quad — parseFields / formatFields / schema / readFields", () => {
        const Item = {
            pos: sparse(vec4),
            velocity: sparse(vec2),
            speed: sparse(f32),
        };

        const ItemPlugin: Plugin = {
            name: "Item",
            components: { Item },
            traits: {
                Item: {
                    defaults: () => ({
                        pos: [0, 0, 0, 0],
                        velocity: [0, 0],
                        speed: 1,
                    }),
                },
            },
        };

        beforeEach(() => {
            register("Item", Item, ItemPlugin.traits!.Item);
        });

        test("parseFields emits dotted lane keys", () => {
            const fields = parseFields("Item", "pos: 1 2 3 4; velocity: 5 6; speed: 7");
            expect(fields["pos.x"]).toBe(1);
            expect(fields["pos.w"]).toBe(4);
            expect(fields["velocity.y"]).toBe(6);
            expect(fields.speed).toBe(7);
        });

        test("formatFields strips trailing-default lanes and collapses equal lanes", () => {
            // vec4 used as vec3: trailing default w lane elides
            expect(
                formatFields("Item", {
                    "pos.x": 1,
                    "pos.y": 2,
                    "pos.z": 3,
                    "pos.w": 0,
                    "velocity.x": 0,
                    "velocity.y": 0,
                    speed: 1,
                }),
            ).toBe("pos: 1 2 3");

            // all lanes equal → single scalar
            expect(
                formatFields("Item", {
                    "pos.x": 5,
                    "pos.y": 5,
                    "pos.z": 5,
                    "pos.w": 5,
                    "velocity.x": 0,
                    "velocity.y": 0,
                    speed: 1,
                }),
            ).toBe("pos: 5");
        });

        test("roundtrip preserves all values", () => {
            const fields = parseFields("Item", "pos: 1.5 2.5 3.5 4.5; velocity: -1 0; speed: 12");
            const reparsed = parseFields(
                "Item",
                formatFields("Item", fields, { stripDefaults: false }),
            );
            // these survive formatNumber's toPrecision(7) without rounding
            expect(reparsed["pos.x"]).toBe(1.5);
            expect(reparsed["pos.w"]).toBe(4.5);
            expect(reparsed["velocity.x"]).toBe(-1);
            expect(reparsed.speed).toBe(12);
        });

        test("vec3-shaped Quad input roundtrips via bulk form", () => {
            // user authors 3 numbers on a Quad; formatter must reassemble
            // to `pos: 1 2 3`, not unroll to `pos.x: 1; pos.y: 2; pos.z: 3`
            expect(
                formatFields("Item", parseFields("Item", "pos: 1 2 3"), { stripDefaults: false }),
            ).toBe("pos: 1 2 3");
        });

        test("sparse lanes fall through to dotted emission", () => {
            // single lane or non-prefix selection: bulk form would change
            // semantics by writing defaults to the trailing lanes
            expect(formatFields("Item", { "pos.y": 5 }, { stripDefaults: false })).toBe("pos.y: 5");
            expect(formatFields("Item", { "pos.x": 1, "pos.z": 3 }, { stripDefaults: false })).toBe(
                "pos.x: 1; pos.z: 3",
            );
        });

        test("schema reports direct Pair/Quad shape with dotted lane field names", () => {
            const s = schema("Item")!;
            const pos = s.fields.find((f) => f.name === "pos")!;
            const vel = s.fields.find((f) => f.name === "velocity")!;
            expect(pos.kind).toBe("vec4");
            expect(pos.fields).toEqual(["pos.x", "pos.y", "pos.z", "pos.w"]);
            expect(vel.kind).toBe("vec2");
            expect(vel.fields).toEqual(["velocity.x", "velocity.y"]);
            expect(s.fields.find((f) => f.name === "speed")?.kind).toBe("float");
        });

        test("readFields emits dotted lane keys", () => {
            const state = new State();
            const eid = state.create();
            state.add(eid, Item);
            Item.pos.set(eid, 1, 2, 3, 4);
            Item.velocity.set(eid, 5, 6);
            Item.speed.set(eid, 7);

            const fields = readFields(Item, eid);
            expect(fields["pos.w"]).toBe(4);
            expect(fields["velocity.y"]).toBe(6);
            expect(fields.speed).toBe(7);
        });
    });

    describe("format idempotence", () => {
        test("stringify(parse(xml)) is idempotent", () => {
            const xml = `<scene>
    <a id="first" />
    <a id="second" />
    <a id="third" />
</scene>`;
            const once = stringify(parse(xml));
            const twice = stringify(parse(once));
            expect(twice).toBe(once);
        });
    });

    describe("diagnose", () => {
        test("reports missing requires", () => {
            const A = { value: [] as number[] };
            const B = { value: [] as number[] };
            register("beta", B);
            register("alpha", A, { requires: [B] });

            const nodes = parse(`<scene><a alpha /></scene>`);
            const results = diagnose(nodes);
            expect(results).toHaveLength(1);
            expect(results[0].kind).toBe("missing-requires");
            expect(results[0].message).toContain("beta");
        });

        test("no missing-requires when dependency present", () => {
            const A = { value: [] as number[] };
            const B = { value: [] as number[] };
            register("beta", B);
            register("alpha", A, { requires: [B] });

            const nodes = parse(`<scene><a alpha beta /></scene>`);
            const results = diagnose(nodes);
            expect(results).toHaveLength(0);
        });

        test("a provider satisfies a requires on the same entity", () => {
            const Xf = { value: [] as number[] };
            const Bod = { value: [] as number[] };
            const Prt = { value: [] as number[] };
            register("xf", Xf);
            register("bod", Bod, { provides: [Xf] }); // Body-provides-Transform shape
            register("prt", Prt, { requires: [Xf] }); // Part-requires-Transform shape

            // bod provides xf, so prt's requires is met without an explicit xf attr
            expect(diagnose(parse(`<scene><a bod prt /></scene>`))).toHaveLength(0);
            // but a bare prt (no provider) still flags
            const bare = diagnose(parse(`<scene><a prt /></scene>`));
            expect(bare).toHaveLength(1);
            expect(bare[0].kind).toBe("missing-requires");
        });

        test("flags an authored derived-trait attr (a system owns it)", () => {
            register("deco", { id: [] as number[] }, { derived: true });

            const results = diagnose(parse(`<scene><a deco="id: 3" /></scene>`));
            expect(results).toHaveLength(1);
            expect(results[0].kind).toBe("derived");
        });

        test("reports unregistered components", () => {
            register("known", { value: [] as number[] });

            const nodes = parse(`<scene><a unknown-thing /></scene>`);
            const results = diagnose(nodes);

            expect(results).toHaveLength(1);
            expect(results[0].kind).toBe("unregistered");
        });

        test("reports excluded-with once per pair when both attrs present", () => {
            const Slab = { value: [] as number[] };
            const Body = { value: [] as number[] };
            register("slab", Slab);
            register("body", Body, { excludes: [Slab] });

            const nodes = parse(`<scene><a body slab /></scene>`);
            const results = diagnose(nodes);

            expect(results).toHaveLength(1);
            expect(results[0].kind).toBe("excluded-with");
            expect(results[0].message).toContain("body");
            expect(results[0].message).toContain("slab");
        });

        test("no excluded-with diagnostic when only one excluder is present", () => {
            const Slab = { value: [] as number[] };
            const Body = { value: [] as number[] };
            register("slab", Slab);
            register("body", Body, { excludes: [Slab] });

            const nodes = parse(`<scene><a body /></scene>`);
            const results = diagnose(nodes);
            expect(results).toHaveLength(0);
        });
    });
});
