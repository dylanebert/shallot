import { test, expect, describe, beforeEach } from "bun:test";
import { pair } from "../src/engine/ecs";
import {
    State,
    parse,
    load,
    serialize,
    parseFields,
    formatFields,
    normalizeAttr,
    diagnose,
    ChildOf,
    traits,
    relation,
    type Plugin,
} from "../src";
import { clearRegistry, registerComponent } from "../src/engine/ecs/component";
import { clearRelations } from "../src/engine/ecs/relation";

describe("XML", () => {
    beforeEach(() => {
        clearRegistry();
        clearRelations();
    });

    describe("parse", () => {
        test("parses entities with hierarchy", () => {
            const Position = { x: [] as number[], y: [] as number[], z: [] as number[] };
            registerComponent("position", Position);

            const nodes = parse(`
                <scene>
                    <a id="parent">
                        <a id="child"></a>
                    </a>
                    <a id="sibling"></a>
                </scene>
            `);
            expect(nodes).toHaveLength(2);
            expect(nodes[0].id).toBe("parent");
            expect(nodes[0].children[0].id).toBe("child");
            expect(nodes[1].id).toBe("sibling");
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
        traits(Position, { defaults: () => ({ x: 0, y: 0, z: 0 }) });

        const TestPlugin: Plugin = { name: "Test", components: { Position } };

        let state: State;

        beforeEach(() => {
            clearRegistry();
            state = new State();
            state.register(TestPlugin);
        });

        test("creates entities with components", () => {
            const nodes = parse(`<scene><a id="player" position="x: 10; y: 20; z: 30" /></scene>`);
            const nodeToEntity = load(nodes, state);

            const eid = nodeToEntity.get(nodes[0])!;
            expect(state.entityExists(eid)).toBe(true);
            expect(Position.x[eid]).toBe(10);
            expect(Position.y[eid]).toBe(20);
            expect(Position.z[eid]).toBe(30);
        });

        test("creates hierarchy with ChildOf relation", () => {
            const nodes = parse(`<scene><a id="parent"><a id="child"></a></a></scene>`);
            const nodeToEntity = load(nodes, state);

            const parent = nodeToEntity.get(nodes[0])!;
            const child = nodeToEntity.get(nodes[0].children[0])!;
            expect(state.hasComponent(child, pair(ChildOf.relation, parent))).toBe(true);
        });

        test("maps vec3 shorthand to component fields", () => {
            const Transform = { posX: [] as number[], posY: [] as number[], posZ: [] as number[] };
            const TransformPlugin: Plugin = { name: "Transform", components: { Transform } };
            state.register(TransformPlugin);

            const nodes = parse(`<scene><a id="p" transform="pos: 1 2 3" /></scene>`);
            const nodeToEntity = load(nodes, state);
            const eid = nodeToEntity.get(nodes[0])!;
            expect(Transform.posX[eid]).toBe(1);
            expect(Transform.posY[eid]).toBe(2);
            expect(Transform.posZ[eid]).toBe(3);
        });

        test("broadcasts single value to all axes", () => {
            const Transform = { posX: [] as number[], posY: [] as number[], posZ: [] as number[] };
            const TransformPlugin: Plugin = { name: "Transform", components: { Transform } };
            state.register(TransformPlugin);

            const nodes = parse(`<scene><a id="p" transform="pos: 2" /></scene>`);
            const nodeToEntity = load(nodes, state);
            const eid = nodeToEntity.get(nodes[0])!;
            expect(Transform.posX[eid]).toBe(2);
            expect(Transform.posY[eid]).toBe(2);
            expect(Transform.posZ[eid]).toBe(2);
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
            traits(Comp, { parse: { mode: (v: string) => lookup.get(v) } });
            state.register({ name: "Comp", components: { Comp } });

            const nodes = parse(`<scene><a comp="mode: fast" /></scene>`);
            const nodeToEntity = load(nodes, state);
            const eid = nodeToEntity.get(nodes[0])!;
            expect(Comp.mode[eid]).toBe(1);
        });

        test("numeric values still work with parse present", () => {
            const Comp = { mode: [] as number[] };
            traits(Comp, { parse: { mode: () => undefined } });
            state.register({ name: "Comp", components: { Comp } });

            const nodes = parse(`<scene><a comp="mode: 42" /></scene>`);
            const nodeToEntity = load(nodes, state);
            const eid = nodeToEntity.get(nodes[0])!;
            expect(Comp.mode[eid]).toBe(42);
        });

        test("unknown parse value errors", () => {
            const Comp = { mode: [] as number[] };
            traits(Comp, { parse: { mode: () => undefined } });
            state.register({ name: "Comp", components: { Comp } });

            const nodes = parse(`<scene><a comp="mode: unknown" /></scene>`);
            expect(() => load(nodes, state)).toThrow("Invalid number");
        });

        test("string proxy field gets string value set directly", () => {
            const content = new Map<number, string>();
            const contentProxy = new Proxy(
                {},
                {
                    get(_, prop) {
                        return content.get(Number(prop));
                    },
                    set(_, prop, value) {
                        content.set(Number(prop), value);
                        return true;
                    },
                },
            );
            const Comp = { label: contentProxy, size: [] as number[] };
            traits(Comp, { defaults: () => ({ size: 1 }) });
            state.register({ name: "Comp", components: { Comp } });

            const nodes = parse(`<scene><a comp="label: hello world; size: 5" /></scene>`);
            const nodeToEntity = load(nodes, state);
            const eid = nodeToEntity.get(nodes[0])!;
            expect(content.get(eid)).toBe("hello world");
            expect(Comp.size[eid]).toBe(5);
        });
    });

    describe("relations", () => {
        let state: State;

        beforeEach(() => {
            clearRegistry();
            clearRelations();
            state = new State();
        });

        test("resolves relation to target entity", () => {
            const Targets = relation("targets", { exclusive: true });

            const nodes = parse(`<scene><a id="cube" /><a id="tween" targets="@cube" /></scene>`);
            const nodeToEntity = load(nodes, state);

            const cube = nodeToEntity.get(nodes[0])!;
            const tween = nodeToEntity.get(nodes[1])!;
            expect(state.hasRelation(tween, Targets, cube)).toBe(true);
        });

        test("resolves forward references", () => {
            const Targets = relation("targets");

            const nodes = parse(`<scene><a id="tween" targets="@cube" /><a id="cube" /></scene>`);
            const nodeToEntity = load(nodes, state);

            const tween = nodeToEntity.get(nodes[0])!;
            const cube = nodeToEntity.get(nodes[1])!;
            expect(state.hasRelation(tween, Targets, cube)).toBe(true);
        });

        test("errors on unknown entity reference", () => {
            relation("targets");
            const nodes = parse(`<scene><a targets="@missing" /></scene>`);
            expect(() => load(nodes, state)).toThrow("@missing");
        });

        test("exclusive relation replaces previous target", () => {
            const Parent = relation("parent", { exclusive: true });

            const a = state.addEntity();
            const b = state.addEntity();
            const c = state.addEntity();

            state.addRelation(c, Parent, a);
            state.addRelation(c, Parent, b);

            expect(state.hasRelation(c, Parent, a)).toBe(false);
            expect(state.hasRelation(c, Parent, b)).toBe(true);
        });
    });

    describe("serialize", () => {
        test("serializes simple nodes", () => {
            const Position = { x: [] as number[], y: [] as number[], z: [] as number[] };
            registerComponent("position", Position);

            const nodes = parse(`<scene><a id="player" position="x: 10" /></scene>`);
            const xml = serialize(nodes);

            expect(xml).toContain("<scene>");
            expect(xml).toContain("</scene>");
            expect(xml).toContain('id="player"');
            expect(xml).toContain("position");
        });

        test("serializes hierarchy", () => {
            const Position = { x: [] as number[], y: [] as number[], z: [] as number[] };
            registerComponent("position", Position);

            const nodes = parse(`<scene><a id="parent"><a id="child" /></a></scene>`);
            const xml = serialize(nodes);

            expect(xml).toContain('id="parent"');
            expect(xml).toContain('id="child"');
            expect(xml).toContain("</a>");
        });

        test("round-trip preserves structure", () => {
            const Position = { x: [] as number[], y: [] as number[], z: [] as number[] };
            registerComponent("position", Position);

            const original = parse(
                `<scene><a id="a" position="x: 1"><a id="b" /></a><a id="c" /></scene>`,
            );
            const xml = serialize(original);
            const reparsed = parse(xml);

            expect(reparsed.length).toBe(original.length);
            expect(reparsed[0].id).toBe(original[0].id);
            expect(reparsed[0].children.length).toBe(original[0].children.length);
            expect(reparsed[1].id).toBe(original[1].id);
        });

        test("serializes relations as refs", () => {
            relation("targets");

            const nodes = parse(`<scene><a id="cube" /><a id="tween" targets="@cube" /></scene>`);
            const xml = serialize(nodes);

            expect(xml).toContain('targets="@cube"');
        });
    });

    describe("formatFields", () => {
        beforeEach(() => {
            clearRegistry();
        });

        test("round-trips with parseFields", () => {
            const Comp = { x: [] as number[], y: [] as number[], z: [] as number[] };
            traits(Comp, { defaults: () => ({ x: 0, y: 0, z: 0 }) });
            registerComponent("comp", Comp);

            const fields = { x: 10, y: 20, z: 30 };
            const str = formatFields("comp", fields);
            const parsed = parseFields("comp", str);

            expect(parsed.x).toBe(10);
            expect(parsed.y).toBe(20);
            expect(parsed.z).toBe(30);
        });

        test("omits default values", () => {
            const Comp = { x: [] as number[], y: [] as number[] };
            traits(Comp, { defaults: () => ({ x: 0, y: 1 }) });
            registerComponent("comp", Comp);

            const str = formatFields("comp", { x: 0, y: 1 });
            expect(str).toBe("");
        });

        test("uses format trait for named lookups", () => {
            const Comp = { mode: [] as number[] };
            const names = new Map([
                [1, "fast"],
                [2, "slow"],
            ]);
            traits(Comp, {
                defaults: () => ({ mode: 0 }),
                parse: { mode: (v: string) => (v === "fast" ? 1 : v === "slow" ? 2 : undefined) },
                format: { mode: (v: number) => names.get(v) },
            });
            registerComponent("comp", Comp);

            const str = formatFields("comp", { mode: 1 });
            expect(str).toBe("mode: fast");
        });

        test("groups vec3 fields", () => {
            const Transform = {
                posX: [] as number[],
                posY: [] as number[],
                posZ: [] as number[],
            };
            traits(Transform, { defaults: () => ({ posX: 0, posY: 0, posZ: 0 }) });
            registerComponent("transform", Transform);

            const str = formatFields("transform", { posX: 1, posY: 2, posZ: 3 });
            expect(str).toBe("pos: 1 2 3");
        });

        test("broadcast shorthand for equal vec3", () => {
            const Transform = {
                posX: [] as number[],
                posY: [] as number[],
                posZ: [] as number[],
            };
            traits(Transform, { defaults: () => ({ posX: 0, posY: 0, posZ: 0 }) });
            registerComponent("transform", Transform);

            const str = formatFields("transform", { posX: 5, posY: 5, posZ: 5 });
            expect(str).toBe("pos: 5");
        });

        test("vec3 round-trip through parseFields", () => {
            const Transform = {
                posX: [] as number[],
                posY: [] as number[],
                posZ: [] as number[],
            };
            traits(Transform, { defaults: () => ({ posX: 0, posY: 0, posZ: 0 }) });
            registerComponent("transform", Transform);

            const str = formatFields("transform", { posX: 1, posY: 2, posZ: 3 });
            const parsed = parseFields("transform", str);
            expect(parsed.posX).toBe(1);
            expect(parsed.posY).toBe(2);
            expect(parsed.posZ).toBe(3);
        });

        test("format trait hex round-trip", () => {
            const Comp = { color: [] as number[] };
            const formatHex = (n: number) => "0x" + (n >>> 0).toString(16).padStart(6, "0");
            traits(Comp, {
                defaults: () => ({ color: 0xffffff }),
                format: { color: formatHex },
            });
            registerComponent("comp", Comp);

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
            traits(Comp, { defaults: () => ({ x: 0, y: 1 }) });
            registerComponent("comp", Comp);

            const str = formatFields("comp", { x: 0, y: 1 }, { stripDefaults: false });
            expect(str).toBe("x: 0; y: 1");
        });

        test("stripDefaults: false preserves default-valued vec groups", () => {
            const Transform = {
                posX: [] as number[],
                posY: [] as number[],
                posZ: [] as number[],
            };
            traits(Transform, { defaults: () => ({ posX: 0, posY: 0, posZ: 0 }) });
            registerComponent("transform", Transform);

            const str = formatFields(
                "transform",
                { posX: 0, posY: 0, posZ: 0 },
                { stripDefaults: false },
            );
            expect(str).toBe("pos: 0");
        });
    });

    describe("normalizeAttr", () => {
        beforeEach(() => {
            clearRegistry();
        });

        test("returns null for unknown components", () => {
            expect(normalizeAttr("nonexistent", "x: 1")).toBeNull();
        });

        test("returns null for empty value", () => {
            registerComponent("comp", { x: [] as number[] });
            expect(normalizeAttr("comp", "")).toBeNull();
        });

        test("preserves authored field set without injecting defaults", () => {
            const Comp = { x: [] as number[], y: [] as number[] };
            traits(Comp, { defaults: () => ({ x: 0, y: 1 }) });
            registerComponent("comp", Comp);

            const result = normalizeAttr("comp", "x: 5");
            expect(result).toBe("x: 5");
            expect(result).not.toContain("y:");
        });

        test("normalizes vec broadcast", () => {
            const Transform = {
                posX: [] as number[],
                posY: [] as number[],
                posZ: [] as number[],
            };
            traits(Transform, { defaults: () => ({ posX: 0, posY: 0, posZ: 0 }) });
            registerComponent("transform", Transform);

            expect(normalizeAttr("transform", "pos: 1 1 1")).toBe("pos: 1");
        });

        test("normalizes precision", () => {
            const Comp = { x: [] as number[] };
            traits(Comp, { defaults: () => ({ x: 0 }) });
            registerComponent("comp", Comp);

            expect(normalizeAttr("comp", "x: 0.50000000")).toBe("x: 0.5");
        });

        test("normalizes kebab-case", () => {
            const Comp = { myField: [] as number[] };
            traits(Comp, { defaults: () => ({ myField: 0 }) });
            registerComponent("comp", Comp);

            expect(normalizeAttr("comp", "myField: 5")).toBe("my-field: 5");
        });

        test("is idempotent", () => {
            const Transform = {
                posX: [] as number[],
                posY: [] as number[],
                posZ: [] as number[],
            };
            traits(Transform, { defaults: () => ({ posX: 0, posY: 0, posZ: 0 }) });
            registerComponent("transform", Transform);

            const first = normalizeAttr("transform", "pos: 1 2 3");
            const second = normalizeAttr("transform", first!);
            expect(second).toBe(first);
        });

        test("preserves fields explicitly set to defaults", () => {
            const Comp = { x: [] as number[], y: [] as number[] };
            traits(Comp, { defaults: () => ({ x: 0, y: 1 }) });
            registerComponent("comp", Comp);

            const result = normalizeAttr("comp", "x: 0; y: 1");
            expect(result).toBe("x: 0; y: 1");
        });
    });

    describe("format idempotence", () => {
        test("serialize(parse(xml)) is idempotent", () => {
            const xml = `<scene>
    <a id="parent">
        <a id="child" />
    </a>
    <a id="sibling" />
</scene>`;
            const once = serialize(parse(xml));
            const twice = serialize(parse(once));
            expect(twice).toBe(once);
        });
    });

    describe("diagnose", () => {
        test("reports missing requires", () => {
            const A = { value: [] as number[] };
            const B = { value: [] as number[] };
            traits(A, { requires: [B] });
            registerComponent("alpha", A);
            registerComponent("beta", B);

            const nodes = parse(`<scene><a alpha /></scene>`);
            const results = diagnose(nodes);
            expect(results).toHaveLength(1);
            expect(results[0].kind).toBe("missing-requires");
            expect(results[0].message).toContain("beta");
        });

        test("no missing-requires when dependency present", () => {
            const A = { value: [] as number[] };
            const B = { value: [] as number[] };
            traits(A, { requires: [B] });
            registerComponent("alpha", A);
            registerComponent("beta", B);

            const nodes = parse(`<scene><a alpha beta /></scene>`);
            const results = diagnose(nodes);
            expect(results).toHaveLength(0);
        });

        test("reports unregistered components", () => {
            registerComponent("known", { value: [] as number[] });

            const nodes = parse(`<scene><a unknown-thing /></scene>`);
            const results = diagnose(nodes);

            expect(results).toHaveLength(1);
            expect(results[0].kind).toBe("unregistered");
        });
    });
});
