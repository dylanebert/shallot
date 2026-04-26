import { test, expect, describe, beforeEach } from "bun:test";
import {
    State,
    parse,
    serialize,
    parseFields,
    formatFields,
    normalizeAttr,
    TransformsPlugin,
    InputPlugin,
    ComputePlugin,
    RenderPlugin,
    RasterPlugin,
    TweenPlugin,
    SkylabPlugin,
    OrbitPlugin,
    PlayerPlugin,
    LinesPlugin,
    ArrowsPlugin,
    TextPlugin,
    type Node,
    type Plugin,
} from "../src";
import { clearRegistry, getComponent, getComponents } from "../src/engine/ecs/component";
import { clearRelations } from "../src/engine/ecs/relation";
import { readFields } from "../src/engine/ecs/reflection";

const SCENE_FILES = [
    "raytracing/public/scenes/raytracing.scene",
    "visualization/tween/scenes/tween.scene",
    "visualization/hierarchy/scenes/hierarchy.scene",
    "visualization/text/scenes/text.scene",
    "visualization/lines/scenes/lines.scene",
];

const PLUGINS: Plugin[] = [
    TransformsPlugin,
    InputPlugin,
    ComputePlugin,
    RenderPlugin,
    RasterPlugin,
    TweenPlugin,
    SkylabPlugin,
    OrbitPlugin,
    PlayerPlugin,
    LinesPlugin,
    ArrowsPlugin,
    TextPlugin,
];

function registerPlugins() {
    const state = new State();
    for (const plugin of PLUGINS) state.register(plugin);
    return state;
}

async function readScene(name: string): Promise<string> {
    const path = `${import.meta.dir}/../../../examples/${name}`;
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
        clearRegistry();
        clearRelations();
        registerPlugins();
    });

    test.each(SCENE_FILES)("node-level roundtrip: %s", async (file) => {
        const xml = await readScene(file);
        const original = parse(xml);
        const serialized = serialize(original);
        const reparsed = parse(serialized);

        expect(countNodes(reparsed)).toBe(countNodes(original));
        compareNodes(original, reparsed);
    });

    test.each(SCENE_FILES)("idempotent serialization: %s", async (file) => {
        const xml = await readScene(file);
        const once = serialize(parse(xml));
        const twice = serialize(parse(once));
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
                            expect(b).toBeCloseTo(a, 5);
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

    describe("readback fidelity", () => {
        test("color sub-fields suppressed across all components", () => {
            const state = registerPlugins();
            for (const { name, component } of getComponents()) {
                if (!("colorR" in component && "colorG" in component && "colorB" in component))
                    continue;
                if (!("color" in component)) continue;

                const reg = getComponent(name)!;
                const eid = state.addEntity();
                state.addComponent(eid, reg.component as never);
                (component as Record<string, number[]>).color[eid] = 0xff0000;

                const defaults = reg.traits?.defaults?.() ?? {};
                const fields = readFields(reg.component, eid);
                const merged = { ...defaults, ...fields };
                const formatted = formatFields(name, merged);
                expect(formatted).not.toContain("color-r");
                expect(formatted).not.toContain("color-g");
                expect(formatted).not.toContain("color-b");
            }
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
                    if (first === null) continue;
                    const second = normalizeAttr(attr.name, first);
                    expect(second).toBe(first);
                }
                for (const child of node.children) checkNode(child);
            }

            for (const node of nodes) checkNode(node);
        });
    });
});
