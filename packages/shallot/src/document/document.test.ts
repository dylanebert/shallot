import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import fc from "fast-check";
import { entity, f32, load, parse, serialize, sparse, stringify, u32, vec4 } from "..";
import { clear as clearRegistry, register } from "../engine/ecs/core";
import { State } from "../engine/ecs/state";
import type { Node } from "../engine/scene";
import { parseFields } from "../engine/scene/core";
import {
    apply,
    type Command,
    clear,
    deselect,
    execute,
    type History,
    redo,
    reverse,
    select,
    undo,
} from "./commands";
import { Document, Readback, ReadbackSystem, Session } from "./index";

// a synthetic component for exercising the editor's component-sync path — the editor
// session is component-agnostic, so this avoids coupling the test to any engine plugin
const Haze = { density: sparse(f32), color: sparse(f32) };
const HazeTraits = { defaults: () => ({ density: 0.005, color: 0x4078d0 }) };

// a Quad + scalar component for the typed-storage value path (dotted lanes), and a u32 entity-ref
// component for ReadbackSystem's `@name` skip
const Glow = { pos: sparse(vec4), intensity: sparse(f32) };
const GlowTraits = { defaults: () => ({ pos: [0, 0, 0, 0], intensity: 1 }) };
const Link = { target: sparse(u32) };
const LinkTraits = { defaults: () => ({ target: -1 }) };

function createNode(id?: string, attrs: { name: string; value: string }[] = []): Node {
    return { id, attrs, children: [] };
}

describe("Editor", () => {
    describe("apply", () => {
        test("apply add inserts node", () => {
            const nodes: Node[] = [];
            const node = createNode("a");
            const cmd: Command = { type: "add", parent: null, node, index: 0 };

            apply(nodes, cmd);

            expect(nodes).toHaveLength(1);
            expect(nodes[0]).toBe(node);
        });

        test("apply add inserts at correct index", () => {
            const first = createNode("first");
            const third = createNode("third");
            const nodes: Node[] = [first, third];
            const second = createNode("second");
            const cmd: Command = { type: "add", parent: null, node: second, index: 1 };

            apply(nodes, cmd);

            expect(nodes).toHaveLength(3);
            expect(nodes[0]).toBe(first);
            expect(nodes[1]).toBe(second);
            expect(nodes[2]).toBe(third);
        });

        test("apply add inserts into parent children", () => {
            const parent = createNode("parent");
            const nodes: Node[] = [parent];
            const child = createNode("child");
            const cmd: Command = { type: "add", parent, node: child, index: 0 };

            apply(nodes, cmd);

            expect(parent.children).toHaveLength(1);
            expect(parent.children[0]).toBe(child);
        });

        test("apply remove deletes node", () => {
            const node = createNode("a");
            const nodes: Node[] = [node];
            const cmd: Command = { type: "remove", parent: null, node, index: 0 };

            apply(nodes, cmd);

            expect(nodes).toHaveLength(0);
        });

        test("apply remove deletes from parent children", () => {
            const child = createNode("child");
            const parent = createNode("parent");
            parent.children.push(child);
            const nodes: Node[] = [parent];
            const cmd: Command = { type: "remove", parent, node: child, index: 0 };

            apply(nodes, cmd);

            expect(parent.children).toHaveLength(0);
        });

        test("apply setAttr updates attr", () => {
            const node = createNode("a", [{ name: "position", value: "x: 0" }]);
            const nodes: Node[] = [node];
            const cmd: Command = {
                type: "setAttr",
                node,
                name: "position",
                prev: "x: 0",
                next: "x: 10",
            };

            apply(nodes, cmd);

            expect(node.attrs[0].value).toBe("x: 10");
        });

        test("apply addAttr adds new attr", () => {
            const node = createNode("a");
            const nodes: Node[] = [node];
            const cmd: Command = {
                type: "addAttr",
                node,
                name: "velocity",
                value: "x: 5",
            };

            apply(nodes, cmd);

            expect(node.attrs).toHaveLength(1);
            expect(node.attrs[0].name).toBe("velocity");
            expect(node.attrs[0].value).toBe("x: 5");
        });
    });

    describe("reverse", () => {
        test("reverse undoes add", () => {
            const node = createNode("a");
            const nodes: Node[] = [node];
            const cmd: Command = { type: "add", parent: null, node, index: 0 };

            reverse(nodes, cmd);

            expect(nodes).toHaveLength(0);
        });

        test("reverse undoes remove", () => {
            const node = createNode("a");
            const nodes: Node[] = [];
            const cmd: Command = { type: "remove", parent: null, node, index: 0 };

            reverse(nodes, cmd);

            expect(nodes).toHaveLength(1);
            expect(nodes[0]).toBe(node);
        });

        test("reverse undoes setAttr", () => {
            const node = createNode("a", [{ name: "position", value: "x: 10" }]);
            const nodes: Node[] = [node];
            const cmd: Command = {
                type: "setAttr",
                node,
                name: "position",
                prev: "x: 0",
                next: "x: 10",
            };

            reverse(nodes, cmd);

            expect(node.attrs[0].value).toBe("x: 0");
        });

        test("reverse removes added attr", () => {
            const node = createNode("a", [{ name: "velocity", value: "x: 5" }]);
            const nodes: Node[] = [node];
            const cmd: Command = {
                type: "addAttr",
                node,
                name: "velocity",
                value: "x: 5",
            };

            reverse(nodes, cmd);

            expect(node.attrs).toHaveLength(0);
        });
    });

    describe("history", () => {
        let history: History;
        let nodes: Node[];

        beforeEach(() => {
            history = { undo: [], redo: [] };
            nodes = [];
        });

        test("execute pushes to undo", () => {
            const node = createNode("a");
            const cmd: Command = { type: "add", parent: null, node, index: 0 };

            execute(history, nodes, cmd);

            expect(history.undo).toHaveLength(1);
            expect(history.undo[0].cmd).toBe(cmd);
            expect(nodes).toHaveLength(1);
        });

        test("execute clears redo", () => {
            const nodeA = createNode("a");
            const nodeB = createNode("b");
            const cmdA: Command = { type: "add", parent: null, node: nodeA, index: 0 };
            const cmdB: Command = { type: "add", parent: null, node: nodeB, index: 1 };

            execute(history, nodes, cmdA);
            execute(history, nodes, cmdB);
            undo(history, nodes);
            expect(history.redo).toHaveLength(1);

            const nodeC = createNode("c");
            const cmdC: Command = { type: "add", parent: null, node: nodeC, index: 1 };
            execute(history, nodes, cmdC);

            expect(history.redo).toHaveLength(0);
        });

        test("undo reverses and pushes to redo", () => {
            const node = createNode("a");
            const cmd: Command = { type: "add", parent: null, node, index: 0 };
            execute(history, nodes, cmd);

            const entry = undo(history, nodes);

            expect(nodes).toHaveLength(0);
            expect(history.undo).toHaveLength(0);
            expect(history.redo).toHaveLength(1);
            expect(history.redo[0].cmd).toBe(cmd);
            expect(entry).toBeDefined();
            expect(entry!.cmd).toBe(cmd);
        });

        test("undo does nothing when stack empty", () => {
            undo(history, nodes);

            expect(history.undo).toHaveLength(0);
            expect(history.redo).toHaveLength(0);
        });

        test("redo reapplies", () => {
            const node = createNode("a");
            const cmd: Command = { type: "add", parent: null, node, index: 0 };
            execute(history, nodes, cmd);
            undo(history, nodes);

            redo(history, nodes);

            expect(nodes).toHaveLength(1);
            expect(nodes[0]).toBe(node);
            expect(history.undo).toHaveLength(1);
            expect(history.redo).toHaveLength(0);
        });

        test("redo does nothing when stack empty", () => {
            redo(history, nodes);

            expect(history.undo).toHaveLength(0);
            expect(history.redo).toHaveLength(0);
        });

        test("execute stores selection snapshot", () => {
            const node = createNode("a");
            const sel = [createNode("sel")];
            const cmd: Command = { type: "add", parent: null, node, index: 0 };

            execute(history, nodes, cmd, sel);

            expect(history.undo[0].selection).toEqual(sel);
        });

        test("undo returns entry with selection", () => {
            const sel = [createNode("sel")];
            const node = createNode("a");
            const cmd: Command = { type: "add", parent: null, node, index: 0 };
            execute(history, nodes, cmd, sel);

            const entry = undo(history, nodes);

            expect(entry).toBeDefined();
            expect(entry!.selection).toEqual(sel);
        });

        test("redo returns entry", () => {
            const node = createNode("a");
            const cmd: Command = { type: "add", parent: null, node, index: 0 };
            execute(history, nodes, cmd);
            undo(history, nodes);

            const entry = redo(history, nodes);

            expect(entry).toBeDefined();
            expect(entry!.cmd).toBe(cmd);
        });
    });

    describe("removeAttr", () => {
        test("apply removes attr", () => {
            const node = createNode("a", [
                { name: "position", value: "x: 10" },
                { name: "velocity", value: "x: 5" },
            ]);
            const nodes: Node[] = [node];
            const cmd: Command = {
                type: "removeAttr",
                node,
                name: "position",
                prev: "x: 10",
                index: 0,
            };

            apply(nodes, cmd);

            expect(node.attrs).toHaveLength(1);
            expect(node.attrs[0].name).toBe("velocity");
        });

        // reverse restores the attr at its original index, not the end — undoing a component removal
        // must not reorder the entity's components.
        test("reverse re-inserts attr at its original index", () => {
            const node = createNode("a", [{ name: "velocity", value: "x: 5" }]);
            const nodes: Node[] = [node];
            const cmd: Command = {
                type: "removeAttr",
                node,
                name: "position",
                prev: "x: 10",
                index: 0,
            };

            reverse(nodes, cmd);

            expect(node.attrs).toHaveLength(2);
            expect(node.attrs[0].name).toBe("position");
            expect(node.attrs[0].value).toBe("x: 10");
            expect(node.attrs[1].name).toBe("velocity");
        });

        test("execute/undo/redo round-trip", () => {
            const history: History = { undo: [], redo: [] };
            const node = createNode("a", [{ name: "position", value: "x: 10" }]);
            const nodes: Node[] = [node];
            const cmd: Command = {
                type: "removeAttr",
                node,
                name: "position",
                prev: "x: 10",
                index: 0,
            };

            execute(history, nodes, cmd);
            expect(node.attrs).toHaveLength(0);

            undo(history, nodes);
            expect(node.attrs).toHaveLength(1);
            expect(node.attrs[0].value).toBe("x: 10");

            redo(history, nodes);
            expect(node.attrs).toHaveLength(0);
        });
    });

    describe("setId", () => {
        test("apply changes id", () => {
            const node = createNode("old-id");
            const nodes: Node[] = [node];
            const cmd: Command = {
                type: "setId",
                node,
                prev: "old-id",
                next: "new-id",
            };

            apply(nodes, cmd);

            expect(node.id).toBe("new-id");
        });

        test("reverse restores id", () => {
            const node = createNode("new-id");
            const nodes: Node[] = [node];
            const cmd: Command = {
                type: "setId",
                node,
                prev: "old-id",
                next: "new-id",
            };

            reverse(nodes, cmd);

            expect(node.id).toBe("old-id");
        });

        test("apply sets id to undefined", () => {
            const node = createNode("some-id");
            const nodes: Node[] = [node];
            const cmd: Command = {
                type: "setId",
                node,
                prev: "some-id",
                next: undefined,
            };

            apply(nodes, cmd);

            expect(node.id).toBeUndefined();
        });

        test("execute/undo/redo round-trip", () => {
            const history: History = { undo: [], redo: [] };
            const node = createNode("old");
            const nodes: Node[] = [node];
            const cmd: Command = {
                type: "setId",
                node,
                prev: "old",
                next: "new",
            };

            execute(history, nodes, cmd);
            expect(node.id).toBe("new");

            undo(history, nodes);
            expect(node.id).toBe("old");

            redo(history, nodes);
            expect(node.id).toBe("new");
        });
    });

    describe("selection", () => {
        let selection: Set<Node>;

        beforeEach(() => {
            selection = new Set();
        });

        test("select adds to set", () => {
            const node = createNode("a");

            select(selection, node);

            expect(selection.has(node)).toBe(true);
            expect(selection.size).toBe(1);
        });

        test("select adds multiple nodes", () => {
            const a = createNode("a");
            const b = createNode("b");

            select(selection, a, b);

            expect(selection.has(a)).toBe(true);
            expect(selection.has(b)).toBe(true);
            expect(selection.size).toBe(2);
        });

        test("deselect removes from set", () => {
            const node = createNode("a");
            selection.add(node);

            deselect(selection, node);

            expect(selection.has(node)).toBe(false);
            expect(selection.size).toBe(0);
        });

        test("deselect removes multiple nodes", () => {
            const a = createNode("a");
            const b = createNode("b");
            const c = createNode("c");
            selection.add(a);
            selection.add(b);
            selection.add(c);

            deselect(selection, a, c);

            expect(selection.has(a)).toBe(false);
            expect(selection.has(b)).toBe(true);
            expect(selection.has(c)).toBe(false);
            expect(selection.size).toBe(1);
        });

        test("clear empties set", () => {
            const a = createNode("a");
            const b = createNode("b");
            selection.add(a);
            selection.add(b);

            clear(selection);

            expect(selection.size).toBe(0);
        });
    });
});

describe("Document", () => {
    test("constructor parses XML string", () => {
        const doc = new Document('<scene><a id="a" /><a id="b" /></scene>');
        expect(doc.nodes).toHaveLength(2);
        expect(doc.nodes[0].id).toBe("a");
    });

    test("constructor accepts Node[]", () => {
        const nodes = [createNode("x")];
        const doc = new Document(nodes);
        expect(doc.nodes).toBe(nodes);
    });

    test("add inserts node and increments version", () => {
        const doc = new Document("<scene></scene>");
        const node = createNode("new");
        doc.add(null, node);
        expect(doc.nodes).toHaveLength(1);
        expect(doc.version).toBe(1);
    });

    test("add inserts at specified index", () => {
        const doc = new Document('<scene><a id="a" /><a id="c" /></scene>');
        const node = createNode("b");
        doc.add(null, node, 1);
        expect(doc.nodes).toHaveLength(3);
        expect(doc.nodes[1].id).toBe("b");
    });

    test("remove deletes node", () => {
        const doc = new Document('<scene><a id="a" /></scene>');
        const node = doc.nodes[0];
        doc.remove(null, node);
        expect(doc.nodes).toHaveLength(0);
    });

    test("remove no-ops for missing node", () => {
        const doc = new Document('<scene><a id="a" /></scene>');
        const missing = createNode("missing");
        doc.remove(null, missing);
        expect(doc.nodes).toHaveLength(1);
        expect(doc.version).toBe(0);
    });

    test("addAttr appends attribute", () => {
        const doc = new Document('<scene><a id="a" /></scene>');
        doc.addAttr(doc.nodes[0], "mesh", "shape: box");
        expect(doc.nodes[0].attrs).toHaveLength(1);
        expect(doc.nodes[0].attrs[0].value).toBe("shape: box");
    });

    test("setAttr updates attribute value", () => {
        const doc = new Document('<scene><a id="a" /></scene>');
        doc.addAttr(doc.nodes[0], "mesh", "shape: box");
        doc.setAttr(doc.nodes[0], "mesh", "shape: sphere");
        expect(doc.nodes[0].attrs[0].value).toBe("shape: sphere");
    });

    test("setAttr no-ops for missing attribute", () => {
        const doc = new Document('<scene><a id="a" /></scene>');
        doc.setAttr(doc.nodes[0], "missing", "value");
        expect(doc.version).toBe(0);
    });

    test("removeAttr deletes attribute", () => {
        const doc = new Document('<scene><a id="a" /></scene>');
        doc.addAttr(doc.nodes[0], "mesh", "shape: box");
        doc.removeAttr(doc.nodes[0], "mesh");
        expect(doc.nodes[0].attrs).toHaveLength(0);
    });

    test("removeAttr no-ops for missing attribute", () => {
        const doc = new Document('<scene><a id="a" /></scene>');
        doc.removeAttr(doc.nodes[0], "missing");
        expect(doc.version).toBe(0);
    });

    test("setId changes node id", () => {
        const doc = new Document('<scene><a id="old" /></scene>');
        doc.setId(doc.nodes[0], "new");
        expect(doc.nodes[0].id).toBe("new");
    });

    test("undo reverses last command", () => {
        const doc = new Document("<scene></scene>");
        doc.add(null, createNode("a"));
        expect(doc.nodes).toHaveLength(1);
        doc.undo();
        expect(doc.nodes).toHaveLength(0);
    });

    test("redo reapplies undone command", () => {
        const doc = new Document("<scene></scene>");
        doc.add(null, createNode("a"));
        doc.undo();
        doc.redo();
        expect(doc.nodes).toHaveLength(1);
    });

    test("serialize produces XML", () => {
        const doc = new Document('<scene><a id="x" /></scene>');
        const xml = doc.serialize();
        expect(xml).toContain('id="x"');
        expect(xml).toContain("<scene>");
    });

    test("full round-trip: mutate then serialize then re-parse", () => {
        const doc = new Document('<scene><a id="a" /></scene>');
        doc.addAttr(doc.nodes[0], "mesh", "shape: box");
        doc.add(null, createNode("b"));
        const xml = doc.serialize();
        const doc2 = new Document(xml);
        expect(doc2.nodes).toHaveLength(2);
    });

    test("version increments on every mutation", () => {
        const doc = new Document('<scene><a id="a" /></scene>');
        expect(doc.version).toBe(0);
        doc.add(null, createNode("b"));
        expect(doc.version).toBe(1);
        doc.undo();
        expect(doc.version).toBe(2);
        doc.redo();
        expect(doc.version).toBe(3);
    });

    test("selection operations", () => {
        const doc = new Document('<scene><a id="a" /><a id="b" /></scene>');
        doc.select(doc.nodes[0]);
        expect(doc.selection.has(doc.nodes[0])).toBe(true);
        doc.deselect(doc.nodes[0]);
        expect(doc.selection.size).toBe(0);
        doc.select(doc.nodes[0], doc.nodes[1]);
        expect(doc.selection.size).toBe(2);
        doc.clearSelection();
        expect(doc.selection.size).toBe(0);
    });

    describe("reorder", () => {
        test("moves node forward", () => {
            const doc = new Document('<scene><a id="a" /><a id="b" /><a id="c" /></scene>');
            doc.reorder(null, doc.nodes[0], 2);
            expect(doc.nodes.map((n) => n.id)).toEqual(["b", "c", "a"]);
        });

        test("moves node backward", () => {
            const doc = new Document('<scene><a id="a" /><a id="b" /><a id="c" /></scene>');
            doc.reorder(null, doc.nodes[2], 0);
            expect(doc.nodes.map((n) => n.id)).toEqual(["c", "a", "b"]);
        });

        test("no-ops for same position", () => {
            const doc = new Document('<scene><a id="a" /><a id="b" /></scene>');
            doc.reorder(null, doc.nodes[0], 0);
            expect(doc.version).toBe(0);
        });

        test("no-ops for missing node", () => {
            const doc = new Document('<scene><a id="a" /></scene>');
            doc.reorder(null, createNode("missing"), 0);
            expect(doc.version).toBe(0);
        });

        test("undo/redo round-trip", () => {
            const doc = new Document('<scene><a id="a" /><a id="b" /><a id="c" /></scene>');
            doc.reorder(null, doc.nodes[0], 2);
            expect(doc.nodes.map((n) => n.id)).toEqual(["b", "c", "a"]);
            doc.undo();
            expect(doc.nodes.map((n) => n.id)).toEqual(["a", "b", "c"]);
            doc.redo();
            expect(doc.nodes.map((n) => n.id)).toEqual(["b", "c", "a"]);
        });

        test("reorders within parent children", () => {
            const doc = new Document('<scene><a id="parent" /></scene>');
            const parent = doc.nodes[0];
            const a: Node = { id: "a", attrs: [], children: [] };
            const b: Node = { id: "b", attrs: [], children: [] };
            const c: Node = { id: "c", attrs: [], children: [] };
            parent.children.push(a, b, c);
            doc.reorder(parent, parent.children[2], 0);
            expect(parent.children.map((n) => n.id)).toEqual(["c", "a", "b"]);
        });
    });

    describe("reparent", () => {
        test("moves node to new parent", () => {
            const doc = new Document('<scene><a id="a" /><a id="b" /></scene>');
            const a = doc.nodes[0];
            const b = doc.nodes[1];
            doc.reparent(b, null, a, 0);
            expect(doc.nodes).toHaveLength(1);
            expect(doc.nodes[0].id).toBe("a");
            expect(a.children).toHaveLength(1);
            expect(a.children[0].id).toBe("b");
        });

        test("moves node to root", () => {
            const doc = new Document('<scene><a id="parent" /></scene>');
            const parent = doc.nodes[0];
            const child: Node = { id: "child", attrs: [], children: [] };
            parent.children.push(child);
            doc.reparent(child, parent, null, 1);
            expect(doc.nodes).toHaveLength(2);
            expect(doc.nodes[1].id).toBe("child");
            expect(parent.children).toHaveLength(0);
        });

        test("undo/redo round-trip", () => {
            const doc = new Document('<scene><a id="a" /><a id="b" /></scene>');
            const a = doc.nodes[0];
            const b = doc.nodes[1];
            doc.reparent(b, null, a, 0);
            expect(doc.nodes).toHaveLength(1);
            doc.undo();
            expect(doc.nodes).toHaveLength(2);
            expect(doc.nodes[0].id).toBe("a");
            expect(doc.nodes[1].id).toBe("b");
            doc.redo();
            expect(doc.nodes).toHaveLength(1);
            expect(a.children[0].id).toBe("b");
        });
    });

    describe("undo/redo selection", () => {
        test("undo restores pre-command selection", () => {
            const doc = new Document('<scene><a id="a" /><a id="b" /></scene>');
            const a = doc.nodes[0];
            const b = doc.nodes[1];
            doc.select(a);
            doc.addAttr(b, "mesh", "shape: box");
            expect(doc.selection.has(a)).toBe(true);

            doc.undo();

            expect(doc.selection.has(a)).toBe(true);
            expect(doc.selection.size).toBe(1);
        });

        test("redo selects command target node", () => {
            const doc = new Document('<scene><a id="a" /><a id="b" /></scene>');
            const a = doc.nodes[0];
            const b = doc.nodes[1];
            doc.select(a);
            doc.addAttr(b, "mesh", "shape: box");
            doc.undo();

            doc.redo();

            expect(doc.selection.has(b)).toBe(true);
            expect(doc.selection.size).toBe(1);
        });

        test("redo of remove clears selection", () => {
            const doc = new Document('<scene><a id="a" /></scene>');
            const a = doc.nodes[0];
            doc.remove(null, a);
            doc.undo();
            expect(doc.nodes).toHaveLength(1);

            doc.redo();

            expect(doc.selection.size).toBe(0);
        });

        test("undo returns command", () => {
            const doc = new Document("<scene></scene>");
            const node = createNode("a");
            doc.add(null, node);

            const cmd = doc.undo();

            expect(cmd).toBeDefined();
            expect(cmd!.type).toBe("add");
        });

        test("redo returns command", () => {
            const doc = new Document("<scene></scene>");
            const node = createNode("a");
            doc.add(null, node);
            doc.undo();

            const cmd = doc.redo();

            expect(cmd).toBeDefined();
            expect(cmd!.type).toBe("add");
        });

        test("undo returns undefined when empty", () => {
            const doc = new Document("<scene></scene>");
            expect(doc.undo()).toBeUndefined();
        });

        test("redo returns undefined when empty", () => {
            const doc = new Document("<scene></scene>");
            expect(doc.redo()).toBeUndefined();
        });
    });

    describe("setAttr with explicit prev", () => {
        test("addAttr then setAttr: first undo reverts value, second removes attr", () => {
            const doc = new Document('<scene><a id="a" /></scene>');
            const node = doc.nodes[0];
            doc.addAttr(node, "haze", "");
            doc.setAttr(node, "haze", "density: 0.5", "");

            doc.undo();
            expect(node.attrs).toHaveLength(1);
            expect(node.attrs[0].name).toBe("haze");
            expect(node.attrs[0].value).toBe("");

            doc.undo();
            expect(node.attrs).toHaveLength(0);
        });

        test("setAttr with explicit prev stores it in history", () => {
            const doc = new Document('<scene><a id="a" /></scene>');
            const node = doc.nodes[0];
            doc.addAttr(node, "mesh", "shape: box");
            node.attrs[0].value = "shape: sphere";
            doc.setAttr(node, "mesh", "shape: cylinder", "shape: box");

            doc.undo();
            expect(node.attrs[0].value).toBe("shape: box");
        });

        test("undo with explicit prev restores prev, not current attr.value", () => {
            const doc = new Document('<scene><a id="a" /></scene>');
            const node = doc.nodes[0];
            doc.addAttr(node, "light", "intensity: 1");
            doc.setAttr(node, "light", "intensity: 5", "intensity: 1");
            expect(node.attrs[0].value).toBe("intensity: 5");

            node.attrs[0].value = "intensity: 99";

            doc.undo();
            expect(node.attrs[0].value).toBe("intensity: 1");
        });
    });

    describe("reorderAttr", () => {
        test("moves attr forward", () => {
            const doc = new Document('<scene><a id="a" /></scene>');
            doc.addAttr(doc.nodes[0], "mesh", "v1");
            doc.addAttr(doc.nodes[0], "transform", "v2");
            doc.addAttr(doc.nodes[0], "light", "v3");
            doc.reorderAttr(doc.nodes[0], 0, 2);
            expect(doc.nodes[0].attrs.map((a) => a.name)).toEqual(["transform", "light", "mesh"]);
        });

        test("moves attr backward", () => {
            const doc = new Document('<scene><a id="a" /></scene>');
            doc.addAttr(doc.nodes[0], "mesh", "v1");
            doc.addAttr(doc.nodes[0], "transform", "v2");
            doc.addAttr(doc.nodes[0], "light", "v3");
            doc.reorderAttr(doc.nodes[0], 2, 0);
            expect(doc.nodes[0].attrs.map((a) => a.name)).toEqual(["light", "mesh", "transform"]);
        });

        test("no-ops for same position", () => {
            const doc = new Document('<scene><a id="a" /></scene>');
            doc.addAttr(doc.nodes[0], "mesh", "v1");
            const v = doc.version;
            doc.reorderAttr(doc.nodes[0], 0, 0);
            expect(doc.version).toBe(v);
        });

        test("no-ops for out of bounds", () => {
            const doc = new Document('<scene><a id="a" /></scene>');
            doc.addAttr(doc.nodes[0], "mesh", "v1");
            const v = doc.version;
            doc.reorderAttr(doc.nodes[0], 0, 5);
            expect(doc.version).toBe(v);
        });

        test("undo/redo round-trip", () => {
            const doc = new Document('<scene><a id="a" /></scene>');
            doc.addAttr(doc.nodes[0], "mesh", "v1");
            doc.addAttr(doc.nodes[0], "transform", "v2");
            doc.reorderAttr(doc.nodes[0], 0, 1);
            expect(doc.nodes[0].attrs.map((a) => a.name)).toEqual(["transform", "mesh"]);
            doc.undo();
            expect(doc.nodes[0].attrs.map((a) => a.name)).toEqual(["mesh", "transform"]);
            doc.redo();
            expect(doc.nodes[0].attrs.map((a) => a.name)).toEqual(["transform", "mesh"]);
        });
    });
});

describe("Session", () => {
    let state: State;
    let doc: Document;
    let node: Node;
    let eid: number;
    let session: Session;

    beforeEach(() => {
        clearRegistry();
        register("Haze", Haze, HazeTraits);
        state = new State();
        node = createNode("cam");
        doc = new Document([node]);
        eid = state.create();
        const nodeMap = new Map<Node, number>([[node, eid]]);
        session = new Session(state, doc, nodeMap);
    });

    test("attachComponent adds component to ECS", () => {
        session.attachComponent("haze", "density: 0.01", eid);
        expect(state.has(eid, Haze as never)).toBe(true);
        expect(Haze.density.get(eid)).toBeCloseTo(0.01);
    });

    test("attachComponent applies defaults when value empty", () => {
        session.attachComponent("haze", "", eid);
        expect(state.has(eid, Haze as never)).toBe(true);
        expect(Haze.density.get(eid)).toBeCloseTo(0.005);
        expect(Haze.color.get(eid)).toBe(0x4078d0);
    });

    test("attachComponent skips unknown component", () => {
        session.attachComponent("nonexistent", "foo: 1", eid);
        expect(state.has(eid, Haze as never)).toBe(false);
    });

    test("syncCommand addAttr adds component", () => {
        doc.addAttr(node, "haze", "density: 0.01");
        const cmd = doc.history.undo[doc.history.undo.length - 1].cmd;
        session.syncCommand(cmd, false);
        expect(state.has(eid, Haze as never)).toBe(true);
        expect(Haze.density.get(eid)).toBeCloseTo(0.01);
    });

    test("syncCommand addAttr undo removes component", () => {
        doc.addAttr(node, "haze", "density: 0.01");
        const cmd = doc.history.undo[doc.history.undo.length - 1].cmd;
        session.syncCommand(cmd, false);
        expect(state.has(eid, Haze as never)).toBe(true);

        session.syncCommand(cmd, true);
        expect(state.has(eid, Haze as never)).toBe(false);
    });

    test("syncCommand removeAttr removes component", () => {
        session.attachComponent("haze", "density: 0.01", eid);
        node.attrs.push({ name: "haze", value: "density: 0.01" });
        doc.removeAttr(node, "haze");
        const cmd = doc.history.undo[doc.history.undo.length - 1].cmd;
        session.syncCommand(cmd, false);
        expect(state.has(eid, Haze as never)).toBe(false);
    });

    test("syncCommand removeAttr undo restores component", () => {
        session.attachComponent("haze", "density: 0.02", eid);
        node.attrs.push({ name: "haze", value: "density: 0.02" });
        doc.removeAttr(node, "haze");
        const cmd = doc.history.undo[doc.history.undo.length - 1].cmd;
        session.syncCommand(cmd, false);
        expect(state.has(eid, Haze as never)).toBe(false);

        session.syncCommand(cmd, true);
        expect(state.has(eid, Haze as never)).toBe(true);
        expect(Haze.density.get(eid)).toBeCloseTo(0.02);
    });

    test("syncCommand compound adds multiple components", () => {
        doc.compound([{ type: "addAttr", node, name: "haze", value: "density: 0.01" }]);
        const cmd = doc.history.undo[doc.history.undo.length - 1].cmd;
        session.syncCommand(cmd, false);
        expect(state.has(eid, Haze as never)).toBe(true);
        expect(Haze.density.get(eid)).toBeCloseTo(0.01);
    });

    test("syncCommand compound undo removes all in reverse", () => {
        doc.compound([{ type: "addAttr", node, name: "haze", value: "density: 0.01" }]);
        const cmd = doc.history.undo[doc.history.undo.length - 1].cmd;
        session.syncCommand(cmd, false);
        expect(state.has(eid, Haze as never)).toBe(true);

        session.syncCommand(cmd, true);
        expect(state.has(eid, Haze as never)).toBe(false);
    });
});

describe("Compound", () => {
    test("apply executes all sub-commands", () => {
        const node = createNode("a");
        const nodes: Node[] = [node];
        const cmd: Command = {
            type: "compound",
            commands: [
                { type: "addAttr", node, name: "mesh", value: "shape: box" },
                { type: "addAttr", node, name: "surface", value: "" },
            ],
        };

        apply(nodes, cmd);

        expect(node.attrs).toHaveLength(2);
        expect(node.attrs[0].name).toBe("mesh");
        expect(node.attrs[1].name).toBe("surface");
    });

    test("reverse undoes sub-commands in reverse order", () => {
        const node = createNode("a", [
            { name: "mesh", value: "shape: box" },
            { name: "surface", value: "" },
        ]);
        const nodes: Node[] = [node];
        const cmd: Command = {
            type: "compound",
            commands: [
                { type: "addAttr", node, name: "mesh", value: "shape: box" },
                { type: "addAttr", node, name: "surface", value: "" },
            ],
        };

        reverse(nodes, cmd);

        expect(node.attrs).toHaveLength(0);
    });

    test("Document.compound creates single undo entry", () => {
        const doc = new Document('<scene><a id="a" /></scene>');
        const node = doc.nodes[0];

        doc.compound([
            { type: "addAttr", node, name: "mesh", value: "shape: box" },
            { type: "addAttr", node, name: "surface", value: "" },
        ]);

        expect(node.attrs).toHaveLength(2);
        expect(doc.history.undo).toHaveLength(1);
        expect(doc.history.undo[0].cmd.type).toBe("compound");
    });

    test("undo of compound removes all attrs in one step", () => {
        const doc = new Document('<scene><a id="a" /></scene>');
        const node = doc.nodes[0];

        doc.compound([
            { type: "addAttr", node, name: "mesh", value: "shape: box" },
            { type: "addAttr", node, name: "surface", value: "" },
        ]);

        expect(node.attrs).toHaveLength(2);

        doc.undo();

        expect(node.attrs).toHaveLength(0);
        expect(doc.history.undo).toHaveLength(0);
        expect(doc.history.redo).toHaveLength(1);
    });

    test("redo of compound re-adds all attrs", () => {
        const doc = new Document('<scene><a id="a" /></scene>');
        const node = doc.nodes[0];

        doc.compound([
            { type: "addAttr", node, name: "mesh", value: "shape: box" },
            { type: "addAttr", node, name: "surface", value: "" },
        ]);

        doc.undo();
        expect(node.attrs).toHaveLength(0);

        doc.redo();
        expect(node.attrs).toHaveLength(2);
        expect(node.attrs[0].name).toBe("mesh");
        expect(node.attrs[1].name).toBe("surface");
    });
});

// the gesture transaction substrate: every field write between begin() and commit() coalesces into one
// undoable entry, prevs captured automatically. The motivating case is a multi-entity drag — without it
// the gizmo commits one setAttr per entity (N entries; one undo reverts one). Every viewport gesture and
// inspector scrub rides this, so a gesture is one undoable unit by construction.
describe("gesture transaction", () => {
    function scene() {
        const doc = new Document(
            '<scene><a id="a" transform="pos: 0 0 0" /><a id="b" transform="pos: 0 0 0" /></scene>',
        );
        return { doc, a: doc.nodes[0], b: doc.nodes[1] };
    }

    // the pre-substrate behaviour this fixes: N bare setAttrs are N history entries, so one undo reverts
    // only the last entity. Documents why the gesture exists.
    test("N bare setAttrs make N entries; one undo reverts only one (the gap)", () => {
        const { doc, a, b } = scene();
        doc.setAttr(a, "transform", "pos: 1 0 0");
        doc.setAttr(b, "transform", "pos: 2 0 0");
        expect(doc.history.undo).toHaveLength(2);
        doc.undo();
        expect(a.attrs[0].value).toBe("pos: 1 0 0");
        expect(b.attrs[0].value).toBe("pos: 0 0 0");
    });

    test("a multi-entity gesture is one compound entry; one undo restores all, redo replays", () => {
        const { doc, a, b } = scene();
        doc.begin();
        doc.setAttr(a, "transform", "pos: 1 0 0");
        doc.setAttr(b, "transform", "pos: 2 0 0");
        doc.commit();

        expect(doc.history.undo).toHaveLength(1);
        expect(doc.history.undo[0].cmd.type).toBe("compound");
        expect(a.attrs[0].value).toBe("pos: 1 0 0");
        expect(b.attrs[0].value).toBe("pos: 2 0 0");

        doc.undo();
        expect(a.attrs[0].value).toBe("pos: 0 0 0");
        expect(b.attrs[0].value).toBe("pos: 0 0 0");

        doc.redo();
        expect(a.attrs[0].value).toBe("pos: 1 0 0");
        expect(b.attrs[0].value).toBe("pos: 2 0 0");
    });

    test("repeated writes to one attr coalesce: prev is first-touch, next is last", () => {
        const { doc, a } = scene();
        doc.begin();
        doc.setAttr(a, "transform", "pos: 1 0 0");
        doc.setAttr(a, "transform", "pos: 2 0 0");
        doc.setAttr(a, "transform", "pos: 3 0 0");
        doc.commit();

        expect(doc.history.undo).toHaveLength(1);
        // a single touched attr → a plain setAttr, not a needless compound wrapper
        expect(doc.history.undo[0].cmd.type).toBe("setAttr");
        expect(a.attrs[0].value).toBe("pos: 3 0 0");
        doc.undo();
        expect(a.attrs[0].value).toBe("pos: 0 0 0");
    });

    test("a single-attr gesture is a plain setAttr entry, not a compound", () => {
        const { doc, a } = scene();
        doc.begin();
        doc.setAttr(a, "transform", "pos: 5 0 0");
        doc.commit();
        expect(doc.history.undo).toHaveLength(1);
        expect(doc.history.undo[0].cmd.type).toBe("setAttr");
    });

    test("an empty gesture, or one whose writes net to no change, records nothing", () => {
        const { doc, a } = scene();
        doc.begin();
        doc.commit();
        expect(doc.history.undo).toHaveLength(0);

        // writing the same value back is prev === next, so it drops
        doc.begin();
        doc.setAttr(a, "transform", "pos: 0 0 0");
        doc.commit();
        expect(doc.history.undo).toHaveLength(0);
    });

    // prev is captured at first touch, so a live mirror writing attr.value mid-gesture (what ReadbackSystem
    // does from the live ECS) can't corrupt the undo target.
    test("auto-prev captures first touch, immune to mid-gesture mirroring", () => {
        const { doc, a } = scene();
        doc.begin();
        doc.setAttr(a, "transform", "pos: 1 0 0");
        a.attrs[0].value = "pos: 1 0 0"; // simulate readback mirroring the live ECS
        doc.setAttr(a, "transform", "pos: 9 0 0");
        doc.commit();
        doc.undo();
        expect(a.attrs[0].value).toBe("pos: 0 0 0");
    });

    test("an explicit prev overrides the auto-capture", () => {
        const { doc, a } = scene();
        doc.begin();
        doc.setAttr(a, "transform", "pos: 7 0 0", "pos: 3 0 0");
        doc.commit();
        doc.undo();
        expect(a.attrs[0].value).toBe("pos: 3 0 0");
    });

    test("cancel discards the gesture with no history entry", () => {
        const { doc, a } = scene();
        doc.begin();
        doc.setAttr(a, "transform", "pos: 4 0 0");
        doc.cancel();
        expect(doc.history.undo).toHaveLength(0);
        // the gesture never applies attr.value (readback owns the live mirror), so the buffer just drops
        expect(a.attrs[0].value).toBe("pos: 0 0 0");
    });

    test("the commit entry carries the selection snapshot", () => {
        const { doc, a, b } = scene();
        doc.select(a);
        doc.begin();
        doc.setAttr(a, "transform", "pos: 1 0 0");
        doc.setAttr(b, "transform", "pos: 1 0 0");
        doc.commit();
        expect(doc.history.undo[0].selection).toContain(a);
    });
});

describe("Session syncFields", () => {
    let state: State;
    let eid: number;
    let session: Session;

    beforeEach(() => {
        clearRegistry();
        register("Glow", Glow, GlowTraits);
        state = new State();
        const node = createNode("e");
        eid = state.create();
        state.add(eid, Glow as never);
        session = new Session(state, new Document([node]), new Map([[node, eid]]));
    });

    test("applies a scalar field to the live ECS", () => {
        session.syncFields("glow", { intensity: 2.5 }, eid);
        expect(Glow.intensity.get(eid)).toBeCloseTo(2.5);
    });

    // the typed-storage write contract the inspector relies on: a Pair/Quad lane is addressed by its
    // dotted key (`pos.x`), and only that lane is written — the siblings keep their value. The inspector
    // must emit dotted keys; a split-suffix `posX` would silently no-op against the typed store.
    test("applies a single Quad lane via its dotted key, leaving siblings", () => {
        Glow.pos.set(eid, 1, 1, 1, 1);
        session.syncFields("glow", { "pos.x": 5 }, eid);
        expect(Glow.pos.x.get(eid)).toBe(5);
        expect(Glow.pos.y.get(eid)).toBe(1);
        expect(Glow.pos.z.get(eid)).toBe(1);
        expect(Glow.pos.w.get(eid)).toBe(1);
    });
});

describe("ReadbackSystem", () => {
    let state: State;
    let node: Node;
    let eid: number;

    beforeEach(() => {
        clearRegistry();
        register("Glow", Glow, GlowTraits);
        register("Link", Link, LinkTraits);
        state = new State();
        node = createNode("e");
        eid = state.create();
        Readback.session = new Session(state, new Document([node]), new Map([[node, eid]]));
        Readback.onUpdate = undefined;
    });

    afterEach(() => {
        Readback.session = null;
        Readback.onUpdate = undefined;
    });

    // ReadbackSystem is the sole writer of attr.value (the ECS→Document reverse channel). It reads each
    // mapped entity's live component fields and formats them back onto the node's attrs.
    test("writes live ECS field values back onto the node attr", () => {
        node.attrs.push({ name: "glow", value: "intensity: 0" });
        state.add(eid, Glow as never);
        Glow.pos.set(eid, 1, 2, 3, 0);
        Glow.intensity.set(eid, 2);

        ReadbackSystem.update!(state);

        const parsed = parseFields("glow", node.attrs[0].value);
        expect(parsed["pos.x"]).toBeCloseTo(1);
        expect(parsed["pos.y"]).toBeCloseTo(2);
        expect(parsed["pos.z"]).toBeCloseTo(3);
        expect(parsed.intensity).toBeCloseTo(2);
    });

    // a component on the entity but unlisted on the node is derived (a `warm` spawn / a system's add).
    // The document is the authoring truth, so readback must not pull it into the saved scene — that would
    // leak derived state to disk and diverge the document path from `serialize(state)`. A legitimately
    // added component is already on the node (the editor's add-component flow writes the doc attr first).
    test("does not pull a derived component (absent from the node) into the document", () => {
        state.add(eid, Glow as never);
        Glow.intensity.set(eid, 3);

        ReadbackSystem.update!(state);

        expect(node.attrs.find((a) => a.name === "glow")).toBeUndefined();
    });

    test("skips attrs holding an @name entity reference", () => {
        node.attrs.push({ name: "link", value: "target: @cam" });
        state.add(eid, Link as never);
        Link.target.set(eid, 7);

        ReadbackSystem.update!(state);

        expect(node.attrs[0].value).toBe("target: @cam");
    });

    test("fires onUpdate only when an attr value actually changed", () => {
        node.attrs.push({ name: "glow", value: "intensity: 0" });
        state.add(eid, Glow as never);
        Glow.intensity.set(eid, 2);
        let calls = 0;
        Readback.onUpdate = () => {
            calls++;
        };

        ReadbackSystem.update!(state);
        expect(calls).toBe(1);

        // a second pass reads the same ECS state, so the formatted value matches and nothing is rewritten
        ReadbackSystem.update!(state);
        expect(calls).toBe(1);
    });
});

// Stage 1 — the editor's save path (`Document.serialize` over the readback-synced nodes) and the engine
// save path (`serialize(state)` over the authored set) are one truth. Components are registered in a
// fixed order so `entries()`-order (which `serialize` emits) matches the fixture's author order, making
// "same bytes" a clean byte comparison. `Ref.target` is `entity`-typed so it round-trips as `@<id>`.
describe("serialize parity (Stage 1)", () => {
    const Mark = { v: sparse(u32) };
    const Ref = { target: sparse(entity) };
    const Tree = { kind: sparse(u32) };

    function setup(fixture: string) {
        clearRegistry();
        register("mark", Mark, { defaults: () => ({ v: 0 }) });
        register("ref", Ref, { defaults: () => ({ target: 0 }) });
        register("tree", Tree, { defaults: () => ({ kind: 0 }) });
        const state = new State();
        const nodes = parse(fixture);
        const nodeMap = load(nodes, state);
        const doc = new Document(nodes);
        Readback.session = new Session(state, doc, nodeMap);
        Readback.onUpdate = undefined;
        return { state, nodes, nodeMap, doc };
    }

    afterEach(() => {
        Readback.session = null;
        Readback.onUpdate = undefined;
    });

    test("both paths emit the same bytes, with an @name ref preserved and a derived entity excluded", () => {
        const fixture =
            '<scene>\n    <a id="anchor" mark="v: 5" />\n    <a id="bob" ref="target: @anchor" />\n</scene>';
        const { state, doc } = setup(fixture);

        // a derived entity (warm's orrstead-tree pattern): created outside `load`, so it's not in the
        // authored set `serialize` captures nor in the node map the document path syncs — both omit it
        const tree = state.create();
        state.add(tree, Tree as never);
        Tree.kind.set(tree, 9);

        ReadbackSystem.update!(state);

        const enginePath = stringify(serialize(state));
        const documentPath = doc.serialize();
        expect(documentPath).toBe(enginePath);
        expect(documentPath).toContain("@anchor"); // ref kept symbolic, not resolved to a raw eid
        expect(documentPath).not.toContain("tree"); // the derived entity is in neither path
    });

    test("a canonical scene is a readback fixed point, and a one-field edit is a one-attr diff", () => {
        const fixture =
            '<scene>\n    <a id="anchor" mark="v: 5" />\n    <a id="bob" mark="v: 2" />\n</scene>';
        const { state, nodes, nodeMap, doc } = setup(fixture);

        // open: one normalizing pass, then the baseline is pinned
        ReadbackSystem.update!(state);
        const baseline = doc.serialize();

        // a second pass mutates nothing — readback is a fixed point, so opening doesn't keep rewriting
        ReadbackSystem.update!(state);
        expect(doc.serialize()).toBe(baseline);

        // edit one field live; the saved diff is exactly that attr's line
        const eid = nodeMap.get(nodes[1])!;
        Mark.v.set(eid, 7);
        ReadbackSystem.update!(state);
        const edited = doc.serialize();

        const before = baseline.split("\n");
        const after = edited.split("\n");
        expect(after.length).toBe(before.length);
        const changed = before.filter((line, i) => line !== after[i]);
        expect(changed.length).toBe(1);
        expect(changed[0]).toContain("v: 2");
    });
});

// the ProseMirror step contract: every command has an exact inverse, so a history is a sound undo
// stack. Property-checked over random command sequences rather than one example per command type —
// the invariant is structural (apply∘reverse = identity), so a counterexample is a real defect.
describe("command invertibility (property)", () => {
    const Seed = '<scene><a id="root" mesh="shape: box" /><a id="other" /></scene>';

    type Placed = { node: Node; parent: Node | null };
    function flatten(nodes: Node[], parent: Node | null = null, out: Placed[] = []): Placed[] {
        for (const node of nodes) {
            out.push({ node, parent });
            flatten(node.children, node, out);
        }
        return out;
    }
    function descendant(ancestor: Node, node: Node): boolean {
        for (const child of ancestor.children) {
            if (child === node || descendant(child, node)) return true;
        }
        return false;
    }

    type Op =
        | { kind: "addRoot"; id?: string }
        | { kind: "addChild"; target: number; id?: string }
        | { kind: "addAttr"; target: number; name: string; value: string }
        | { kind: "setAttr"; target: number; value: string }
        | { kind: "removeAttr"; target: number }
        | { kind: "setId"; target: number; id?: string }
        | { kind: "reorder"; target: number; to: number }
        | { kind: "reorderAttr"; target: number; to: number }
        | { kind: "remove"; target: number }
        | { kind: "reparent"; target: number; into: number };

    // resolve generated ops against the live tree, skipping ones with no valid target (a no-op records
    // no history entry). reparent guards cycles the way the outliner's drop logic does.
    function applyOp(doc: Document, op: Op): void {
        const placed = flatten(doc.nodes);
        const pick = (i: number): Placed | null =>
            placed.length ? placed[i % placed.length] : null;
        switch (op.kind) {
            case "addRoot":
                doc.add(null, { id: op.id, attrs: [], children: [] });
                break;
            case "addChild": {
                const t = pick(op.target);
                if (t) doc.add(t.node, { id: op.id, attrs: [], children: [] });
                break;
            }
            case "addAttr": {
                const t = pick(op.target);
                if (t && !t.node.attrs.some((a) => a.name === op.name))
                    doc.addAttr(t.node, op.name, op.value);
                break;
            }
            case "setAttr": {
                const t = pick(op.target);
                if (t?.node.attrs.length) doc.setAttr(t.node, t.node.attrs[0].name, op.value);
                break;
            }
            case "removeAttr": {
                const t = pick(op.target);
                if (t?.node.attrs.length) doc.removeAttr(t.node, t.node.attrs[0].name);
                break;
            }
            case "setId": {
                const t = pick(op.target);
                if (t) doc.setId(t.node, op.id);
                break;
            }
            case "reorder": {
                const t = pick(op.target);
                if (!t) break;
                const siblings = t.parent ? t.parent.children : doc.nodes;
                if (siblings.length > 1) doc.reorder(t.parent, t.node, op.to % siblings.length);
                break;
            }
            case "reorderAttr": {
                const t = pick(op.target);
                if (t && t.node.attrs.length > 1)
                    doc.reorderAttr(t.node, 0, op.to % t.node.attrs.length);
                break;
            }
            case "remove": {
                const t = pick(op.target);
                if (t) doc.remove(t.parent, t.node);
                break;
            }
            case "reparent": {
                const t = pick(op.target);
                const into = pick(op.into);
                if (!t || !into || into.node === t.node || descendant(t.node, into.node)) break;
                doc.reparent(t.node, t.parent, into.node, into.node.children.length);
                break;
            }
        }
    }

    const Id = fc.constantFrom<string | undefined>("a", "b", "rig", undefined);
    const Attr = fc.constantFrom("mesh", "light", "tint", "wobble");
    const Value = fc.constantFrom("", "shape: box", "intensity: 2", "density: 0.5");
    const Nat = fc.nat({ max: 16 });

    const flatOp = fc.oneof(
        fc.record({ kind: fc.constant("addRoot" as const), id: Id }),
        fc.record({ kind: fc.constant("addAttr" as const), target: Nat, name: Attr, value: Value }),
        fc.record({ kind: fc.constant("setAttr" as const), target: Nat, value: Value }),
        fc.record({ kind: fc.constant("removeAttr" as const), target: Nat }),
        fc.record({ kind: fc.constant("setId" as const), target: Nat, id: Id }),
        fc.record({ kind: fc.constant("reorder" as const), target: Nat, to: Nat }),
        fc.record({ kind: fc.constant("reorderAttr" as const), target: Nat, to: Nat }),
        fc.record({ kind: fc.constant("remove" as const), target: Nat }),
    );
    const treeOp = fc.oneof(
        flatOp,
        fc.record({ kind: fc.constant("addChild" as const), target: Nat, id: Id }),
        fc.record({ kind: fc.constant("reparent" as const), target: Nat, into: Nat }),
    );

    test("undo restores the exact prior serialization (apply∘reverse = identity)", () => {
        clearRegistry();
        fc.assert(
            fc.property(fc.array(treeOp as fc.Arbitrary<Op>, { maxLength: 30 }), (ops) => {
                const doc = new Document(Seed);
                const priors: string[] = [];
                for (const op of ops) {
                    const snapshot = doc.serialize();
                    const v = doc.version;
                    applyOp(doc, op);
                    if (doc.version > v) priors.push(snapshot);
                }
                while (priors.length > 0) {
                    const expected = priors.pop()!;
                    doc.undo();
                    expect(doc.serialize()).toBe(expected);
                }
                expect(doc.serialize()).toBe(new Document(Seed).serialize());
            }),
            { numRuns: 200 },
        );
    });

    test("undo-all then redo-all replays back to the post-sequence state", () => {
        clearRegistry();
        fc.assert(
            fc.property(fc.array(treeOp as fc.Arbitrary<Op>, { maxLength: 30 }), (ops) => {
                const doc = new Document(Seed);
                let count = 0;
                for (const op of ops) {
                    const v = doc.version;
                    applyOp(doc, op);
                    if (doc.version > v) count++;
                }
                const after = doc.serialize();
                for (let i = 0; i < count; i++) doc.undo();
                for (let i = 0; i < count; i++) doc.redo();
                expect(doc.serialize()).toBe(after);
            }),
            { numRuns: 100 },
        );
    });

    test("serialize∘parse is a fixed point for the flat scene format", () => {
        clearRegistry();
        fc.assert(
            fc.property(fc.array(flatOp as fc.Arbitrary<Op>, { maxLength: 30 }), (ops) => {
                const doc = new Document(Seed);
                for (const op of ops) applyOp(doc, op);
                const once = doc.serialize();
                expect(new Document(once).serialize()).toBe(once);
            }),
            { numRuns: 100 },
        );
    });
});
