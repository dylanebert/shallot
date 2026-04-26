import { test, expect, describe, beforeEach } from "bun:test";
import type { Node } from "../src/engine/scene";
import {
    apply,
    reverse,
    execute,
    undo,
    redo,
    select,
    deselect,
    clear,
    Document,
    Session,
    type Command,
    type History,
} from "../src/editor/document";
import { State } from "../src/engine/ecs/state";
import { clearRegistry, registerComponent } from "../src/engine/ecs/component";
import { Haze } from "../src/standard/render/camera";

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
            };

            apply(nodes, cmd);

            expect(node.attrs).toHaveLength(1);
            expect(node.attrs[0].name).toBe("velocity");
        });

        test("reverse re-inserts attr", () => {
            const node = createNode("a", [{ name: "velocity", value: "x: 5" }]);
            const nodes: Node[] = [node];
            const cmd: Command = {
                type: "removeAttr",
                node,
                name: "position",
                prev: "x: 10",
            };

            reverse(nodes, cmd);

            expect(node.attrs).toHaveLength(2);
            expect(node.attrs[1].name).toBe("position");
            expect(node.attrs[1].value).toBe("x: 10");
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
            const doc = new Document(
                '<scene><a id="parent"><a id="a" /><a id="b" /><a id="c" /></a></scene>',
            );
            const parent = doc.nodes[0];
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
            const doc = new Document('<scene><a id="parent"><a id="child" /></a></scene>');
            const parent = doc.nodes[0];
            const child = parent.children[0];
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
        registerComponent("Haze", Haze);
        state = new State();
        node = createNode("cam");
        doc = new Document([node]);
        eid = state.addEntity();
        const nodeMap = new Map<Node, number>([[node, eid]]);
        session = new Session(state, doc, nodeMap);
    });

    test("attachComponent adds component to ECS", () => {
        session.attachComponent("haze", "density: 0.01", eid);
        expect(state.hasComponent(eid, Haze as never)).toBe(true);
        expect(Haze.density[eid]).toBeCloseTo(0.01);
    });

    test("attachComponent applies defaults when value empty", () => {
        session.attachComponent("haze", "", eid);
        expect(state.hasComponent(eid, Haze as never)).toBe(true);
        expect(Haze.density[eid]).toBeCloseTo(0.005);
        expect(Haze.color[eid]).toBe(0x4078d0);
    });

    test("attachComponent skips unknown component", () => {
        session.attachComponent("nonexistent", "foo: 1", eid);
        expect(state.getEntityComponents(eid)).toHaveLength(0);
    });

    test("syncCommand addAttr adds component", () => {
        doc.addAttr(node, "haze", "density: 0.01");
        const cmd = doc.history.undo[doc.history.undo.length - 1].cmd;
        session.syncCommand(cmd, false);
        expect(state.hasComponent(eid, Haze as never)).toBe(true);
        expect(Haze.density[eid]).toBeCloseTo(0.01);
    });

    test("syncCommand addAttr undo removes component", () => {
        doc.addAttr(node, "haze", "density: 0.01");
        const cmd = doc.history.undo[doc.history.undo.length - 1].cmd;
        session.syncCommand(cmd, false);
        expect(state.hasComponent(eid, Haze as never)).toBe(true);

        session.syncCommand(cmd, true);
        expect(state.hasComponent(eid, Haze as never)).toBe(false);
    });

    test("syncCommand removeAttr removes component", () => {
        session.attachComponent("haze", "density: 0.01", eid);
        node.attrs.push({ name: "haze", value: "density: 0.01" });
        doc.removeAttr(node, "haze");
        const cmd = doc.history.undo[doc.history.undo.length - 1].cmd;
        session.syncCommand(cmd, false);
        expect(state.hasComponent(eid, Haze as never)).toBe(false);
    });

    test("syncCommand removeAttr undo restores component", () => {
        session.attachComponent("haze", "density: 0.02", eid);
        node.attrs.push({ name: "haze", value: "density: 0.02" });
        doc.removeAttr(node, "haze");
        const cmd = doc.history.undo[doc.history.undo.length - 1].cmd;
        session.syncCommand(cmd, false);
        expect(state.hasComponent(eid, Haze as never)).toBe(false);

        session.syncCommand(cmd, true);
        expect(state.hasComponent(eid, Haze as never)).toBe(true);
        expect(Haze.density[eid]).toBeCloseTo(0.02);
    });

    test("syncCommand compound adds multiple components", () => {
        doc.compound([{ type: "addAttr", node, name: "haze", value: "density: 0.01" }]);
        const cmd = doc.history.undo[doc.history.undo.length - 1].cmd;
        session.syncCommand(cmd, false);
        expect(state.hasComponent(eid, Haze as never)).toBe(true);
        expect(Haze.density[eid]).toBeCloseTo(0.01);
    });

    test("syncCommand compound undo removes all in reverse", () => {
        doc.compound([{ type: "addAttr", node, name: "haze", value: "density: 0.01" }]);
        const cmd = doc.history.undo[doc.history.undo.length - 1].cmd;
        session.syncCommand(cmd, false);
        expect(state.hasComponent(eid, Haze as never)).toBe(true);

        session.syncCommand(cmd, true);
        expect(state.hasComponent(eid, Haze as never)).toBe(false);
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
