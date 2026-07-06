import { describe, expect, test } from "bun:test";
import type { Diagnostic } from "@dylanebert/shallot";
import type { Node } from "@dylanebert/shallot/editor";
import { rows } from "./rows";

function node(id: string | undefined, attrs: string[] = [], children: Node[] = []): Node {
    return { id, attrs: attrs.map((name) => ({ name, value: "" })), children };
}

const NONE = new WeakSet<Node>();

describe("rows — labels", () => {
    test("a node with an id titles by id", () => {
        const [row] = rows([node("camera-rig", ["camera"])], new Set(), NONE, []);
        expect(row.label).toBe("camera-rig");
    });

    test("an id-less node titles by its hero component type, not 'entity'", () => {
        const [row] = rows([node(undefined, ["camera"])], new Set(), NONE, []);
        expect(row.label).toBe("camera");
    });

    test("an id-less node with no hero type falls back to its first component", () => {
        const [row] = rows([node(undefined, ["transform"])], new Set(), NONE, []);
        expect(row.label).toBe("transform");
    });

    test("a truly empty node titles 'entity'", () => {
        const [row] = rows([node(undefined, [])], new Set(), NONE, []);
        expect(row.label).toBe("entity");
    });
});

describe("rows — tree", () => {
    test("flattens depth-first with depth + parent tags", () => {
        const child = node("child", ["part"]);
        const parent = node("parent", ["part"], [child]);

        const result = rows([parent], new Set(), NONE, []);

        expect(result.map((r) => [r.label, r.depth])).toEqual([
            ["parent", 0],
            ["child", 1],
        ]);
        expect(result[1].parent).toBe(parent);
    });

    test("a collapsed node hides its children and reports expanded=false", () => {
        const child = node("child", ["part"]);
        const parent = node("parent", ["part"], [child]);
        const collapsed = new WeakSet<Node>([parent]);

        const result = rows([parent], new Set(), collapsed, []);

        expect(result.map((r) => r.label)).toEqual(["parent"]);
        expect(result[0]).toMatchObject({ hasChildren: true, expanded: false });
    });

    test("marks selection and diagnostics per row", () => {
        const a = node("a", ["part"]);
        const b = node("b", ["wobble"]);
        const diag: Diagnostic = { node: b, attr: "wobble", kind: "unregistered", message: "x" };

        const result = rows([a, b], new Set([a]), NONE, [diag]);

        expect(result.find((r) => r.node === a)).toMatchObject({ selected: true, warning: false });
        expect(result.find((r) => r.node === b)).toMatchObject({ selected: false, warning: true });
    });
});
