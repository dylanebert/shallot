import { describe, expect, test } from "bun:test";
import { Document, type Node } from "@dylanebert/shallot/editor";
import { applyCommand, type CommandContext, queryEntities } from "./commands";

function spyCtx() {
    const calls = { undo: 0, redo: 0, bump: 0, sync: [] as [unknown, boolean][] };
    const ctx: CommandContext = {
        undo: () => {
            calls.undo++;
        },
        redo: () => {
            calls.redo++;
        },
        sync: (cmd, isUndo) => {
            calls.sync.push([cmd, isUndo]);
            return true;
        },
        bump: () => {
            calls.bump++;
        },
    };
    return { ctx, calls };
}

const boxScene = (): Node[] => [
    { id: "box", attrs: [{ name: "mesh", value: "shape: box" }], children: [] },
];

describe("applyCommand", () => {
    test("setAttr mutates the doc, syncs the new command, and bumps the version", () => {
        const doc = new Document(boxScene());
        const { ctx, calls } = spyCtx();
        const r = applyCommand(
            doc,
            { method: "setAttr", args: { id: "box", name: "mesh", value: "shape: sphere" } },
            ctx,
        );
        expect(r.ok).toBe(true);
        expect(doc.nodes[0].attrs[0].value).toBe("shape: sphere");
        expect(calls.sync.length).toBe(1);
        expect(calls.sync[0][1]).toBe(false);
        expect(calls.bump).toBe(1);
    });

    test("add inserts the parsed node and syncs it", () => {
        const doc = new Document([] as Node[]);
        const { ctx, calls } = spyCtx();
        const r = applyCommand(doc, { method: "add", args: { xml: '<a id="new" />' } }, ctx);
        expect(r.ok).toBe(true);
        expect(doc.nodes.length).toBe(1);
        expect(doc.nodes[0].id).toBe("new");
        expect(calls.sync.length).toBe(1);
    });

    test("add with no xml inserts a blank entity from the default", () => {
        const doc = new Document([] as Node[]);
        const { ctx } = spyCtx();
        const r = applyCommand(doc, { method: "add" }, ctx);
        expect(r.ok).toBe(true);
        expect(doc.nodes.length).toBe(1);
    });

    test("select updates the selection and bumps, without syncing the ECS", () => {
        const doc = new Document(boxScene());
        const { ctx, calls } = spyCtx();
        const r = applyCommand(doc, { method: "select", args: { ids: ["box"] } }, ctx);
        expect(r.ok).toBe(true);
        expect(doc.selection.has(doc.nodes[0])).toBe(true);
        expect(calls.bump).toBe(1);
        expect(calls.sync.length).toBe(0);
    });

    test("undo/redo delegate to the context rather than the doc directly", () => {
        const doc = new Document(boxScene());
        const { ctx, calls } = spyCtx();
        applyCommand(doc, { method: "undo" }, ctx);
        applyCommand(doc, { method: "redo" }, ctx);
        expect(calls.undo).toBe(1);
        expect(calls.redo).toBe(1);
    });

    test("an unknown method is a structured failure", () => {
        const doc = new Document([] as Node[]);
        const { ctx } = spyCtx();
        const r = applyCommand(doc, { method: "frobnicate" }, ctx);
        expect(r.ok).toBe(false);
        expect(r.error).toContain("Unknown method");
    });

    test("a missing node id is a structured failure naming the id", () => {
        const doc = new Document([] as Node[]);
        const { ctx } = spyCtx();
        const r = applyCommand(
            doc,
            { method: "setAttr", args: { id: "ghost", name: "mesh", value: "x" } },
            ctx,
        );
        expect(r.ok).toBe(false);
        expect(r.error).toContain("ghost");
    });
});

describe("queryEntities", () => {
    test("reports no engine when the ECS is absent", () => {
        expect(queryEntities(null, null)).toEqual({ error: "No engine running" });
    });
});
