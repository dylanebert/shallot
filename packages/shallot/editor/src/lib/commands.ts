import { parse, type State } from "@dylanebert/shallot";
import { find, inspect, snapshot } from "@dylanebert/shallot/ecs/core";
import type { Command, Document, Node } from "@dylanebert/shallot/editor";
import { findNodeById, findParent } from "@dylanebert/shallot/scene/core";

/**
 * the editor's control seam: the request payloads an agent (or a test) sends over `shallot:request`,
 * and later the command palette's action registry, both route through here. {@link applyCommand} is
 * pure over the Document with its side effects injected as {@link CommandContext}, so the dispatch is
 * `bun test`-covered without the editor mounted.
 */
export interface CommandPayload {
    method: string;
    args?: Record<string, unknown>;
}

export interface CommandResult {
    ok: boolean;
    version?: number;
    error?: string;
}

/** the App-side effects applyCommand reaches back into: undo/redo, the ECS session sync, the version bump */
export interface CommandContext {
    undo(): void;
    redo(): void;
    sync(cmd: Command, isUndo: boolean): boolean;
    bump(): void;
}

export function applyCommand(
    doc: Document,
    payload: CommandPayload,
    ctx: CommandContext,
): CommandResult {
    const { method, args = {} } = payload;

    if (method === "undo") {
        ctx.undo();
        return { ok: true, version: doc.version };
    }
    if (method === "redo") {
        ctx.redo();
        return { ok: true, version: doc.version };
    }
    if (method === "clearSelection") {
        doc.clearSelection();
        ctx.bump();
        return { ok: true, version: doc.version };
    }
    if (method === "select" || method === "deselect") {
        const nodes = ((args.ids ?? []) as string[])
            .map((id) => findNodeById(id, doc.nodes))
            .filter(Boolean) as Node[];
        if (method === "select") doc.select(...nodes);
        else doc.deselect(...nodes);
        ctx.bump();
        return { ok: true, version: doc.version };
    }

    const node = args.id ? findNodeById(args.id as string, doc.nodes) : undefined;
    const prevLen = doc.history.undo.length;

    switch (method) {
        case "add": {
            const newNode = parse((args.xml as string) ?? "<a />")[0];
            if (!newNode) return { ok: false, error: "Invalid XML" };
            const parent = args.parent
                ? (findNodeById(args.parent as string, doc.nodes) ?? null)
                : null;
            doc.add(parent, newNode, args.index as number | undefined);
            break;
        }
        case "remove": {
            if (!node) return { ok: false, error: `Node not found: ${args.id}` };
            const parent = findParent(node, doc.nodes, null);
            if (parent === undefined) return { ok: false, error: "Node not in tree" };
            doc.remove(parent, node);
            break;
        }
        case "setAttr": {
            if (!node) return { ok: false, error: `Node not found: ${args.id}` };
            doc.setAttr(node, args.name as string, args.value as string);
            break;
        }
        case "addAttr": {
            if (!node) return { ok: false, error: `Node not found: ${args.id}` };
            doc.addAttr(node, args.name as string, (args.value as string) ?? "");
            break;
        }
        case "removeAttr": {
            if (!node) return { ok: false, error: `Node not found: ${args.id}` };
            doc.removeAttr(node, args.name as string);
            break;
        }
        case "setId": {
            if (!node) return { ok: false, error: `Node not found: ${args.id}` };
            doc.setId(node, args.newId as string | undefined);
            break;
        }
        case "reorder": {
            if (!node) return { ok: false, error: `Node not found: ${args.id}` };
            const parent = args.parent
                ? (findNodeById(args.parent as string, doc.nodes) ?? null)
                : null;
            doc.reorder(parent, node, args.to as number);
            break;
        }
        default:
            return { ok: false, error: `Unknown method: ${method}` };
    }

    // a mutation that recorded history syncs its command into the live ECS, the way a UI gesture would
    if (doc.history.undo.length > prevLen) {
        const cmd = doc.history.undo[doc.history.undo.length - 1].cmd;
        ctx.sync(cmd, false);
    }
    ctx.bump();
    return { ok: true, version: doc.version };
}

/** the control seam's read side: inspect one entity, find by component, or snapshot the whole ECS */
export function queryEntities(
    ecs: State | null,
    payload: { component?: string; eid?: number } | null,
): unknown {
    if (!ecs) return { error: "No engine running" };
    if (payload?.eid != null) return inspect(ecs, payload.eid);
    if (payload?.component) return find(ecs, payload.component);
    return snapshot(ecs);
}
