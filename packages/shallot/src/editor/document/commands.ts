import type { Node } from "../../engine/scene";

type Add = { type: "add"; parent: Node | null; node: Node; index: number };
type Remove = { type: "remove"; parent: Node | null; node: Node; index: number };
type AddAttr = { type: "addAttr"; node: Node; name: string; value: string };
type RemoveAttr = { type: "removeAttr"; node: Node; name: string; prev: string; index: number };
type SetAttr = { type: "setAttr"; node: Node; name: string; prev: string; next: string };
type SetId = { type: "setId"; node: Node; prev: string | undefined; next: string | undefined };
type Reorder = { type: "reorder"; parent: Node | null; node: Node; from: number; to: number };
type Reparent = {
    type: "reparent";
    node: Node;
    oldParent: Node | null;
    oldIndex: number;
    newParent: Node | null;
    newIndex: number;
};
type ReorderAttr = { type: "reorderAttr"; node: Node; from: number; to: number };
export type Compound = { type: "compound"; commands: Command[] };

/**
 * a reversible scene-tree edit: add/remove a node, add/remove/set an attribute, set an id, reorder or
 * reparent, or a `compound` batch. Every case names the data needed to both apply and reverse it, so the
 * undo stack replays without re-deriving state.
 */
export type Command =
    | Add
    | Remove
    | AddAttr
    | RemoveAttr
    | SetAttr
    | SetId
    | Reorder
    | Reparent
    | ReorderAttr
    | Compound;

export type Entry = { cmd: Command; selection: Node[] };
export type History = { undo: Entry[]; redo: Entry[] };

function getChildren(parent: Node | null, nodes: Node[]): Node[] {
    return parent ? parent.children : nodes;
}

export function apply(nodes: Node[], cmd: Command): void {
    switch (cmd.type) {
        case "add": {
            const children = getChildren(cmd.parent, nodes);
            children.splice(cmd.index, 0, cmd.node);
            break;
        }
        case "remove": {
            const children = getChildren(cmd.parent, nodes);
            children.splice(cmd.index, 1);
            break;
        }
        case "addAttr": {
            cmd.node.attrs.push({ name: cmd.name, value: cmd.value });
            break;
        }
        case "removeAttr": {
            cmd.node.attrs.splice(cmd.index, 1);
            break;
        }
        case "setAttr": {
            const idx = cmd.node.attrs.findIndex((a) => a.name === cmd.name);
            if (idx >= 0) cmd.node.attrs[idx].value = cmd.next;
            break;
        }
        case "setId": {
            cmd.node.id = cmd.next;
            break;
        }
        case "reorder": {
            const children = getChildren(cmd.parent, nodes);
            children.splice(cmd.from, 1);
            children.splice(cmd.to, 0, cmd.node);
            break;
        }
        case "reparent": {
            const oldChildren = getChildren(cmd.oldParent, nodes);
            oldChildren.splice(cmd.oldIndex, 1);
            const newChildren = getChildren(cmd.newParent, nodes);
            newChildren.splice(cmd.newIndex, 0, cmd.node);
            break;
        }
        case "reorderAttr": {
            const [attr] = cmd.node.attrs.splice(cmd.from, 1);
            cmd.node.attrs.splice(cmd.to, 0, attr);
            break;
        }
        case "compound":
            for (const sub of cmd.commands) apply(nodes, sub);
            break;
    }
}

export function reverse(nodes: Node[], cmd: Command): void {
    switch (cmd.type) {
        case "add": {
            const children = getChildren(cmd.parent, nodes);
            children.splice(cmd.index, 1);
            break;
        }
        case "remove": {
            const children = getChildren(cmd.parent, nodes);
            children.splice(cmd.index, 0, cmd.node);
            break;
        }
        case "addAttr": {
            const idx = cmd.node.attrs.findIndex((a) => a.name === cmd.name);
            if (idx >= 0) cmd.node.attrs.splice(idx, 1);
            break;
        }
        case "removeAttr": {
            cmd.node.attrs.splice(cmd.index, 0, { name: cmd.name, value: cmd.prev });
            break;
        }
        case "setAttr": {
            const idx = cmd.node.attrs.findIndex((a) => a.name === cmd.name);
            if (idx >= 0) cmd.node.attrs[idx].value = cmd.prev;
            break;
        }
        case "setId": {
            cmd.node.id = cmd.prev;
            break;
        }
        case "reorder": {
            const children = getChildren(cmd.parent, nodes);
            children.splice(cmd.to, 1);
            children.splice(cmd.from, 0, cmd.node);
            break;
        }
        case "reparent": {
            const newChildren = getChildren(cmd.newParent, nodes);
            newChildren.splice(cmd.newIndex, 1);
            const oldChildren = getChildren(cmd.oldParent, nodes);
            oldChildren.splice(cmd.oldIndex, 0, cmd.node);
            break;
        }
        case "reorderAttr": {
            const [attr] = cmd.node.attrs.splice(cmd.to, 1);
            cmd.node.attrs.splice(cmd.from, 0, attr);
            break;
        }
        case "compound":
            for (let i = cmd.commands.length - 1; i >= 0; i--) reverse(nodes, cmd.commands[i]);
            break;
    }
}

const MAX_UNDO = 256;

export function execute(history: History, nodes: Node[], cmd: Command, selection?: Node[]): void {
    apply(nodes, cmd);
    history.undo.push({ cmd, selection: selection ?? [] });
    if (history.undo.length > MAX_UNDO) history.undo.shift();
    history.redo.length = 0;
}

export function undo(history: History, nodes: Node[]): Entry | undefined {
    const entry = history.undo.pop();
    if (!entry) return;
    reverse(nodes, entry.cmd);
    history.redo.push(entry);
    return entry;
}

export function redo(history: History, nodes: Node[]): Entry | undefined {
    const entry = history.redo.pop();
    if (!entry) return;
    apply(nodes, entry.cmd);
    history.undo.push(entry);
    return entry;
}

export function select(selection: Set<Node>, ...nodesToSelect: Node[]): void {
    for (const node of nodesToSelect) {
        selection.add(node);
    }
}

export function deselect(selection: Set<Node>, ...nodesToDeselect: Node[]): void {
    for (const node of nodesToDeselect) {
        selection.delete(node);
    }
}

export function clear(selection: Set<Node>): void {
    selection.clear();
}
