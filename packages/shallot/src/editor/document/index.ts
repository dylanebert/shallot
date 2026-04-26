import { type Node, parse, serialize } from "../../engine/scene";
export type { Node };

export { Session, ReadbackSystem, Readback } from "./session";

type Add = { type: "add"; parent: Node | null; node: Node; index: number };
type Remove = { type: "remove"; parent: Node | null; node: Node; index: number };
type AddAttr = { type: "addAttr"; node: Node; name: string; value: string };
type RemoveAttr = { type: "removeAttr"; node: Node; name: string; prev: string };
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
type Compound = { type: "compound"; commands: Command[] };

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
            const idx = cmd.node.attrs.findIndex((a) => a.name === cmd.name);
            if (idx >= 0) cmd.node.attrs.splice(idx, 1);
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
            cmd.node.attrs.push({ name: cmd.name, value: cmd.prev });
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

export class Document {
    nodes: Node[];
    history: History = { undo: [], redo: [] };
    selection: Set<Node> = new Set();
    version = 0;

    constructor(source: string | Node[]) {
        this.nodes = typeof source === "string" ? parse(source) : source;
    }

    add(parent: Node | null, node: Node, index?: number): void {
        const children = parent ? parent.children : this.nodes;
        const idx = index ?? children.length;
        execute(this.history, this.nodes, { type: "add", parent, node, index: idx }, [
            ...this.selection,
        ]);
        this.version++;
    }

    remove(parent: Node | null, node: Node): void {
        const children = parent ? parent.children : this.nodes;
        const index = children.indexOf(node);
        if (index < 0) return;
        execute(this.history, this.nodes, { type: "remove", parent, node, index }, [
            ...this.selection,
        ]);
        this.version++;
    }

    addAttr(node: Node, name: string, value: string): void {
        execute(this.history, this.nodes, { type: "addAttr", node, name, value }, [
            ...this.selection,
        ]);
        this.version++;
    }

    removeAttr(node: Node, name: string): void {
        const attr = node.attrs.find((a) => a.name === name);
        if (!attr) return;
        execute(this.history, this.nodes, { type: "removeAttr", node, name, prev: attr.value }, [
            ...this.selection,
        ]);
        this.version++;
    }

    setAttr(node: Node, name: string, value: string, prev?: string): void {
        const attr = node.attrs.find((a) => a.name === name);
        if (!attr) return;
        execute(
            this.history,
            this.nodes,
            {
                type: "setAttr",
                node,
                name,
                prev: prev ?? attr.value,
                next: value,
            },
            [...this.selection],
        );
        this.version++;
    }

    setId(node: Node, id: string | undefined): void {
        execute(this.history, this.nodes, { type: "setId", node, prev: node.id, next: id }, [
            ...this.selection,
        ]);
        this.version++;
    }

    reorder(parent: Node | null, node: Node, to: number): void {
        const children = parent ? parent.children : this.nodes;
        const from = children.indexOf(node);
        if (from < 0 || from === to) return;
        execute(this.history, this.nodes, { type: "reorder", parent, node, from, to }, [
            ...this.selection,
        ]);
        this.version++;
    }

    reparent(node: Node, oldParent: Node | null, newParent: Node | null, newIndex: number): void {
        const oldChildren = oldParent ? oldParent.children : this.nodes;
        const oldIndex = oldChildren.indexOf(node);
        if (oldIndex < 0) return;
        execute(
            this.history,
            this.nodes,
            {
                type: "reparent",
                node,
                oldParent,
                oldIndex,
                newParent,
                newIndex,
            },
            [...this.selection],
        );
        this.version++;
    }

    compound(commands: Command[]): void {
        const cmd: Compound = { type: "compound", commands };
        execute(this.history, this.nodes, cmd, [...this.selection]);
        this.version++;
    }

    reorderAttr(node: Node, from: number, to: number): void {
        if (
            from < 0 ||
            to < 0 ||
            from >= node.attrs.length ||
            to >= node.attrs.length ||
            from === to
        )
            return;
        execute(this.history, this.nodes, { type: "reorderAttr", node, from, to }, [
            ...this.selection,
        ]);
        this.version++;
    }

    undo(): Command | undefined {
        const entry = undo(this.history, this.nodes);
        if (!entry) return;
        this.selection.clear();
        for (const node of entry.selection) this.selection.add(node);
        this.version++;
        return entry.cmd;
    }

    redo(): Command | undefined {
        const entry = redo(this.history, this.nodes);
        if (!entry) return;
        this.selection.clear();
        if (entry.cmd.type === "compound") {
            const first = entry.cmd.commands[0];
            if (first && first.type !== "remove" && first.type !== "compound") {
                this.selection.add(first.node);
            }
        } else if (entry.cmd.type !== "remove") {
            this.selection.add(entry.cmd.node);
        }
        this.version++;
        return entry.cmd;
    }

    select(...nodesToSelect: Node[]): void {
        select(this.selection, ...nodesToSelect);
        this.version++;
    }

    deselect(...nodesToDeselect: Node[]): void {
        deselect(this.selection, ...nodesToDeselect);
        this.version++;
    }

    clearSelection(): void {
        clear(this.selection);
        this.version++;
    }

    serialize(): string {
        return serialize(this.nodes);
    }
}
