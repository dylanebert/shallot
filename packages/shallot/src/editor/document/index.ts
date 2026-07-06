import { type Node, parse, stringify } from "../../engine/scene";
import {
    type Command,
    type Compound,
    clear,
    deselect,
    execute,
    type History,
    redo,
    select,
    undo,
} from "./commands";

export type { Command } from "./commands";
export { Readback, ReadbackSystem, Session } from "./session";
export type { Node };

/**
 * a scene document: the parsed node tree plus its undo/redo history, selection, and gesture buffer. Every
 * mutating method (`add`, `setAttr`, `reorder`, …) records a reversible {@link Command} and bumps
 * `version` so a UI can react. Wrap a drag or multi-field scrub in `begin`/`commit` to coalesce its writes
 * into one undo step.
 *
 * @example
 * ```
 * const doc = new Document(sceneXml);
 * doc.setAttr(node, "camera", "fov: 60");
 * doc.undo();
 * const xml = doc.serialize();
 * ```
 */
export class Document {
    nodes: Node[];
    history: History = { undo: [], redo: [] };
    selection: Set<Node> = new Set();
    version = 0;
    // an open gesture's buffered field writes, keyed node → attr → { prev (first touch), next (last) };
    // null when no gesture is in flight. See begin / commit / cancel.
    private _gesture: Map<Node, Map<string, { prev: string; next: string }>> | null = null;

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
        const index = node.attrs.findIndex((a) => a.name === name);
        if (index < 0) return;
        execute(
            this.history,
            this.nodes,
            { type: "removeAttr", node, name, prev: node.attrs[index].value, index },
            [...this.selection],
        );
        this.version++;
    }

    setAttr(node: Node, name: string, value: string, prev?: string): void {
        const attr = node.attrs.find((a) => a.name === name);
        if (!attr) return;
        if (this._gesture) {
            let attrs = this._gesture.get(node);
            if (!attrs) {
                attrs = new Map();
                this._gesture.set(node, attrs);
            }
            const buffered = attrs.get(name);
            // first touch holds the pristine prev; the gesture coalesces to one prev→next. attr.value is
            // left untouched here — readback is its sole live mirror; commit records + applies the final.
            if (buffered) buffered.next = value;
            else attrs.set(name, { prev: prev ?? attr.value, next: value });
            return;
        }
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

    /**
     * open a gesture: `setAttr` calls until {@link commit} coalesce into one undoable entry, so a
     * multi-entity drag or an inspector scrub is a single undo step. Prevs are captured automatically at
     * first touch — issue the first write before any live mirror moves the value. Gestures don't nest; a
     * `begin` while one is open is a no-op.
     */
    begin(): void {
        if (!this._gesture) this._gesture = new Map();
    }

    /**
     * close the open gesture, pushing its buffered field writes as one undoable entry: nothing if they net
     * to no change, a plain `setAttr` for a single touched attr, a `compound` for several.
     */
    commit(): void {
        const gesture = this._gesture;
        if (!gesture) return;
        this._gesture = null;
        const commands: Command[] = [];
        for (const [node, attrs] of gesture) {
            for (const [name, { prev, next }] of attrs) {
                if (prev !== next) commands.push({ type: "setAttr", node, name, prev, next });
            }
        }
        if (commands.length === 0) return;
        const cmd: Command = commands.length === 1 ? commands[0] : { type: "compound", commands };
        execute(this.history, this.nodes, cmd, [...this.selection]);
        this.version++;
    }

    /**
     * abort the open gesture, recording no history. The gesture never wrote `attr.value` (readback owns the
     * live mirror), so dropping the buffer is the rollback; a caller restoring the live ECS to the
     * pre-gesture pose lets readback un-mirror it.
     */
    cancel(): void {
        if (!this._gesture) return;
        this._gesture = null;
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
        return stringify(this.nodes);
    }
}
