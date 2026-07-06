import type { State, System } from "../../engine";
import { getComponent, getTraits } from "../../engine/ecs/core";
import type { Node } from "../../engine/scene";
import { parseFields, readComponent, setFieldValue } from "../../engine/scene/core";
import type { Command } from "./commands";
import type { Document } from "./index";

const EDIT_SKIP = new Set(["transform", "orbit"]);

/** the live-edit readback singleton: the active {@link Session} (null when no scene is open) and an
 *  optional callback fired when a frame's readback mutated any authored attribute. */
export interface Readback {
    session: Session | null;
    onUpdate?: () => void;
}

export const Readback: Readback = {
    session: null,
};

/** reflects live ECS field values back onto the scene attributes each node already authors, so an edit
 *  driven through the engine (a gizmo drag, a running system) shows up in the document. Runs last in the
 *  `simulation` group; a no-op until a scene sets {@link Readback}`.session`. */
export const ReadbackSystem: System = {
    group: "simulation",
    last: true,
    annotations: { mode: "always" },
    update(state: State) {
        const { session, onUpdate } = Readback;
        if (!session) return;
        const editing = state.mode === "edit";
        let changed = false;
        // reflect live field values back onto the attrs the node *already authors* — never auto-add a
        // component the node doesn't list. A component on the entity but absent from the node is derived
        // (a `warm` spawn / a system's add), and the document is the authoring truth: the editor's
        // add-component flow writes the doc attr first (Inspector `addAttr`), so a legitimately-added
        // component is already here. Auto-adding the rest leaks derived state into the saved file and
        // diverges the document path from `serialize(state)` (which emits the authored set).
        for (const [node, eid] of session.nodeMap) {
            const isCamera = editing && node.attrs.some((a) => a.name === "camera");
            for (let i = 0; i < node.attrs.length; i++) {
                const attr = node.attrs[i];
                if (isCamera && EDIT_SKIP.has(attr.name)) continue;
                if (attr.value.includes("@")) continue;
                const component = getComponent(attr.name);
                if (!component) continue;
                if (!session.state.has(eid, component as never)) continue;

                const formatted = readComponent(attr.name, component, eid);
                if (formatted !== attr.value) {
                    attr.value = formatted;
                    changed = true;
                }
            }
        }
        if (changed) onUpdate?.();
    },
};

/**
 * the editor↔engine bridge: maps document {@link Node}s to live entity ids and applies document commands
 * to the running {@link State}. It owns the reflection reach (getComponent / setFieldValue) so the rest of
 * the editor stays out of the ECS internals. The inspector's live field preview, component attach/detach,
 * and command replay for undo/redo all route through here.
 */
export class Session {
    state: State;
    document: Document;
    nodeMap: Map<Node, number>;

    constructor(state: State, document: Document, nodeMap: Map<Node, number>) {
        this.state = state;
        this.document = document;
        this.nodeMap = nodeMap;
    }

    syncAttr(name: string, value: string, eid: number): void {
        const component = getComponent(name);
        if (!component) return;
        const traits = getTraits(name);
        const defaults = traits?.defaults?.() ?? {};
        const parsed: Record<string, number | string | readonly number[]> = value
            ? { ...defaults, ...parseFields(name, value) }
            : { ...defaults };
        for (const [field, val] of Object.entries(parsed)) {
            if (typeof val === "number" || Array.isArray(val)) {
                setFieldValue(component, field, eid, val as number | number[]);
            } else if (typeof val === "string") {
                const parseFn = traits?.parse?.[field];
                const id = parseFn?.(val);
                if (id !== undefined) setFieldValue(component, field, eid, id);
            }
        }
    }

    /**
     * apply a partial field map to the live ECS — the inspector's per-field live preview during a drag /
     * input gesture. Keys are field names (`intensity`) or dotted Pair/Quad lanes (`pos.x`), matching what
     * the inspector emits; string values resolve through the trait parser (`@name` refs, named enums). The
     * reflection reach (getComponent / setFieldValue) lives here so the editor↔engine seam stays in Session.
     */
    syncFields(name: string, fields: Record<string, number | string>, eid: number): void {
        const component = getComponent(name);
        if (!component) return;
        const traits = getTraits(name);
        for (const [field, value] of Object.entries(fields)) {
            if (typeof value === "number") {
                setFieldValue(component, field, eid, value);
            } else {
                const id = traits?.parse?.[field]?.(value);
                if (id !== undefined) setFieldValue(component, field, eid, id);
            }
        }
    }

    attachComponent(name: string, value: string, eid: number): void {
        const component = getComponent(name);
        if (!component) return;
        this.state.add(eid, component as never);
        const defaults = getTraits(name)?.defaults?.() ?? {};
        for (const [field, val] of Object.entries(defaults)) {
            setFieldValue(component, field, eid, val as number | number[]);
        }
        if (value) this.syncAttr(name, value, eid);
    }

    loadNode(node: Node, _parent: Node | null): void {
        const eid = this.state.create();
        this.nodeMap.set(node, eid);
        for (const attr of node.attrs) this.attachComponent(attr.name, attr.value, eid);
    }

    unloadNode(node: Node): void {
        const eid = this.nodeMap.get(node);
        if (eid !== undefined) {
            this.state.destroy(eid);
            this.nodeMap.delete(node);
        }
    }

    syncCommand(cmd: Command, isUndo: boolean): boolean {
        switch (cmd.type) {
            case "setAttr": {
                const eid = this.nodeMap.get(cmd.node);
                if (eid === undefined) return false;
                this.syncAttr(cmd.name, isUndo ? cmd.prev : cmd.next, eid);
                return true;
            }
            case "setId":
                return true;
            case "addAttr": {
                const eid = this.nodeMap.get(cmd.node);
                if (eid === undefined) return false;
                if (isUndo) {
                    const component = getComponent(cmd.name);
                    if (component) this.state.remove(eid, component as never);
                } else {
                    this.attachComponent(cmd.name, cmd.value, eid);
                }
                return true;
            }
            case "removeAttr": {
                const eid = this.nodeMap.get(cmd.node);
                if (eid === undefined) return false;
                if (isUndo) {
                    this.attachComponent(cmd.name, cmd.prev, eid);
                } else {
                    const component = getComponent(cmd.name);
                    if (component) this.state.remove(eid, component as never);
                }
                return true;
            }
            case "add": {
                if (isUndo) this.unloadNode(cmd.node);
                else this.loadNode(cmd.node, cmd.parent);
                return true;
            }
            case "remove": {
                if (isUndo) this.loadNode(cmd.node, cmd.parent);
                else this.unloadNode(cmd.node);
                return true;
            }
            case "reparent": {
                return true;
            }
            case "reorder":
            case "reorderAttr":
                return true;
            case "compound": {
                const subs = isUndo ? [...cmd.commands].reverse() : cmd.commands;
                return subs.every((sub) => this.syncCommand(sub, isUndo));
            }
        }
    }
}
