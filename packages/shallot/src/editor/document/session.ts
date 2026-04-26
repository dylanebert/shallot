import { resource, ChildOf, type State, type System } from "../../engine";
import type { Node } from "../../engine/scene";
import { getComponent, getComponents, readFields } from "../../engine/ecs/core";
import { setFieldValue, setString, parseFields, formatFields } from "../../engine/scene";
import type { Document, Command } from "./index";

const EDIT_SKIP = new Set(["transform", "orbit"]);

interface ReadbackState {
    session: Session;
    onUpdate?: () => void;
}

export const Readback = resource<ReadbackState>("readback");

export const ReadbackSystem: System = {
    group: "simulation",
    last: true,
    annotations: { mode: "always" },
    update(state: State) {
        const res = Readback.from(state);
        if (!res) return;
        const { session, onUpdate } = res;
        const editing = state.scheduler.mode === "edit";
        let changed = false;
        for (const [node, eid] of session.nodeMap) {
            const isCamera = editing && node.attrs.some((a) => a.name === "camera");
            for (let i = 0; i < node.attrs.length; i++) {
                const attr = node.attrs[i];
                if (isCamera && EDIT_SKIP.has(attr.name)) continue;
                if (attr.value.includes("@")) continue;
                const reg = getComponent(attr.name);
                if (!reg) continue;
                if (!session.state.hasComponent(eid, reg.component as never)) continue;

                const defaults = reg.traits?.defaults?.() ?? {};
                const fields = readFields(reg.component, eid);
                const merged = { ...defaults, ...fields };
                const formatted = formatFields(attr.name, merged);

                if (formatted !== attr.value) {
                    attr.value = formatted;
                    changed = true;
                }
            }
        }
        for (const [node, eid] of session.nodeMap) {
            const existing = new Set(node.attrs.map((a) => a.name));
            for (const { name, component } of getComponents()) {
                if (existing.has(name)) continue;
                if (!session.state.hasComponent(eid, component as never)) continue;
                const reg = getComponent(name);
                if (!reg) continue;
                const defaults = reg.traits?.defaults?.() ?? {};
                const fields = readFields(reg.component, eid);
                const merged = { ...defaults, ...fields };
                node.attrs.push({ name, value: formatFields(name, merged) });
                changed = true;
            }
        }
        if (changed) onUpdate?.();
    },
};

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
        const reg = getComponent(name);
        if (!reg) return;
        const defaults = reg.traits?.defaults?.() ?? {};
        const parsed = value ? { ...defaults, ...parseFields(name, value) } : defaults;
        for (const [field, val] of Object.entries(parsed)) {
            if (typeof val === "number") setFieldValue(reg.component, field, eid, val);
            else if (typeof val === "string") setString(reg.component, field, eid, val);
        }
    }

    attachComponent(name: string, value: string, eid: number): void {
        const reg = getComponent(name);
        if (!reg) return;
        this.state.addComponent(eid, reg.component as never);
        const defaults = reg.traits?.defaults?.() ?? {};
        for (const [field, val] of Object.entries(defaults)) {
            setFieldValue(reg.component, field, eid, val as number);
        }
        if (value) this.syncAttr(name, value, eid);
    }

    loadNode(node: Node, parent: Node | null): void {
        const eid = this.state.addEntity();
        this.nodeMap.set(node, eid);
        if (parent) {
            const parentEid = this.nodeMap.get(parent);
            if (parentEid !== undefined) this.state.addRelation(eid, ChildOf, parentEid);
        }
        for (const attr of node.attrs) this.attachComponent(attr.name, attr.value, eid);
        for (const child of node.children) this.loadNode(child, node);
    }

    unloadNode(node: Node): void {
        for (const child of node.children) this.unloadNode(child);
        const eid = this.nodeMap.get(node);
        if (eid !== undefined) {
            this.state.removeEntity(eid);
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
                    const reg = getComponent(cmd.name);
                    if (reg) this.state.removeComponent(eid, reg.component as never);
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
                    const reg = getComponent(cmd.name);
                    if (reg) this.state.removeComponent(eid, reg.component as never);
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
                const eid = this.nodeMap.get(cmd.node);
                if (eid === undefined) return false;
                const parent = isUndo ? cmd.oldParent : cmd.newParent;
                if (parent) {
                    const parentEid = this.nodeMap.get(parent);
                    if (parentEid === undefined) return false;
                    this.state.addRelation(eid, ChildOf, parentEid);
                } else {
                    const oldParent = isUndo ? cmd.newParent : cmd.oldParent;
                    if (oldParent) {
                        const oldParentEid = this.nodeMap.get(oldParent);
                        if (oldParentEid !== undefined) {
                            this.state.removeRelation(eid, ChildOf, oldParentEid);
                        }
                    }
                }
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
