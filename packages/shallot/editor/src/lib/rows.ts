import type { Diagnostic } from "@dylanebert/shallot";
import type { Node } from "@dylanebert/shallot/editor";
import { type ComponentMeta, heroMeta, nodeLabel } from "./components";

export type RowView = {
    node: Node;
    parent: Node | null;
    depth: number;
    label: string;
    meta: ComponentMeta;
    selected: boolean;
    hasChildren: boolean;
    expanded: boolean;
    warning: boolean;
};

/**
 * the outliner's tree→rows derivation: the document tree flattened to a depth-tagged row list,
 * skipping the children of collapsed nodes. Pure over (tree, selection, collapsed, diagnostics) —
 * `bun test`-covered, so a mistitled node or a wrong depth is a unit failure, not a visual one.
 */
export function rows(
    nodes: Node[],
    selection: ReadonlySet<Node>,
    collapsed: { has(node: Node): boolean },
    diagnostics: readonly Diagnostic[],
): RowView[] {
    const warned = new Set<Node>();
    for (const d of diagnostics) warned.add(d.node);

    const result: RowView[] = [];
    function walk(list: Node[], parent: Node | null, depth: number) {
        for (const node of list) {
            const hasChildren = node.children.length > 0;
            const expanded = hasChildren && !collapsed.has(node);
            result.push({
                node,
                parent,
                depth,
                label: nodeLabel(node),
                meta: heroMeta(node.attrs),
                selected: selection.has(node),
                hasChildren,
                expanded,
                warning: warned.has(node),
            });
            if (expanded) walk(node.children, node, depth + 1);
        }
    }
    walk(nodes, null, 0);
    return result;
}
