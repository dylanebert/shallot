// Lift a floating node up to the `.editor` root so its own z-index is authoritative. A
// `position: fixed` dropdown nested inside a `position: sticky` / `overflow` ancestor inherits that
// ancestor's stacking level (sticky always forms a stacking context), so a sibling with a higher
// z-index paints over it — it looks see-through and clicks pass through. The target is `.editor`, not
// <body>, because the color tokens are CSS custom properties scoped to `.editor`: a node outside it
// resolves them to nothing and renders transparent. Pair the portaled node's selector into the menu's
// dismiss check, since it sits outside the trigger's DOM subtree.
export function portal(node: HTMLElement): { destroy(): void } {
    const target = node.closest(".editor") ?? document.body;
    target.appendChild(node);
    return {
        destroy() {
            node.remove();
        },
    };
}
