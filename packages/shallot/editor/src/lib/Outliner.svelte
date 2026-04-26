<script lang="ts">
    import type { Document, Node } from "@dylanebert/shallot/editor";
    import { type Diagnostic, findParent } from "@dylanebert/shallot";
    import { heroMeta, ChevronRight, Plus, TriangleAlert, type ComponentMeta } from "./components";
    import Icon from "./Icon.svelte";
    import { dismissOnClickOutside } from "./dismiss";

    let { doc, version, diagnostics, renameSignal, onselect, oncreate, ondelete, onreorder, onrename }: {
        doc: Document;
        version: number;
        diagnostics: Diagnostic[];
        renameSignal: number;
        onselect: () => void;
        oncreate: () => Node;
        ondelete: (node: Node) => void;
        onreorder: () => void;
        onrename: () => void;
    } = $props();

    import { tick } from "svelte";

    let contextMenu: { x: number; y: number; node: Node } | null = $state.raw(null);
    let collapsed: WeakSet<Node> = new WeakSet();
    let collapseVersion = $state(0);
    let editingNode: Node | null = $state.raw(null);
    let editValue = $state("");
    let lastClickTime = 0;
    let lastClickNode: Node | null = null;

    type RowView = {
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

    let rows = $derived.by((): RowView[] => {
        void version;
        void collapseVersion;
        const result: RowView[] = [];
        function walk(nodes: Node[], parent: Node | null, depth: number) {
            for (const node of nodes) {
                const hasChildren = node.children.length > 0;
                const expanded = hasChildren && !collapsed.has(node);
                result.push({
                    node,
                    parent,
                    depth,
                    label: node.id || "entity",
                    meta: heroMeta(node.attrs),
                    selected: doc.selection.has(node),
                    hasChildren,
                    expanded,
                    warning: diagnostics.some((d) => d.node === node),
                });
                if (expanded) walk(node.children, node, depth + 1);
            }
        }
        walk(doc.nodes, null, 0);
        return result;
    });

    function revealSelection() {
        if (doc.selection.size === 0) return;
        let revealed = false;
        function walkAncestors(nodes: Node[], parent: Node | null): boolean {
            for (const node of nodes) {
                if (doc.selection.has(node)) return true;
                if (node.children.length > 0 && walkAncestors(node.children, node)) {
                    if (collapsed.has(node)) {
                        collapsed.delete(node);
                        revealed = true;
                    }
                    return true;
                }
            }
            return false;
        }
        walkAncestors(doc.nodes, null);
        if (revealed) collapseVersion++;
    }

    $effect(() => {
        void version;
        revealSelection();
        tick().then(() => {
            if (!outlinerEl || doc.selection.size === 0) return;
            const rowEls = outlinerEl.querySelectorAll(".row.selected");
            if (rowEls.length > 0) {
                rowEls[0].scrollIntoView({ block: "nearest" });
            }
        });
    });

    function toggleCollapse(e: MouseEvent, node: Node) {
        e.stopPropagation();
        e.preventDefault();
        if (collapsed.has(node)) collapsed.delete(node);
        else collapsed.add(node);
        collapseVersion++;
    }

    function selectNode(e: MouseEvent, node: Node) {
        if (e.shiftKey && doc.selection.has(node)) {
            doc.deselect(node);
        } else {
            doc.clearSelection();
            doc.select(node);
        }
        onselect();
    }

    function clearSelection() {
        if (justSelected) { justSelected = false; return; }
        if (doc.selection.size === 0) return;
        doc.clearSelection();
        onselect();
    }

    function startRename(node: Node) {
        editingNode = node;
        editValue = node.id || "entity";
    }

    function focusSelect(el: HTMLInputElement) {
        el.focus();
        el.select();
    }

    function commitRename() {
        if (!editingNode) return;
        const node = editingNode;
        const trimmed = editValue.trim();
        const newId = trimmed || undefined;
        if (newId !== node.id) {
            doc.setId(node, newId);
            onrename();
        }
        editingNode = null;
    }

    function cancelRename() {
        editingNode = null;
    }

    function handleRenameKeydown(e: KeyboardEvent) {
        if (e.key === "Enter") {
            e.preventDefault();
            commitRename();
        } else if (e.key === "Escape") {
            e.preventDefault();
            cancelRename();
        }
    }

    function handleCreate(e: MouseEvent) {
        e.stopPropagation();
        oncreate();
    }

    function showContextMenu(e: MouseEvent, node: Node) {
        e.preventDefault();
        if (!doc.selection.has(node)) {
            doc.clearSelection();
            doc.select(node);
            onselect();
        }
        contextMenu = { x: e.clientX, y: e.clientY, node };
    }

    function handleContextRename() {
        if (!contextMenu) return;
        const { node } = contextMenu;
        contextMenu = null;
        requestAnimationFrame(() => startRename(node));
    }

    function handleContextDelete() {
        if (!contextMenu) return;
        const { node } = contextMenu;
        contextMenu = null;
        ondelete(node);
    }

    $effect(() => {
        if (!contextMenu) return;
        return dismissOnClickOutside(() => { contextMenu = null; }, ".context-menu");
    });

    $effect(() => {
        if (renameSignal > 0 && doc.selection.size === 1) {
            const node = doc.selection.values().next().value!;
            requestAnimationFrame(() => startRename(node));
        }
    });

    type DropTarget = { parent: Node | null; index: number; reparent: boolean };
    let dragNode: Node | null = $state.raw(null);
    let dropTarget: DropTarget | null = $state.raw(null);
    let dragStartY = 0;
    let dragStartX = 0;
    let dragging = $state(false);
    let justSelected = false;
    let outlinerEl: HTMLElement = $state.raw(undefined!);

    function isDescendant(ancestor: Node, node: Node): boolean {
        for (const child of ancestor.children) {
            if (child === node || isDescendant(child, node)) return true;
        }
        return false;
    }

    function getRowElements(): HTMLElement[] {
        return Array.from(outlinerEl.querySelectorAll(".row"));
    }

    function computeDropTarget(clientY: number): DropTarget | null {
        if (!dragNode) return null;
        const rowEls = getRowElements();
        if (rowEls.length !== rows.length) return null;

        for (let i = 0; i < rowEls.length; i++) {
            const rect = rowEls[i].getBoundingClientRect();
            if (clientY < rect.top || clientY > rect.bottom) continue;
            const entry = rows[i];
            const relY = clientY - rect.top;
            const zone = rect.height;

            if (relY < zone * 0.25) {
                const parent = entry.parent;
                const siblings = parent ? parent.children : doc.nodes;
                const idx = siblings.indexOf(entry.node);
                if (entry.node === dragNode) return null;
                const dragParent = findParent(dragNode, doc.nodes, null);
                if (dragParent === parent) {
                    const fromIdx = siblings.indexOf(dragNode);
                    if (fromIdx >= 0 && (fromIdx === idx || fromIdx === idx - 1)) return null;
                }
                return { parent, index: idx, reparent: false };
            } else if (relY > zone * 0.75) {
                if (entry.expanded) {
                    if (entry.node === dragNode || isDescendant(dragNode, entry.node)) return null;
                    return { parent: entry.node, index: 0, reparent: false };
                }
                const parent = entry.parent;
                const siblings = parent ? parent.children : doc.nodes;
                const idx = siblings.indexOf(entry.node) + 1;
                if (entry.node === dragNode) return null;
                const dragParent = findParent(dragNode, doc.nodes, null);
                if (dragParent === parent) {
                    const fromIdx = siblings.indexOf(dragNode);
                    if (fromIdx >= 0 && (fromIdx === idx || fromIdx === idx - 1)) return null;
                }
                return { parent, index: idx, reparent: false };
            } else {
                if (entry.node === dragNode || isDescendant(dragNode, entry.node)) return null;
                const children = entry.node.children;
                return { parent: entry.node, index: children.length, reparent: true };
            }
        }
        return null;
    }

    function getDropIndicatorY(target: DropTarget): number | null {
        if (target.reparent) return null;
        const rowEls = getRowElements();
        const siblings = target.parent ? target.parent.children : doc.nodes;
        if (target.index < siblings.length) {
            const node = siblings[target.index];
            const fi = rows.findIndex((r) => r.node === node);
            if (fi >= 0 && rowEls[fi]) {
                return rowEls[fi].getBoundingClientRect().top;
            }
        } else if (siblings.length > 0) {
            const lastNode = siblings[siblings.length - 1];
            const lastFlat = findLastRowIndex(lastNode);
            if (lastFlat >= 0 && rowEls[lastFlat]) {
                return rowEls[lastFlat].getBoundingClientRect().bottom;
            }
        } else if (target.parent) {
            const fi = rows.findIndex((r) => r.node === target.parent);
            if (fi >= 0 && rowEls[fi]) {
                return rowEls[fi].getBoundingClientRect().bottom;
            }
        }
        return null;
    }

    function findLastRowIndex(node: Node): number {
        let last = rows.findIndex((r) => r.node === node);
        if (last < 0) return -1;
        for (let i = last + 1; i < rows.length; i++) {
            let ancestor = rows[i].parent;
            let isDesc = false;
            while (ancestor) {
                if (ancestor === node) { isDesc = true; break; }
                const p = findParent(ancestor, doc.nodes, null);
                ancestor = p !== undefined ? p : null;
            }
            if (isDesc) last = i;
            else break;
        }
        return last;
    }

    function handleDragStart(e: PointerEvent, node: Node) {
        if (e.button !== 0) return;
        dragNode = node;
        dragStartY = e.clientY;
        dragStartX = e.clientX;
        dragging = false;
        outlinerEl.setPointerCapture(e.pointerId);
    }

    function handleDragMove(e: PointerEvent) {
        if (!dragNode) return;
        if (!dragging) {
            const dx = e.clientX - dragStartX;
            const dy = e.clientY - dragStartY;
            if (Math.abs(dx) < 4 && Math.abs(dy) < 4) return;
            dragging = true;
        }
        dropTarget = computeDropTarget(e.clientY);
    }

    function handleDragEnd(e: PointerEvent) {
        if (!dragNode) return;
        if (!dragging) {
            const node = dragNode;
            dragNode = null;
            dropTarget = null;
            const now = Date.now();
            if (lastClickNode === node && now - lastClickTime < 400) {
                lastClickNode = null;
                lastClickTime = 0;
                startRename(node);
                return;
            }
            lastClickNode = node;
            lastClickTime = now;
            justSelected = true;
            selectNode(e, node);
            return;
        }
        if (dropTarget && dragNode) {
            const dragParent = findParent(dragNode, doc.nodes, null);
            if (dragParent === undefined) { dragNode = null; dropTarget = null; dragging = false; return; }
            if (dropTarget.reparent || dropTarget.parent !== dragParent) {
                doc.reparent(dragNode, dragParent, dropTarget.parent, dropTarget.index);
            } else {
                const siblings = dragParent ? dragParent.children : doc.nodes;
                const fromIdx = siblings.indexOf(dragNode);
                let toIdx = dropTarget.index;
                if (fromIdx < toIdx) toIdx--;
                if (fromIdx !== toIdx) doc.reorder(dragParent, dragNode, toIdx);
            }
            onreorder();
        }
        dragNode = null;
        dropTarget = null;
        dragging = false;
    }
</script>

<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
<div
    class="outliner"
    class:dragging
    bind:this={outlinerEl}
    onclick={clearSelection}
    onpointermove={handleDragMove}
    onpointerup={handleDragEnd}
>
    {#if rows.length > 0}
        {#each rows as row}
            {@const isReparentTarget = dragging && dropTarget?.reparent && dropTarget.parent === row.node}
            <div
                class="row"
                class:selected={row.selected}
                class:reparent-target={isReparentTarget}
                class:drag-source={dragging && dragNode === row.node}
                style="padding-left: {2 + row.depth * 16}px"
                onclick={(e: MouseEvent) => e.stopPropagation()}
                onpointerdown={(e: PointerEvent) => { e.stopPropagation(); handleDragStart(e, row.node); }}
                oncontextmenu={(e: MouseEvent) => { e.stopPropagation(); showContextMenu(e, row.node); }}
            >
                <button
                    class="disclosure"
                    class:visible={row.hasChildren}
                    class:expanded={row.expanded}
                    onclick={(e: MouseEvent) => toggleCollapse(e, row.node)}
                    onpointerdown={(e: PointerEvent) => e.stopPropagation()}
                    tabindex={-1}
                >
                    <Icon icon={ChevronRight} size={12} strokeWidth={2} class="disclosure-icon" />
                </button>
                <Icon icon={row.meta.icon} size={12} strokeWidth={1.5} class="icon-hero" style="color: {row.meta.color}" />
                {#if editingNode === row.node}
                    <input
                        class="rename-input"
                        type="text"
                        bind:value={editValue}
                        use:focusSelect
                        onkeydown={handleRenameKeydown}
                        onblur={commitRename}
                        onpointerdown={(e: PointerEvent) => e.stopPropagation()}
                        onclick={(e: MouseEvent) => e.stopPropagation()}
                        placeholder="entity"
                        spellcheck={false}
                    />
                {:else}
                    <span class="label">{row.label}</span>
                {/if}
                {#if row.warning}
                    <Icon icon={TriangleAlert} size={11} strokeWidth={1.5} class="row-warning" />
                {/if}
            </div>
        {/each}
    {:else}
        <div class="empty">No entities</div>
    {/if}
    <div class="add-entity">
        <button class="add-entity-btn" onclick={handleCreate}>
            <Icon icon={Plus} size={14} strokeWidth={1.5} />
            <span>Add Entity</span>
        </button>
    </div>
    {#if dragging && dropTarget && !dropTarget.reparent}
        {@const y = getDropIndicatorY(dropTarget)}
        {#if y !== null}
            {@const containerRect = outlinerEl.getBoundingClientRect()}
            <div class="drop-indicator" style="top: {y - containerRect.top + outlinerEl.scrollTop}px"></div>
        {/if}
    {/if}
</div>

{#if contextMenu}
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div
        class="context-menu"
        style="left: {contextMenu.x}px; top: {contextMenu.y}px"
    >
        <button class="context-item" onmousedown={handleContextRename}>
            Rename
        </button>
        <button class="context-item delete" onmousedown={handleContextDelete}>
            Delete
        </button>
    </div>
{/if}

<style>
    .outliner {
        position: relative;
        flex: 1;
        min-height: 100%;
    }

    .row {
        display: flex;
        align-items: center;
        width: 100%;
        height: 26px;
        padding-right: 12px;
        border: none;
        border-radius: 0;
        background: transparent;
        color: var(--text-secondary);
        font-size: 12px;
        text-align: left;
        cursor: pointer;
        transition: background 120ms var(--ease-out), color 120ms var(--ease-out);
    }

    .row:hover {
        background: rgba(255, 255, 255, 0.04);
        color: var(--text);
    }

    .row.selected {
        background: rgba(212, 149, 96, 0.1);
        color: var(--accent);
    }

    .row.selected:hover {
        background: rgba(212, 149, 96, 0.15);
        color: var(--accent-hover);
    }

    .row:active {
        background: rgba(212, 149, 96, 0.08);
    }

    .row.selected:active {
        background: rgba(212, 149, 96, 0.18);
    }

    .disclosure {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 26px;
        flex-shrink: 0;
        border: none;
        background: transparent;
        padding: 0;
        cursor: pointer;
        color: inherit;
        opacity: 0;
        pointer-events: none;
        transition: opacity 120ms var(--ease-out);
    }

    .disclosure.visible {
        opacity: 0.25;
        pointer-events: auto;
    }

    .row:hover .disclosure.visible {
        opacity: 0.65;
    }

    .disclosure.visible:hover {
        opacity: 0.9;
    }

    .disclosure :global(.disclosure-icon) {
        transition: transform 150ms var(--ease-out);
    }

    .disclosure.expanded :global(.disclosure-icon) {
        transform: rotate(90deg);
    }

    .row :global(.icon-hero) {
        flex-shrink: 0;
        opacity: 0.75;
        margin-right: 6px;
    }

    .row.selected :global(.icon-hero) {
        opacity: 1;
    }

    .label {
        flex: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .row :global(.row-warning) {
        flex-shrink: 0;
        color: var(--text-muted);
        opacity: 0.6;
        margin-left: 4px;
    }

    .rename-input {
        flex: 1;
        min-width: 0;
        height: 20px;
        padding: 0 4px;
        margin: 0;
        border: 1px solid var(--accent);
        border-radius: 3px;
        background: var(--bg, #0e0d0c);
        color: var(--text);
        font-size: 12px;
        font-family: inherit;
        outline: none;
        box-shadow: 0 0 0 1px rgba(212, 149, 96, 0.3), 0 0 0 3px rgba(212, 149, 96, 0.15);
    }

    .rename-input::placeholder {
        color: var(--text-muted);
        opacity: 0.5;
    }

    .empty {
        padding: 16px 12px;
        font-size: 12px;
        color: var(--text-muted);
    }

    .add-entity {
        padding: 10px 12px 8px;
    }

    .add-entity-btn {
        display: flex;
        align-items: center;
        gap: 6px;
        width: 100%;
        height: 28px;
        padding: 0 8px;
        border: 1px dashed var(--border);
        border-radius: 4px;
        background: transparent;
        color: var(--text-muted);
        font-size: 11px;
        font-family: inherit;
        cursor: pointer;
        transition: all 150ms var(--ease-out);
    }

    .add-entity-btn:hover {
        border-color: var(--accent);
        color: var(--accent);
        background: rgba(212, 149, 96, 0.04);
    }

    .add-entity-btn:active {
        transform: scale(0.95);
    }

    .add-entity-btn :global(svg) {
        flex-shrink: 0;
    }

    .context-menu {
        position: fixed;
        min-width: 120px;
        padding: 4px;
        background: rgba(38, 37, 36, 0.85);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border);
        border-radius: 6px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3), 0 1px 4px rgba(0, 0, 0, 0.2);
        z-index: 100;
    }

    .context-item {
        display: flex;
        align-items: center;
        width: 100%;
        padding: 4px 8px;
        border: none;
        border-radius: 4px;
        background: transparent;
        color: var(--text-secondary);
        font-size: 12px;
        font-family: inherit;
        text-align: left;
        cursor: pointer;
        white-space: nowrap;
        transition: background 150ms var(--ease-out), color 150ms var(--ease-out);
    }

    .context-item:hover {
        background: var(--surface-2);
        color: var(--text);
    }

    .context-item.delete:hover {
        background: rgba(220, 80, 60, 0.12);
        color: #e05545;
    }

    .context-item:active {
        background: rgba(212, 149, 96, 0.08);
        transform: scale(0.95);
    }

    .context-item:focus-visible {
        outline: 1px solid var(--accent);
        outline-offset: -1px;
    }

    .outliner.dragging {
        cursor: grabbing;
    }

    .row.drag-source {
        opacity: 0.4;
    }

    .row.reparent-target {
        background: rgba(212, 149, 96, 0.12);
        outline: 1px solid var(--accent);
        outline-offset: -1px;
    }

    .drop-indicator {
        position: absolute;
        left: 8px;
        right: 8px;
        height: 2px;
        background: var(--accent);
        border-radius: 1px;
        pointer-events: none;
        z-index: 10;
    }
</style>
