<script lang="ts">
    import { tick } from "svelte";
    import type { Document, Node } from "@dylanebert/shallot/editor";
    import type { Diagnostic } from "@dylanebert/shallot";
    import { findParent } from "@dylanebert/shallot/scene/core";
    import { Import } from "lucide-static";
    import { ChevronRight, Plus, TriangleAlert } from "./components";
    import { type Bundle, menuGroups } from "./bundles";
    import { nextSelection } from "./pick";
    import { rows, type RowView } from "./rows";
    import Icon from "./Icon.svelte";
    import SectionLabel from "./SectionLabel.svelte";
    import { dismissOnClickOutside } from "./dismiss";
    import { fit, type Rect } from "./place";
    import { portal } from "./portal";

    let { doc, version, diagnostics, renameSignal, onselect, oncreate, onimport, ondelete, onreorder, onrename }: {
        doc: Document;
        version: number;
        diagnostics: Diagnostic[];
        renameSignal: number;
        onselect: () => void;
        oncreate: (bundle: Bundle) => void;
        onimport: () => void;
        ondelete: (node: Node) => void;
        onreorder: () => void;
        onrename: () => void;
    } = $props();

    let contextMenu: { x: number; y: number; node: Node } | null = $state.raw(null);
    let collapsed: WeakSet<Node> = new WeakSet();
    let collapseVersion = $state(0);
    let editingNode: Node | null = $state.raw(null);
    let editValue = $state("");
    let lastClickTime = 0;
    let lastClickNode: Node | null = null;

    let rowViews = $derived.by((): RowView[] => {
        void version;
        void collapseVersion;
        return rows(doc.nodes, doc.selection, collapsed, diagnostics);
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
        // shift / ctrl / cmd toggles the row in or out of the selection; a plain click selects only
        // it — the same toggle-everywhere rule the viewport pick uses (nextSelection, lib/pick).
        const additive = e.shiftKey || e.ctrlKey || e.metaKey;
        const next = nextSelection([...doc.selection], node, additive);
        doc.clearSelection();
        doc.select(...next);
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

    // the Add menu: the outliner `+` summons a grouped bundle picker (Empty first), keyboard-navigable,
    // fitted above the button. Picking a bundle hands it to `oncreate`, which instantiates + selects it.
    let addOpen = $state(false);
    let addAnchor = $state.raw<Rect>({ left: 0, top: 0, right: 0, bottom: 0 });
    let addFocus = $state(-1);
    let addBtnEl: HTMLElement;

    let addGroups = $derived.by(() => {
        void version; // re-derive on rebuild / plugin toggle — availability keys on the live registry
        return menuGroups();
    });
    let addFlat = $derived(addGroups.flatMap((g) => g.items));

    function openAdd(e: MouseEvent) {
        e.stopPropagation();
        const r = addBtnEl.getBoundingClientRect();
        addAnchor = { left: r.left, top: r.top, right: r.right, bottom: r.bottom };
        addFocus = -1;
        addOpen = true;
    }

    function pickBundle(bundle: Bundle) {
        addOpen = false;
        oncreate(bundle);
    }

    function pickImport() {
        addOpen = false;
        onimport();
    }

    function handleAddKey(e: KeyboardEvent) {
        // the summoned menu owns the keyboard — nothing leaks to the window keymap (a stray Delete
        // would otherwise delete the selection underneath; a digit would switch tools)
        e.stopPropagation();
        // the Model… import row sits after the bundles, at index addFlat.length
        const total = addFlat.length + 1;
        if (e.key === "Escape") {
            addOpen = false;
        } else if (e.key === "ArrowDown" && total) {
            e.preventDefault();
            addFocus = addFocus < total - 1 ? addFocus + 1 : 0;
        } else if (e.key === "ArrowUp" && total) {
            e.preventDefault();
            addFocus = addFocus > 0 ? addFocus - 1 : total - 1;
        } else if (e.key === "Enter" && addFocus >= 0 && addFocus < total) {
            e.preventDefault();
            if (addFocus === addFlat.length) pickImport();
            else pickBundle(addFlat[addFocus]);
        }
    }

    function focusMenu(node: HTMLElement) {
        node.focus();
    }

    $effect(() => {
        if (!addOpen) return;
        return dismissOnClickOutside(() => { addOpen = false; }, ".add-entity-menu", ".add-entity-btn");
    });

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
        if (rowEls.length !== rowViews.length) return null;

        for (let i = 0; i < rowEls.length; i++) {
            const rect = rowEls[i].getBoundingClientRect();
            if (clientY < rect.top || clientY > rect.bottom) continue;
            const entry = rowViews[i];
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
            const fi = rowViews.findIndex((r) => r.node === node);
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
            const fi = rowViews.findIndex((r) => r.node === target.parent);
            if (fi >= 0 && rowEls[fi]) {
                return rowEls[fi].getBoundingClientRect().bottom;
            }
        }
        return null;
    }

    function findLastRowIndex(node: Node): number {
        let last = rowViews.findIndex((r) => r.node === node);
        if (last < 0) return -1;
        for (let i = last + 1; i < rowViews.length; i++) {
            let ancestor = rowViews[i].parent;
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
    <SectionLabel label="Entities" />
    {#if rowViews.length > 0}
        {#each rowViews as row}
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
        <button class="add-entity-btn" class:active={addOpen} onclick={openAdd} bind:this={addBtnEl}>
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
    <div class="context-menu" use:fit={{ anchor: { left: contextMenu.x, top: contextMenu.y, right: contextMenu.x, bottom: contextMenu.y } }}>
        <button class="context-item" onmousedown={handleContextRename}>
            Rename
        </button>
        <button class="context-item delete" onmousedown={handleContextDelete}>
            Delete
        </button>
    </div>
{/if}

{#if addOpen}
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div
        class="add-entity-menu"
        tabindex="-1"
        use:portal
        use:focusMenu
        use:fit={{ anchor: addAnchor, side: "above", align: "start" }}
        onkeydown={handleAddKey}
    >
        {#each addGroups as group}
            {#if group.category}
                <div class="add-entity-group-header">
                    <span class="add-entity-dot" style="background: {group.category.color}"></span>
                    <span class="add-entity-group-label">{group.category.label}</span>
                </div>
            {/if}
            {#each group.items as bundle}
                {@const idx = addFlat.indexOf(bundle)}
                <button
                    class="add-entity-item"
                    class:focused={addFocus === idx}
                    onmousedown={() => pickBundle(bundle)}
                    onmouseenter={() => { addFocus = idx; }}
                >
                    <Icon icon={bundle.icon} size={13} strokeWidth={1.5} class="add-entity-icon" style="color: {bundle.color}" />
                    <span>{bundle.label}</span>
                </button>
            {/each}
        {/each}
        <div class="add-entity-group-header">
            <span class="add-entity-dot" style="background: var(--cat-rendering)"></span>
            <span class="add-entity-group-label">Import</span>
        </div>
        <button
            class="add-entity-item"
            class:focused={addFocus === addFlat.length}
            onmousedown={pickImport}
            onmouseenter={() => { addFocus = addFlat.length; }}
        >
            <Icon icon={Import} size={13} strokeWidth={1.5} class="add-entity-icon" style="color: var(--cat-rendering)" />
            <span>Model…</span>
        </button>
    </div>
{/if}

<style>
    .outliner {
        position: relative;
        flex: 1;
        /* fill the panel below the menu header so empty space clears selection, without overflowing the
           shared scroll container (the header is a sibling in the same .sidebar) */
        min-height: calc(100% - var(--header-h));
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
        background: var(--surface-1);
        color: var(--text);
    }

    /* selection is the accent-tinted background; the text stays neutral (readable) rather than accent —
       gold-on-gold has too little chromatic contrast (VS Code likewise: tinted row, neutral foreground) */
    .row.selected {
        background: color-mix(in srgb, var(--accent) 16%, transparent);
        color: var(--text);
    }

    .row.selected:hover {
        background: color-mix(in srgb, var(--accent) 22%, transparent);
        color: var(--text);
    }

    .row:active {
        background: color-mix(in srgb, var(--accent) 8%, transparent);
    }

    .row.selected:active {
        background: color-mix(in srgb, var(--accent) 26%, transparent);
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
        box-shadow: 0 0 0 1px color-mix(in srgb, var(--accent) 30%, transparent), 0 0 0 3px color-mix(in srgb, var(--accent) 15%, transparent);
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
        background: color-mix(in srgb, var(--accent) 4%, transparent);
    }

    .add-entity-btn:active {
        transform: scale(0.95);
    }

    .add-entity-btn.active {
        border-color: var(--accent);
        color: var(--accent);
        border-style: solid;
    }

    .add-entity-btn :global(svg) {
        flex-shrink: 0;
    }

    .add-entity-menu {
        display: flex;
        flex-direction: column;
        min-width: 176px;
        padding: 4px;
        background: var(--surface-3-solid);
        border: 1px solid var(--border);
        border-radius: 6px;
        z-index: 100;
        max-height: 60vh;
        overflow-y: auto;
        outline: none;
        animation: add-entity-appear 150ms var(--ease-out);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3), 0 1px 4px rgba(0, 0, 0, 0.2);
    }

    @keyframes add-entity-appear {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .add-entity-group-header {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 6px 8px 2px;
    }

    .add-entity-group-header:first-child {
        padding-top: 2px;
    }

    .add-entity-dot {
        width: 5px;
        height: 5px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    .add-entity-group-label {
        font-size: 9px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
    }

    .add-entity-item {
        display: flex;
        align-items: center;
        gap: 8px;
        width: 100%;
        padding: 4px 8px;
        border: none;
        border-radius: 4px;
        background: transparent;
        color: var(--text-secondary);
        font-size: 11px;
        font-family: inherit;
        text-align: left;
        cursor: pointer;
        transition: background 120ms var(--ease-out), color 120ms var(--ease-out);
    }

    .add-entity-item:hover,
    .add-entity-item.focused {
        background: var(--surface-2);
        color: var(--text);
    }

    .add-entity-item :global(.add-entity-icon) {
        flex-shrink: 0;
    }

    .context-menu {
        position: fixed;
        min-width: 120px;
        padding: 4px;
        background: var(--surface-3-solid);
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
        background: color-mix(in srgb, var(--error) 12%, transparent);
        color: var(--error);
    }

    .context-item:active {
        background: color-mix(in srgb, var(--accent) 8%, transparent);
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
        background: color-mix(in srgb, var(--accent) 12%, transparent);
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
