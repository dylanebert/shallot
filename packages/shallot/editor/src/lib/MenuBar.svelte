<script lang="ts">
    import type { EditorEntry } from "../plugins";
    import { dismissOnClickOutside } from "./dismiss";
    import { portal } from "./portal";
    import { fit, type Rect } from "./place";

    interface Props {
        entries: EditorEntry[];
        ontoggle: (name: string) => void;
        hasProject: boolean;
        scenes: string[];
        activeScene: string | null;
        onscene: (path: string) => void;
        onundo: () => void;
        onredo: () => void;
        onsave: () => void;
        ondocs: () => void;
        canUndo: boolean;
        canRedo: boolean;
        canSave: boolean;
        themes: { id: string; label: string }[];
        activeTheme: string;
        ontheme: (id: string) => void;
    }

    let { entries, ontoggle, hasProject, scenes, activeScene, onscene, onundo, onredo, onsave, ondocs, canUndo, canRedo, canSave, themes, activeTheme, ontheme }: Props = $props();

    const mod = navigator.platform.startsWith("Mac") ? "⌘" : "Ctrl+";

    function sceneLabel(path: string): string {
        const parts = path.split("/");
        return parts[parts.length - 1].replace(/\.scene$/, "");
    }

    type Menu = "file" | "edit" | "preferences" | "plugins";
    let open = $state(false);
    let cat = $state<Menu | null>(null);
    // second-level selection within a category (e.g. Preferences → Theme), so a category can hold
    // sub-menus that each fly out their own options
    let sub = $state<string | null>(null);
    let btnEl: HTMLElement;
    let anchor = $state.raw<Rect>({ left: 0, top: 0, right: 0, bottom: 0 });

    function openCat(c: Menu) {
        cat = c;
        sub = null;
    }

    const project = $derived(entries.filter((e) => e.source === "project"));
    const defaults = $derived(entries.filter((e) => e.source === "default"));
    const extras = $derived(entries.filter((e) => e.source === "extra"));
    const enabledCount = $derived(entries.filter((e) => e.enabled).length);
    const totalCount = $derived(entries.length);
    const hasScenes = $derived(scenes.length > 1);

    function toggle() {
        open = !open;
        cat = null;
        sub = null;
        // the dropdown is portaled to `.editor` (use:portal) to escape the sidebar's sticky-header stacking
        // context + overflow clip, then fitted against the button rect (use:fit); submenus fly out from
        // the rows relative to the dropdown
        if (open) {
            const r = btnEl.getBoundingClientRect();
            anchor = { left: r.left, top: r.top, right: r.right, bottom: r.bottom };
        }
    }

    function close() {
        open = false;
        cat = null;
        sub = null;
    }

    $effect(() => {
        if (!open) return;
        // the dropdown is portaled out of `.logo-menu`, so its own selector joins the outside-click guard
        return dismissOnClickOutside(close, ".logo-menu", ".menu-dropdown");
    });
</script>

<div class="logo-menu">
    <button class="logo-btn" class:active={open} onclick={toggle} bind:this={btnEl}>
        <img src="/icon.svg" alt="" class="brand-icon" />
        <span class="brand-name">shallot</span>
        <svg class="caret" viewBox="0 0 8 6" fill="currentColor">
            <path d="M1 1.5 L4 4.5 L7 1.5" />
        </svg>
    </button>
    {#if open}
        <div class="menu-dropdown" use:portal use:fit={{ anchor }}>
            <div class="cat">
                <button class="cat-row menu-file" class:active={cat === "file"} onpointerenter={() => openCat("file")} onclick={() => openCat("file")}>
                    <span class="cat-name">File</span>
                    <svg class="chevron" viewBox="0 0 6 8" fill="currentColor"><path d="M1 1 L5 4 L1 7 Z" /></svg>
                </button>
                {#if cat === "file"}
                    <div class="submenu">
                        <div class="menu-actions">
                            <button class="menu-action" disabled={!canSave} onclick={() => { onsave(); close(); }}>
                                <span class="action-name">Save</span>
                                <span class="shortcut">{mod}S</span>
                            </button>
                        </div>
                        {#if hasScenes}
                            <div class="menu-divider"></div>
                            <div class="menu-section-label">Scenes</div>
                            <div class="plugin-section">
                                {#each scenes as scene}
                                    {@const active = scene === activeScene}
                                    <button class="plugin-row" class:active onclick={() => { onscene(scene); close(); }}>
                                        <span class="dot"></span>
                                        <span class="plugin-name">{sceneLabel(scene)}</span>
                                    </button>
                                {/each}
                            </div>
                        {/if}
                    </div>
                {/if}
            </div>

            <div class="cat">
                <button class="cat-row menu-edit" class:active={cat === "edit"} onpointerenter={() => openCat("edit")} onclick={() => openCat("edit")}>
                    <span class="cat-name">Edit</span>
                    <svg class="chevron" viewBox="0 0 6 8" fill="currentColor"><path d="M1 1 L5 4 L1 7 Z" /></svg>
                </button>
                {#if cat === "edit"}
                    <div class="submenu">
                        <div class="menu-actions">
                            <button class="menu-action" disabled={!canUndo} onclick={() => { onundo(); close(); }}>
                                <span class="action-name">Undo</span>
                                <span class="shortcut">{mod}Z</span>
                            </button>
                            <button class="menu-action" disabled={!canRedo} onclick={() => { onredo(); close(); }}>
                                <span class="action-name">Redo</span>
                                <span class="shortcut">{mod}Y</span>
                            </button>
                        </div>
                    </div>
                {/if}
            </div>

            <div class="cat">
                <button class="cat-row menu-docs" onpointerenter={() => { cat = null; sub = null; }} onclick={() => { ondocs(); close(); }}>
                    <span class="cat-name">Docs</span>
                </button>
            </div>

            <div class="cat">
                <button class="cat-row menu-preferences" class:active={cat === "preferences"} onpointerenter={() => openCat("preferences")} onclick={() => openCat("preferences")}>
                    <span class="cat-name">Preferences</span>
                    <svg class="chevron" viewBox="0 0 6 8" fill="currentColor"><path d="M1 1 L5 4 L1 7 Z" /></svg>
                </button>
                {#if cat === "preferences"}
                    <div class="submenu">
                        <div class="cat">
                            <button class="cat-row" class:active={sub === "theme"} onpointerenter={() => (sub = "theme")} onclick={() => (sub = "theme")}>
                                <span class="cat-name">Theme</span>
                                <svg class="chevron" viewBox="0 0 6 8" fill="currentColor"><path d="M1 1 L5 4 L1 7 Z" /></svg>
                            </button>
                            {#if sub === "theme"}
                                <div class="submenu">
                                    <div class="plugin-section">
                                        {#each themes as t}
                                            {@const active = t.id === activeTheme}
                                            <button class="plugin-row theme-row" class:active onclick={() => { ontheme(t.id); close(); }}>
                                                <span class="dot"></span>
                                                <span class="plugin-name">{t.label}</span>
                                            </button>
                                        {/each}
                                    </div>
                                </div>
                            {/if}
                        </div>
                    </div>
                {/if}
            </div>

            {#if hasProject}
                <div class="cat">
                    <button class="cat-row menu-plugins" class:active={cat === "plugins"} onpointerenter={() => openCat("plugins")} onclick={() => openCat("plugins")}>
                        <span class="cat-name">Plugins</span>
                        <span class="plugin-count">{enabledCount}/{totalCount}</span>
                        <svg class="chevron" viewBox="0 0 6 8" fill="currentColor"><path d="M1 1 L5 4 L1 7 Z" /></svg>
                    </button>
                    {#if cat === "plugins"}
                        <div class="submenu">
                            <div class="plugin-section tall">
                                {#each project as entry}
                                    <button class="plugin-row" class:active={entry.enabled} onclick={() => ontoggle(entry.name)}>
                                        <span class="dot"></span>
                                        <span class="plugin-name">{entry.name}</span>
                                        <span class="tag project">project</span>
                                    </button>
                                {/each}
                                {#each defaults as entry}
                                    <button class="plugin-row" class:active={entry.enabled} onclick={() => ontoggle(entry.name)}>
                                        <span class="dot"></span>
                                        <span class="plugin-name">{entry.name}</span>
                                        <span class="tag standard">default</span>
                                    </button>
                                {/each}
                                {#each extras as entry}
                                    <button class="plugin-row" class:active={entry.enabled} onclick={() => ontoggle(entry.name)}>
                                        <span class="dot"></span>
                                        <span class="plugin-name">{entry.name}</span>
                                    </button>
                                {/each}
                            </div>
                        </div>
                    {/if}
                </div>
            {/if}
        </div>
    {/if}
</div>

<style>
    .logo-menu {
        position: relative;
    }

    /* a compact menu-bar lockup, not the docs hero: the icon anchors it (larger than the wordmark at this
       size), the icon↔wordmark pairing stays tight, and the caret sits apart (its own margin) so it reads
       as the menu affordance rather than part of the name */
    .logo-btn {
        display: flex;
        align-items: center;
        gap: 2px;
        height: 28px;
        padding: 0 8px 0 6px;
        border: none;
        border-radius: 4px;
        background: transparent;
        font-size: 13px;
        font-weight: 700;
        line-height: 1;
        color: var(--text);
        cursor: pointer;
        transition: background 150ms var(--ease-out);
    }

    .logo-btn:hover,
    .logo-btn.active {
        background: var(--surface-2);
    }

    .logo-btn:active {
        transform: scale(0.95);
    }

    .logo-btn:focus-visible {
        outline: 1px solid var(--accent);
        outline-offset: -1px;
    }

    .brand-icon {
        height: 22px;
        width: auto;
    }

    .brand-name {
        user-select: none;
    }

    .caret {
        width: 9px;
        height: 6px;
        margin-left: 4px;
        color: var(--text-muted);
    }

    .menu-dropdown,
    .submenu {
        position: fixed;
        min-width: 200px;
        padding: 4px 0;
        border: 1px solid var(--border);
        border-radius: 6px;
        z-index: 100;
        background: var(--surface-3-solid);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3), 0 1px 4px rgba(0, 0, 0, 0.2);
        transform-origin: top left;
        animation: menu-appear 150ms var(--ease-out);
    }

    .menu-dropdown {
        padding: 4px;
    }

    /* fly out to the right of the hovered category row, top-aligned with it (Figma main-menu shape) */
    .submenu {
        position: absolute;
        top: -5px;
        left: 100%;
        margin-left: 4px;
    }

    @keyframes menu-appear {
        from { opacity: 0; transform: scale(0.97); }
        to { opacity: 1; transform: scale(1); }
    }

    .cat {
        position: relative;
    }

    .cat-row {
        display: flex;
        align-items: center;
        gap: 8px;
        width: 100%;
        padding: 4px 8px;
        border: none;
        border-radius: 4px;
        background: transparent;
        color: var(--text-secondary);
        font-size: 12px;
        font-family: inherit;
        cursor: pointer;
        text-align: left;
        transition: background 120ms var(--ease-out), color 120ms var(--ease-out);
    }

    .cat-row:hover,
    .cat-row.active {
        background: var(--surface-2);
        color: var(--text);
    }

    .cat-name {
        flex: 1;
    }

    .chevron {
        width: 6px;
        height: 8px;
        opacity: 0.5;
        flex-shrink: 0;
    }

    .menu-divider {
        height: 1px;
        margin: 4px 8px;
        background: var(--border);
    }

    .menu-actions {
        padding: 0 4px;
    }

    .menu-action {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        width: 100%;
        padding: 3px 8px;
        border: none;
        border-radius: 4px;
        background: transparent;
        color: var(--text-secondary);
        font-size: 11px;
        font-family: inherit;
        cursor: pointer;
        text-align: left;
        transition: background 120ms var(--ease-out), color 120ms var(--ease-out);
    }

    .menu-action:hover:not(:disabled) {
        background: var(--surface-2);
        color: var(--text);
    }

    .menu-action:active:not(:disabled) {
        background: color-mix(in srgb, var(--accent) 8%, transparent);
        transform: scale(0.95);
    }

    .menu-action:disabled {
        opacity: 0.35;
        cursor: default;
    }

    .shortcut {
        color: var(--text-muted);
        font-size: 10px;
        font-variant-numeric: tabular-nums;
    }

    .menu-section-label {
        display: flex;
        align-items: center;
        padding: 4px 12px 2px;
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
    }

    .plugin-count {
        font-variant-numeric: tabular-nums;
        font-size: 10px;
        opacity: 0.6;
    }

    .plugin-section {
        padding: 0 4px 2px;
    }

    .plugin-section.tall {
        max-height: 240px;
        overflow-y: auto;
    }

    .plugin-section::-webkit-scrollbar {
        width: 4px;
    }

    .plugin-section::-webkit-scrollbar-track {
        background: transparent;
    }

    .plugin-section::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 2px;
    }

    .plugin-row {
        display: flex;
        align-items: center;
        gap: 6px;
        width: 100%;
        padding: 3px 8px;
        border: none;
        border-radius: 4px;
        background: transparent;
        color: var(--text-muted);
        font-size: 11px;
        font-family: inherit;
        cursor: pointer;
        text-align: left;
        white-space: nowrap;
        transition: background 120ms var(--ease-out), color 120ms var(--ease-out);
    }

    .plugin-row:hover {
        background: var(--surface-2);
    }

    .plugin-row.active {
        color: var(--text-secondary);
    }

    .plugin-row.active:hover {
        color: var(--text);
    }

    .plugin-row:active {
        background: color-mix(in srgb, var(--accent) 8%, transparent);
        transform: scale(0.95);
    }

    /* theme options are a short preference list, not the dense plugin/scene roster — give them a
       comfortable, readable size instead of inheriting the compact roster row */
    .theme-row {
        font-size: 12px;
        padding: 5px 8px;
        color: var(--text-secondary);
    }

    .theme-row.active {
        color: var(--text);
    }

    .dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        flex-shrink: 0;
        background: var(--text-muted);
        opacity: 0.3;
        transition: background 150ms var(--ease-out), opacity 150ms var(--ease-out), transform 150ms var(--ease-out);
    }

    .plugin-row.active .dot {
        background: var(--accent);
        opacity: 1;
    }

    .plugin-row:hover .dot {
        transform: scale(1.2);
    }

    .plugin-name {
        flex: 1;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .tag {
        font-size: 9px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 1px 4px;
        border-radius: 3px;
        flex-shrink: 0;
    }

    .tag.project {
        color: var(--cat-spatial);
        background: color-mix(in srgb, var(--cat-spatial) 12%, transparent);
    }

    .tag.standard {
        color: var(--cat-rendering);
        background: color-mix(in srgb, var(--cat-rendering) 12%, transparent);
    }
</style>
