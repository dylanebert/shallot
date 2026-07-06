<script lang="ts">
    import { Code as CodeIcon, MousePointerClick, Search as SearchIcon, Settings, X } from "lucide-static";
    import { search, type DocIndex, type DocPage } from "./docs";
    import Icon from "./Icon.svelte";

    // The in-editor docs reader: a pure view over the same docs/dist index the site reads (docs.ts), so
    // the block model + tab markers render identically here and on the web (distribution boundary: bundled
    // offline, never the site renderer). Summoned over the inspector, dismissed back to it (editor-ui.md).
    let { index, target, onclose }: {
        index: DocIndex;
        /** a page (+ optional ref anchor) to open, set by a context request from the inspector */
        target: { slug: string; anchor?: string } | null;
        onclose: () => void;
    } = $props();

    let query = $state("");
    let slug: string | null = $state(null);
    let activeTab = $state("");
    let pendingAnchor: string | null = $state.raw(null);
    let bodyEl: HTMLElement | undefined = $state();

    const tabIcon: Record<string, string> = {
        Editor: MousePointerClick,
        Code: CodeIcon,
        Internals: Settings,
    };

    let results = $derived(query.trim() ? search(index, query) : []);
    let page: DocPage | null = $derived(slug ? (index.pages.find((p) => p.slug === slug) ?? null) : null);

    // pages grouped by their top-level group, the browse list shown when nothing is searched
    let groups = $derived.by(() => {
        const by = new Map<string, DocPage[]>();
        for (const p of index.pages) {
            const g = p.group || "general";
            let arr = by.get(g);
            if (!arr) by.set(g, (arr = []));
            arr.push(p);
        }
        return [...by.entries()].map(([name, pages]) => ({ name, pages }));
    });

    // a context request from the inspector (a new {slug, anchor} each open): navigate + scroll to the ref
    $effect(() => {
        const t = target;
        if (t) open(t.slug, t.anchor);
    });

    // keep the audience tab valid as the page changes; default to the first
    $effect(() => {
        const p = page;
        if (p && !p.tabNames.includes(activeTab)) activeTab = p.tabNames[0] ?? "";
    });

    // once the page (and the right tab) render, scroll a pending ref anchor into view
    $effect(() => {
        void page;
        void activeTab;
        const a = pendingAnchor;
        if (!a || !bodyEl) return;
        const el = bodyEl;
        requestAnimationFrame(() => el.querySelector(`#${CSS.escape(a)}`)?.scrollIntoView({ block: "start" }));
        pendingAnchor = null;
    });

    function open(s: string, anchor?: string): void {
        slug = s;
        query = "";
        if (anchor) {
            const tab = tabForAnchor(s, anchor);
            if (tab) activeTab = tab;
            pendingAnchor = anchor;
        }
    }

    // which audience tab holds a ref anchor, so navigating to a symbol lands on the tab that documents it
    function tabForAnchor(s: string, anchor: string): string | null {
        const p = index.pages.find((pp) => pp.slug === s);
        if (!p) return null;
        for (const block of p.blocks) {
            if (block.type !== "tabs") continue;
            for (const t of block.tabs)
                for (const b of t.blocks)
                    if (b.type === "content" && b.html.includes(`id="${anchor}"`)) return t.name;
        }
        return null;
    }

    function home(): void {
        slug = null;
        query = "";
    }

    // links inside the rendered {@html} stay in the panel: a `doc:slug#anchor` link opens another page
    // (the cross-page convention, docs.md), a ref-type `#anchor` link jumps to a symbol on this page
    // (switching tab if it lives on another). An action, not an element handler, so the content div needs
    // no interactive role.
    function refNav(node: HTMLElement) {
        function handler(e: MouseEvent): void {
            const a = (e.target as HTMLElement).closest("a");
            const href = a?.getAttribute("href");
            if (!href) return;
            if (href.startsWith("doc:")) {
                e.preventDefault();
                const [s, anchor] = href.slice(4).split("#");
                open(s, anchor);
            } else if (a?.classList.contains("ref-type") && href.startsWith("#") && slug) {
                e.preventDefault();
                open(slug, href.slice(1));
            }
        }
        node.addEventListener("click", handler);
        return { destroy: () => node.removeEventListener("click", handler) };
    }

    // inline os/pick groups carry their own selection, defaulting to the running OS (mirrors the site)
    let inlineSel: Record<string, string> = $state({});
    function inlineKey(tabs: { name: string }[]): string {
        return tabs.map((t) => t.name).join("|");
    }
    function inlineTab(tabs: { name: string }[]): string {
        const key = inlineKey(tabs);
        if (inlineSel[key]) return inlineSel[key];
        const os = /(Mac|iPhone|iPad|Linux|Android)/.test(navigator.userAgent) ? "Mac/Linux" : "Windows";
        return tabs.some((t) => t.name === os) ? os : (tabs[0]?.name ?? "");
    }

    function focusOnMount(node: HTMLInputElement): void {
        // search is the first action on the index only; opening to a page (inspector "?") shouldn't grab it.
        // preventScroll: focusing must not scroll the page to bring the right-edge overlay into view
        if (!target) node.focus({ preventScroll: true });
    }
</script>

<div class="docs-reader">
    <div class="docs-head">
        <span class="docs-search-icon"><Icon icon={SearchIcon} size={14} strokeWidth={1.6} /></span>
        <input
            class="docs-search"
            type="search"
            placeholder="Search docs..."
            bind:value={query}
            use:focusOnMount
        />
        <button class="docs-icon-btn" onclick={onclose} title="Close docs" aria-label="Close docs">
            <Icon icon={X} size={15} strokeWidth={1.6} />
        </button>
    </div>

    <div class="docs-body" bind:this={bodyEl}>
        {#if query.trim()}
            {#if results.length === 0}
                <div class="docs-empty">No matches for “{query.trim()}”</div>
            {:else}
                <ul class="docs-results">
                    {#each results as hit (hit.kind + hit.slug + (hit.anchor ?? ""))}
                        <li>
                            <button class="docs-result" onclick={() => open(hit.slug, hit.anchor)}>
                                <span class="docs-result-title" class:symbol={hit.kind === "symbol"}>{hit.title}</span>
                                <span class="docs-result-meta">{hit.kind === "symbol" ? "ref" : hit.group}</span>
                            </button>
                        </li>
                    {/each}
                </ul>
            {/if}
        {:else if page}
            <nav class="docs-crumbs" aria-label="Breadcrumb">
                <button class="docs-crumb" onclick={home}>Docs</button>
                {#if page.group}<span class="docs-crumb-sep">›</span><span class="docs-crumb-cur">{page.group}</span>{/if}
                <span class="docs-crumb-sep">›</span><span class="docs-crumb-cur">{page.title}</span>
            </nav>
            <div class="docs-page-head">
                <h1>{page.title}</h1>
                {#if page.description}<p class="docs-desc">{page.description}</p>{/if}
            </div>
            <div class="docs-content" use:refNav>
                {#each page.blocks as block, i}
                    {#if block.type === "content"}
                        <article class="doc">{@html block.html}</article>
                    {:else if block.type === "inline"}
                        {@render inlineGroup(block.tabs)}
                    {:else}
                        <div class="doc-tabs">
                            {#if i === page.blocks.findIndex((b) => b.type === "tabs")}
                                <nav class="doc-tabbar" aria-label="Documentation level">
                                    {#each page.tabNames as name}
                                        {#if tabIcon[name]}
                                            <button class="doc-tab" class:active={activeTab === name} onclick={() => (activeTab = name)}>
                                                <Icon icon={tabIcon[name]} size={14} strokeWidth={1.6} />
                                                <span>{name}</span>
                                            </button>
                                        {/if}
                                    {/each}
                                </nav>
                            {/if}
                            {#each block.tabs as tab}
                                {#if activeTab === tab.name}
                                    {#each tab.blocks as inner}
                                        {#if inner.type === "content"}
                                            <article class="doc">{@html inner.html}</article>
                                        {:else}
                                            {@render inlineGroup(inner.tabs)}
                                        {/if}
                                    {/each}
                                {/if}
                            {/each}
                        </div>
                    {/if}
                {/each}
            </div>
        {:else}
            {#each groups as group}
                <div class="docs-group">
                    <div class="docs-group-label">{group.name}</div>
                    {#each group.pages as p}
                        <button class="docs-result" onclick={() => open(p.slug)}>
                            <span class="docs-result-title">{p.title}</span>
                            <span class="docs-result-meta">{p.description}</span>
                        </button>
                    {/each}
                </div>
            {/each}
        {/if}
    </div>
</div>

{#snippet inlineGroup(tabs: { name: string; html: string }[])}
    <div class="inline-group">
        <nav class="inline-bar">
            {#each tabs as t}
                <button class="inline-tab" class:active={inlineTab(tabs) === t.name} onclick={() => (inlineSel[inlineKey(tabs)] = t.name)}>
                    {t.name}
                </button>
            {/each}
        </nav>
        {#each tabs as t}
            {#if inlineTab(tabs) === t.name}
                <article class="doc">{@html t.html}</article>
            {/if}
        {/each}
    </div>
{/snippet}

<style>
    .docs-reader {
        display: flex;
        flex-direction: column;
        height: 100%;
        min-height: 0;
    }

    .docs-head {
        display: flex;
        align-items: center;
        gap: 4px;
        height: var(--header-h);
        padding: 0 8px;
        background: var(--surface-1-solid);
        border-bottom: 1px solid var(--border);
        position: sticky;
        top: 0;
        z-index: 1;
    }

    .docs-search-icon {
        display: flex;
        color: var(--text-muted);
        padding: 0 2px;
    }

    .docs-search {
        flex: 1;
        min-width: 0;
        height: 24px;
        border: none;
        background: transparent;
        color: var(--text);
        font-size: 12px;
        font-family: inherit;
        outline: none;
    }

    .docs-search::placeholder {
        color: var(--text-muted);
    }

    .docs-search::-webkit-search-cancel-button {
        -webkit-appearance: none;
        appearance: none;
    }

    .docs-icon-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        width: 24px;
        height: 24px;
        padding: 0;
        border: none;
        border-radius: 4px;
        background: transparent;
        color: var(--text-muted);
        cursor: pointer;
        transition: background 150ms var(--ease-out), color 150ms var(--ease-out);
    }

    .docs-icon-btn:hover {
        background: var(--surface-2);
        color: var(--text-secondary);
    }

    .docs-icon-btn:active {
        background: color-mix(in srgb, var(--accent) 8%, transparent);
        transform: scale(0.95);
    }

    .docs-body {
        flex: 1;
        min-height: 0;
        overflow-y: auto;
        padding: 10px 14px 32px;
    }

    .docs-empty {
        padding: 16px 4px;
        font-size: 12px;
        color: var(--text-muted);
        text-align: center;
    }

    .docs-results {
        list-style: none;
        margin: 0;
        padding: 0;
    }

    .docs-group + .docs-group {
        margin-top: 10px;
    }

    .docs-group-label {
        font-size: 9px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
        padding: 6px 6px 4px;
    }

    .docs-result {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 1px;
        width: 100%;
        padding: 6px 8px;
        border: none;
        border-radius: 4px;
        background: transparent;
        text-align: left;
        cursor: pointer;
        transition: background 120ms var(--ease-out);
    }

    .docs-result:hover {
        background: var(--surface-2);
    }

    .docs-result:active {
        background: color-mix(in srgb, var(--accent) 8%, transparent);
    }

    .docs-result-title {
        font-size: 12px;
        color: var(--text);
    }

    .docs-result-title.symbol {
        font-family: "JetBrains Mono", monospace;
        font-size: 11px;
        color: var(--accent);
    }

    .docs-result-meta {
        font-size: 10px;
        color: var(--text-muted);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: 100%;
    }

    /* breadcrumb: the sole within-docs nav. `Docs` returns to the index; the trailing crumbs are static.
       Leaving docs entirely is the header ✕ / Esc — the two intents stay distinct (no ambiguous back). */
    .docs-crumbs {
        display: flex;
        align-items: center;
        gap: 5px;
        font-size: 11px;
        color: var(--text-muted);
        margin-bottom: 10px;
    }

    .docs-crumb {
        padding: 0;
        border: none;
        background: none;
        font: inherit;
        color: var(--text-muted);
        cursor: pointer;
        transition: color 150ms var(--ease-out);
    }

    .docs-crumb:hover {
        color: var(--accent);
    }

    .docs-crumb-sep {
        opacity: 0.5;
    }

    .docs-crumb-cur {
        color: var(--text-secondary);
    }

    /* docs are reading material — opt out of the editor's global user-select: none so prose is selectable */
    .docs-page-head,
    .docs-content {
        user-select: text;
        -webkit-user-select: text;
    }

    .docs-page-head {
        padding-bottom: 10px;
        margin-bottom: 4px;
        border-bottom: 1px solid var(--border);
    }

    .docs-page-head h1 {
        font-size: 18px;
        font-weight: 600;
        color: var(--text);
        margin: 0;
    }

    .docs-desc {
        font-size: 12px;
        color: var(--text-muted);
        margin: 4px 0 0;
    }

    .doc-tabs {
        margin-top: 12px;
    }

    .doc-tabbar {
        display: flex;
        gap: 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 12px;
    }

    .doc-tab {
        display: flex;
        align-items: center;
        gap: 5px;
        font-family: inherit;
        font-size: 12px;
        font-weight: 500;
        color: var(--text-muted);
        background: none;
        border: none;
        padding: 7px 10px;
        cursor: pointer;
        border-bottom: 2px solid transparent;
        margin-bottom: -1px;
        transition: color 150ms var(--ease-out), border-color 150ms var(--ease-out);
    }

    .doc-tab:hover {
        color: var(--text-secondary);
    }

    .doc-tab.active {
        color: var(--text);
        border-bottom-color: var(--accent);
    }

    .doc-tab :global(.icon) {
        opacity: 0.6;
    }

    .doc-tab.active :global(.icon) {
        opacity: 1;
    }

    /* the rendered doc HTML — a compact port of the site's reader styles (sites/shallot app.css) onto the
       editor's tokens, sized for the narrow panel. Code colors arrive inline from Shiki. */
    .docs-content {
        font-size: 13px;
        line-height: 1.6;
        color: var(--text-secondary);
    }

    .docs-content :global(p) {
        margin: 0.9em 0;
    }

    .docs-content :global(h2) {
        font-size: 1.25em;
        font-weight: 600;
        color: var(--text);
        margin: 1.8em 0 0.8em;
        padding-bottom: 0.3em;
        border-bottom: 2px solid var(--accent);
    }

    .docs-content :global(h3) {
        font-size: 1.05em;
        font-weight: 600;
        color: var(--text);
        margin: 1.4em 0 0.4em;
    }

    .docs-content :global(h4) {
        color: var(--text);
        margin: 1.2em 0 0.3em;
    }

    .docs-content :global(a) {
        color: var(--accent);
        text-decoration: none;
    }

    .docs-content :global(a:hover) {
        text-decoration: underline;
    }

    .docs-content :global(strong) {
        color: var(--text);
        font-weight: 600;
    }

    .docs-content :global(code) {
        font-family: "JetBrains Mono", monospace;
        font-size: 0.85em;
        background: var(--surface-2-solid);
        padding: 0.12em 0.34em;
        border-radius: 4px;
    }

    .docs-content :global(pre) {
        background: var(--bg) !important;
        border-radius: 4px;
        border-left: 3px solid color-mix(in srgb, var(--accent) 22%, transparent);
        padding: 0.9em 1em;
        overflow-x: auto;
        margin: 1.2em 0;
    }

    .docs-content :global(pre code) {
        background: none;
        padding: 0;
        font-size: 0.85em;
    }

    .docs-content :global(ul),
    .docs-content :global(ol) {
        margin: 0.9em 0;
        padding-left: 1.4em;
    }

    .docs-content :global(li) {
        margin: 0.2em 0;
    }

    .docs-content :global(table) {
        border-collapse: collapse;
        width: 100%;
        margin: 1em 0;
        font-size: 0.9em;
    }

    .docs-content :global(th),
    .docs-content :global(td) {
        padding: 0.4em 0.6em;
        text-align: left;
        border-bottom: 1px solid var(--border);
    }

    .docs-content :global(thead) {
        display: none;
    }

    /* reference list — the Internals tab's generated API entries */
    .docs-content :global(.ref-list) {
        margin: 1.2em 0;
    }

    .docs-content :global(.ref-item),
    .docs-content :global(.ref-entry > summary) {
        display: flex;
        align-items: baseline;
        gap: 0.6em;
        padding: 0.5em 0.2em;
        list-style: none;
        border-bottom: 1px solid var(--border);
    }

    .docs-content :global(.ref-entry > summary) {
        cursor: pointer;
    }

    .docs-content :global(.ref-entry > summary::-webkit-details-marker) {
        display: none;
    }

    .docs-content :global(.ref-entry > summary::before) {
        content: "›";
        font-family: "JetBrains Mono", monospace;
        color: var(--text-muted);
        flex-shrink: 0;
        width: 0.75em;
        transition: transform 150ms var(--ease-out);
    }

    .docs-content :global(.ref-entry[open] > summary::before) {
        transform: rotate(90deg);
        color: var(--accent);
    }

    .docs-content :global(.ref-item code),
    .docs-content :global(.ref-entry > summary code) {
        background: none;
        padding: 0;
        white-space: nowrap;
    }

    .docs-content :global(.ref-params) {
        color: var(--text-muted);
    }

    .docs-content :global(.ref-desc) {
        color: var(--text-muted);
        font-size: 0.85em;
        padding-top: 0.3em;
    }

    .docs-content :global(.ref-methods) {
        padding-left: 1em;
        border-left: 2px solid var(--surface-3);
        margin: 0 0 0.6em 0.5em;
    }

    .docs-content :global(.ref-type) {
        font-family: "JetBrains Mono", monospace;
        font-size: 0.8em;
        color: var(--accent);
        opacity: 0.75;
    }

    .docs-content :global(.ref-type:hover) {
        opacity: 1;
    }

    /* kind badge (component / plugin / enum / function …) — tells entries apart at a glance */
    .docs-content :global(.ref-kind) {
        font-family: "JetBrains Mono", monospace;
        font-size: 0.62em;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--accent);
        background: color-mix(in srgb, var(--accent) 12%, transparent);
        border-radius: 3px;
        padding: 0.15em 0.4em;
        flex-shrink: 0;
        align-self: center;
    }

    /* a component's field reference: field, type, default, description */
    .docs-content :global(.ref-fields) {
        width: 100%;
        border-collapse: collapse;
        margin: 0.2em 0 0.4em;
        font-size: 0.82em;
    }

    .docs-content :global(.ref-fields th) {
        text-align: left;
        font-weight: 500;
        font-size: 0.85em;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        padding: 0 0.7em 0.35em 0;
        border-bottom: 1px solid var(--border);
        white-space: nowrap;
    }

    .docs-content :global(.ref-fields td) {
        padding: 0.25em 0.7em 0.25em 0;
        vertical-align: top;
        border-bottom: 1px solid var(--surface-3);
    }

    .docs-content :global(.ref-fields code) {
        background: none;
        padding: 0;
        white-space: nowrap;
    }

    .docs-content :global(.ref-ftype) {
        font-family: "JetBrains Mono", monospace;
        color: var(--accent);
        opacity: 0.75;
        white-space: nowrap;
    }

    .docs-content :global(.ref-fdefault) {
        font-family: "JetBrains Mono", monospace;
        color: var(--text-muted);
        white-space: nowrap;
    }

    .docs-content :global(.ref-fdesc) {
        color: var(--text-muted);
        width: 100%;
    }

    /* a plugin's bundled parts: a row-labelled table (Components / Systems / Dependencies) */
    .docs-content :global(.ref-parts) {
        margin: 0.2em 0 0.4em;
        font-size: 0.85em;
    }

    .docs-content :global(.ref-parts th) {
        text-align: left;
        font-weight: 500;
        color: var(--text-muted);
        padding: 0.25em 0.8em 0.25em 0;
        vertical-align: top;
        white-space: nowrap;
    }

    .docs-content :global(.ref-parts td) {
        padding: 0.25em 0;
        vertical-align: top;
    }

    .docs-content :global(.ref-item:target),
    .docs-content :global(.ref-entry:target > summary) {
        background: color-mix(in srgb, var(--accent) 10%, transparent);
    }

    .docs-content :global(.ref-src) {
        display: flex;
        flex-shrink: 0;
        margin-left: auto;
        /* a guaranteed gap from the signature: margin-left:auto collapses to 0 when a long nowrap
           signature fills the row, so the padding keeps the icon off the text */
        padding-left: 0.75em;
        color: var(--text-muted);
        opacity: 0.4;
    }

    .docs-content :global(.ref-src:hover) {
        opacity: 1;
        color: var(--accent);
    }

    /* inline os/pick groups */
    .inline-group {
        margin: 1.2em 0;
        border-left: 3px solid var(--surface-3);
        border-radius: 4px;
        background: var(--surface-1-solid);
    }

    .inline-bar {
        display: flex;
        padding: 0 0.25em;
        border-bottom: 1px solid var(--border);
    }

    .inline-tab {
        font-family: "JetBrains Mono", monospace;
        font-size: 0.78em;
        font-weight: 500;
        color: var(--text-muted);
        background: none;
        border: none;
        padding: 0.45em 0.7em;
        cursor: pointer;
        border-bottom: 2px solid transparent;
        margin-bottom: -1px;
        transition: color 150ms var(--ease-out), border-color 150ms var(--ease-out);
    }

    .inline-tab.active {
        color: var(--text);
        border-bottom-color: var(--accent);
    }

    .inline-group :global(pre) {
        margin: 0;
        border-left: none;
        border-radius: 0;
    }
</style>
