<script lang="ts">
    import type { Plugin } from "@dylanebert/shallot";
    import type { DiscoveredPlugin } from "../project";
    import { pluginName } from "../plugins";
    import { dismissOnClickOutside } from "./dismiss";

    interface Props {
        custom: DiscoveredPlugin[];
        standard: DiscoveredPlugin[];
        shallot: DiscoveredPlugin[];
        enabled: Set<Plugin>;
        ontoggle: (plugin: Plugin) => void;
        hasProject: boolean;
        scenes: string[];
        activeScene: string | null;
        onscene: (path: string) => void;
    }

    let { custom, standard, shallot, enabled, ontoggle, hasProject, scenes, activeScene, onscene }: Props = $props();

    function sceneLabel(path: string): string {
        const parts = path.split("/");
        return parts[parts.length - 1].replace(/\.scene$/, "");
    }

    let open = $state(false);

    const enabledCount = $derived(
        [...custom, ...standard, ...shallot].filter(dp => enabled.has(dp.plugin)).length
    );
    const totalCount = $derived(custom.length + standard.length + shallot.length);

    function toggle() {
        open = !open;
    }

    function close() {
        open = false;
    }

    $effect(() => {
        if (!open) return;
        return dismissOnClickOutside(close, ".logo-menu");
    });
</script>

<div class="logo-menu">
    <button class="logo-btn" class:active={open} onclick={toggle}>
        <img src="/logo.svg" alt="shallot" class="logo" />
        <svg class="caret" viewBox="0 0 8 6" fill="currentColor">
            <path d="M1 1.5 L4 4.5 L7 1.5" />
        </svg>
    </button>
    {#if open}
        <div class="menu-dropdown">
            {#if hasProject}
                <div class="menu-section-label">
                    Plugins
                    <span class="plugin-count">{enabledCount}/{totalCount}</span>
                </div>
                <div class="plugin-section">
                    {#each custom as dp}
                        {@const active = enabled.has(dp.plugin)}
                        <button class="plugin-row" class:active onclick={() => ontoggle(dp.plugin)}>
                            <span class="dot"></span>
                            <span class="plugin-name">{pluginName(dp)}</span>
                            <span class="tag project">project</span>
                        </button>
                    {/each}
                    {#each standard as dp}
                        {@const active = enabled.has(dp.plugin)}
                        <button class="plugin-row" class:active onclick={() => ontoggle(dp.plugin)}>
                            <span class="dot"></span>
                            <span class="plugin-name">{pluginName(dp)}</span>
                            <span class="tag standard">standard</span>
                        </button>
                    {/each}
                    {#each shallot as dp}
                        {@const active = enabled.has(dp.plugin)}
                        <button class="plugin-row" class:active onclick={() => ontoggle(dp.plugin)}>
                            <span class="dot"></span>
                            <span class="plugin-name">{pluginName(dp)}</span>
                        </button>
                    {/each}
                </div>
            {/if}
            {#if scenes.length > 1}
                {#if hasProject}
                    <div class="menu-divider"></div>
                {/if}
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

<style>
    .logo-menu {
        position: relative;
    }

    .logo-btn {
        display: flex;
        align-items: center;
        gap: 4px;
        height: 28px;
        padding: 0 8px;
        border: none;
        border-radius: 4px;
        background: transparent;
        color: var(--text-muted);
        cursor: pointer;
        transition: background 150ms var(--ease-out), color 150ms var(--ease-out);
    }

    .logo-btn:hover,
    .logo-btn.active {
        background: var(--surface-2);
        color: var(--text-secondary);
    }

    .logo-btn:active {
        transform: scale(0.95);
    }

    .logo-btn:focus-visible {
        outline: 1px solid var(--accent);
        outline-offset: -1px;
    }

    .logo {
        height: 18px;
        width: auto;
    }

    .caret {
        width: 8px;
        height: 6px;
        opacity: 0.6;
    }

    .menu-dropdown {
        position: absolute;
        top: 100%;
        left: 0;
        margin-top: 4px;
        min-width: 200px;
        padding: 4px 0;
        border: 1px solid var(--border);
        border-radius: 6px;
        z-index: 100;
        background: rgba(38, 37, 36, 0.85);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3), 0 1px 4px rgba(0, 0, 0, 0.2);
        transform-origin: top left;
        animation: menu-appear 150ms var(--ease-out);
    }

    @keyframes menu-appear {
        from { opacity: 0; transform: scale(0.97); }
        to { opacity: 1; transform: scale(1); }
    }

    .menu-divider {
        height: 1px;
        margin: 4px 8px;
        background: var(--border);
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
        margin-left: auto;
        font-variant-numeric: tabular-nums;
        opacity: 0.6;
    }

    .plugin-section {
        max-height: 240px;
        overflow-y: auto;
        padding: 0 4px 2px;
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
        background: rgba(212, 149, 96, 0.08);
        transform: scale(0.95);
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
        background: rgba(74, 144, 226, 0.12);
    }

    .tag.standard {
        color: var(--cat-rendering);
        background: rgba(80, 200, 120, 0.12);
    }
</style>
