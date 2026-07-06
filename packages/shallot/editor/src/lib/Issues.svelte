<script lang="ts">
    import type { Diagnostic } from "@dylanebert/shallot";
    import type { Node } from "@dylanebert/shallot/editor";
    import { dismissOnClickOutside } from "./dismiss";
    import { nodeLabel } from "./components";

    let { diagnostics, onselect }: { diagnostics: Diagnostic[]; onselect: (node: Node) => void } = $props();

    let open = $state(false);

    $effect(() => {
        if (!open) return;
        return dismissOnClickOutside(() => { open = false; }, ".issues");
    });

    // the issue set is the live scene diagnostics, so it empties itself as the user fixes each — no
    // separate stream to clear. The badge stays hidden at zero (quiet when silent).
    $effect(() => {
        if (diagnostics.length === 0) open = false;
    });

    function pick(node: Node) {
        onselect(node);
        open = false;
    }
</script>

{#if diagnostics.length > 0}
    <div class="issues">
        <button class="badge" class:active={open} onclick={() => open = !open} title="Issues">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">
                <path d="M8 2 L14.5 13.5 L1.5 13.5 Z" />
                <path d="M8 6.5 V9.5" />
                <circle cx="8" cy="11.6" r="0.4" fill="currentColor" stroke="none" />
            </svg>
            <span class="count">{diagnostics.length}</span>
        </button>
        {#if open}
            <div class="popover">
                {#each diagnostics as d}
                    <button class="row" onclick={() => pick(d.node)}>
                        <span class="message">{d.message}</span>
                        <span class="node">{nodeLabel(d.node)}</span>
                    </button>
                {/each}
            </div>
        {/if}
    </div>
{/if}

<style>
    .issues {
        position: relative;
    }

    .badge {
        display: flex;
        align-items: center;
        gap: 4px;
        height: 22px;
        padding: 0 7px 0 5px;
        border: none;
        border-radius: 4px;
        background: transparent;
        color: var(--warning);
        font-size: 11px;
        font-variant-numeric: tabular-nums;
        cursor: pointer;
        transition: all 150ms var(--ease-out);
    }

    .badge:hover { background: var(--surface-2); }

    .badge.active { background: var(--surface-2); }

    .badge:active {
        background: color-mix(in srgb, var(--accent) 8%, transparent);
        transform: scale(0.95);
    }

    .badge svg {
        width: 13px;
        height: 13px;
    }

    .popover {
        position: absolute;
        top: 100%;
        right: 0;
        margin-top: 6px;
        width: 320px;
        max-height: 280px;
        overflow-y: auto;
        padding: 4px;
        border: 1px solid var(--border);
        border-radius: 6px;
        z-index: 10;
        background: var(--surface-3-solid);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3), 0 1px 4px rgba(0, 0, 0, 0.2);
        transform-origin: top right;
        animation: popover-in 150ms var(--ease-out);
    }

    .row {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 1px;
        width: 100%;
        padding: 5px 8px;
        border: none;
        border-radius: 4px;
        background: transparent;
        color: var(--text-secondary);
        font-size: 11px;
        text-align: left;
        cursor: pointer;
        transition: background 120ms var(--ease-out);
    }

    .row:hover { background: var(--surface-2); }

    .row:active { background: color-mix(in srgb, var(--accent) 8%, transparent); }

    .message {
        line-height: 1.35;
        word-break: break-word;
    }

    .node {
        color: var(--text-muted);
        font-size: 10px;
        font-family: ui-monospace, "SF Mono", "Cascadia Code", monospace;
    }

    @keyframes popover-in {
        from { opacity: 0; transform: scale(0.97); }
        to { opacity: 1; transform: scale(1); }
    }
</style>
