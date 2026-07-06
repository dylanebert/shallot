<script lang="ts">
    import { getBanners, clearBanner } from "./notify.svelte.js";

    let banners = $derived(getBanners());
</script>

<div class="banners">
    {#each banners as b (b.id)}
        <div class="banner" class:error={b.severity === "error"} class:warning={b.severity === "warning"}>
            <span class="dot"></span>
            <span class="text">{b.text}</span>
            {#if b.actions}
                {#each b.actions as action (action.label)}
                    <button class="action" onclick={action.fn}>{action.label}</button>
                {/each}
            {/if}
            <button class="dismiss" onclick={() => clearBanner(b.id)} title="Dismiss">
                <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
                    <path d="M4 4l8 8M12 4l-8 8" />
                </svg>
            </button>
        </div>
    {/each}
</div>

<style>
    .banners {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        z-index: 4;
        display: flex;
        flex-direction: column;
    }

    .banner {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 7px 8px 7px 12px;
        background: var(--surface-2-solid);
        border-bottom: 1px solid var(--border);
        box-shadow: inset 2px 0 0 var(--accent);
        color: var(--text-secondary);
        font-size: 12px;
        animation: banner-in 200ms var(--ease-out);
    }

    .banner.error { box-shadow: inset 2px 0 0 var(--error); }
    .banner.warning { box-shadow: inset 2px 0 0 var(--warning); }

    .dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        flex-shrink: 0;
        background: var(--accent);
    }

    .banner.error .dot { background: var(--error); }
    .banner.warning .dot { background: var(--warning); }

    .text {
        flex: 1;
        min-width: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .action {
        flex-shrink: 0;
        padding: 2px 8px;
        border: 1px solid var(--border);
        border-radius: 4px;
        background: transparent;
        color: var(--text-secondary);
        font-size: 11px;
        cursor: pointer;
        transition: all 120ms var(--ease-out);
    }

    .action:hover {
        color: var(--text);
        border-color: var(--text-muted);
        background: var(--surface-2);
    }

    .action:active {
        background: color-mix(in srgb, var(--accent) 8%, transparent);
        transform: scale(0.95);
    }

    .dismiss {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        flex-shrink: 0;
        border: none;
        border-radius: 4px;
        background: transparent;
        color: var(--text-muted);
        cursor: pointer;
        transition: all 120ms var(--ease-out);
    }

    .dismiss:hover {
        color: var(--text);
        background: var(--surface-2);
    }

    .dismiss svg {
        width: 11px;
        height: 11px;
    }

    @keyframes banner-in {
        from { opacity: 0; transform: translateY(-4px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
