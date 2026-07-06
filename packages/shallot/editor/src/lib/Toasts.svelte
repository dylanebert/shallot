<script lang="ts">
    import { getToasts, dismissToast } from "./notify.svelte.js";

    let toasts = $derived(getToasts());
</script>

<div class="toasts">
    {#each toasts as t (t.id)}
        <button
            class="toast"
            class:error={t.severity === "error"}
            class:warning={t.severity === "warning"}
            onclick={() => dismissToast(t.id)}
        >
            <span class="dot"></span>
            <span class="text">{t.text}</span>
        </button>
    {/each}
</div>

<style>
    .toasts {
        position: absolute;
        bottom: 16px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 5;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
        pointer-events: none;
    }

    .toast {
        display: flex;
        align-items: center;
        gap: 8px;
        max-width: 420px;
        padding: 7px 12px 7px 10px;
        border: 1px solid var(--border);
        border-radius: 8px;
        background: var(--surface-2-solid);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3), 0 1px 4px rgba(0, 0, 0, 0.2);
        color: var(--text-secondary);
        font-size: 12px;
        cursor: pointer;
        pointer-events: auto;
        animation: toast-in 200ms var(--ease-out);
    }

    .dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        flex-shrink: 0;
        background: var(--accent);
    }

    .toast.error .dot { background: var(--error); }
    .toast.warning .dot { background: var(--warning); }

    .text {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    @keyframes toast-in {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
