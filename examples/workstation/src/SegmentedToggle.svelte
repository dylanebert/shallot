<script lang="ts">
    let {
        options,
        selected = $bindable(),
        onchange,
    }: {
        options: string[];
        selected: number;
        onchange?: (index: number) => void;
    } = $props();
</script>

<div class="toggle-group">
    {#each options as opt, i}
        <button
            class="toggle-option"
            class:active={selected === i}
            onclick={() => { selected = i; onchange?.(i); }}
        >
            {opt}
        </button>
    {/each}
</div>

<style>
    .toggle-group {
        display: flex;
        gap: 2px;
        background: var(--surface-1);
        border-radius: 6px;
        padding: 2px;
    }

    .toggle-option {
        flex: 1;
        padding: 5px 10px;
        font-size: 11px;
        font-family: "Outfit", system-ui, sans-serif;
        font-weight: 500;
        color: var(--text-muted);
        background: transparent;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        transition: all 150ms var(--ease-out);
    }

    .toggle-option:hover {
        color: var(--text-secondary);
    }

    .toggle-option.active {
        background: var(--accent);
        color: var(--bg);
    }

    .toggle-option:active {
        transform: scale(0.95);
    }
</style>
