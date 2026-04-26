<script lang="ts">
    import type { GraphPreset } from "./presets";

    let {
        presets,
        presetIndex = $bindable(),
        onpresetchange,
    }: {
        presets: GraphPreset[];
        presetIndex: number;
        onpresetchange: (index: number) => void;
    } = $props();
</script>

<div class="toolbar">
    <div class="toolbar-left">
        <span class="section-label">Preset</span>
        <select class="preset-select" bind:value={presetIndex} onchange={() => onpresetchange(presetIndex)}>
            {#each presets as preset, i}
                <option value={i}>{preset.name}</option>
            {/each}
        </select>
    </div>
    <div class="toolbar-right">
        <span class="hint">Z-M keys to play</span>
    </div>
</div>

<style>
    .toolbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        height: 40px;
        padding: 0 16px;
        background: var(--surface-2-solid);
        border-bottom: 1px solid var(--border);
        flex-shrink: 0;
    }

    .toolbar-left {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .toolbar-right {
        display: flex;
        align-items: center;
    }

    .section-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--accent);
        flex-shrink: 0;
    }

    .preset-select {
        min-width: 140px;
    }

    .hint {
        font-size: 11px;
        color: var(--text-muted);
        font-family: "JetBrains Mono", monospace;
    }
</style>
