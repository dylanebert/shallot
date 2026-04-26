<script lang="ts">
    import type { SlotConfig, AddableType } from "./graph";
    import { canAdd } from "./graph";

    let {
        slots,
        userSlots,
        selected,
        onselect,
        onadd,
        onremove,
    }: {
        slots: SlotConfig[];
        userSlots: SlotConfig[];
        selected: string;
        onselect: (id: string) => void;
        onadd: (type: AddableType) => void;
        onremove: (id: string) => void;
    } = $props();

    const addableTypes: { type: AddableType; label: string }[] = [
        { type: "oscillator", label: "Osc" },
        { type: "sample", label: "Sample" },
        { type: "filter", label: "Filter" },
        { type: "envelope", label: "Mod Env" },
        { type: "constant", label: "Const" },
    ];

    let menuOpen = $state(false);
</script>

<div class="slot-list">
    <div class="header">
        <span class="section-label">Nodes</span>
        <div class="add-wrap">
            <button class="add-btn" onclick={() => menuOpen = !menuOpen}>+</button>
            {#if menuOpen}
                <div class="add-menu">
                    {#each addableTypes as at}
                        <button
                            class="add-option"
                            disabled={!canAdd(userSlots, at.type)}
                            onclick={() => { onadd(at.type); menuOpen = false; }}
                        >{at.label}</button>
                    {/each}
                </div>
            {/if}
        </div>
    </div>
    <div class="items">
        {#each slots as slot}
            <div
                class="slot-item"
                class:active={slot.id === selected}
                onclick={() => onselect(slot.id)}
                onkeydown={(e) => { if (e.key === "Enter") onselect(slot.id); }}
                role="button"
                tabindex="0"
            >
                <span class="dot" class:osc={slot.type === "oscillator"} class:sample={slot.type === "sample"} class:filter={slot.type === "filter"} class:env={slot.type === "envelope"} class:const={slot.type === "constant"} class:gain={slot.type === "gain"} class:mix={slot.type === "mix"}></span>
                <span class="slot-label">{slot.label}</span>
                {#if slot.removable}
                    <button class="remove-btn" onclick={(e) => { e.stopPropagation(); onremove(slot.id); }}>×</button>
                {/if}
            </div>
        {/each}
    </div>
</div>

<style>
    .slot-list {
        display: flex;
        flex-direction: column;
        gap: 6px;
    }

    .header {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .section-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--accent);
    }

    .add-wrap {
        position: relative;
    }

    .add-btn {
        width: 22px;
        height: 22px;
        background: transparent;
        border: 1px solid var(--border);
        border-radius: 4px;
        color: var(--text-muted);
        font-size: 14px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0;
        line-height: 1;
        transition: color 150ms var(--ease-out), background 150ms var(--ease-out);
    }

    .add-btn:hover {
        background: var(--surface-2);
        color: var(--text);
    }

    .add-menu {
        position: absolute;
        top: 100%;
        right: 0;
        margin-top: 4px;
        background: var(--surface-2-solid);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 2px;
        z-index: 10;
        min-width: 90px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }

    .add-option {
        display: block;
        width: 100%;
        padding: 4px 8px;
        font-size: 11px;
        font-family: "JetBrains Mono", monospace;
        color: var(--text-secondary);
        background: none;
        border: none;
        border-radius: 3px;
        cursor: pointer;
        text-align: left;
    }

    .add-option:hover:not(:disabled) {
        background: var(--surface-3);
        color: var(--text);
    }

    .add-option:disabled {
        opacity: 0.3;
        cursor: default;
    }

    .items {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }

    .slot-item {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 5px 8px;
        background: transparent;
        border: 1px solid transparent;
        border-radius: 4px;
        cursor: pointer;
        color: var(--text-muted);
        font-size: 11px;
        font-family: "JetBrains Mono", monospace;
        transition: all 150ms var(--ease-out);
    }

    .slot-item:hover {
        background: var(--surface-2);
        color: var(--text-secondary);
    }

    .slot-item.active {
        background: var(--surface-2);
        border-color: var(--accent);
        color: var(--text);
    }

    .dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        flex-shrink: 0;
        background: var(--text-muted);
    }

    .dot.osc { background: #d49560; }
    .dot.sample { background: #c97060; }
    .dot.filter { background: #60b0d4; }
    .dot.env { background: #60d480; }
    .dot.const { background: #d460b0; }
    .dot.gain { background: #d4d460; }
    .dot.mix { background: #9060d4; }

    .slot-label {
        flex: 1;
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .remove-btn {
        width: 14px;
        height: 14px;
        background: none;
        border: none;
        color: var(--text-muted);
        font-size: 12px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0;
        border-radius: 2px;
        opacity: 0;
        margin-left: auto;
        flex-shrink: 0;
        line-height: 1;
        transition: opacity 100ms;
    }

    .slot-item:hover .remove-btn {
        opacity: 0.5;
    }

    .remove-btn:hover {
        background: var(--surface-4);
        opacity: 1 !important;
        color: var(--text);
    }
</style>
