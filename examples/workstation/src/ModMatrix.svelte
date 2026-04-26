<script lang="ts">
    import Knob from "./Knob.svelte";
    import SegmentedToggle from "./SegmentedToggle.svelte";
    import { getModSources, getModulableParams, depthKey, type SlotConfig, type ModRoute } from "./graph";
    import { fmtPct, parsePct } from "./presets";

    let {
        slots,
        modRoutes = $bindable(),
        values = $bindable(),
        ontopologychange,
        onvaluechange,
    }: {
        slots: SlotConfig[];
        modRoutes: ModRoute[];
        values: Record<string, number>;
        ontopologychange: () => void;
        onvaluechange: (key: string, value: number) => void;
    } = $props();

    let sources = $derived(getModSources(slots));
    let targets = $derived(getModulableParams(slots));
    const modes: ("linear" | "semitone")[] = ["linear", "semitone"];

    function addRoute() {
        if (sources.length === 0 || targets.length === 0) return;
        const route: ModRoute = {
            source: sources[0].id,
            target: targets[0].slotId,
            param: targets[0].param,
            depth: 0.5,
            mode: "linear",
        };
        modRoutes = [...modRoutes, route];
        values[depthKey(route)] = route.depth;
        ontopologychange();
    }

    function removeRoute(idx: number) {
        const route = modRoutes[idx];
        const key = depthKey(route);
        delete values[key];
        modRoutes = modRoutes.filter((_, i) => i !== idx);
        ontopologychange();
    }

    function updateRoute(idx: number) {
        modRoutes = [...modRoutes];
        const route = modRoutes[idx];
        values[depthKey(route)] = route.depth;
        ontopologychange();
    }

    function updateDepth(idx: number, v: number) {
        modRoutes[idx].depth = v;
        const key = depthKey(modRoutes[idx]);
        values[key] = v;
        onvaluechange(key, v);
    }
</script>

<div class="mod-matrix">
    <div class="header">
        <span class="section-label">Modulation</span>
        <button
            class="add-btn"
            onclick={addRoute}
            disabled={sources.length === 0 || targets.length === 0}
        >+</button>
    </div>
    {#if modRoutes.length === 0}
        <div class="empty">no modulations</div>
    {:else}
        {#each modRoutes as route, idx}
            <div class="route">
                <div class="route-row">
                    <select bind:value={route.source} onchange={() => updateRoute(idx)}>
                        {#each sources as src}
                            <option value={src.id}>{src.label}</option>
                        {/each}
                    </select>
                    <span class="arrow">&rarr;</span>
                    <select value={`${route.target}.${route.param}`} onchange={(e) => {
                        const val = (e.target as HTMLSelectElement).value;
                        const dot = val.lastIndexOf(".");
                        route.target = val.slice(0, dot);
                        route.param = val.slice(dot + 1);
                        updateRoute(idx);
                    }}>
                        {#each targets as t}
                            <option value={`${t.slotId}.${t.param}`}>{t.slotLabel}.{t.param}</option>
                        {/each}
                    </select>
                    <button class="remove-btn" onclick={() => removeRoute(idx)}>×</button>
                </div>
                <div class="route-row">
                    <Knob
                        label="Depth"
                        value={route.depth}
                        min={0}
                        max={1}
                        step={0.01}
                        fmt={fmtPct}
                        parse={parsePct}
                        defaultValue={0.5}
                        onchange={(v) => updateDepth(idx, v)}
                    />
                    <SegmentedToggle
                        options={["Lin", "Semi"]}
                        selected={modes.indexOf(route.mode)}
                        onchange={(i) => { route.mode = modes[i]; updateRoute(idx); }}
                    />
                </div>
            </div>
        {/each}
    {/if}
</div>

<style>
    .mod-matrix {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }

    .header {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .section-label {
        font-size: 11px;
        font-weight: 600;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.04em;
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

    .add-btn:hover:not(:disabled) {
        background: var(--surface-2);
        color: var(--text);
    }

    .add-btn:disabled {
        opacity: 0.3;
        cursor: default;
    }

    .empty {
        color: var(--text-muted);
        font-size: 11px;
    }

    .route {
        display: flex;
        flex-direction: column;
        gap: 6px;
        padding: 10px 12px;
        background: var(--surface-1);
        border-radius: 4px;
    }

    .route-row {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .route-row select {
        flex: 1;
        min-width: 0;
        font-size: 11px;
        padding: 5px 22px 5px 8px;
    }

    .arrow {
        color: var(--text-muted);
        font-size: 11px;
        flex-shrink: 0;
    }

    .remove-btn {
        width: 18px;
        height: 18px;
        background: none;
        border: none;
        color: var(--text-muted);
        font-size: 14px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0;
        border-radius: 2px;
        flex-shrink: 0;
        line-height: 1;
    }

    .remove-btn:hover {
        background: var(--surface-3);
        color: var(--text);
    }
</style>
