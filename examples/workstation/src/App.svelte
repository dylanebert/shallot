<script lang="ts" module>
    export interface SampleBuffer {
        id: number;
        name: string;
        data: Float32Array;
        sampleRate: number;
    }
</script>

<script lang="ts">
    import type { AudioState } from "@dylanebert/shallot/audio/core";
    import TransportBar from "./TransportBar.svelte";
    import SlotList from "./SlotList.svelte";
    import SlotEditor from "./SlotEditor.svelte";
    import ModMatrix from "./ModMatrix.svelte";
    import SignalFlow from "./SignalFlow.svelte";
    import Keyboard from "./Keyboard.svelte";
    import Knob from "./Knob.svelte";
    import {
        type SlotConfig,
        type ModRoute,
        type AddableType,
        getDisplaySlots,
        createSlot,
        removeSlot,
        compile,
        recompile,
        applyValues,
        getInstId,
    } from "./graph";
    import { graphPresets, keyboardKnobs } from "./presets";

    let { audio }: { audio: AudioState } = $props();

    const initSlots = graphPresets[0].slots.map((s) => ({ ...s }));
    const initValues = { ...graphPresets[0].values };
    compile(initSlots, [], initValues);

    let userSlots = $state<SlotConfig[]>(initSlots);
    let modRoutes = $state<ModRoute[]>([]);
    let values = $state<Record<string, number>>(initValues);
    let presetIndex = $state(0);
    let selectedSlotId = $state("osc1");
    let sampleBuffers = $state<SampleBuffer[]>([]);

    let displaySlots = $derived(getDisplaySlots(userSlots));
    let selectedSlot = $derived(displaySlots.find((s) => s.id === selectedSlotId) ?? null);

    let keyboard: { onkeydown: (e: KeyboardEvent) => void; onkeyup: (e: KeyboardEvent) => void };

    function onValueChange(key: string, value: number) {
        values[key] = value;
        applyValues(audio, { [key]: value });
    }

    function onTopologyChange() {
        recompile(audio, userSlots, modRoutes, values);
    }

    function onAddSlot(type: AddableType) {
        const slot = createSlot(userSlots, type);
        if (!slot) return;
        userSlots = [...userSlots, slot];
        selectedSlotId = slot.id;
        onTopologyChange();
    }

    function onRemoveSlot(id: string) {
        const result = removeSlot(userSlots, modRoutes, id);
        userSlots = result.slots;
        modRoutes = result.modRoutes;
        if (selectedSlotId === id) {
            selectedSlotId = userSlots[0]?.id ?? "";
        }
        onTopologyChange();
    }

    function loadPreset(index: number) {
        const preset = graphPresets[index];
        if (!preset) return;
        userSlots = preset.slots.map((s) => ({ ...s }));
        modRoutes = preset.modRoutes.map((r) => ({ ...r }));
        values = { ...preset.values };
        selectedSlotId = userSlots[0]?.id ?? "";
        recompile(audio, userSlots, modRoutes, values);
    }

    function onKeydown(e: KeyboardEvent) {
        keyboard?.onkeydown(e);
    }

    function onKeyup(e: KeyboardEvent) {
        keyboard?.onkeyup(e);
    }
</script>

<svelte:window onkeydown={onKeydown} onkeyup={onKeyup} />

<div class="synth">
    <header class="bar">
        <TransportBar presets={graphPresets} bind:presetIndex onpresetchange={loadPreset} />
    </header>

    <nav class="nodes">
        <SlotList
            slots={displaySlots}
            {userSlots}
            selected={selectedSlotId}
            onselect={(id) => selectedSlotId = id}
            onadd={onAddSlot}
            onremove={onRemoveSlot}
        />
    </nav>

    <main class="center">
        <SlotEditor slot={selectedSlot} bind:values bind:sampleBuffers onchange={onValueChange} />
    </main>

    <aside class="routing">
        <SignalFlow slots={userSlots} {modRoutes} />
        <ModMatrix
            slots={userSlots}
            bind:modRoutes
            bind:values
            ontopologychange={onTopologyChange}
            onvaluechange={onValueChange}
        />
    </aside>

    <footer class="perform">
        <div class="master">
            <span class="master-label">Master</span>
            <div class="master-knobs">
                {#each keyboardKnobs as knob}
                    <Knob
                        label={knob.label}
                        bind:value={values[knob.field]}
                        min={knob.min}
                        max={knob.max}
                        step={knob.step}
                        fmt={knob.fmt}
                        parse={knob.parse}
                        defaultValue={knob.default}
                        onchange={(v) => onValueChange(knob.field, v)}
                    />
                {/each}
            </div>
        </div>
        <div class="keys">
            <Keyboard
                bind:this={keyboard}
                {audio}
                instrumentId={getInstId()}
                octave={values.octave ?? 0}
                semitone={values.semitone ?? 0}
                fine={values.fine ?? 0}
                volume={values.volume ?? 0.7}
                {values}
            />
        </div>
    </footer>
</div>

<style>
    :global(body) {
        user-select: none;
        -webkit-user-select: none;
    }

    :global(select:focus, button:focus) {
        outline: none;
    }

    :global(select) {
        font-family: "JetBrains Mono", monospace;
        font-size: 12px;
        background: var(--surface-2);
        color: var(--text-secondary);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 5px 8px;
        cursor: pointer;
        appearance: none;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%23a09890'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-position: right 8px center;
        padding-right: 24px;
        transition: border-color 150ms var(--ease-out);
    }

    :global(select:hover) {
        border-color: var(--text-muted);
    }

    :global(select option) {
        background: var(--surface-2-solid);
        color: var(--text-secondary);
    }

    :global(:root) {
        --bg: #0e0d0c;
        --text: #f0ece8;
        --text-secondary: #cdc5bc;
        --text-muted: #a09890;
        --accent: #d49560;
        --accent-hover: #e8a86b;
        --surface-1: rgba(255, 255, 255, 0.03);
        --surface-2: rgba(255, 255, 255, 0.07);
        --surface-3: rgba(255, 255, 255, 0.12);
        --surface-4: rgba(255, 255, 255, 0.18);
        --surface-1-solid: #161514;
        --surface-2-solid: #1f1e1d;
        --border: rgba(255, 255, 255, 0.09);
        --ease-out: cubic-bezier(0.34, 0, 0, 1);
    }

    .synth {
        height: 100vh;
        display: grid;
        grid-template-rows: 40px 1fr auto;
        grid-template-columns: 200px 1fr 420px;
        grid-template-areas:
            "bar     bar     bar"
            "nodes   center  routing"
            "perform perform perform";
        background: var(--bg);
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
    }

    .bar {
        grid-area: bar;
    }

    .nodes {
        grid-area: nodes;
        padding: 12px;
        border-right: 1px solid var(--border);
        background: var(--surface-1);
        overflow-y: auto;
    }

    .nodes::-webkit-scrollbar { width: 6px; }
    .nodes::-webkit-scrollbar-track { background: transparent; }
    .nodes::-webkit-scrollbar-thumb { background: transparent; border-radius: 4px; }
    .nodes:hover::-webkit-scrollbar-thumb { background: var(--border); }

    .center {
        grid-area: center;
        overflow-y: auto;
        padding: 20px 24px;
        scrollbar-gutter: stable;
    }

    .center::-webkit-scrollbar { width: 6px; }
    .center::-webkit-scrollbar-track { background: transparent; }
    .center::-webkit-scrollbar-thumb { background: transparent; border-radius: 4px; }
    .center:hover::-webkit-scrollbar-thumb { background: var(--border); }

    .routing {
        grid-area: routing;
        border-left: 1px solid var(--border);
        overflow-y: auto;
        padding: 16px 20px;
        display: flex;
        flex-direction: column;
        gap: 16px;
        scrollbar-gutter: stable;
    }

    .routing::-webkit-scrollbar { width: 6px; }
    .routing::-webkit-scrollbar-track { background: transparent; }
    .routing::-webkit-scrollbar-thumb { background: transparent; border-radius: 4px; }
    .routing:hover::-webkit-scrollbar-thumb { background: var(--border); }

    .perform {
        grid-area: perform;
        display: flex;
        align-items: stretch;
        border-top: 1px solid var(--border);
        background: var(--surface-1-solid);
    }

    .master {
        width: 200px;
        flex-shrink: 0;
        padding: 12px 16px;
        border-right: 1px solid var(--border);
        display: flex;
        flex-direction: column;
        gap: 4px;
    }

    .master-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--accent);
        margin-bottom: 2px;
    }

    .master-knobs {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }

    .keys {
        flex: 1;
        min-width: 0;
        display: flex;
        justify-content: center;
    }
</style>
