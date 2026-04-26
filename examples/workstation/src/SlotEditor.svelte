<script lang="ts">
    import type { SlotConfig } from "./graph";
    import type { SampleBuffer } from "./App.svelte";
    import OscEditor from "./OscEditor.svelte";
    import FilterEditor from "./FilterEditor.svelte";
    import EnvelopeEditor from "./EnvelopeEditor.svelte";
    import ConstantEditor from "./ConstantEditor.svelte";
    import GainEditor from "./GainEditor.svelte";
    import MixEditor from "./MixEditor.svelte";
    import SampleEditor from "./SampleEditor.svelte";

    let {
        slot,
        values = $bindable(),
        sampleBuffers = $bindable(),
        onchange,
    }: {
        slot: SlotConfig | null;
        values: Record<string, number>;
        sampleBuffers: SampleBuffer[];
        onchange: (key: string, value: number) => void;
    } = $props();
</script>

<div class="slot-editor">
    {#if slot}
        <div class="section-label">{slot.label}</div>
        <div class="editor-body">
            {#if slot.type === "oscillator"}
                <OscEditor slotId={slot.id} bind:values {onchange} />
            {:else if slot.type === "filter"}
                <FilterEditor slotId={slot.id} bind:values {onchange} />
            {:else if slot.type === "envelope"}
                <EnvelopeEditor slotId={slot.id} bind:values {onchange} />
            {:else if slot.type === "constant"}
                <ConstantEditor slotId={slot.id} bind:values {onchange} />
            {:else if slot.type === "gain"}
                <GainEditor slotId={slot.id} bind:values {onchange} />
            {:else if slot.type === "mix"}
                <MixEditor slotId={slot.id} bind:values {onchange} />
            {:else if slot.type === "sample"}
                <SampleEditor slotId={slot.id} bind:values bind:buffers={sampleBuffers} {onchange} />
            {/if}
        </div>
    {:else}
        <div class="empty">select a node</div>
    {/if}
</div>

<style>
    .slot-editor {
        display: flex;
        flex-direction: column;
        gap: 12px;
        max-width: 560px;
    }

    .section-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--accent);
    }

    .empty {
        color: var(--text-muted);
        font-size: 12px;
        padding: 20px 0;
    }
</style>
