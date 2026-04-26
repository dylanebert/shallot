<script lang="ts">
    import Knob from "./Knob.svelte";
    import { mixKnobs } from "./presets";

    let {
        slotId,
        values = $bindable(),
        onchange,
    }: {
        slotId: string;
        values: Record<string, number>;
        onchange: (key: string, value: number) => void;
    } = $props();
</script>

<div class="editor">
    {#each mixKnobs as knob}
        {@const key = `${slotId}.${knob.field}`}
        <Knob
            label={knob.label}
            bind:value={values[key]}
            min={knob.min}
            max={knob.max}
            step={knob.step}
            fmt={knob.fmt}
            parse={knob.parse}
            defaultValue={knob.default}
            onchange={() => onchange(key, values[key])}
        />
    {/each}
</div>

<style>
    .editor {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
</style>
