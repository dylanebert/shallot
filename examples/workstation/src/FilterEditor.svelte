<script lang="ts">
    import Knob from "./Knob.svelte";
    import SegmentedToggle from "./SegmentedToggle.svelte";
    import FilterViz from "./FilterViz.svelte";
    import { filterModes, filterKnobs } from "./presets";

    let {
        slotId,
        values = $bindable(),
        onchange,
    }: {
        slotId: string;
        values: Record<string, number>;
        onchange: (key: string, value: number) => void;
    } = $props();

    let mode = $derived(values[`${slotId}.mode`] ?? 0);
    let cutoff = $derived(values[`${slotId}.cutoff`] ?? 22050);
    let res = $derived(values[`${slotId}.q`] ?? 0);
    let mix = $derived(values[`${slotId}.mix`] ?? 0);

    function setMode(idx: number) {
        values[`${slotId}.mode`] = idx;
        onchange(`${slotId}.mode`, idx);
    }
</script>

<div class="editor">
    <FilterViz mode={mode + 1} {cutoff} {res} {mix} />
    <SegmentedToggle options={filterModes.slice(1)} selected={mode} onchange={(i) => setMode(i)} />
    {#each filterKnobs as knob}
        {@const key = `${slotId}.${knob.field}`}
        {@const displayVal = knob.fromAudio ? knob.fromAudio(values[key] ?? knob.default) : (values[key] ?? knob.default)}
        <Knob
            label={knob.label}
            value={displayVal}
            min={knob.min}
            max={knob.max}
            step={knob.step}
            scale={knob.scale}
            fmt={knob.fmt}
            parse={knob.parse}
            defaultValue={knob.default}
            onchange={(v) => {
                const audioVal = knob.toAudio ? knob.toAudio(v) : v;
                values[key] = audioVal;
                onchange(key, audioVal);
            }}
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
