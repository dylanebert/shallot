<script lang="ts">
    import Knob from "./Knob.svelte";
    import SegmentedToggle from "./SegmentedToggle.svelte";
    import WaveformViz from "./WaveformViz.svelte";
    import { waveforms, oscKnobs, oscPitchKnobs } from "./presets";

    let {
        slotId,
        values = $bindable(),
        onchange,
    }: {
        slotId: string;
        values: Record<string, number>;
        onchange: (key: string, value: number) => void;
    } = $props();

    $effect(() => {
        for (const knob of [...oscPitchKnobs, ...oscKnobs]) {
            const key = `${slotId}.${knob.field}`;
            if (values[key] === undefined) values[key] = knob.default;
        }
    });

    let waveform = $derived(values[`${slotId}.waveform`] ?? 0);

    function setWaveform(idx: number) {
        values[`${slotId}.waveform`] = idx;
        onchange(`${slotId}.waveform`, idx);
    }
</script>

<div class="editor">
    <WaveformViz {waveform} />
    <SegmentedToggle options={waveforms} selected={waveform} onchange={(i) => setWaveform(i)} />
    {#each oscPitchKnobs as knob}
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
    {#each oscKnobs as knob}
        {@const key = `${slotId}.${knob.field}`}
        <Knob
            label={knob.label}
            bind:value={values[key]}
            min={knob.min}
            max={knob.max}
            step={knob.step}
            scale={knob.scale}
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
