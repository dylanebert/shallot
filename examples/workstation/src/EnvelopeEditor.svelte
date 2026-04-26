<script lang="ts">
    import Knob from "./Knob.svelte";
    import EnvelopeViz from "./EnvelopeViz.svelte";
    import { ampEnvKnobs } from "./presets";

    let {
        slotId,
        values = $bindable(),
        onchange,
    }: {
        slotId: string;
        values: Record<string, number>;
        onchange: (key: string, value: number) => void;
    } = $props();

    let attack = $derived(values[`${slotId}.attack`] ?? 0.01);
    let decay = $derived(values[`${slotId}.decay`] ?? 0.1);
    let sustain = $derived(values[`${slotId}.sustain`] ?? 0.7);
    let release = $derived(values[`${slotId}.release`] ?? 0.3);
    let attackCurve = $derived(values[`${slotId}.attackCurve`] ?? 0);
    let decayCurve = $derived(values[`${slotId}.decayCurve`] ?? 0);
    let releaseCurve = $derived(values[`${slotId}.releaseCurve`] ?? 0);
</script>

<div class="editor">
    <EnvelopeViz {attack} {decay} {sustain} {release} {attackCurve} {decayCurve} {releaseCurve} />
    {#each ampEnvKnobs as knob}
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
