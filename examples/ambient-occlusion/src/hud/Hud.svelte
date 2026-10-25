<script lang="ts">
    import type { World } from "@dylanebert/shallot";
    import "./controls.css";
    import { setOcclusion } from "../scene";
    import { settings, setSetting, setQuality } from "../settings";
    let { world }: { world: World } = $props();
    let enabled = $state(true);
    let values = $state(settings.map(control => Number(control.value)));
    let mode = $state(2);
    function toggle() {
        enabled = !enabled;
        setOcclusion(world, enabled);
        if (enabled) {
            values.forEach((value, index) => setSetting(world, index, value));
            setQuality(world, mode);
        }
    }
</script>

<details onmousedown={(event) => { if (!(event.target instanceof HTMLInputElement && event.target.type === "range") && !(event.target instanceof HTMLSelectElement)) event.preventDefault(); }} open class="ec controls" role="region" aria-label="Ambient occlusion controls">
    <summary tabindex="-1" onfocus={(event) => event.currentTarget.blur()} class="panel-title">Ambient occlusion</summary>
    <div class="fields">
        <section class="section" aria-label="Ambient occlusion">
            <label class="section-title" for="occlusion"><span role="heading" aria-level="2">Ambient occlusion</span><input tabindex="-1" onfocus={(event) => event.currentTarget.blur()} id="occlusion" type="checkbox" checked={enabled} onchange={toggle} /></label>
            <fieldset class="section dependents" disabled={!enabled} aria-label="Ambient occlusion settings">
                <div class="row">
                    <label for="mode">Quality</label>
                    <select tabindex="-1" onfocus={(event) => event.currentTarget.blur()} id="mode" value={mode} onchange={(event) => { mode = Number(event.currentTarget.value); setQuality(world, mode); }}>
                        <option value={0}>Low</option>
                        <option value={1}>Medium</option>
                        <option value={2}>High</option>
                        <option value={3}>Ultra</option>
                    </select>
                </div>
                {#each settings as control, index}
                    <div class="row">
                        <label for={`control-${index}`}>{control.label}</label>
                        <div class="slider">
                            <input tabindex="-1" onfocus={(event) => event.currentTarget.blur()} id={`control-${index}`} type="range" min={control.min} max={control.max} step={control.step} value={values[index]} oninput={(event) => { values[index] = Number(event.currentTarget.value); setSetting(world, index, values[index]); }} />
                            <output for={`control-${index}`}>{values[index].toFixed(4)}</output>
                        </div>
                    </div>
                {/each}
            </fieldset>
        </section>
    </div>
</details>
