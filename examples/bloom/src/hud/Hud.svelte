<script lang="ts">
    import type { World } from "@dylanebert/shallot";
    import "./controls.css";
    import { setBloom } from "../scene";
    import { settings, setSetting, setMode } from "../settings";
    let { world }: { world: World } = $props();
    let enabled = $state(true);
    let values = $state(settings.map(control => Number(control.value)));
    let mode = $state(0);
    function toggle() {
        enabled = !enabled;
        setBloom(world, enabled);
        if (enabled) {
            values.forEach((value, index) => setSetting(world, index, value));
            setMode(world, mode);
        }
    }
</script>

<details onmousedown={(event) => { if (!(event.target instanceof HTMLInputElement && event.target.type === "range") && !(event.target instanceof HTMLSelectElement)) event.preventDefault(); }} open class="ec controls" role="region" aria-label="Bloom controls">
    <summary tabindex="-1" onfocus={(event) => event.currentTarget.blur()} class="panel-title">Bloom</summary>
    <div class="fields">
        <section class="section" aria-label="Bloom">
            <label class="section-title" for="bloom"><span role="heading" aria-level="2">Bloom</span><input tabindex="-1" onfocus={(event) => event.currentTarget.blur()} id="bloom" type="checkbox" checked={enabled} onchange={toggle} /></label>
            <fieldset class="section dependents" disabled={!enabled} aria-label="Bloom settings">
                <div class="row">
                    <label for="mode">Mode</label>
                    <select tabindex="-1" onfocus={(event) => event.currentTarget.blur()} id="mode" value={mode} onchange={(event) => { mode = Number(event.currentTarget.value); setMode(world, mode); }}>
                        <option value={0}>Energy-conserving</option>
                        <option value={1}>Additive</option>
                    </select>
                </div>
                {#each settings as control, index}
                    <div class="row">
                        <label for={`control-${index}`}>{control.label}</label>
                        <div class="slider">
                            <input tabindex="-1" onfocus={(event) => event.currentTarget.blur()} id={`control-${index}`} type="range" min={control.min} max={control.max} step="0.01" value={values[index]} oninput={(event) => { values[index] = Number(event.currentTarget.value); setSetting(world, index, values[index]); }} />
                            <output for={`control-${index}`}>{values[index].toFixed(2)}</output>
                        </div>
                    </div>
                {/each}
            </fieldset>
        </section>
    </div>
</details>
