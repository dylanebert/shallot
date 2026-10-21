<script lang="ts">
    import type { World } from "@dylanebert/shallot";
    import "./controls.css";
    import { effects, setEffect } from "../scene";
    import { settings, setSetting } from "../settings";
    let { world }: { world: World } = $props();
    let enabled = $state([true, true, true]);
    let values = $state(settings.map(group => group.map(control => Number(control.value))));
    function toggle(index: number) {
        enabled[index] = !enabled[index];
        setEffect(world, index, enabled[index]);
        if (enabled[index]) values[index].forEach((value, control) => setSetting(world, index, control, value));
    }
    function update(effect: number, control: number, value: number) {
        values[effect][control] = value;
        setSetting(world, effect, control, value);
    }
</script>

<details onmousedown={(event) => { if (!(event.target instanceof HTMLInputElement && event.target.type === "range")) event.preventDefault(); }} open class="ec controls" role="region" aria-label="Post-processing controls">
    <summary tabindex="-1" onfocus={(event) => event.currentTarget.blur()} class="panel-title">Post-processing</summary>
    <div class="fields">
        {#each effects as effect, index}
            <section class="section" aria-label={effect.name}>
                <label class="section-title" for={`effect-${index}`}><span role="heading" aria-level="2">{effect.name}</span><input tabindex="-1" onfocus={(event) => event.currentTarget.blur()} id={`effect-${index}`} type="checkbox" checked={enabled[index]} onchange={() => toggle(index)} /></label>
                <fieldset class="section dependents" disabled={!enabled[index]} aria-label={effect.name}>
                    {#each settings[index] as control, setting}
                        <div class="row">
                            <label for={`control-${index}-${setting}`}>{control.label}</label>
                            <div class="slider">
                                <input tabindex="-1" onfocus={(event) => event.currentTarget.blur()} id={`control-${index}-${setting}`} type="range" min={control.min} max={control.max} step="0.01" value={values[index][setting]} oninput={(event) => update(index, setting, Number(event.currentTarget.value))} />
                                <output for={`control-${index}-${setting}`}>{values[index][setting].toFixed(2)}</output>
                            </div>
                        </div>
                    {/each}
                </fieldset>
            </section>
        {/each}
    </div>
</details>
