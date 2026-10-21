<script lang="ts">
    import type { World } from "@dylanebert/shallot";
    import "./controls.css";
    import { settings, setSetting } from "../settings";
    let { world }: { world: World } = $props();
    let values = $state(settings.map(control => control.value));
</script>

<details onmousedown={(event) => { if (!(event.target instanceof HTMLInputElement && event.target.type === "range")) event.preventDefault(); }} open class="ec controls" role="region" aria-label="Color grading controls">
    <summary tabindex="-1" onfocus={(event) => event.currentTarget.blur()} class="panel-title">Color grading</summary>
    <div class="fields">
        {#each ["Global", "Highlights", "Midtones", "Shadows"] as group}
            <section class="section" aria-label={group}>
                <h2 class="section-title">{group}</h2>
                {#each settings as control, index}
                    {#if control.group === group}
                        <div class="row">
                            <label for={`control-${index}`}>{control.label}</label>
                            <div class="slider">
                                <input tabindex="-1" onfocus={(event) => event.currentTarget.blur()} id={`control-${index}`} aria-label={`${group} ${control.label}`} type="range" min={control.min} max={control.max} step="0.01" value={values[index]} oninput={(event) => { values[index] = Number(event.currentTarget.value); setSetting(world, index, values[index]); }} />
                                <output for={`control-${index}`}>{values[index].toFixed(2)}</output>
                            </div>
                        </div>
                    {/if}
                {/each}
            </section>
        {/each}
    </div>
</details>
