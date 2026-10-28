<script lang="ts">
    import type { World } from "@dylanebert/shallot";
    import "./controls.css";
    import { settings, setSetting, setDither } from "../settings";
    let { world }: { world: World } = $props();
    let values = $state(settings.map(control => Number(control.value)));
    let dither = $state(true);
</script>

<details onmousedown={(event) => { if (!(event.target instanceof HTMLInputElement && event.target.type === "range")) event.preventDefault(); }} open class="ec controls" role="region" aria-label="Low-res controls">
    <summary tabindex="-1" onfocus={(event) => event.currentTarget.blur()} class="panel-title">Low-res</summary>
    <div class="fields">
        {#each settings as control, index}
            <div class="row">
                <label for={`control-${index}`}>{control.label}</label>
                <div class="slider">
                    <input tabindex="-1" onfocus={(event) => event.currentTarget.blur()} id={`control-${index}`} type="range" min={control.min} max={control.max} step="1" bind:value={values[index]} oninput={(event) => setSetting(world, index, Number(event.currentTarget.value))} />
                    <output for={`control-${index}`}>{values[index]}</output>
                </div>
            </div>
        {/each}
        <label class="section-title" for="dither">Dither<input tabindex="-1" onfocus={(event) => event.currentTarget.blur()} id="dither" type="checkbox" bind:checked={dither} onchange={(event) => setDither(world, event.currentTarget.checked)} /></label>
    </div>
</details>
