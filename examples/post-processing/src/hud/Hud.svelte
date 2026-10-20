<script lang="ts">
    import type { World } from "@dylanebert/shallot";
    import { effects, setEffect } from "../scene";
    import { settings, setSetting } from "../settings";
    let { world }: { world: World } = $props();
    let enabled = $state([false, false, false]);
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

<section class="controls" style:--label="calc(20ch + 8px)" aria-label="Post-processing controls">
    <div class="fields">
        {#each effects as effect, index}
            <div class="group">
                <label class="toggle"><span>{effect.name}</span><input type="checkbox" checked={enabled[index]} onchange={() => toggle(index)} /></label>
                {#if enabled[index]}
                    <div class="dependents" role="group" aria-label={effect.name}>
                        {#each settings[index] as control, setting}
                            <div class="group">
                                <label for={`control-${index}-${setting}`}>{control.label}</label>
                                <input id={`control-${index}-${setting}`} type="range" min={control.min} max={control.max} step="0.01" value={values[index][setting]} style:--fill={`${100 * (values[index][setting] - control.min) / (control.max - control.min)}%`} oninput={(event) => update(index, setting, Number(event.currentTarget.value))} />
                            </div>
                        {/each}
                    </div>
                {/if}
            </div>
        {/each}
    </div>
</section>

<style>
    .controls { pointer-events: auto; position: absolute; color: #eee9df; font: 400 14px/20px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; --top: calc(24px + env(safe-area-inset-top)); --right: calc(24px + env(safe-area-inset-right)); --bottom: calc(24px + env(safe-area-inset-bottom)); }
    .controls { top: var(--top); right: var(--right); padding: 8px; background: #282622; border-radius: 2px; max-height: calc(100dvh - var(--top) - var(--bottom) - 48px); overflow-y: auto; box-sizing: border-box; }
    .fields { display: grid; width: calc(var(--label) + 120px); gap: 8px; }
    .group { display: grid; grid-template-columns: minmax(0, 1fr) 112px; align-items: center; row-gap: 4px; column-gap: 8px; }
    .group > label:not(.toggle) { line-height: 24px; }
    .dependents { grid-column: 1 / -1; display: grid; gap: 4px; padding-left: 8px; }
    .toggle { grid-column: 1 / -1; display: grid; grid-template-columns: minmax(0, 1fr) 112px; align-items: center; column-gap: 8px; height: 24px; cursor: pointer; }
    input[type="checkbox"] { appearance: none; display: grid; place-content: center; width: 20px; height: 20px; margin: 0; border: 0; border-radius: 2px; background: #191816; cursor: pointer; }
    input[type="checkbox"]:checked { background-color: #797369; }
    input[type="checkbox"]:hover:not(:checked) { background: #454440; }
    input[type="checkbox"]:checked::after { content: ""; width: 8px; height: 4px; border: solid #eee9df; border-width: 0 0 2px 2px; transform: translateY(-1px) rotate(-45deg); }
    input[type="range"] { appearance: none; width: 112px; height: 24px; margin: 0; border-radius: 2px; background: linear-gradient(to right, #454440 var(--fill), #3a3835 var(--fill)); cursor: ew-resize; }
    input[type="range"]:hover { background: linear-gradient(to right, #52504c var(--fill), #454440 var(--fill)); }
    input::-webkit-slider-thumb { appearance: none; width: 2px; height: 24px; border: 0; background: #eee9df; }
    input::-moz-range-thumb { width: 2px; height: 24px; border: 0; background: #eee9df; }
    input:focus-visible { outline: 1px solid #eee9df; outline-offset: 0; }
</style>
