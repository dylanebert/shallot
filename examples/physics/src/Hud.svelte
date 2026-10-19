<script lang="ts">
    import { tick } from "svelte";
    import { categories, controlValues, scenes, visibleControls, type Scene } from "./scenes";
    let { scene, select, change }: { scene: Scene; select: (scene: Scene) => void; change: (name: string, value: number, values: Record<string, number>) => void } = $props();
    let values = $state(controlValues(scene));
    const controls = $derived(visibleControls(scene, values));
    const labelWidth = $derived(`max(0px, ${Object.values(scenes[scene].controls ?? {}).map(control => `calc(${control.label.length}ch + ${(control.when?.length ?? 0) * 8}px)`).join(", ") || "0px"})`);
    function update(name: string, value: number) { change(name, value, values); }
    let open = $state(false);
    let picker: HTMLElement;
    let trigger: HTMLButtonElement;
    const samples = categories.flatMap(category => category.scenes);
    const nameWidth = Math.max(...samples.map(sample => sample.name.length));
    function step(direction: number) {
        open = false;
        select(samples[(samples.findIndex(sample => sample.id === scene) + direction + samples.length) % samples.length].id);
    }
    function close() { open = false; trigger.focus(); }
    async function toggle() {
        open = !open;
        if (open) {
            await tick();
            picker.querySelector<HTMLButtonElement>('.scenes button[aria-current="true"]')?.focus();
        }
    }
</script>

<svelte:window onclick={(event) => { if (open && !picker.contains(event.target as Node)) open = false; }} onkeydown={(event) => { if (open && event.key === "Escape") { event.preventDefault(); close(); } }} />

<section bind:this={picker} class="picker" aria-label="Physics scenes">
    {#if open}
        <div id="scene-list" class="scenes" style:--categories={categories.length} aria-label="Scenes">
            <div class="categories">
                {#each categories as category}
                    <section aria-label={category.name}>
                        <h2>{category.name}</h2>
                        {#each category.scenes as sample}
                            <button aria-current={scene === sample.id ? "true" : undefined} onclick={() => { close(); if (scene !== sample.id) select(sample.id); }}>{sample.name}</button>
                        {/each}
                    </section>
                {/each}
            </div>
        </div>
    {/if}
    <nav aria-label="Scene navigation">
        <button aria-label="Previous scene" onclick={() => step(-1)}>←</button>
        <button bind:this={trigger} class="name" style:width="calc({nameWidth}ch + 16px)" aria-label="Scene" aria-controls="scene-list" aria-expanded={open} onclick={toggle}>{samples.find(sample => sample.id === scene)!.name}</button>
        <button aria-label="Next scene" onclick={() => step(1)}>→</button>
    </nav>
</section>
{#if scene === "contact"}<span class="input-help">WASD to roll</span>{/if}
{#if scenes[scene].controls}
    {#snippet fields(parent?: string)}
        {#each Object.entries(scenes[scene].controls ?? {}).filter(([, control]) => control.when?.at(-1) === parent) as [name, control]}
            {#if name in controls}
                <div class="group">
                {#if control.button}
                    <button class="action" onclick={() => update(name, 1)}>{control.label}</button>
                {:else if control.min === undefined}
                    <label class="toggle"><span>{control.label}</span><input type="checkbox" checked={!!values[name]} onchange={(event) => update(name, Number(event.currentTarget.checked))} /></label>
                {:else}
                    <label for={`control-${name}`}>{control.label}</label>
                    <input id={`control-${name}`} type="range" style:--fill={`${100 * (values[name] - control.min) / (control.max! - control.min)}%`} min={control.min} max={control.max} step={control.step} value={values[name]} oninput={(event) => update(name, Number(event.currentTarget.value))} />
                {/if}
                {#if Object.entries(controls).some(([, child]) => child.when?.at(-1) === name)}
                    <div class="dependents">{@render fields(name)}</div>
                {/if}
                </div>
            {/if}
        {/each}
    {/snippet}
    <section class="controls" style:--label={labelWidth} aria-label={`${samples.find(sample => sample.id === scene)!.name} controls`}>
        <div class="fields">{@render fields()}</div>
    </section>
{/if}

<style>
    .picker, .controls, .input-help { pointer-events: auto; position: absolute; color: #eee9df; font: 400 14px/20px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .picker, .controls, .input-help { --top: calc(24px + env(safe-area-inset-top)); --right: calc(24px + env(safe-area-inset-right)); --bottom: calc(24px + env(safe-area-inset-bottom)); --left: calc(24px + env(safe-area-inset-left)); }
    .picker { z-index: 1; bottom: var(--bottom); left: var(--left); right: var(--right); }
    nav { display: flex; justify-content: center; align-items: center; gap: 4px; }
    button { min-height: 24px; padding: 0 8px; color: inherit; background: transparent; border: 0; border-radius: 2px; font: 400 14px/20px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; cursor: pointer; }
    nav button { min-height: 24px; }
    .name { flex: none; white-space: nowrap; }
    nav button:hover { text-decoration: underline; text-underline-offset: 2px; }
    button:focus-visible, input:focus-visible { outline: 1px solid #eee9df; outline-offset: 2px; }
    .scenes { position: absolute; bottom: calc(100% + 8px); left: 50%; transform: translateX(-50%); width: max-content; max-width: 100%; max-height: calc(100dvh - var(--top) - var(--bottom) - 32px); overflow-y: auto; padding: 4px; box-sizing: border-box; background: #282622; border-radius: 2px; }
    .categories { display: grid; grid-template-columns: repeat(var(--categories), max-content); gap: 8px; }
    h2 { font: 400 14px/24px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin: 0 4px 4px; color: #797369; }
    .scenes button { display: block; width: 100%; text-align: left; }
    .controls { top: var(--top); right: var(--right); padding: 4px; background: #282622; border-radius: 2px; max-height: calc(100dvh - var(--top) - var(--bottom) - 48px); overflow-y: auto; box-sizing: border-box; }
    .fields { display: grid; width: calc(var(--label) + 116px); gap: 8px; }
    .group { display: grid; grid-template-columns: minmax(0, 1fr) 112px; align-items: center; gap: 4px; }
    .group > label:not(.toggle) { line-height: 24px; }
    .dependents { grid-column: 1 / -1; display: grid; gap: 4px; padding-left: 8px; }
    .action { grid-column: 1 / -1; }
    .controls button { height: 24px; background: #191816; }
    .scenes button { height: 24px; }
    .controls button:hover, .scenes button:hover, input:hover { background-color: #454440; }
    .scenes button[aria-current="true"], input[type="checkbox"]:checked { background-color: #797369; }
    .controls button:focus-visible, .scenes button:focus-visible, input:focus-visible { outline: 1px solid #eee9df; outline-offset: 0; }
    .scenes button + button { margin-top: 4px; }
    .input-help { top: var(--top); left: var(--left); pointer-events: none; }
    .toggle { grid-column: 1 / -1; display: grid; grid-template-columns: minmax(0, 1fr) 112px; align-items: center; gap: 4px; height: 24px; cursor: pointer; }
    input[type="checkbox"] { appearance: none; display: grid; place-content: center; width: 20px; height: 20px; margin: 0; border: 0; border-radius: 2px; background: #191816; cursor: pointer; }
    input[type="checkbox"]:hover:not(:checked) { background: #454440; }
    input[type="checkbox"]:checked::after { content: ""; width: 8px; height: 4px; border: solid #eee9df; border-width: 0 0 2px 2px; transform: translateY(-1px) rotate(-45deg); }
    input[type="range"] { appearance: none; width: 112px; height: 24px; margin: 0; border-radius: 2px; background: linear-gradient(to right, #454440 var(--fill), #3a3835 var(--fill)); cursor: ew-resize; }
    input[type="range"]:hover { background: linear-gradient(to right, #52504c var(--fill), #454440 var(--fill)); }
    input::-webkit-slider-thumb { appearance: none; width: 2px; height: 24px; border: 0; background: #eee9df; }
    input::-moz-range-thumb { width: 2px; height: 24px; border: 0; background: #eee9df; }
    @media (max-width: 600px) { .categories { grid-template-columns: 1fr; } }
    @media (pointer: coarse) { nav button { min-height: 44px; } }
</style>
