<script lang="ts">
    import { tick } from "svelte";
    import { categories, controlValues, visibleControls, type Scene } from "./scenes";
    let { scene, select, change }: { scene: Scene; select: (scene: Scene) => void; change: (name: string, value: number, values: Record<string, number>) => void } = $props();
    let values = $state(controlValues(scene));
    const controls = $derived(visibleControls(scene, values));
    function update(name: string, value: number) { change(name, value, values); }
    let open = $state(false);
    let picker: HTMLElement;
    let trigger: HTMLButtonElement;
    const samples = categories.flatMap(category => category.scenes);
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
        <button bind:this={trigger} class="name" aria-label="Scene" aria-controls="scene-list" aria-expanded={open} onclick={toggle}>{samples.find(sample => sample.id === scene)!.name}</button>
        <button aria-label="Next scene" onclick={() => step(1)}>→</button>
    </nav>
</section>
{#if Object.keys(controls).length}
    <section class="controls" class:joint-controls={scene !== "bridge"} aria-label={`${samples.find(sample => sample.id === scene)!.name} controls`}>
        {#each Object.entries(controls) as [name, control]}
                {#if control.min === undefined}
                    <label class="toggle"><input type="checkbox" checked={!!values[name]} onchange={(event) => update(name, Number(event.currentTarget.checked))} />{control.label}</label>
                {:else}
                    <label for={`control-${name}`}>{control.label}</label>
                    <input id={`control-${name}`} type="range" min={control.min} max={control.max} step={control.step} value={values[name]} oninput={(event) => update(name, Number(event.currentTarget.value))} />
                {/if}
        {/each}
    </section>
{/if}

<style>
    .picker, .controls { pointer-events: auto; position: absolute; color: #eee9df; font: 400 14px/20px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .picker, .controls { --top: calc(24px + env(safe-area-inset-top)); --right: calc(24px + env(safe-area-inset-right)); --bottom: calc(24px + env(safe-area-inset-bottom)); --left: calc(24px + env(safe-area-inset-left)); }
    .picker { bottom: var(--bottom); left: var(--left); right: var(--right); }
    nav { display: flex; justify-content: center; align-items: center; gap: 4px; }
    button { min-height: 24px; padding: 0 8px; color: inherit; background: transparent; border: 0; border-radius: 2px; font: 400 14px/20px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; cursor: pointer; }
    .name { min-width: 112px; }
    button:hover, button[aria-current="true"] { text-decoration: underline; text-underline-offset: 2px; }
    button:focus-visible, input:focus-visible { outline: 1px solid #eee9df; outline-offset: 2px; }
    .scenes { position: absolute; bottom: calc(100% + 8px); left: 50%; transform: translateX(-50%); width: min(100%, calc(var(--categories) * 192px + 8px), 1000px); max-height: calc(100dvh - var(--top) - var(--bottom) - 32px); overflow-y: auto; padding: 8px; box-sizing: border-box; background: #282622; border-radius: 2px; }
    .categories { column-width: 176px; column-gap: 16px; }
    .scenes section { break-inside: avoid; margin-bottom: 16px; }
    h2 { font: 400 14px/20px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin: 0 8px 4px; color: #797369; }
    .scenes button { display: block; width: 100%; text-align: left; }
    .controls { top: var(--top); right: var(--right); display: grid; gap: 4px; padding: 8px; background: #282622; border-radius: 2px; max-height: calc(100dvh - var(--top) - var(--bottom) - 48px); overflow-y: auto; box-sizing: border-box; }
    .joint-controls { grid-template-columns: max-content 144px; align-items: center; column-gap: 8px; }
    .toggle { grid-column: 1 / -1; display: flex; align-items: center; gap: 8px; min-height: 24px; cursor: pointer; }
    input[type="checkbox"] { width: 16px; height: 16px; margin: 0; accent-color: #eee9df; cursor: pointer; }
    input[type="range"] { appearance: none; width: 144px; height: 24px; margin: 0; background: transparent; cursor: pointer; }
    input::-webkit-slider-runnable-track { height: 2px; background: #797369; }
    input::-moz-range-track { height: 2px; background: #797369; }
    input::-webkit-slider-thumb { appearance: none; width: 12px; height: 16px; margin-top: -7px; border: 0; border-radius: 2px; background: #eee9df; }
    input::-moz-range-thumb { width: 12px; height: 16px; border: 0; border-radius: 2px; background: #eee9df; }
    @media (max-width: 480px) { .categories { column-count: 1; } }
    @media (pointer: coarse) {
        button, input[type="range"], .toggle { min-height: 44px; }
        .scenes { max-height: calc(100dvh - var(--top) - var(--bottom) - 52px); }
    }
</style>
