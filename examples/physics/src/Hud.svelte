<script lang="ts">
    import { tick } from "svelte";
    import type { Scene } from "./scenes";
    let { scene, select, gravity }: { scene: Scene; select: (scene: Scene) => void; gravity: (value: number) => void } = $props();
    let open = $state(false);
    let picker: HTMLElement;
    let trigger: HTMLButtonElement;
    const categories = [
        { name: "Joints", scenes: [{ id: "bridge" as Scene, name: "Bridge" }] },
        { name: "Stacking", scenes: [{ id: "boxStack" as Scene, name: "Box Stack" }] },
    ];
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
            {#each categories as category}
                <section aria-label={category.name}>
                    <h2>{category.name}</h2>
                    {#each category.scenes as sample}
                        <button aria-current={scene === sample.id ? "true" : undefined} onclick={() => { close(); if (scene !== sample.id) select(sample.id); }}>{sample.name}</button>
                    {/each}
                </section>
            {/each}
        </div>
    {/if}
    <nav aria-label="Scene navigation">
        <button aria-label="Previous scene" onclick={() => step(-1)}>←</button>
        <button bind:this={trigger} class="name" aria-label="Scene" aria-controls="scene-list" aria-expanded={open} onclick={toggle}>{scene === "bridge" ? "Bridge" : "Box Stack"}</button>
        <button aria-label="Next scene" onclick={() => step(1)}>→</button>
    </nav>
</section>
{#if scene === "bridge"}
    <section class="controls" aria-label="Bridge controls">
        <label for="gravity">Gravity scale</label>
        <input id="gravity" type="range" min="-1" max="1" step="0.1" value="1" oninput={(event) => gravity(Number(event.currentTarget.value))} />
    </section>
{/if}

<style>
    .picker, .controls { pointer-events: auto; position: absolute; color: #eee9df; font: 400 14px/20px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .picker, .controls { --top: calc(24px + env(safe-area-inset-top)); --right: calc(24px + env(safe-area-inset-right)); --bottom: calc(24px + env(safe-area-inset-bottom)); --left: calc(24px + env(safe-area-inset-left)); }
    .picker { bottom: var(--bottom); left: var(--left); right: var(--right); }
    nav { display: flex; justify-content: center; align-items: center; gap: 4px; }
    button { min-height: 24px; padding: 0 8px; color: inherit; background: transparent; border: 0; border-radius: 2px; font: inherit; cursor: pointer; }
    .name { min-width: 112px; }
    button:hover, button[aria-current="true"] { text-decoration: underline; text-underline-offset: 2px; }
    button:focus-visible, input:focus-visible { outline: 1px solid #eee9df; outline-offset: 2px; }
    .scenes { position: absolute; bottom: calc(100% + 8px); left: 50%; transform: translateX(-50%); width: min(100%, calc(var(--categories) * 192px + 8px), 1000px); max-height: calc(100dvh - var(--top) - var(--bottom) - 32px); overflow-y: auto; display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 176px), 1fr)); align-items: start; gap: 16px; padding: 8px; box-sizing: border-box; background: #282622; border-radius: 2px; }
    .scenes section { min-width: 0; }
    h2 { font: inherit; margin: 0 8px 4px; color: #797369; }
    .scenes button { display: block; width: 100%; text-align: left; }
    .controls { top: var(--top); right: var(--right); display: grid; gap: 4px; padding: 8px; background: #282622; border-radius: 2px; }
    input { appearance: none; width: 144px; height: 24px; margin: 0; background: transparent; cursor: pointer; }
    input::-webkit-slider-runnable-track { height: 2px; background: #797369; }
    input::-moz-range-track { height: 2px; background: #797369; }
    input::-webkit-slider-thumb { appearance: none; width: 12px; height: 16px; margin-top: -7px; border: 0; border-radius: 2px; background: #eee9df; }
    input::-moz-range-thumb { width: 12px; height: 16px; border: 0; border-radius: 2px; background: #eee9df; }
    @media (max-width: 480px) { .scenes { grid-template-columns: 1fr; } }
    @media (pointer: coarse) {
        button, input { min-height: 44px; }
        .scenes { max-height: calc(100dvh - var(--top) - var(--bottom) - 52px); }
    }
</style>
