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
        <div id="scene-list" class="scenes" aria-label="Scenes">
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
    .picker, .controls { pointer-events: auto; position: absolute; color: #eee9df; font: 400 14px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .picker { bottom: 16px; left: 50%; transform: translateX(-50%); }
    nav { display: flex; align-items: center; gap: 12px; }
    button { min-height: 28px; padding: 0 8px; color: inherit; background: transparent; border: 0; border-radius: 2px; font: 400 14px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; cursor: pointer; white-space: nowrap; }
    .name { min-width: 12ch; }
    button:hover, button[aria-current="true"] { text-decoration: underline; text-underline-offset: 2px; }
    button:focus-visible, input:focus-visible { outline: 1px solid #eee9df; outline-offset: 2px; }
    .scenes { position: absolute; bottom: calc(100% + 8px); left: 50%; transform: translateX(-50%); width: 240px; max-width: calc(100vw - 32px); max-height: calc(100dvh - 96px); overflow-y: auto; padding: 12px; box-sizing: border-box; background: #282622; border-radius: 2px; }
    .scenes section + section { margin-top: 12px; }
    h2 { font: 400 14px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin: 0 0 8px; color: #797369; }
    .scenes button { display: block; width: 100%; text-align: left; }
    .controls { top: 16px; right: 16px; display: grid; gap: 8px; padding: 10px 12px; background: #282622; border-radius: 2px; }
    input { appearance: none; width: 140px; height: 28px; margin: 0; background: transparent; cursor: pointer; }
    input::-webkit-slider-runnable-track { height: 2px; background: #797369; }
    input::-moz-range-track { height: 2px; background: #797369; }
    input::-webkit-slider-thumb { appearance: none; width: 12px; height: 16px; margin-top: -7px; border: 0; border-radius: 2px; background: #eee9df; }
    input::-moz-range-thumb { width: 12px; height: 16px; border: 0; border-radius: 2px; background: #eee9df; }
</style>
