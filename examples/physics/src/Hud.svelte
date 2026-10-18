<script lang="ts">
    import type { Scene } from "./scenes";
    let { scene, select, gravity }: { scene: Scene; select: (scene: Scene) => void; gravity: (value: number) => void } = $props();
    let open = $state(false);
    const categories = [
        { name: "Joints", scenes: [{ id: "bridge" as Scene, name: "Bridge" }] },
        { name: "Stacking", scenes: [{ id: "boxStack" as Scene, name: "Box Stack" }] },
    ];
</script>

<svelte:window onkeydown={(event) => { if (event.key === "Escape") open = false; }} />

<section class="hud" aria-label="Physics controls">
    {#if open}
        <div class="scenes" aria-label="Scenes">
            {#each categories as category}
                <section aria-label={category.name}>
                    <h2>{category.name}</h2>
                    {#each category.scenes as sample}
                        <button aria-current={scene === sample.id ? "true" : undefined} onclick={() => { open = false; if (scene !== sample.id) select(sample.id); }}>{sample.name}</button>
                    {/each}
                </section>
            {/each}
        </div>
    {/if}
    <div class="controls">
        <button aria-label="Scene" aria-expanded={open} onclick={() => open = !open}>{scene === "bridge" ? "Bridge" : "Box Stack"} <span aria-hidden="true">{open ? "▾" : "▴"}</span></button>
        {#if scene === "bridge"}
            <input aria-label="Gravity scale" title="Gravity scale" type="range" min="-1" max="1" step="0.1" value="1" oninput={(event) => gravity(Number(event.currentTarget.value))} />
        {/if}
    </div>
</section>

<style>
    .hud { pointer-events: auto; position: absolute; bottom: 16px; left: 50%; transform: translateX(-50%); color: #eee9df; font: 400 14px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .controls { display: flex; align-items: center; gap: 12px; padding: 10px 12px; background: #282622; border-radius: 2px; }
    button { display: flex; align-items: center; justify-content: space-between; gap: 12px; min-height: 28px; padding: 0 8px; color: inherit; background: #282622; border: 1px solid #797369; border-radius: 2px; font: 400 14px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; cursor: pointer; white-space: nowrap; }
    button:hover, button:focus-visible, button[aria-current="true"] { border-color: #eee9df; outline: none; }
    .scenes { position: absolute; bottom: calc(100% + 8px); left: 50%; transform: translateX(-50%); width: 280px; max-width: calc(100vw - 32px); max-height: min(480px, calc(100dvh - 96px)); overflow-y: auto; padding: 12px; box-sizing: border-box; background: #282622; border: 1px solid #797369; border-radius: 2px; }
    .scenes section + section { margin-top: 12px; }
    h2 { font: 400 14px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin: 0 0 8px; color: #eee9df; }
    .scenes button { width: 100%; margin-top: 8px; }
    input { appearance: none; width: 140px; height: 28px; margin: 0; background: transparent; cursor: pointer; }
    input::-webkit-slider-runnable-track { height: 2px; background: #797369; }
    input::-moz-range-track { height: 2px; background: #797369; }
    input::-webkit-slider-thumb { appearance: none; width: 12px; height: 16px; margin-top: -7px; border: 1px solid #eee9df; border-radius: 2px; background: #eee9df; }
    input::-moz-range-thumb { width: 12px; height: 16px; border: 1px solid #eee9df; border-radius: 2px; background: #eee9df; }
    input:focus-visible { outline: 1px solid #eee9df; outline-offset: 2px; }
</style>
