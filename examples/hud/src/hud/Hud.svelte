<script lang="ts">
    import { Materials, type System, type World } from "@dylanebert/shallot";
    import { onMount, untrack } from "svelte";
    import { CubeMaterial, Turn } from "../turntable";

    let { world }: { world: World } = $props();
    let degrees = $state(0);
    let speed = $state(untrack(() => world.resource(Turn).speed));
    let selected = $state(0);
    const swatches = [
        { name: "Orange", color: [0.621407, 0.26791815, 0.11756791, 1] },
        { name: "Blue", color: [0.15270135, 0.36628627, 0.70424525, 1] },
    ];

    function srgb(color: number[]) {
        return `rgb(${color.slice(0, 3).map(value =>
            255 * (value <= 0.0031308 ? 12.92 * value : 1.055 * value ** (1 / 2.4) - 0.055)
        ).join(" ")})`;
    }

    $effect(() => { world.resource(Turn).speed = speed; });

    function recolor(index: number) {
        world.resource(Materials).update(world.resource(CubeMaterial).handle!, {
            baseColor: swatches[index].color as [number, number, number, number],
        });
        selected = index;
    }

    onMount(() => {
        const read: System = {
            name: "Hud/Read",
            group: "draw",
            update(world) {
                degrees = Math.floor((world.resource(Turn).angle * 180) / Math.PI) % 360;
            },
        };
        world.addSystem(read);
        return () => world.removeSystem(read);
    });
</script>

<section class="hud" aria-label="Cube controls">
    <div class="swatches" aria-label="Cube colour">
        {#each swatches as swatch, index}
            <button aria-label={swatch.name} aria-pressed={selected === index}
                style:background={srgb(swatch.color)} onclick={() => recolor(index)}></button>
        {/each}
    </div>
    <input id="speed" aria-label="Turn speed" type="range" min="0" max="3" step="0.1" bind:value={speed} />
    <span class="angle" data-hud="angle">{degrees}<small>°</small></span>
</section>

<style>
    .hud {
        pointer-events: auto;
        position: absolute;
        bottom: 16px;
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 12px;
        border-radius: 6px;
        background: #373635;
    }
    .swatches { display: flex; gap: 8px; }
    button { width: 28px; height: 28px; border: 2px solid transparent; border-radius: 4px; cursor: pointer; }
    button[aria-pressed="true"] { border-color: #efe3cf; }
    .angle { min-width: 4ch; color: #efe3cf; font: 14px/1 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; text-align: right; }
    small { font-size: 0.7em; vertical-align: top; }
    input { width: 140px; margin: 0; accent-color: #efe3cf; }
</style>
