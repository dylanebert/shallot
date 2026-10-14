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
    <div>Angle <span data-hud="angle">{degrees}°</span></div>
    <div class="swatches" aria-label="Cube colour">
        {#each swatches as swatch, index}
            <button aria-label={swatch.name} aria-pressed={selected === index}
                style:background={srgb(swatch.color)} onclick={() => recolor(index)}></button>
        {/each}
    </div>
    <label for="speed">Speed <span>{speed.toFixed(1)} rad/s</span></label>
    <input id="speed" type="range" min="0" max="3" step="0.1" bind:value={speed} />
</section>

<style>
    .hud {
        pointer-events: auto;
        position: absolute;
        top: 16px;
        left: 16px;
        display: grid;
        gap: 8px;
        width: 180px;
        padding: 10px 12px;
        border-radius: 6px;
        background: #373635;
        color: #efe3cf;
        font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    span { float: right; font-variant-numeric: tabular-nums; }
    .swatches { display: flex; gap: 8px; }
    button { width: 28px; height: 28px; border: 2px solid transparent; border-radius: 4px; cursor: pointer; }
    button[aria-pressed="true"] { border-color: #efe3cf; }
    input { width: 100%; margin: 0; accent-color: #efe3cf; }
</style>
