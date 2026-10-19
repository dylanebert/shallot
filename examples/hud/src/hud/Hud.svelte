<script lang="ts">
    import { Materials, type System, type World } from "@dylanebert/shallot";
    import { onMount, untrack } from "svelte";
    import { CubeMaterial, Turn } from "../turntable";

    let { world }: { world: World } = $props();
    let degrees = $state(0);
    let speed = $state(untrack(() => world.resource(Turn).speed));
    let selected = $state(0);
    const swatches = [
        { name: "Amber", color: [0.72, 0.4, 0.05, 1] },
        { name: "Cyan", color: [0.04, 0.43, 0.53, 1] },
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
    <input id="speed" aria-label="Turn speed" type="range" style:--fill={`${100 * speed / 3}%`} min="0" max="3" step="0.1" bind:value={speed} />
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
        gap: 8px;
        padding: 4px;
        border-radius: 2px;
        background: #282622;
    }
    .swatches { display: flex; gap: 8px; }
    button { width: 24px; height: 24px; border: 2px solid transparent; border-radius: 2px; cursor: pointer; }
    button[aria-pressed="true"] { border-color: #797369; }
    .angle { min-width: 4ch; color: #eee9df; font: 400 14px/1 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; text-align: right; }
    small { font-size: 14px; vertical-align: top; }
    input { appearance: none; width: 112px; height: 24px; margin: 0; border-radius: 2px; background: linear-gradient(to right, #454440 var(--fill), #3a3835 var(--fill)); cursor: ew-resize; }
    input:hover { background: linear-gradient(to right, #52504c var(--fill), #454440 var(--fill)); }
    input::-webkit-slider-thumb { appearance: none; width: 2px; height: 24px; background: #eee9df; }
    input::-moz-range-thumb { width: 2px; height: 24px; border: 0; background: #eee9df; }
    button:hover { box-shadow: inset 0 0 0 2px #454440; }
    button:focus-visible, input:focus-visible { outline: 1px solid #eee9df; outline-offset: 0; }
</style>
