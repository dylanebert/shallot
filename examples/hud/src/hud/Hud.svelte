<script lang="ts">
    import { Materials, type System, type World } from "@dylanebert/shallot";
    import { onMount, untrack } from "svelte";
    import { CubeMaterial, Turn } from "../turntable";
    import "./controls.css";

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

<section onmousedown={(event) => { if (!(event.target instanceof HTMLInputElement && event.target.type === "range")) event.preventDefault(); }} class="ec controls hud" aria-label="Cube controls">
  <div class="fields">
    <div class="swatches" aria-label="Cube colour">
        {#each swatches as swatch, index}
            <button tabindex="-1" onfocus={(event) => event.currentTarget.blur()} aria-label={swatch.name} aria-pressed={selected === index}
                onclick={() => recolor(index)}><span class="swatch" style:background={srgb(swatch.color)}></span></button>
        {/each}
    </div>
    <div class="row">
        <label for="speed">Turn speed</label>
        <div class="slider">
            <input tabindex="-1" onfocus={(event) => event.currentTarget.blur()} id="speed" type="range" min="0" max="3" step="0.1" bind:value={speed} />
            <output for="speed">{speed.toFixed(1)}</output>
        </div>
    </div>
    <div class="row"><span id="angle-label">Angle</span><output aria-labelledby="angle-label" data-hud="angle">{degrees}°</output></div>
  </div>
</section>
