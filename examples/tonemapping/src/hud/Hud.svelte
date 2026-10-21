<script lang="ts">
    import { TonemappingMethod, type World } from "@dylanebert/shallot";
    import { methods, setMethod, setScene } from "../settings";
    import "./controls.css";
    let { world }: { world: World } = $props();
    let method = $state<number>(TonemappingMethod.TonyMcMapface);
    let scene = $state("basic");
</script>

<details class="ec controls" open aria-label="Tonemapping controls">
    <summary tabindex="-1" onmousedown={(event) => event.preventDefault()} onfocus={(event) => event.currentTarget.blur()} class="panel-title">Tonemapping</summary>
    <div class="fields">
        <div class="row">
            <label for="method">Method</label>
            <select tabindex="-1" id="method" bind:value={method} onchange={(event) => { setMethod(world, method); event.currentTarget.blur(); }}>
                {#each methods as [name, value]}<option {value}>{name}</option>{/each}
            </select>
        </div>
        <div class="row">
            <label for="scene">Scene</label>
            <select tabindex="-1" id="scene" bind:value={scene} onchange={(event) => { setScene(world, scene === "pattern"); event.currentTarget.blur(); }}>
                <option value="basic">Basic scene</option>
                <option value="pattern">Test pattern</option>
            </select>
        </div>
    </div>
</details>
