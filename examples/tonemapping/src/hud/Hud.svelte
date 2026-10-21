<script lang="ts">
    import { TonemappingMethod, type World } from "@dylanebert/shallot";
    import { methods, setMethod, setScene } from "../settings";
    import "./controls.css";
    let { world }: { world: World } = $props();
    let method = $state<number>(TonemappingMethod.TonyMcMapface);
    let scene = $state("basic");
</script>

<details class="ec controls" open aria-label="Tonemapping controls">
    <summary class="panel-title">Tonemapping</summary>
    <div class="fields">
        <div class="row">
            <label for="method">Method</label>
            <select id="method" bind:value={method} onchange={() => setMethod(world, method)}>
                {#each methods as [name, value]}<option {value}>{name}</option>{/each}
            </select>
        </div>
        <div class="row">
            <label for="scene">Scene</label>
            <select id="scene" bind:value={scene} onchange={() => setScene(world, scene === "pattern")}>
                <option value="basic">Basic scene</option>
                <option value="pattern">Test pattern</option>
            </select>
        </div>
    </div>
</details>
