<script lang="ts">
    import "./controls.css";
    import { onMount } from "svelte";
    let outputs = $state<Record<string, number | string | null>>({});
    let liveControls = $state<Record<string, boolean>>({});
    let liveBounds = $state<Record<string, [number, number]>>({});
    onMount(() => {
        const rows = Object.entries(scenes[scene].controls ?? {}).filter(([, control]) => control.read);
        const live = Object.entries(scenes[scene].controls ?? {}).filter(([, control]) => control.enabled);
        const bounded = Object.entries(scenes[scene].controls ?? {}).filter(([, control]) => control.bounds);
        if (!rows.length && !live.length && !bounded.length) return;
        let frame: number;
        function read() {
            outputs = Object.fromEntries(rows.map(([name, control]) => [name, control.read!(world)]));
            liveControls = Object.fromEntries(live.map(([name, control]) => [name, control.enabled!(world)]));
            liveBounds = Object.fromEntries(bounded.map(([name, control]) => [name, control.bounds!(values, world)]));
            frame = requestAnimationFrame(read);
        }
        read();
        return () => cancelAnimationFrame(frame);
    });
    import type { World } from "@dylanebert/shallot";
    import { formatReadout, formatValue } from "./control-format";
    import { manifoldScenes } from "./manifold";
    import { categories, controlValues, scenes, enabledControls, type Scene } from "./scenes";
    let { world, scene, select, change }: { world: World; scene: Scene; select: (scene: Scene) => void; change: (name: string, value: number, values: Record<string, number>) => void } = $props();
    let values = $state(controlValues(scene));
    const controls = $derived(Object.fromEntries(Object.entries(enabledControls(scene, values)).filter(([name]) => liveControls[name] !== false)));
    const isManifold = $derived(scene in manifoldScenes);
    const sectionTitle = $derived(isManifold ? "Manifold" : scene === "boxHull" || scene === "hullReduction" || scene === "hullTransform" || scene === "capsuleMass" ? "Geometry" : scene === "meshViewer" ? "Mesh settings" : scene === "meshCreation" ? "Build cost" : scene === "gyroscopicPrecession" ? "Precession, rad/s" : scene === "moverOverlap" ? "Planes and normals" : "Simulation");
    const separateReadouts = $derived(isManifold || scene === "hullReduction" || scene === "hullTransform" || scene === "capsuleMass" || scene === "sensorHits" || scene === "meshViewer" || scene === "castWorld" || scene === "longRayCast" || scene === "shapeDistance" || scene === "distanceDebug" || scene === "timeOfImpact");
    const readoutTitle = $derived(isManifold ? "Contact points" : scene === "capsuleMass" ? "Mass and inertia" : scene === "hullReduction" ? "Reduced hull" : scene === "hullTransform" ? "Hull geometry" : scene === "shapeDistance" || scene === "distanceDebug" ? "Distance and vertex identity" : scene === "timeOfImpact" ? "Time of impact" : scene === "meshViewer" ? "Mesh statistics" : scene === "castWorld" ? "Cast results" : scene === "longRayCast" ? "Ray accuracy" : "Touch counts");
    function update(name: string, value: number) { change(name, value, values); }
    let open = $state(false);
    let picker: HTMLElement;
    const samples = categories.flatMap(category => category.scenes);
    function step(direction: number) {
        open = false;
        select(samples[(samples.findIndex(sample => sample.id === scene) + direction + samples.length) % samples.length].id);
    }
    function close() { open = false; }
    function toggle() { open = !open; }
</script>

<svelte:window onclick={(event) => { if (open && !picker.contains(event.target as Node)) close(); }} onkeydown={(event) => { if (open && event.key === "Escape") { event.preventDefault(); close(); } }} />
<section onmousedown={(event) => event.preventDefault()} bind:this={picker} class="ec picker" aria-label="Physics scenes">
    {#if open}
        <div id="scene-list" class="scenes" aria-label="Scenes">
            <div class="categories">
                {#each categories as category}
                    <section aria-label={category.name}>
                        <h2>{category.name}</h2>
                        {#each category.scenes as sample}
                            <button tabindex="-1" onfocus={(event) => event.currentTarget.blur()} aria-current={scene === sample.id ? "true" : undefined} onclick={() => { close(); if (scene !== sample.id) select(sample.id); }}>{sample.name}</button>
                        {/each}
                    </section>
                {/each}
            </div>
        </div>
    {/if}
    <nav aria-label="Scene navigation">
        <button tabindex="-1" onfocus={(event) => event.currentTarget.blur()} aria-label="Previous scene" onclick={() => step(-1)}>&lt;</button>
        <button tabindex="-1" onfocus={(event) => event.currentTarget.blur()} class="name" aria-label="Scene" aria-controls="scene-list" aria-expanded={open} onclick={toggle}>{samples.find(sample => sample.id === scene)!.name}</button>
        <button tabindex="-1" onfocus={(event) => event.currentTarget.blur()} aria-label="Next scene" onclick={() => step(1)}>&gt;</button>
    </nav>
</section>
{#if scene === "contact"}<span class="ec input-help">WASD to roll</span>{/if}
{#if scene === "driving"}<span class="ec input-help">WASD to drive</span>{/if}
{#if scene === "geometricMover" || scene === "rigidBody" || scene === "village"}<span class="ec input-help">WASD to move · Space to jump · Shift to sprint</span>{/if}
{#if scenes[scene].controls}
    {#snippet fields(parent?: string, readOnly = false)}
        {#each Object.entries(scenes[scene].controls ?? {}).filter(([, control]) => control.when?.at(-1) === parent && (!separateReadouts || Boolean(control.read) === readOnly)) as [name, control]}
                <fieldset class="section" disabled={!(name in controls)}>
                    {#if control.read}
                        {#if outputs[name] != null || isManifold || scene === "castWorld" || scene === "longRayCast" || scene === "shapeDistance" || scene === "timeOfImpact"}
                            <div class="row"><span>{control.label}</span><output aria-label={control.label}>{outputs[name] == null ? "—" : formatReadout(outputs[name]!, control.decimals, control.precision)}</output></div>
                        {/if}
                    {:else if control.button}
                        <button tabindex="-1" onfocus={(event) => event.currentTarget.blur()} onclick={() => update(name, 1)}>{control.label}</button>
                    {:else if control.options && control.select}
                        <div class="row">
                            <label for={`control-${name}`}>{control.label}</label>
                            <select tabindex="-1" onfocus={(event) => event.currentTarget.blur()} id={`control-${name}`} value={values[name]} onchange={(event) => { update(name, Number(event.currentTarget.value)); event.currentTarget.blur(); }}>
                                {#each control.options as option}
                                    <option value={option.value}>{option.label}</option>
                                {/each}
                            </select>
                        </div>
                    {:else if control.options}
                        <div class="row" role="radiogroup" aria-label={control.label}>
                            <span>{control.label}</span>
                            <div class="options">
                                {#each control.options as option}
                                    <button tabindex="-1" role="radio" aria-checked={values[name] === option.value} onfocus={(event) => event.currentTarget.blur()} onclick={() => update(name, option.value)}>{option.label}</button>
                                {/each}
                            </div>
                        </div>
                    {:else if control.min === undefined}
                        <label class="row toggle"><span>{control.label}</span><input tabindex="-1" onfocus={(event) => event.currentTarget.blur()} type="checkbox" style:font-family="var(--mono)" checked={!!values[name]} onchange={(event) => update(name, Number(event.currentTarget.checked))} /></label>
                    {:else}
                        <div class="row">
                            <label for={`control-${name}`}>{control.label}</label>
                            <div class="slider">
                                <input tabindex="-1" onfocus={(event) => event.currentTarget.blur()} id={`control-${name}`} type="range" min={control.logarithmic ? Math.log10(control.min) : liveBounds[name]?.[0] ?? control.bounds?.(values, world)[0] ?? control.min} max={control.logarithmic ? Math.log10(control.max!) : liveBounds[name]?.[1] ?? control.bounds?.(values, world)[1] ?? control.max} step={control.logarithmic ? "any" : control.step} value={control.logarithmic ? Math.log10(values[name]) : values[name]} oninput={(event) => { const value = Number(event.currentTarget.value); update(name, control.logarithmic ? Math.round(10 ** value / control.step!) * control.step! : value); }}  />
                                <output for={`control-${name}`}>{formatValue(values[name], control.decimals)}</output>
                            </div>
                        </div>
                    {/if}
                    {#if Object.entries(scenes[scene].controls ?? {}).some(([, child]) => child.when?.at(-1) === name)}
                        <div class="dependents">{@render fields(name)}</div>
                    {/if}
                </fieldset>
        {/each}
    {/snippet}
    <details onmousedown={(event) => { if (!(event.target instanceof HTMLInputElement && event.target.type === "range")) event.preventDefault(); }} open class="ec controls" aria-label={`${samples.find(sample => sample.id === scene)!.name} controls`}>
        <summary tabindex="-1" onfocus={(event) => event.currentTarget.blur()} class="panel-title">Physics</summary>
        <div class="fields">
            <section class="section" aria-label={sectionTitle}>
                <h2 class="section-title">{sectionTitle}</h2>
                {@render fields()}
            </section>
            {#if separateReadouts}
                <section class="section" aria-label={readoutTitle}>
                    <h2 class="section-title">{readoutTitle}</h2>
                    {@render fields(undefined, true)}
                </section>
            {/if}
        </div>
    </details>
{/if}
