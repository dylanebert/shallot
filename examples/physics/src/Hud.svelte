<script lang="ts">
    import "./controls.css";
    import { onMount } from "svelte";
    let outputs = $state<Record<string, number | null>>({});
    let liveControls = $state<Record<string, boolean>>({});
    onMount(() => {
        const rows = Object.entries(scenes[scene].controls ?? {}).filter(([, control]) => control.read);
        const live = Object.entries(scenes[scene].controls ?? {}).filter(([, control]) => control.enabled);
        if (!rows.length && !live.length) return;
        let frame: number;
        function read() {
            outputs = Object.fromEntries(rows.map(([name, control]) => [name, control.read!(world)]));
            liveControls = Object.fromEntries(live.map(([name, control]) => [name, control.enabled!(world)]));
            frame = requestAnimationFrame(read);
        }
        read();
        return () => cancelAnimationFrame(frame);
    });
    import type { World } from "@dylanebert/shallot";
    import { formatValue } from "./control-format";
    import { categories, controlValues, scenes, enabledControls, type Scene } from "./scenes";
    let { world, scene, select, change }: { world: World; scene: Scene; select: (scene: Scene) => void; change: (name: string, value: number, values: Record<string, number>) => void } = $props();
    let values = $state(controlValues(scene));
    const controls = $derived(Object.fromEntries(Object.entries(enabledControls(scene, values)).filter(([name]) => liveControls[name] !== false)));
    const sectionTitle = $derived(scene === "gyroscopicPrecession" ? "Precession, rad/s" : "Simulation");
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
{#if scenes[scene].controls}
    {#snippet fields(parent?: string)}
        {#each Object.entries(scenes[scene].controls ?? {}).filter(([, control]) => control.when?.at(-1) === parent) as [name, control]}
                <fieldset class="section" disabled={!(name in controls)}>
                    {#if control.read}
                        {#if outputs[name] != null}
                            <div class="row"><span>{control.label}</span><output aria-label={control.label}>{formatValue(control.decimals === undefined ? Number(outputs[name]!.toPrecision(3)) : outputs[name]!, control.decimals)}</output></div>
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
                                <input tabindex="-1" onfocus={(event) => event.currentTarget.blur()} id={`control-${name}`} type="range" min={control.min} max={control.max} step={control.step} value={values[name]} oninput={(event) => update(name, Number(event.currentTarget.value))} />
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
        </div>
    </details>
{/if}
