<script lang="ts">
    import { tick } from "svelte";
    import "./controls.css";
    import { formatValue } from "./control-format";
    import { categories, controlValues, scenes, enabledControls, type Scene } from "./scenes";
    let { scene, select, change }: { scene: Scene; select: (scene: Scene) => void; change: (name: string, value: number, values: Record<string, number>) => void } = $props();
    let values = $state(controlValues(scene));
    const controls = $derived(enabledControls(scene, values));
    function update(name: string, value: number) { change(name, value, values); }
    let open = $state(false);
    let picker: HTMLElement;
    let trigger: HTMLButtonElement;
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

<svelte:window onclick={(event) => { if (open && !picker.contains(event.target as Node)) close(); }} onkeydown={(event) => { if (open && event.key === "Escape") { event.preventDefault(); close(); } }} />
<section bind:this={picker} class="ec picker" aria-label="Physics scenes">
    {#if open}
        <div id="scene-list" class="scenes" aria-label="Scenes">
            <div class="categories">
                {#each categories as category}
                    <section aria-label={category.name}>
                        <h2>{category.name}</h2>
                        {#each category.scenes as sample}
                            <button aria-current={scene === sample.id ? "true" : undefined} onclick={() => { close(); if (scene !== sample.id) select(sample.id); }}>{sample.name}</button>
                        {/each}
                    </section>
                {/each}
            </div>
        </div>
    {/if}
    <nav aria-label="Scene navigation">
        <button aria-label="Previous scene" onclick={() => step(-1)}>&lt;</button>
        <button bind:this={trigger} class="name" aria-label="Scene" aria-controls="scene-list" aria-expanded={open} onclick={toggle}>{samples.find(sample => sample.id === scene)!.name}</button>
        <button aria-label="Next scene" onclick={() => step(1)}>&gt;</button>
    </nav>
</section>
{#if scene === "contact"}<span class="ec input-help">WASD to roll</span>{/if}
{#if scenes[scene].controls}
    {#snippet fields(parent?: string)}
        {#each Object.entries(scenes[scene].controls ?? {}).filter(([, control]) => control.when?.at(-1) === parent) as [name, control]}
                <fieldset class="section" disabled={!(name in controls)}>
                    {#if control.button}
                        <button onclick={() => update(name, 1)}>{control.label}</button>
                    {:else if control.min === undefined}
                        <label class="row toggle"><span>{control.label}</span><input type="checkbox" style:font-family="var(--mono)" checked={!!values[name]} onchange={(event) => update(name, Number(event.currentTarget.checked))} /></label>
                    {:else}
                        <div class="row">
                            <label for={`control-${name}`}>{control.label}</label>
                            <div class="slider">
                                <input id={`control-${name}`} type="range" min={control.min} max={control.max} step={control.step} value={values[name]} oninput={(event) => update(name, Number(event.currentTarget.value))} />
                                <output for={`control-${name}`}>{formatValue(values[name])}</output>
                            </div>
                        </div>
                    {/if}
                    {#if Object.entries(scenes[scene].controls ?? {}).some(([, child]) => child.when?.at(-1) === name)}
                        <div class="dependents">{@render fields(name)}</div>
                    {/if}
                </fieldset>
        {/each}
    {/snippet}
    <details open class="ec controls" aria-label={`${samples.find(sample => sample.id === scene)!.name} controls`}>
        <summary class="panel-title">Physics</summary>
        <div class="fields">
            <section class="section" aria-label="Simulation">
                <h2 class="section-title">Simulation</h2>
                {@render fields()}
            </section>
        </div>
    </details>
{/if}
