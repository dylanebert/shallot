<script lang="ts">
    import "./controls.css";
    import { categories, type Scene } from "./scenes";
    let { scene, select }: { scene: Scene; select: (scene: Scene) => void } = $props();
    let open = $state(false);
    let picker: HTMLElement;
    const samples = [...new Map(categories.flatMap(category => category.scenes).map(sample => [sample.id, sample])).values()];
    function step(direction: number) {
        open = false;
        select(samples[(samples.findIndex(sample => sample.id === scene) + direction + samples.length) % samples.length].id);
    }
    function fitColumns(node: HTMLElement) {
        let disposed = false;
        function fit() {
            if (disposed) return;
            node.style.width = "100%";
            const style = getComputedStyle(node);
            const columns = node.querySelector<HTMLElement>(".categories")!;
            const gap = parseFloat(getComputedStyle(columns).columnGap);
            const padding = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
            const chrome = node.offsetWidth - node.clientWidth + padding;
            const available = node.clientWidth - padding;
            const width = Math.min(available, Math.ceil(Math.max(...Array.from(
                columns.querySelectorAll("button, h2"), (item) => {
                    const range = document.createRange();
                    range.selectNodeContents(item);
                    const style = getComputedStyle(item);
                    return range.getBoundingClientRect().width + parseFloat(style.paddingLeft)
                        + parseFloat(style.paddingRight) + parseFloat(style.borderLeftWidth || "0")
                        + parseFloat(style.borderRightWidth || "0");
                },
            ))));
            const count = matchMedia("(max-width: 480px)").matches ? 1 : Math.max(1, Math.floor((available + gap) / (width + gap)));
            columns.style.columnCount = String(count);
            node.style.width = `${count * width + (count - 1) * gap + chrome}px`;
        }
        const observer = new ResizeObserver(fit);
        observer.observe(picker);
        void document.fonts.ready.then(fit);
        return { destroy() { disposed = true; observer.disconnect(); } };
    }
    function close() { open = false; }
    function toggle() { open = !open; }
 </script>
<svelte:window onclick={(event) => { if (open && !picker.contains(event.target as Node)) close(); }} onkeydown={(event) => { if (open && event.key === "Escape") { event.preventDefault(); close(); } }} />
<section onmousedown={(event) => event.preventDefault()} bind:this={picker} class="ec picker" aria-label="glTF models">
    {#if open}
        <div use:fitColumns id="scene-list" class="scenes" aria-label="Scenes">
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
