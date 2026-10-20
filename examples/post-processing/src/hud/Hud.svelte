<script lang="ts">
    import type { World } from "@dylanebert/shallot";
    import { effects, setEffect } from "../scene";

    let { world }: { world: World } = $props();
    let enabled = $state([false, false, false]);

    function toggle(index: number) {
        enabled[index] = !enabled[index];
        setEffect(world, index, enabled[index]);
    }
</script>

<section aria-label="Post-processing controls">
    {#each effects as effect, index}
        <button aria-pressed={enabled[index]} onclick={() => toggle(index)}>{effect.name}</button>
    {/each}
</section>

<style>
    section {
        pointer-events: auto;
        position: absolute;
        bottom: 16px;
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        flex-wrap: wrap;
        width: max-content;
        max-width: calc(100% - 32px);
        box-sizing: border-box;
        gap: 8px;
        padding: 4px;
        border-radius: 2px;
        background: #282622;
    }
    button {
        height: 24px;
        padding: 0 8px;
        white-space: nowrap;
        color: #eee9df;
        background: #191816;
        border: 1px solid transparent;
        border-radius: 2px;
        font: 400 14px/20px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        cursor: pointer;
    }
    button[aria-pressed="true"] { background: #454440; border-color: #797369; }
    button:hover { background: #454440; }
    button:focus-visible { outline: 1px solid #eee9df; outline-offset: 0; }
</style>
