<script lang="ts">
    import type { System, World } from "@dylanebert/shallot";
    import { onMount } from "svelte";
    import { Turn } from "../turntable";

    let { world }: { world: World } = $props();
    let degrees = $state(0);
    let turns = $state(0);

    onMount(() => {
        const read: System = {
            name: "Hud/Read",
            group: "draw",
            update(world) {
                const turn = world.resource(Turn);
                degrees = Math.floor((turn.angle * 180) / Math.PI) % 360;
                turns = turn.turns;
            },
        };
        world.addSystem(read);
        return () => world.removeSystem(read);
    });
</script>

<dl class="hud">
    <dt>Angle</dt>
    <dd data-hud="angle">{degrees}°</dd>
    <dt>Turns</dt>
    <dd data-hud="turns">{turns}</dd>
</dl>

<style>
    .hud {
        position: absolute;
        top: 16px;
        left: 16px;
        display: grid;
        grid-template-columns: max-content max-content;
        gap: 4px 16px;
        padding: 10px 12px;
        border-radius: 6px;
        background: #373635;
        color: #efe3cf;
        font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    dd {
        margin: 0;
        font-variant-numeric: tabular-nums;
    }
</style>
