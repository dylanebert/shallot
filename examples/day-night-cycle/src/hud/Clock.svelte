<script lang="ts">
    import { type System, type World } from "@dylanebert/shallot";
    import { onMount, untrack } from "svelte";
    import { Clock } from "../cycle";
    let { world }: { world: World } = $props();
    let hour = $state(untrack(() => world.resource(Clock).hour));
    onMount(() => {
        const read: System = { name: "Clock/Read", group: "draw", update(world) { hour = world.resource(Clock).hour; } };
        world.addSystem(read);
        return () => world.removeSystem(read);
    });
</script>

<section aria-label="Time of day">
    <input aria-label="Hour" type="range" min="0" max="23.99" step="0.01" value={hour}
        oninput={(event) => { world.resource(Clock).hour = Number(event.currentTarget.value); }} />
</section>

<style>
    section { pointer-events: auto; position: absolute; bottom: 16px; left: 50%; transform: translateX(-50%); padding: 12px; border-radius: 6px; background: #373635; }
    input { width: 240px; margin: 0; accent-color: #efe3cf; }
</style>
