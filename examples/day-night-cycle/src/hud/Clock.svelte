<script lang="ts">
    import { type System, type World } from "@dylanebert/shallot";
    import { onMount, untrack } from "svelte";
    import { Clock } from "../cycle";
    import "./controls.css";

    function time(hour: number) {
        const minutes = Math.floor(hour * 60);
        return `${String(Math.floor(minutes / 60)).padStart(2, "0")}:${String(minutes % 60).padStart(2, "0")}`;
    }
    let { world }: { world: World } = $props();
    let hour = $state(untrack(() => world.resource(Clock).hour));
    onMount(() => {
        const read: System = { name: "Clock/Read", group: "draw", update(world) { hour = world.resource(Clock).hour; } };
        world.addSystem(read);
        return () => world.removeSystem(read);
    });
</script>

<section onmousedown={(event) => { if (!(event.target instanceof HTMLInputElement && event.target.type === "range")) event.preventDefault(); }} class="ec controls hud" aria-label="Time of day">
    <div class="fields">
        <div class="row">
            <label for="hour">Hour</label>
            <div class="slider">
                <input tabindex="-1" onfocus={(event) => event.currentTarget.blur()} id="hour" type="range" min="0" max="23.99" step="0.01" value={hour}
                    oninput={(event) => { world.resource(Clock).hour = Number(event.currentTarget.value); }} />
                <output for="hour">{time(hour)}</output>
            </div>
        </div>
    </div>
</section>
