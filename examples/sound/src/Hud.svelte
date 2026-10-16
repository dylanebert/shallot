<script lang="ts">
    import { onMount } from 'svelte';
    import type { World, System } from '@dylanebert/shallot';
    import { Hint, setVolume } from './sound';
    import { Yard } from './yard';
    let { world }: { world: World } = $props();
    let action = $state('');
    let volume = $state(0.7);
    onMount(() => {
        const read: System = {name:'SoundHud/Read', group:'draw', update(world) { action=world.resource(Hint).action; }};
        world.addSystem(read);
        return () => world.removeSystem(read);
    });
</script>
<div class="instructions">Click to look · WASD to walk · E to use</div>
{#if action}<div class="hint"><kbd>E</kbd> {action}</div>{/if}
<div class="volume"><input aria-label="Music volume" type="range" min="0" max="1" step="0.01" bind:value={volume} oninput={() => setVolume(world, world.resource(Yard).speaker, volume)} /></div>
<details><summary>Credits</summary><p>Project Utopia — congusbongus · CC0<br/><a href="https://opengameart.org/content/project-utopia-seamless-loop">OpenGameArt</a></p><p>bong_001.ogg — Kenney · CC0<br/><a href="https://kenney.nl/assets/interface-sounds">Interface Sounds</a></p></details>
<style>
    .instructions,.hint,.volume,details {position:absolute;color:#eee9df;font:14px system-ui;}
    .instructions {top:24px;left:24px;opacity:0.7;}
    .hint {top:calc(50% + 32px);left:50%;transform:translateX(-50%);}
    .hint::before {content:'';position:fixed;left:50%;top:-34px;width:4px;height:4px;border-radius:50%;background:#eee9df;}
    kbd {padding:7px 10px;border:1px solid #eee9df;border-radius:4px;margin-right:8px;}
    .volume {bottom:32px;left:50%;transform:translateX(-50%);pointer-events:auto;}
    input {width:240px;accent-color:#d97843;cursor:pointer;}
    details {bottom:24px;right:24px;pointer-events:auto;background:#282622ee;padding:8px;border-radius:4px;}
    a {color:#eee9df;}
</style>
