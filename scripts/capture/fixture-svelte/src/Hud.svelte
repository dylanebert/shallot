<!-- The HUD proves a project framework component compiles + mounts in the editor: the editor's svelte
     compiles this .svelte (the project's own svelte plugin deduped out by name), and it mounts over the
     canvas. Static label, not a `define` — vite's dev server doesn't text-replace `define` in client
     modules (it applies at build time), so the flow reads this literal. -->
<div class="game-hud" data-framework="svelte">svelte</div>

<style>
    .game-hud {
        position: fixed;
        top: 8px;
        left: 8px;
        z-index: 9999;
        font: 12px monospace;
        color: #fff;
        pointer-events: none;
    }
</style>
