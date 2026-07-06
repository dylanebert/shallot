import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

const dir = fileURLToPath(new URL(".", import.meta.url));

// A standard vite+svelte project — the shape a user authors, not the editor-first layout. The editor host
// (packages/shallot/bin/edit.ts) loads + merges this config: the svelte plugin dedups by name against the
// editor's own (one runes-configured instance compiles the project's .svelte), and resolve.alias + define
// overlay. This is what makes shallot-as-tool framework-aware. (define is carried into the config and
// applies at `shallot build`; vite's dev server doesn't text-replace define in client modules.)
export default defineConfig({
    plugins: [svelte({ compilerOptions: { runes: true } })],
    resolve: { alias: { $game: resolve(dir, "lib") } },
    define: { __GAME_BUILD__: JSON.stringify("svelte-fixture") },
});
