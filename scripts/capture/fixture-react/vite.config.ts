import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const dir = fileURLToPath(new URL(".", import.meta.url));

// A standard vite+react project — proving a user can build their game UI in React and still open the
// project in the editor. The editor host merges this config: @vitejs/plugin-react has no name collision
// with the editor's svelte plugin, so it runs and transforms the project's .tsx (JSX), and resolve.alias
// + define overlay. react-refresh's HMR preamble auto-injects via the plugin's transformIndexHtml hook.
// (define is carried into the config and applies at `shallot build`; vite's dev server doesn't text-replace
// define in client modules.)
export default defineConfig({
    plugins: [react()],
    resolve: { alias: { $game: resolve(dir, "lib") } },
    define: { __GAME_BUILD__: JSON.stringify("react-fixture") },
});
