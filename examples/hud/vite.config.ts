import { shallot } from "@dylanebert/shallot/vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default {
    plugins: [shallot(), svelte()],
};
