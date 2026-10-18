import { svelte } from "@sveltejs/vite-plugin-svelte";
import { shallot } from "@dylanebert/shallot/vite";

export default {
    plugins: [shallot(), svelte()],
};
