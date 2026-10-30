import { fileURLToPath } from "node:url";
import { shallot } from "@dylanebert/shallot/vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import manifest from "./assets.json";
import { syncFiles } from "./sync";

export default {
    plugins: [
        shallot(),
        svelte(),
        {
            name: "gltf-assets",
            async buildStart() {
                await syncFiles(
                    manifest.models.flatMap((model) => model.files),
                    new URL(`${manifest.commit}/`, manifest.source),
                    fileURLToPath(new URL("./public/assets/", import.meta.url)),
                );
            },
        },
    ],
};
