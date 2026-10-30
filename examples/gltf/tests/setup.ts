import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import manifest from "../assets.json";
import { syncFiles } from "../sync";

export const assets = new URL("../public/assets/", import.meta.url);
export async function syncAssets() {
    await syncFiles(
        manifest.models.flatMap((model) => model.files),
        new URL(`${manifest.commit}/`, manifest.source),
        fileURLToPath(assets),
    );
}
const networkFetch = globalThis.fetch;
export const fileFetch: typeof fetch = (async (input, init) => {
    const url = new URL(String(input));
    return url.protocol === "file:" ? new Response(await readFile(url)) : networkFetch(input, init);
}) as typeof fetch;
