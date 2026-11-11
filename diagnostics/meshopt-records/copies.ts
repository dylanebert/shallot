import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import type { GltfJson } from "../../src/extras/gltf/gltf";
import { loadRecords, readInput, referenceOutput } from "./records";

/** Writes standalone decompressed copies using only the pinned reference decoder and corpus. */
export async function decompressedCopies() {
    const records = await loadRecords();
    await readInput({ path: "MeshoptCubeTest/glTF/MeshoptCubeTestFallback.bin", sha256: "8d3d779653780e85a75eda988110ab235ea85cd3d174361ffb318c6b657dee07" });
    const assets = [...new Set(records.map((record) => record.asset))];
    const cache = resolve(import.meta.dir, "../../test-results/meshopt-assets");
    const destination = resolve(import.meta.dir, "../../test-results/meshopt-copies");
    const copies: { asset: string; compressed: URL; decompressed: URL }[] = [];
    for (const asset of assets) {
        const model = asset.split("/")[0];
        const path = join(cache, asset, `${model}.gltf`);
        const json: GltfJson = JSON.parse(readFileSync(path, "utf8"));
        for (const image of json.images ?? [])
            if (image.uri) image.uri = new URL(image.uri, pathToFileURL(path)).href;
        for (const buffer of json.buffers ?? []) {
            if (buffer.uri && !buffer.uri.startsWith("data:"))
                buffer.uri = pathToFileURL(join(dirname(path), buffer.uri)).href;
            delete buffer.extensions;
        }
        for (const record of records.filter((record) => record.asset === asset)) {
            const output = await referenceOutput(record);
            const view = json.bufferViews![record.index];
            view.buffer = json.buffers!.length;
            view.byteOffset = 0;
            delete view.extensions;
            json.buffers!.push({ byteLength: output.length,
                uri: `data:application/octet-stream;base64,${Buffer.from(output).toString("base64")}` });
        }
        for (const key of ["extensionsRequired", "extensionsUsed"] as const)
            json[key] = json[key]?.filter((name) => !["KHR_meshopt_compression", "EXT_meshopt_compression"].includes(name));
        // Unreferenced placeholders remain valid buffers for a generic, non-meshopt loader.
        for (const buffer of json.buffers ?? [])
            if (!buffer.uri) buffer.uri = "data:application/octet-stream;base64,";
        const copy = join(destination, asset, `${model}.gltf`);
        mkdirSync(dirname(copy), { recursive: true });
        writeFileSync(copy, JSON.stringify(json, (_, value) => Object.is(value, -0) ? (JSON as typeof JSON & { rawJSON(text: string): unknown }).rawJSON("-0") : value));
        copies.push({ asset, compressed: pathToFileURL(path), decompressed: pathToFileURL(copy) });
    }
    return copies;
}

if (import.meta.main) await decompressedCopies();
