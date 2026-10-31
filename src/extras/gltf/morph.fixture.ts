import { readFileSync } from "node:fs";
import type { GltfJson } from "./gltf";

export function morphFixture() {
    const json = JSON.parse(
        readFileSync(new URL("./fixtures/morph-skin.gltf", import.meta.url), "utf8"),
    ) as GltfJson;
    const buffers = json.buffers!.map(
        (buffer) => Uint8Array.from(Buffer.from(buffer.uri!.split(",")[1], "base64")).buffer,
    );
    return { json, buffers };
}

export function morphFixtureUrl(json: GltfJson): string {
    return `data:model/gltf+json;base64,${Buffer.from(JSON.stringify(json)).toString("base64")}`;
}
