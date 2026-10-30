import { isGlb, parseGlb } from "./glb";
import { type GltfJson, type GltfScene, generateSceneTangents, parse, validateAsset } from "./gltf";

/** Reads a glTF or GLB URL and resolves external and data-URI buffers with the supplied fetch.
 * Generates missing normal-texture tangents using that texture's coordinate set.
 * Image sources are returned without decoding; loading performs no GPU work.
 */
export async function loadGltf(url: string | URL, read: typeof fetch = fetch): Promise<GltfScene> {
    const base = new URL(url);
    const response = await read(base);
    if (!response.ok) throw new Error(`glTF load failed: ${base} (${response.status})`);
    const bytes = await response.arrayBuffer();
    const { json, bin } = isGlb(bytes)
        ? parseGlb(bytes)
        : { json: JSON.parse(new TextDecoder().decode(bytes)) as GltfJson, bin: undefined };
    validateAsset(json);
    const buffers = await Promise.all(
        (json.buffers ?? []).map(async (buffer) => {
            if (!buffer.uri) {
                if (!bin) throw new Error("glTF buffer has no URI or BIN chunk");
                return bin;
            }
            const response = await read(new URL(buffer.uri, base));
            if (!response.ok)
                throw new Error(`glTF buffer load failed: ${buffer.uri} (${response.status})`);
            return response.arrayBuffer();
        }),
    );
    const scene = parse(json, buffers);
    await generateSceneTangents(scene, json, buffers);
    scene.images = scene.images.map((image) =>
        image.uri ? { ...image, uri: new URL(image.uri, base).href } : image,
    );
    return scene;
}
