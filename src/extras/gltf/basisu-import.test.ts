import { expect, spyOn, test } from "bun:test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import manifest from "../../../examples/gltf/assets.json";
import { syncFiles } from "../../../examples/gltf/sync";
import { DataImages } from "../../core/rendering/image";
import { World } from "../../engine";
import { parseGltf } from "./gltf";
import { basisImageSource } from "./image";
import { loadGltf } from "./load";
import { loadGltfScene } from "./scene";

await syncFiles(
    manifest.models
        .filter((model) => ["StainedGlassLamp", "AnisotropyBarnLamp"].includes(model.name))
        .flatMap((model) => model.files),
    new URL(`${manifest.commit}/`, manifest.source),
    fileURLToPath(new URL("../../../examples/gltf/public/assets/", import.meta.url)),
);

const read: typeof fetch = (async (url: URL | string) =>
    new Response(readFileSync(new URL(String(url))))) as typeof fetch;
for (const name of ["StainedGlassLamp", "AnisotropyBarnLamp"]) {
    test(`${name}: BasisU scene, geometry, materials, samplers and world transforms equal PNG except image sources`, async () => {
        const base = new URL(
            `../../../examples/gltf/public/assets/Models/${name}/`,
            import.meta.url,
        );
        const actual = await loadGltf(new URL(`glTF-KTX-BasisU/${name}.gltf`, base), read);
        const expected = await loadGltf(new URL(`glTF/${name}.gltf`, base), read);
        const names = (scene: typeof actual) =>
            scene.images.map((image) =>
                image
                    .uri!.split("/")
                    .pop()!
                    .replace(/\.(png|ktx2)$/, ""),
            );
        expect(names(actual)).toEqual(names(expected));
        expect({ ...actual, images: [] }).toEqual({ ...expected, images: [] });
    });
}

test("extension source wins and no fallback image is fetched", async () => {
    const requests: string[] = [];
    const world = new World();
    const read = spyOn(globalThis, "fetch").mockImplementation((async (url: URL | string) => {
        requests.push(String(url));
        if (String(url) === "https://example.test/scene.gltf")
            return Response.json({
                materials: [{ normalTexture: { index: 0 } }],
                textures: [{ source: 0, extensions: { ["KHR_texture_basisu"]: { source: 1 } } }],
                images: [{ uri: "fallback.png" }, { uri: "basis" }],
            });
        if (String(url) !== "https://example.test/basis")
            throw new Error(`unexpected fetch ${url}`);
        return new Response(
            readFileSync(
                new URL("../../../crates/basisu/tests/fixtures/uastc-none.ktx2", import.meta.url),
            ),
        );
    }) as typeof fetch);
    try {
        await loadGltfScene(world, "https://example.test/scene.gltf");
        const images = [...world.resource(DataImages)];
        expect(images).toHaveLength(1);
        const source = images[0].source;
        if (typeof source !== "function") throw new Error("KTX2 must register a level source");
        expect((await source(7)).format).toBe("rgba8unorm");
    } finally {
        read.mockRestore();
        world.dispose();
    }
    expect(requests).toEqual(["https://example.test/scene.gltf", "https://example.test/basis"]);
});

for (const source of [-1, 1, 0.5, undefined, null])
    test(`KHR_texture_basisu refuses invalid source ${source} by texture name`, () => {
        expect(() =>
            parseGltf(
                {
                    textures: [{ source: 0, extensions: { ["KHR_texture_basisu"]: { source } } }],
                    images: [{ uri: "fallback.png" }],
                },
                [],
            ),
        ).toThrow("texture 0: KHR_texture_basisu invalid source");
    });

test("KHR_texture_basisu refuses a non-KTX2 MIME type", () => {
    expect(() =>
        parseGltf(
            {
                textures: [{ extensions: { ["KHR_texture_basisu"]: { source: 0 } } }],
                images: [{ uri: "wrong.png", mimeType: "image/png" }],
            },
            [],
        ),
    ).toThrow("KHR_texture_basisu requires image/ktx2");
});

test("contradicting DFD transferFunction is refused by name", async () => {
    const bytes = readFileSync(
        new URL("../../../crates/basisu/tests/fixtures/uastc-none.ktx2", import.meta.url),
    );
    await expect(basisImageSource({ bytes }, true, false)(0)).rejects.toThrow(
        "DFD transferFunction contradicts texture role",
    );
});
