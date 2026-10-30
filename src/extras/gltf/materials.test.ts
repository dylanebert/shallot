import { expect, test } from "bun:test";
import { parse } from "./gltf";

test("material textures retain their independent UV set, factors and scales without unsupported reports", () => {
    const scene = parse(
        {
            materials: [
                {
                    pbrMetallicRoughness: {
                        baseColorTexture: { index: 2, texCoord: 1 },
                        metallicRoughnessTexture: { index: 3, texCoord: 1 },
                    },
                    normalTexture: { index: 4, texCoord: 1, scale: 0.4 },
                    occlusionTexture: { index: 5, texCoord: 1, strength: 0.7 },
                    emissiveTexture: { index: 6, texCoord: 1 },
                    emissiveFactor: [0.1, 0.2, 0.3],
                },
            ],
        },
        [],
    );
    expect(scene.unsupported).toEqual([]);
    expect(scene.materials[0]).toMatchObject({
        texture: 2,
        baseColorChannel: 1,
        metallicRoughnessTexture: 3,
        metallicRoughnessChannel: 1,
        normalMapTexture: 4,
        normalMapChannel: 1,
        normalMapScale: 0.4,
        occlusionTexture: 5,
        occlusionChannel: 1,
        occlusionStrength: 0.7,
        emissiveTexture: 6,
        emissiveChannel: 1,
        emissive: [0.1, 0.2, 0.3],
    });
});

test("texture channels, normal scale and occlusion strength take the specification defaults", () => {
    const material = parse({ materials: [{}] }, []).materials[0];
    expect(material).toMatchObject({
        baseColorChannel: 0,
        normalMapChannel: 0,
        normalMapScale: 1,
        metallicRoughnessChannel: 0,
        occlusionChannel: 0,
        occlusionStrength: 1,
        emissiveChannel: 0,
    });
    for (const field of [
        "texture",
        "normalMapTexture",
        "metallicRoughnessTexture",
        "occlusionTexture",
        "emissiveTexture",
    ] as const)
        expect(material[field]).toBeUndefined();
});
