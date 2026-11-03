import { expect, test } from "bun:test";
import { parseGltf } from "./gltf";

for (const slot of [
    "baseColor",
    "metallicRoughness",
    "normalMap",
    "occlusion",
    "emissive",
] as const)
    test(`${slot} KHR_texture_transform retains its own transform and overrides texCoord`, () => {
        const transform = {
            offset: [0.2, 0.3],
            rotation: 0.7,
            scale: [-2, 3],
            texCoord: 1,
        } as const;
        // biome-ignore lint/style/useNamingConvention: glTF owns its extension names.
        const info = { index: 4, texCoord: 0, extensions: { KHR_texture_transform: transform } };
        const source =
            slot === "baseColor" || slot === "metallicRoughness"
                ? { pbrMetallicRoughness: { [`${slot}Texture`]: info } }
                : { [slot === "normalMap" ? "normalTexture" : `${slot}Texture`]: info };
        const scene = parseGltf(
            { extensionsRequired: ["KHR_texture_transform"], materials: [source] },
            [],
        );
        expect(scene.unsupported).toEqual([]);
        expect(scene.materials[0][`${slot}Transform`]).toEqual(transform);
        expect(scene.materials[0][`${slot}Channel`]).toBe(1);
        for (const other of [
            "baseColor",
            "metallicRoughness",
            "normalMap",
            "occlusion",
            "emissive",
        ] as const)
            if (other !== slot) expect(scene.materials[0][`${other}Transform`]).toBeUndefined();
    });

const emissiveStrengthExtension = "KHR_materials_emissive_strength";
for (const strength of [undefined, 0, 0.5, 8])
    test(`emissive factor is multiplied by strength ${strength ?? "default"}`, () => {
        const extensions =
            strength === undefined
                ? {}
                : {
                      [emissiveStrengthExtension]: { emissiveStrength: strength },
                  };
        const scene = parseGltf(
            { materials: [{ emissiveFactor: [0.25, 0.5, 1], extensions }] },
            [],
        );
        expect([...scene.materials[0].emissive]).toEqual(
            [0.25, 0.5, 1].map((value) => value * (strength ?? 1)),
        );
        expect(scene.unsupported).toEqual([]);
    });

test("emissive strength defaults to one and does not supply an absent factor", () => {
    const scene = parseGltf(
        {
            extensionsRequired: ["KHR_materials_emissive_strength"],
            materials: [
                {
                    emissiveFactor: [0.25, 0.5, 1],
                    extensions: { [emissiveStrengthExtension]: {} },
                },
                { extensions: { [emissiveStrengthExtension]: { emissiveStrength: 8 } } },
            ],
        },
        [],
    );
    expect(scene.materials.map((material) => material.emissive)).toEqual([
        [0.25, 0.5, 1],
        [0, 0, 0],
    ]);
    expect(scene.unsupported).toEqual([]);
});

test("material textures retain their independent UV set, factors and scales without unsupported reports", () => {
    const scene = parseGltf(
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
    const material = parseGltf({ materials: [{}] }, []).materials[0];
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
