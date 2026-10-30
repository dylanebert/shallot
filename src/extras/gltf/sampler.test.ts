import { expect, test } from "bun:test";
import { type GltfJson, parse } from "./gltf";

function model(minFilter?: number): GltfJson {
    return {
        materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
        textures: [{ source: 0, sampler: 0 }],
        images: [{ uri: "color.png" }],
        samplers: [{ wrapS: 33648, wrapT: 33071, magFilter: 9728, minFilter }],
    };
}

for (const min of [9728, 9729, 9984, 9985, 9986, 9987, undefined]) {
    test(`minFilter ${min}: effective texel and mip filters without unsupported reports`, () => {
        const scene = parse(model(min), []);
        expect(scene.textures[0].sampler).toEqual({
            addressModeU: 2,
            addressModeV: 0,
            magFilter: 0,
            minFilter: min !== undefined && [9728, 9984, 9986].includes(min) ? 0 : 1,
            mipmapFilter: min === undefined || min === 9986 || min === 9987 ? 1 : 0,
        });
        expect(scene.unsupported).toEqual([]);
    });
}

test("unused textures and the second UV set do not report mip selection", () => {
    const json = model();
    json.materials = [];
    expect(parse(json, []).unsupported).toEqual([]);
    const extraUv = model();
    extraUv.materials![0].pbrMetallicRoughness!.baseColorTexture!.texCoord = 1;
    expect(parse(extraUv, []).unsupported).toEqual([]);
    expect(parse(extraUv, []).materials[0].baseColorChannel).toBe(1);
});

test("two textures sharing one image retain distinct effective samplers", () => {
    const json = model(9985);
    json.textures!.push({ source: 0 });
    const scene = parse(json, []);
    expect(scene.textures.map((texture) => texture.image)).toEqual([0, 0]);
    expect(scene.textures[1].sampler).toEqual({
        addressModeU: 1,
        addressModeV: 1,
        magFilter: 1,
        minFilter: 1,
        mipmapFilter: 1,
    });
});
