import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { generateTangents } from "../../core/mesh";
import { type GltfJson, generateSceneTangents, parse } from "./gltf";
import { loadGltf } from "./load";

async function load(json: GltfJson) {
    return loadGltf("https://fixture.test/model.gltf", (async (url) =>
        String(url) === "https://fixture.test/model.gltf"
            ? Response.json(json)
            : fetch(url)) as typeof fetch);
}
function model(): GltfJson {
    return JSON.parse(
        readFileSync(new URL("./fixtures/normal-uv1.gltf", import.meta.url), "utf8"),
    ) as GltfJson;
}
function buffers(json: GltfJson) {
    return [
        Uint8Array.from(atob(json.buffers![0].uri!.split(",")[1]), (c) => c.charCodeAt(0)).buffer,
    ];
}

for (const set of [0, 1])
    for (const scale of [
        [1, 1],
        [-1, 2],
    ])
        for (const morph of [false, true])
            test(`normal texture UV${set} rotation and scale ${scale} generate ${morph ? "morph and base" : "base"} tangents from transformed coordinates`, async () => {
                const json = model();
                const texture = json.materials![0].normalTexture!;
                texture.texCoord = 1 - set;
                texture.extensions = {
                    // biome-ignore lint/style/useNamingConvention: glTF owns its extension names.
                    KHR_texture_transform: {
                        texCoord: set,
                        rotation: Math.PI / 2,
                        scale: scale as [number, number],
                        offset: [0.2, 0.4],
                    },
                };
                const scene = parse(json, buffers(json));
                const source = scene.meshes[0];
                const coordinates =
                    set === 1
                        ? source.uv1!
                        : Float32Array.from(
                              { length: source.vertices.length / 4 },
                              (_, i) =>
                                  source.vertices[Math.floor(i / 2) * 8 + (i % 2 === 0 ? 3 : 7)],
                          );
                // Independent homogeneous T * R * S, for a quarter turn: x' = -sy*y + ox; y' = sx*x + oy.
                const transformed = Float32Array.from(coordinates, (_, i) =>
                    i % 2 === 0
                        ? -scale[1] * coordinates[i + 1] + 0.2
                        : scale[0] * coordinates[i - 1] + 0.4,
                );
                const expected = await generateTangents({ ...source, uv1: transformed }, 1);
                const vertices = source.vertices.slice();
                const delta = new Float32Array([0, 0, 0, 0, 0, 0.4, 0.2, 0, 0]);
                if (morph) source.morphTargets = [{ position: delta }];
                for (let v = 0; v < vertices.length / 8; v++)
                    for (let a = 0; a < 3; a++) vertices[v * 8 + a] += delta[v * 3 + a];
                const target = await generateTangents({ ...source, vertices, uv1: transformed }, 1);
                await generateSceneTangents(scene, json, buffers(json));
                const actual = scene.meshes[0];
                for (let corner = 0; corner < actual.indices.length; corner++) {
                    const a = actual.indices[corner],
                        e = expected.indices[corner],
                        t = target.indices[corner];
                    expect(actual.tangent!.subarray(a * 4, a * 4 + 4)).toEqual(
                        expected.tangent.subarray(e * 4, e * 4 + 4),
                    );
                    expect(actual.uv1!.subarray(a * 2, a * 2 + 2)).toEqual(
                        source.uv1!.subarray(
                            source.indices[corner] * 2,
                            source.indices[corner] * 2 + 2,
                        ),
                    );
                    if (morph)
                        for (let axis = 0; axis < 3; axis++)
                            expect(actual.morphTargets![0].tangent![a * 3 + axis]).toBeCloseTo(
                                target.tangent[t * 4 + axis] - expected.tangent[e * 4 + axis],
                                6,
                            );
                }
            });

test("normalTexture.texCoord selects UV_1 for MikkTSpace, not UV_0", async () => {
    const json = model();
    const source = parse(json, buffers(json)).meshes[0];
    const expected = await generateTangents(source, 1);
    const wrong = await generateTangents(source, 0);
    expect(Array.from(expected.tangent)).not.toEqual(Array.from(wrong.tangent));
    const scene = await load(json);
    expect(scene.meshes[0].tangent).toEqual(expected.tangent);
    expect(scene.meshes[0].uv1).toEqual(source.uv1);
    expect(scene.unsupported).toEqual([]);
});

test("normal textures default to UV_0 and generate tangents with flat normals when NORMAL is absent", async () => {
    const json = model();
    delete json.materials![0].normalTexture!.texCoord;
    delete json.meshes![0].primitives[0].attributes.NORMAL;
    const source = parse(json, buffers(json)).meshes[0];
    const expected = await generateTangents(source);
    expect((await load(json)).meshes[0].tangent).toEqual(expected.tangent);
});

for (const flat of [false, true])
    test(`normalTexture UV_2 generation preserves UV_1 with ${flat ? "flat" : "authored"} normals`, async () => {
        const json = model();
        const attributes = json.meshes![0].primitives[0].attributes as Record<string, number>;
        attributes.TEXCOORD_2 = attributes.TEXCOORD_0;
        json.materials![0].normalTexture!.texCoord = 2;
        if (flat) delete attributes.NORMAL;
        const source = parse(json, buffers(json)).meshes[0];
        const expected = await generateTangents(source, 0);
        const scene = await load(json);
        expect(scene.meshes[0].tangent).toEqual(expected.tangent);
        expect(scene.meshes[0].uv1).toEqual(source.uv1);
        expect(scene.unsupported).toEqual(["TEXCOORD_2", "normalTexture.texCoord"]);
    });

for (const set of [0, 1, 2])
    test(`normalTexture refuses missing TEXCOORD_${set} by name`, async () => {
        const json = model();
        json.materials![0].normalTexture!.texCoord = set;
        delete (json.meshes![0].primitives[0].attributes as Record<string, number>)[
            `TEXCOORD_${set}`
        ];
        await expect(load(json)).rejects.toThrow(`normalTexture names missing TEXCOORD_${set}`);
    });

for (const componentType of [5126, 5121, 5123])
    for (const width of [3, 4]) {
        test(`COLOR_0 VEC${width} and TEXCOORD_1 decode ${componentType} with normalized integer components`, () => {
            const json = model();
            delete json.materials;
            const values = [0, 1, 0.5, 1];
            const max = componentType === 5121 ? 255 : componentType === 5123 ? 65535 : 1;
            const make = (count: number) => {
                const v = Array.from({ length: count }, (_, i) => Math.round(values[i % 4] * max));
                return componentType === 5121
                    ? new Uint8Array(v)
                    : componentType === 5123
                      ? new Uint16Array(v)
                      : new Float32Array(Array.from({ length: count }, (_, i) => values[i % 4]));
            };
            const color = make(3 * width);
            const uv = make(6);
            const data = new Uint8Array(color.byteLength + uv.byteLength);
            data.set(new Uint8Array(color.buffer));
            data.set(new Uint8Array(uv.buffer), color.byteLength);
            json.bufferViews!.push(
                { buffer: 1, byteOffset: 0, byteLength: color.byteLength },
                { buffer: 1, byteOffset: color.byteLength, byteLength: uv.byteLength },
            );
            json.accessors!.push(
                {
                    bufferView: 4,
                    componentType,
                    normalized: componentType !== 5126,
                    count: 3,
                    type: width === 3 ? "VEC3" : "VEC4",
                },
                {
                    bufferView: 5,
                    componentType,
                    normalized: componentType !== 5126,
                    count: 3,
                    type: "VEC2",
                },
            );
            json.meshes![0].primitives[0].attributes.COLOR_0 = 4;
            json.meshes![0].primitives[0].attributes.TEXCOORD_1 = 5;
            const scene = parse(json, [...buffers(json), data.buffer]);
            const mesh = scene.meshes[0];
            expect(mesh.color).toEqual(
                Float32Array.from({ length: 12 }, (_, i) =>
                    i % 4 === 3 && width === 3
                        ? 1
                        : color[Math.floor(i / 4) * width + (i % 4)] / max,
                ),
            );
            expect(mesh.uv1).toEqual(Float32Array.from(uv, (v) => v / max));
            expect(scene.unsupported).toEqual([]);
        });
    }

test("authored tangents retain all four components, but are ignored without NORMAL", async () => {
    const json = model();
    const tangent = new Float32Array([1, 0, 0, -1, 0, 1, 0, 1, 0, 0, 1, -1]);
    json.bufferViews!.push({ buffer: 1, byteLength: tangent.byteLength });
    json.accessors!.push({ bufferView: 4, componentType: 5126, count: 3, type: "VEC4" });
    json.meshes![0].primitives[0].attributes.TANGENT = 4;
    const data = [...buffers(json), tangent.buffer];
    expect(parse(json, data).meshes[0].tangent).toEqual(tangent);
    delete json.meshes![0].primitives[0].attributes.NORMAL;
    delete json.materials;
    const mesh = parse(json, data).meshes[0];
    expect(mesh.tangent).toBeUndefined();
    expect(mesh.uv1).toEqual(new Float32Array([0, 0, 0, 1, 1, 0]));
});

test("higher UV and color sets remain unsupported; absent streams stay absent", () => {
    const json = model();
    delete json.materials;
    const attributes = json.meshes![0].primitives[0].attributes as Record<string, number>;
    attributes.TEXCOORD_2 = 3;
    attributes.COLOR_1 = 1;
    delete attributes.TEXCOORD_1;
    const scene = parse(json, buffers(json));
    expect(scene.unsupported).toEqual(["COLOR_1", "TEXCOORD_2"]);
    expect(scene.meshes[0].uv1).toBeUndefined();
    expect(scene.meshes[0].color).toBeUndefined();
    expect(scene.meshes[0].tangent).toBeUndefined();
});
