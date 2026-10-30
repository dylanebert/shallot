import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { recordArray } from "../../../examples/gltf/tests/records";
import records from "./fixtures/records.json";
import { loadGltf, parse } from "./index";

const localFetch: typeof fetch = (async (input: URL | string) => {
    const url = new URL(String(input));
    return url.protocol === "file:" ? new Response(readFileSync(fileURLToPath(url))) : fetch(url);
}) as typeof fetch;
function near(actual: ArrayLike<number>, expected: ArrayLike<number>) {
    expect(actual.length).toBe(expected.length);
    for (let i = 0; i < actual.length; i++) expect(actual[i]).toBeCloseTo(expected[i], 5);
}
function array(
    actual: number[] | Uint32Array,
    expected: ReturnType<typeof recordArray>,
    width: number,
    integer = false,
) {
    if (Array.isArray(expected)) near(actual, expected);
    else expect(recordArray(actual, width, integer)).toEqual(expected);
}
for (const { fixture, textures, instances } of records.records) {
    test(`${fixture}: geometry, world placement and mapped material match specification records`, async () => {
        const path = resolve(import.meta.dir, "fixtures", fixture);
        const result = await loadGltf(new URL(`file://${path}`), localFetch);
        expect<unknown>(result.textures).toEqual(textures);
        expect(result.instances.length).toBe(instances.length);
        for (const [i, instance] of result.instances.entries()) {
            const mesh = result.meshes[instance.mesh];
            const expected = instances[i];
            near(instance.matrix, expected.matrix);
            const positions: number[] = [],
                normals: number[] = [],
                uvs: number[] = [];
            for (let v = 0; v < mesh.vertices.length; v += 8) {
                positions.push(...mesh.vertices.slice(v, v + 3));
                normals.push(...mesh.vertices.slice(v + 4, v + 7));
                uvs.push(mesh.vertices[v + 3], mesh.vertices[v + 7]);
            }
            array(positions, expected.positions, 3);
            array(normals, expected.normals, 3);
            array(uvs, expected.uvs, 2);
            array(mesh.indices, expected.indices, 1, true);
            if (mesh.material < 0) continue;
            const material = result.materials[mesh.material];
            const ref = expected.material as {
                color: number[];
                metallic?: number;
                roughness?: number;
                emissive?: number[];
                alphaMode: string;
                cutoff?: number;
                textureImage: number | null;
                unlit: boolean;
                doubleSided: boolean;
            };
            near(material.color, ref.color);
            if (!ref.unlit) {
                near([material.metallic, material.roughness], [ref.metallic!, ref.roughness!]);
                near(material.emissive, ref.emissive!);
            }
            expect<string>(material.alphaMode).toBe(ref.alphaMode);
            if (material.alphaMode === "MASK") expect(material.cutoff).toBe(ref.cutoff!);
            expect(material.texture !== undefined).toBe(ref.textureImage !== null);
            if (ref.textureImage !== null) {
                expect(result.textures[material.texture!].image).toBe(ref.textureImage);
            }
            expect(material.unlit).toBe(ref.unlit);
            expect(material.doubleSided).toBe(ref.doubleSided);
            expect(result.unsupported).not.toContain("doubleSided");
        }
    });
}
test("required cut extension refuses by name; optional cuts are listed and normal maps are supported", () => {
    const required = JSON.parse(
        readFileSync(resolve(import.meta.dir, "fixtures/required-draco.gltf"), "utf8"),
    );
    expect(() => parse(required, [])).toThrow("KHR_draco_mesh_compression");
    const scene = parse(
        {
            extensionsUsed: ["KHR_materials_clearcoat"],
            // biome-ignore lint/style/useNamingConvention: glTF owns its attribute and extension names.
            materials: [{ normalTexture: { index: 0 }, extensions: { KHR_materials_unlit: {} } }],
        },
        [],
    );
    expect(scene.unsupported).toEqual(["KHR_materials_clearcoat"]);
    expect(scene.materials[0].unlit).toBe(true);
});

test("loadGltf refuses required meshopt by name before resolving its URI-less fallback buffer", async () => {
    const url = new URL("./fixtures/required-meshopt.gltf", import.meta.url);
    const requests: string[] = [];
    const read: typeof fetch = (async (input: URL | string) => {
        requests.push(String(input));
        return String(input) === url.href
            ? new Response(readFileSync(url))
            : new Response(new Uint8Array(16));
    }) as typeof fetch;
    await expect(loadGltf(url, read)).rejects.toThrow("EXT_meshopt_compression");
    expect(requests).toEqual([url.href]);
});

test("missing normals split shared corners into flat faces and preserve UVs", () => {
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const uv = new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]);
    const indices = new Uint16Array([0, 1, 2, 0, 3, 1]);
    const scene = parse(
        {
            bufferViews: [
                { buffer: 0, byteLength: positions.byteLength },
                { buffer: 1, byteLength: uv.byteLength },
                { buffer: 2, byteLength: indices.byteLength },
            ],
            accessors: [
                { bufferView: 0, count: 4, type: "VEC3", componentType: 5126 },
                { bufferView: 1, count: 4, type: "VEC2", componentType: 5126 },
                { bufferView: 2, count: 6, type: "SCALAR", componentType: 5123 },
            ],
            meshes: [{ primitives: [{ attributes: { POSITION: 0, TEXCOORD_0: 1 }, indices: 2 }] }],
            nodes: [{ mesh: 0 }],
        },
        [positions.buffer, uv.buffer, indices.buffer],
    );
    const mesh = scene.meshes[0];
    near(mesh.indices, [0, 1, 2, 3, 4, 5]);
    for (let v = 0; v < 6; v++) {
        near(mesh.vertices.slice(v * 8 + 4, v * 8 + 7), v < 3 ? [0, 0, 1] : [0, 1, 0]);
        near(
            [mesh.vertices[v * 8 + 3], mesh.vertices[v * 8 + 7]],
            uv.slice(indices[v] * 2, indices[v] * 2 + 2),
        );
    }
});
