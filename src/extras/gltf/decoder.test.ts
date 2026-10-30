import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
    DoubleSide,
    LoadingManager,
    Mesh,
    MeshBasicMaterial,
    type MeshStandardMaterial,
} from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { loadGltf, parse } from "./index";

// Image pixels are outside the deviceless claim; GLTFLoader still resolves texture objects and UVs.
globalThis.self = globalThis as typeof self;
globalThis.ProgressEvent = class extends Event {} as unknown as typeof ProgressEvent;
globalThis.createImageBitmap = (async () => ({
    width: 1,
    height: 1,
    close() {},
})) as typeof createImageBitmap;
const localFetch: typeof fetch = (async (input: URL | string) => {
    const url = new URL(String(input));
    return url.protocol === "file:" ? new Response(readFileSync(fileURLToPath(url))) : fetch(url);
}) as typeof fetch;
const fixtures = [
    "default-sampler.gltf",
    "double-sided.gltf",
    "BoxInterleaved/glTF/BoxInterleaved.gltf",
    "BoxTextured/glTF/BoxTextured.gltf",
    "BoxTextured/glTF-Binary/BoxTextured.glb",
    "BoxTextured/glTF-Embedded/BoxTextured.gltf",
    "AlphaBlendModeTest/glTF/AlphaBlendModeTest.gltf",
    "SimpleMeshes/glTF/SimpleMeshes.gltf",
    "CesiumMilkTruck/glTF/CesiumMilkTruck.gltf",
    "UnlitTest/glTF/UnlitTest.gltf",
];
function near(actual: ArrayLike<number>, expected: ArrayLike<number>) {
    expect(actual.length).toBe(expected.length);
    for (let i = 0; i < actual.length; i++) expect(actual[i]).toBeCloseTo(expected[i], 5);
}
for (const fixture of fixtures) {
    test(`${fixture}: geometry, world placement and mapped material match GLTFLoader`, async () => {
        const path = resolve(import.meta.dir, "fixtures", fixture);
        const manager = new LoadingManager();
        manager.setURLModifier((url) => {
            if (url.startsWith("data:") || url.startsWith("blob:")) return url;
            return `data:application/octet-stream;base64,${readFileSync(resolve(dirname(path), url)).toString("base64")}`;
        });
        const bytes = readFileSync(path);
        const reference = await new GLTFLoader(manager).parseAsync(
            bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
            "",
        );
        reference.scene.updateMatrixWorld(true);
        const expected: Mesh[] = [];
        reference.scene.traverse((object) => {
            if (object instanceof Mesh) expected.push(object);
        });
        const result = await loadGltf(new URL(`file://${path}`), localFetch);
        const wrap = (value: number) =>
            (({ 1000: 1, 1001: 0, 1002: 2 }) as const)[value as 1000 | 1001 | 1002];
        const filter = (value: number) => ([1003, 1004, 1005].includes(value) ? 0 : 1);
        for (const [index, texture] of result.textures.entries()) {
            const ref = await reference.parser.getDependency("texture", index);
            expect(texture.image).toBe(reference.parser.json.textures[index].source);
            expect(texture.sampler).toEqual({
                addressModeU: wrap(ref.wrapS),
                addressModeV: wrap(ref.wrapT),
                magFilter: filter(ref.magFilter),
                minFilter: filter(ref.minFilter),
                mipmapFilter: [1003, 1004, 1006, 1007].includes(ref.minFilter) ? 0 : 1,
            });
        }
        expect(result.instances.length).toBe(expected.length);
        for (const [i, instance] of result.instances.entries()) {
            const mesh = result.meshes[instance.mesh];
            const other = expected[i];
            near(instance.matrix, other.matrixWorld.elements);
            // glTF mandates flat normals for missing NORMAL; three.js computes smooth normals.
            // Deindex the reference before computing that specification-required fallback.
            let geometry = other.geometry;
            const normalMissing = !geometry.getAttribute("normal");
            if (normalMissing) {
                geometry = geometry.index ? geometry.toNonIndexed() : geometry.clone();
                geometry.computeVertexNormals();
            }
            const count = geometry.getAttribute("position").count;
            expect(mesh.vertices.length).toBe(count * 8);
            for (let v = 0; v < count; v++) {
                const p = geometry.getAttribute("position");
                near(mesh.vertices.slice(v * 8, v * 8 + 3), [p.getX(v), p.getY(v), p.getZ(v)]);
                const n = geometry.getAttribute("normal");
                near(mesh.vertices.slice(v * 8 + 4, v * 8 + 7), [n.getX(v), n.getY(v), n.getZ(v)]);
                const uv = geometry.getAttribute("uv");
                near(
                    [mesh.vertices[v * 8 + 3], mesh.vertices[v * 8 + 7]],
                    uv ? [uv.getX(v), uv.getY(v)] : [0, 0],
                );
            }
            near(
                mesh.indices,
                geometry.index?.array ?? Uint32Array.from({ length: count }, (_, i) => i),
            );
            if (mesh.material < 0) continue;
            const material = result.materials[mesh.material];
            const ref = other.material as MeshStandardMaterial;
            near(material.color, [...ref.color.toArray(), ref.opacity]);
            const unlit = other.material instanceof MeshBasicMaterial;
            if (!unlit) {
                near([material.metallic, material.roughness], [ref.metalness, ref.roughness]);
                near(material.emissive, ref.emissive.toArray());
            }
            expect(material.alphaMode).toBe(
                ref.transparent ? "BLEND" : ref.alphaTest > 0 ? "MASK" : "OPAQUE",
            );
            if (material.alphaMode === "MASK") expect(material.cutoff).toBe(ref.alphaTest);
            expect(material.texture !== undefined).toBe(ref.map !== null);
            if (ref.map) {
                const texture = reference.parser.associations.get(ref.map)!.textures!;
                expect(result.textures[material.texture!].image).toBe(
                    reference.parser.json.textures[texture].source,
                );
            }
            expect(material.unlit).toBe(unlit);
            expect(material.doubleSided).toBe(ref.side === DoubleSide);
            expect(result.unsupported).not.toContain("doubleSided");
        }
    });
}
test("required cut extension refuses by name; optional cuts and ignored maps are listed", () => {
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
    expect(scene.unsupported).toEqual(["KHR_materials_clearcoat", "normalTexture"]);
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
