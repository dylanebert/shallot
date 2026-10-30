import { beforeAll, expect, setDefaultTimeout, spyOn, test } from "bun:test";
import {
    composeMat4,
    createApp,
    decomposeMat4,
    type ImageSamplerDescriptor,
    Materials,
    Meshes,
    MeshInstance,
    MeshMaterial,
    MeshPlugin,
    multiplyMat4,
    registerImage,
    Transform,
    TransformPlugin,
} from "@dylanebert/shallot";
import { loadGltf, loadGltfScene, spawnGltfScene } from "@dylanebert/shallot/gltf";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { categories, modelPath } from "../src/scenes";
import gold from "./models.gold.json";
import { decodeQuantized } from "./quantization";
import { assets, fileFetch, syncAssets } from "./setup";

setDefaultTimeout(TIMEOUT.node);
function near(actual: ArrayLike<number>, expected: ArrayLike<number>, tolerance: number) {
    expect(actual.length).toBe(expected.length);
    for (let i = 0; i < actual.length; i++)
        expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(tolerance);
}
function materialValues(actual: ArrayLike<number>, expected: number[]) {
    expect(Array.from(actual)).toEqual(expected.map(Math.fround));
}
function quantized(code: number, offset: number, extent: number, expected: number) {
    const { value, bound } = decodeQuantized(code, offset, extent);
    expect(Math.abs(value - expected)).toBeLessThanOrEqual(bound);
}
function recordedMatrix(
    path: { matrix?: number[]; translation?: number[]; rotation?: number[]; scale?: number[] }[],
) {
    let matrix = composeMat4(0, 0, 0, 0, 0, 0, 1, 1, 1, 1);
    for (const node of path) {
        const [x, y, z] = node.translation ?? [0, 0, 0];
        const [qx, qy, qz, qw] = node.rotation ?? [0, 0, 0, 1];
        const [sx, sy, sz] = node.scale ?? [1, 1, 1];
        const local = node.matrix
            ? new Float32Array(node.matrix)
            : composeMat4(x, y, z, qx, qy, qz, qw, sx, sy, sz);
        matrix = multiplyMat4(matrix, local);
    }
    return matrix;
}
export function parity(category: string) {
    beforeAll(async () => {
        await syncAssets();
        await setupGlobals();
    });
    for (const { id } of categories.find((entry) => entry.name === category)!.scenes) {
        test(`${category}/${id}: registered geometry, materials, samplers and node worlds equal the specification records`, async () => {
            const reference = gold.records.find(
                (record) => record.fixture === `Models/${id}/glTF/${id}.gltf`,
            )!;
            const fetch = spyOn(globalThis, "fetch").mockImplementation(fileFetch);
            let model: Awaited<ReturnType<typeof loadGltfScene>>;
            let entities: number[];
            const url = new URL(modelPath(id).slice("assets/".length), assets);
            const decoded = await loadGltf(url);
            const app = await createApp({
                defaults: false,
                plugins: [
                    MeshPlugin,
                    TransformPlugin,
                    {
                        name: "Parity",
                        async initialize(world) {
                            model = await loadGltfScene(world, url);
                            entities = spawnGltfScene(world, model);
                        },
                    },
                ],
            }).finally(() => fetch.mockRestore());
            try {
                console.log(`glTF parity ${category}/${id}`, app.world.gpu.adapter);
                expect(model!.primitives.length).toBe(reference.instances.length);
                expect<unknown>(decoded.textures).toEqual(reference.textures);
                for (const [i, primitive] of model!.primitives.entries()) {
                    const ref = reference.instances[i];
                    const mesh = app.world.resource(Meshes).get(primitive.mesh)!;
                    expect(mesh.pending).toBeUndefined();
                    const vertices = await mesh.vertices.read();
                    const quant = await mesh.quant!.read();
                    const indices = (await mesh.indices.read()).slice(
                        mesh.indexBase,
                        mesh.indexBase + mesh.indexCount,
                    );
                    const meshId = vertices[indices[0]].y >>> 16;
                    const vertexBase = vertices.findIndex((vertex) => vertex.y >>> 16 === meshId);
                    const localVertices = vertices.filter((vertex) => vertex.y >>> 16 === meshId);
                    expect(localVertices.length * 3).toBe(ref.positions.length);
                    expect(Array.from(indices, (index) => index - vertexBase)).toEqual(ref.indices);
                    for (const [source, v] of localVertices.entries()) {
                        const q = quant[meshId];
                        const position = [v.x & 65535, v.x >>> 16, v.y & 65535];
                        const offset = [q.posOffset.x, q.posOffset.y, q.posOffset.z];
                        const extent = [q.posScale.x, q.posScale.y, q.posScale.z];
                        for (let axis = 0; axis < 3; axis++) {
                            quantized(
                                position[axis],
                                offset[axis],
                                extent[axis],
                                ref.positions[source * 3 + axis],
                            );
                        }
                        quantized(v.w & 65535, q.posOffset.w, q.uvScale.x, ref.uvs[source * 2]);
                        quantized(v.w >>> 16, q.posScale.w, q.uvScale.y, ref.uvs[source * 2 + 1]);
                        let x = ((v.z << 16) >> 16) / 32767;
                        let y = (v.z >> 16) / 32767;
                        const z = 1 - Math.abs(x) - Math.abs(y);
                        if (z < 0) {
                            const old = x;
                            x = (1 - Math.abs(y)) * (x >= 0 ? 1 : -1);
                            y = (1 - Math.abs(old)) * (y >= 0 ? 1 : -1);
                        }
                        const length = Math.hypot(x, y, z);
                        // Snorm16 oct coordinates each err by at most h = 1/(2*32767).
                        // Unfolding gives length error <= sqrt(6)*h; normalization magnifies
                        // it by at most 2*sqrt(3), since an oct surface vector has length >= 1/sqrt(3).
                        near(
                            [x / length, y / length, z / length],
                            ref.normals.slice(source * 3, source * 3 + 3),
                            Math.sqrt(18) / 32767,
                        );
                    }
                    // Evaluate the recorded authored node path at the client's storage boundaries:
                    // local composition and every parent product store float32 matrices; spawn
                    // stores decomposed TRS in float32 columns, then composition stores float32
                    // again. These identical arithmetic paths permit exact equality, including
                    // translation, rather than a tolerance on the double-precision oracle matrix.
                    const expectedWorld = recordedMatrix(ref.nodePath);
                    expect(Array.from(primitive.matrix)).toEqual(Array.from(expectedWorld));
                    const expectedTrs = decomposeMat4(expectedWorld);
                    const expectedMatrix = composeMat4(
                        expectedTrs[0],
                        expectedTrs[1],
                        expectedTrs[2],
                        expectedTrs[3],
                        expectedTrs[4],
                        expectedTrs[5],
                        expectedTrs[6],
                        expectedTrs[7],
                        expectedTrs[8],
                        expectedTrs[9],
                    );
                    const t = app.world.storage(Transform);
                    const eid = entities![i];
                    expect(app.world.storage(MeshInstance).mesh.get(eid)).toBe(primitive.mesh);
                    const bindings = app.world.storage(MeshMaterial);
                    for (const field of [
                        "type",
                        "material",
                        "alphaMode",
                        "alphaCutoff",
                        "cullMode",
                    ] as const) {
                        expect(bindings[field].get(eid)).toBe(primitive.material[field]);
                    }
                    expect(
                        Array.from(
                            composeMat4(
                                t.translation.x.get(eid),
                                t.translation.y.get(eid),
                                t.translation.z.get(eid),
                                t.rotation.x.get(eid),
                                t.rotation.y.get(eid),
                                t.rotation.z.get(eid),
                                t.rotation.w.get(eid),
                                t.scale.x.get(eid),
                                t.scale.y.get(eid),
                                t.scale.z.get(eid),
                            ),
                        ),
                    ).toEqual(Array.from(expectedMatrix));
                    const material = app.world.resource(Materials).get(primitive.material)!;
                    const refMaterial = ref.material as {
                        color: number[];
                        metallic?: number;
                        roughness?: number;
                        emissive?: number[];
                        alphaMode: string;
                        cutoff?: number;
                        textureImage: number | null;
                        textureIndex: number | null;
                        unlit: boolean;
                        doubleSided: boolean;
                    } | null;
                    if (refMaterial) {
                        materialValues(material.baseColor, refMaterial.color);
                        if (!refMaterial.unlit) {
                            materialValues(
                                [material.metallic, material.perceptualRoughness],
                                [refMaterial.metallic!, refMaterial.roughness!],
                            );
                            materialValues(material.emissive, refMaterial.emissive!);
                        }
                        expect(material.unlit).toBe(Number(refMaterial.unlit));
                        expect(material.doubleSided).toBe(Number(refMaterial.doubleSided));
                        expect(primitive.material.cullMode).toBe(
                            refMaterial.doubleSided ? 1 : ref.frontFace === "cw" ? 2 : 0,
                        );
                        expect(primitive.material.alphaMode).toBe(
                            ({ OPAQUE: 0, MASK: 1, BLEND: 2 } as Record<string, number>)[
                                refMaterial.alphaMode
                            ]!,
                        );
                        if (refMaterial.alphaMode === "MASK")
                            expect(primitive.material.alphaCutoff).toBe(refMaterial.cutoff!);
                        expect(material.baseColorTexture >= 0).toBe(
                            refMaterial.textureImage !== null,
                        );
                        const parsedMaterial =
                            decoded.materials[decoded.meshes[decoded.instances[i].mesh].material];
                        expect(parsedMaterial.texture ?? null).toBe(refMaterial.textureIndex);
                        if (refMaterial.textureIndex !== null) {
                            const texture = reference.textures[refMaterial.textureIndex];
                            expect(decoded.textures[parsedMaterial.texture!].image).toBe(
                                refMaterial.textureImage!,
                            );
                            const source = new URL(reference.images[texture.image], url).href;
                            const registered = registerImage(app.world, source, {
                                sampler: texture.sampler as ImageSamplerDescriptor,
                            });
                            expect(material.baseColorTexture).toBe(registered);
                        }
                    } else {
                        materialValues(material.baseColor, [1, 1, 1, 1]);
                        materialValues([material.metallic, material.perceptualRoughness], [1, 1]);
                        materialValues(material.emissive, [0, 0, 0]);
                        expect(material.unlit).toBe(0);
                        expect(material.doubleSided).toBe(0);
                        expect(material.baseColorTexture).toBe(-1);
                        expect(primitive.material.alphaMode).toBe(0);
                        expect(primitive.material.cullMode).toBe(ref.frontFace === "cw" ? 2 : 0);
                    }
                }
            } finally {
                app.dispose();
            }
        });
    }
}
