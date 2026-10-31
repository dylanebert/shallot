import { beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { readFile } from "node:fs/promises";
import { loadGltf } from "@dylanebert/shallot/gltf";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import gold from "./quantization.gold.json";
import { assets, fileFetch, syncAssets } from "./setup";

setDefaultTimeout(TIMEOUT.node);
beforeAll(syncAssets);
const transform = (m: ArrayLike<number>, v: ArrayLike<number>, w: number) =>
    [0, 1, 2].map((a) => m[a] * v[0] + m[4 + a] * v[1] + m[8 + a] * v[2] + m[12 + a] * w);
const direction = (m: ArrayLike<number>, v: ArrayLike<number>) => {
    const scales = [0, 4, 8].map((a) => Math.hypot(m[a], m[a + 1], m[a + 2]));
    return transform(
        m,
        Array.from(v, (x, a) => x / scales[a]),
        0,
    );
};
const distance = (a: number[], b: number[]) => Math.max(...a.map((x, i) => Math.abs(x - b[i])));

for (const record of gold.records) {
    const name = record.fixture.split("/")[1];
    test(`${name}: quantized triangle geometry and morph deltas equal unquantized records within their steps`, async () => {
        const path = record.fixture.replace("/glTF/", "/glTF-Quantized/");
        const url = new URL(path, assets);
        const model = await loadGltf(url, fileFetch);
        expect(model.unsupported).not.toContain("KHR_mesh_quantization");
        const json = JSON.parse(await readFile(url, "utf8"));
        type Corner = {
            p: number[];
            n: number[];
            uv: number[];
            t?: number[];
            morph: { p?: number[]; n?: number[]; t?: number[] }[];
        };
        const references: Corner[][] = record.instances.flatMap((instance) => {
            const corners = instance.indices.map(
                (v): Corner => ({
                    p: transform(instance.matrix, instance.positions.slice(v * 3, v * 3 + 3), 1),
                    n: direction(instance.matrix, instance.normals.slice(v * 3, v * 3 + 3)),
                    uv: instance.uvs.slice(v * 2, v * 2 + 2),
                    t: instance.tangent
                        ? [
                              ...direction(
                                  instance.matrix,
                                  instance.tangent.slice(v * 4, v * 4 + 3),
                              ),
                              instance.tangent[v * 4 + 3],
                          ]
                        : undefined,
                    morph: (instance.morphTargets ?? []).map((target) => ({
                        p: target.position
                            ? transform(instance.matrix, target.position.slice(v * 3, v * 3 + 3), 0)
                            : undefined,
                        n: target.normal
                            ? direction(instance.matrix, target.normal.slice(v * 3, v * 3 + 3))
                            : undefined,
                        t: target.tangent
                            ? direction(instance.matrix, target.tangent.slice(v * 3, v * 3 + 3))
                            : undefined,
                    })),
                }),
            );
            return Array.from({ length: corners.length / 3 }, (_, i) =>
                corners.slice(i * 3, i * 3 + 3),
            );
        });
        const unused = new Set(references.map((_, i) => i));
        for (const instance of model.instances) {
            const mesh = model.meshes[instance.mesh];
            const sourceNode = json.nodes.find((n: { mesh?: number }) => n.mesh !== undefined);
            const primitive = json.meshes[sourceNode.mesh].primitives[0];
            const posAccessor = json.accessors[primitive.attributes.POSITION];
            const maxInteger =
                posAccessor.componentType === 5120
                    ? 127
                    : posAccessor.componentType === 5121
                      ? 255
                      : posAccessor.componentType === 5122
                        ? 32767
                        : 65535;
            const unit = posAccessor.normalized ? 1 / maxInteger : 1;
            const positionStep =
                unit *
                Math.max(
                    ...[0, 1, 2].map(
                        (a) =>
                            Math.abs(instance.matrix[a]) +
                            Math.abs(instance.matrix[a + 4]) +
                            Math.abs(instance.matrix[a + 8]),
                    ),
                );
            const directionStep =
                Math.max(
                    ...[0, 1, 2].map(
                        (a) =>
                            direction(instance.matrix, [1, 0, 0]).map(Math.abs)[a] +
                            Math.abs(direction(instance.matrix, [0, 1, 0])[a]) +
                            Math.abs(direction(instance.matrix, [0, 0, 1])[a]),
                    ),
                ) /
                    127 +
                8 * 2 ** -23;
            // World matrices and reference attributes meet at float32 storage boundaries.
            const roundoff =
                Math.max(1, ...references.flat(2).flatMap((c) => c.p.map(Math.abs))) *
                16 *
                2 ** -23;
            const tolerance = positionStep + roundoff;
            const material = json.materials?.[primitive.material];
            const uvTransform =
                material?.pbrMetallicRoughness?.baseColorTexture?.extensions?.KHR_texture_transform;
            const uvScale: number[] = uvTransform?.scale ?? [1, 1];
            const uvOffset: number[] = uvTransform?.offset ?? [0, 0];
            const uvStep = Math.max(...uvScale.map(Math.abs)) + 8 * 2 ** -23;
            const corners = Array.from(
                mesh.indices,
                (v): Corner => ({
                    p: transform(instance.matrix, mesh.vertices.subarray(v * 8, v * 8 + 3), 1),
                    n: direction(instance.matrix, mesh.vertices.subarray(v * 8 + 4, v * 8 + 7)),
                    uv: [
                        mesh.vertices[v * 8 + 3] * uvScale[0] + uvOffset[0],
                        mesh.vertices[v * 8 + 7] * uvScale[1] + uvOffset[1],
                    ],
                    t: mesh.tangent
                        ? [
                              ...direction(
                                  instance.matrix,
                                  mesh.tangent.subarray(v * 4, v * 4 + 3),
                              ),
                              mesh.tangent[v * 4 + 3],
                          ]
                        : undefined,
                    morph: (mesh.morphTargets ?? []).map((target) => ({
                        p: target.position
                            ? transform(
                                  instance.matrix,
                                  target.position.subarray(v * 3, v * 3 + 3),
                                  0,
                              )
                            : undefined,
                        n: target.normal
                            ? direction(instance.matrix, target.normal.subarray(v * 3, v * 3 + 3))
                            : undefined,
                        t: target.tangent
                            ? direction(instance.matrix, target.tangent.subarray(v * 3, v * 3 + 3))
                            : undefined,
                    })),
                }),
            );
            for (let i = 0; i < corners.length; i += 3) {
                let match: { index: number; rotation: number } | undefined;
                for (const index of unused) {
                    for (let rotation = 0; rotation < 3; rotation++) {
                        if (
                            [0, 1, 2].every((a) => {
                                const actual = corners[i + a],
                                    expected = references[index][(a + rotation) % 3];
                                return (
                                    distance(actual.p, expected.p) <= tolerance &&
                                    distance(actual.n, expected.n) <= directionStep &&
                                    distance(actual.uv, expected.uv) <= uvStep
                                );
                            })
                        ) {
                            match = { index, rotation };
                            break;
                        }
                    }
                    if (match) break;
                }
                expect(match, `${name} triangle ${i / 3}`).toBeDefined();
                unused.delete(match!.index);
                for (let a = 0; a < 3; a++) {
                    const actual = corners[i + a],
                        expected = references[match!.index][(a + match!.rotation) % 3];
                    if (actual.t)
                        expect(distance(actual.t, expected.t!)).toBeLessThanOrEqual(directionStep);
                    expect(actual.morph.length).toBe(expected.morph.length);
                    for (let target = 0; target < actual.morph.length; target++) {
                        for (const semantic of ["p", "n", "t"] as const) {
                            const delta = actual.morph[target][semantic];
                            if (delta)
                                expect(
                                    distance(delta, expected.morph[target][semantic]!),
                                ).toBeLessThanOrEqual(semantic === "p" ? tolerance : directionStep);
                        }
                    }
                }
            }
        }
        expect(unused.size).toBe(0);
    });
}
