import { expect } from "bun:test";
import type { GltfMesh } from "@dylanebert/shallot/gltf";
import { recordArray } from "./records";

export function decodedArrays(mesh: GltfMesh) {
    const count = mesh.vertices.length / 8;
    return {
        positions: Float32Array.from(
            { length: count * 3 },
            (_, i) => mesh.vertices[Math.floor(i / 3) * 8 + (i % 3)],
        ),
        normals: Float32Array.from(
            { length: count * 3 },
            (_, i) => mesh.vertices[Math.floor(i / 3) * 8 + 4 + (i % 3)],
        ),
        uvs: Float32Array.from(
            { length: count * 2 },
            (_, i) => mesh.vertices[Math.floor(i / 2) * 8 + (i % 2 ? 7 : 3)],
        ),
        indices: mesh.indices,
        uv1: mesh.uv1 ?? null,
        color: mesh.color ?? null,
        tangent: mesh.tangent ?? null,
    };
}

export function decodedArray(
    actual: ArrayLike<number> | null,
    expected: ReturnType<typeof recordArray> | null,
    width: number,
    label: string,
    integer = false,
) {
    expect(actual === null ? null : recordArray(actual, width, integer), label).toEqual(expected);
}

export function decodeConformance(
    mesh: GltfMesh,
    reference: Record<
        keyof ReturnType<typeof decodedArrays>,
        ReturnType<typeof recordArray> | null
    >,
    label: string,
    generatedTangents?: Float32Array,
) {
    const arrays = decodedArrays(mesh);
    for (const name of Object.keys(arrays) as (keyof typeof arrays)[]) {
        const width =
            name === "positions" || name === "normals"
                ? 3
                : name === "color" || name === "tangent"
                  ? 4
                  : name === "indices"
                    ? 1
                    : 2;
        if (name === "tangent" && reference.tangent === null && generatedTangents) {
            expect(arrays.tangent, `${label} generated tangent`).not.toBeNull();
            const corners = Float32Array.from(
                { length: arrays.indices.length * 4 },
                (_, i) => arrays.tangent![arrays.indices[Math.floor(i / 4)] * 4 + (i % 4)],
            );
            decodedArray(
                corners,
                recordArray(generatedTangents, 4),
                4,
                `${label} generated tangent`,
            );
        } else
            decodedArray(
                arrays[name],
                reference[name],
                width,
                `${label} ${name}`,
                name === "indices",
            );
    }
    return arrays;
}
