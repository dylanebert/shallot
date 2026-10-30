import { createHash } from "node:crypto";

/** Large arrays use little-endian float32 (attributes) or uint32 (indices), without rounding. */
export function recordArray(values: ArrayLike<number>, width: number, integer = false) {
    const array = Array.from(values);
    if (array.length <= 128) return array;
    const bytes = Buffer.alloc(array.length * 4);
    for (let i = 0; i < array.length; i++) {
        if (integer) bytes.writeUInt32LE(array[i], i * 4);
        else bytes.writeFloatLE(array[i], i * 4);
    }
    const min = Array<number>(width).fill(Infinity);
    const max = Array<number>(width).fill(-Infinity);
    for (let i = 0; i < array.length; i++) {
        min[i % width] = Math.min(min[i % width], array[i]);
        max[i % width] = Math.max(max[i % width], array[i]);
    }
    return { count: array.length / width, min, max, sha256: createHash("sha256").update(bytes).digest("hex") };
}

export const fixtures = [
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
