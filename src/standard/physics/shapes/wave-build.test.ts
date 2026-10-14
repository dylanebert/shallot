import { expect, test } from "bun:test";
import { createWave } from "./heightfield";
import { createWaveMesh } from "./mesh";
import gold from "./wave-build.gold.json";

const word = new DataView(new ArrayBuffer(4));
function bits(value: number): number {
    word.setFloat32(0, value, true);
    return word.getUint32(0, true);
}

test("mesh wave heights equal native Box3D's portable trigonometry", () => {
    const mesh = createWaveMesh(6, 5, 1, 0.4, 0.05, 0.1);
    expect(mesh.vertices.map((vertex) => bits(vertex.y))).toEqual(gold.meshHeights);
});

test("height-field wave quantization equals native Box3D's portable trigonometry", () => {
    const field = createWave(32, 32, { x: 2, y: 1.5, z: 2 }, 0.1, 0.03333, false);
    expect(field.compressedHeights).toEqual(gold.fieldHeights);
});
