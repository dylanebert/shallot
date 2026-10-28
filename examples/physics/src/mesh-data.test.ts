import { expect, test } from "bun:test";
import { decodeMesh } from "./mesh-data";

test("mesh arrays decode little-endian floats and full-width unsigned indices", () => {
    const mesh = decodeMesh({
        vertices: "AACAPwAAIMAAAACA",
        indices: "eFY0kv////8=",
        materialIndices: "AP8B",
    });
    expect(Array.from(mesh.vertices)).toEqual([1, -2.5, -0]);
    expect(Array.from(mesh.indices)).toEqual([0x92345678, 0xffffffff]);
    expect(Array.from(mesh.materialIndices)).toEqual([0, 255, 1]);
});
