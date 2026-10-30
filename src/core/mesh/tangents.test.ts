import { expect, test } from "bun:test";
import { generateTangents } from "@dylanebert/shallot/mesh";
import * as d from "typegpu/data";

function seam() {
    return {
        name: "mirrored seam",
        vertices: new Float32Array([
            0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, -1, 0, 0, 1, 0,
            0, 1, 0, 0, -1, 0, 0, 0, 0, 1, -1,
        ]),
        indices: new Uint32Array([0, 1, 2, 0, 3, 4]),
        uv1: new Float32Array([0, 0, 1, 0, 0, 1, -1, 0, 0, -1]),
        color: Float32Array.from({ length: 20 }, (_, i) => i / 20),
        attributes: { tag: { element: d.u32, data: new Uint32Array([0xffffffff, 1, 2, 3, 4]) } },
    };
}

test("MikkTSpace splits only the vertex whose mirrored corners disagree, duplicating every stream", async () => {
    const input = seam();
    const result = await generateTangents(input);
    expect(result.name).toBe(input.name);
    expect(result.indices).toEqual(new Uint32Array([0, 1, 2, 5, 3, 4]));
    expect(result.vertices.length).toBe(6 * 8);
    expect(result.vertices.subarray(0, 40)).toEqual(input.vertices);
    expect(result.vertices.subarray(40)).toEqual(input.vertices.subarray(0, 8));
    expect(result.uv1).toEqual(new Float32Array([...input.uv1, ...input.uv1.subarray(0, 2)]));
    expect(result.color).toEqual(new Float32Array([...input.color, ...input.color.subarray(0, 4)]));
    expect(new Uint32Array(result.attributes!.tag.data.buffer)).toEqual(
        new Uint32Array([0xffffffff, 1, 2, 3, 4, 0xffffffff]),
    );
    expect(result.tangent.subarray(0, 4)).toEqual(new Float32Array([1, 0, 0, -1]));
    expect(result.tangent.subarray(20, 24)).toEqual(new Float32Array([-1, 0, 0, 1]));
    expect(input.indices).toEqual(new Uint32Array([0, 1, 2, 0, 3, 4]));
    expect(input.vertices.length).toBe(40);
});

test("MikkTSpace uses UV_1 when selected and retains the original indices when corners agree", async () => {
    const input = seam();
    const result = await generateTangents(input, 1);
    expect(result.indices).toEqual(input.indices);
    expect(result.vertices).toEqual(input.vertices);
    expect(result.uv1).toEqual(input.uv1);
    expect(result.tangent.subarray(0, 4)).toEqual(new Float32Array([1, 0, 0, -1]));
    const again = await Promise.all([generateTangents(input, 1), generateTangents(input, 1)]);
    expect(again[0].tangent).toEqual(again[1].tangent);
});

test("tangent generation refuses malformed staging inputs before entering WASM", async () => {
    const input = seam();
    await expect(generateTangents({ ...input, vertices: new Float32Array(9) })).rejects.toThrow(
        "eight-f32",
    );
    await expect(generateTangents({ ...input, indices: new Uint32Array([0]) })).rejects.toThrow(
        "triangle",
    );
    await expect(
        generateTangents({ ...input, indices: new Uint32Array([0, 1, 5]) }),
    ).rejects.toThrow("outside");
    await expect(generateTangents({ ...input, uv1: undefined }, 1)).rejects.toThrow("requires uv1");
    await expect(generateTangents({ ...input, color: new Float32Array(1) })).rejects.toThrow(
        "color",
    );
    input.vertices[0] = Number.NaN;
    await expect(generateTangents(input)).rejects.toThrow("finite");
});
