import { expect, test } from "bun:test";
import { type GltfJson, parse, readAccessor } from "./gltf";

// KHR_mesh_quantization, Extending Mesh Attributes / Extending Morph Target Attributes.
const signed = [5120, 5122];
const integers = [5120, 5121, 5122, 5123];
const formats = (types: number[], normalized: boolean[]) =>
    types.flatMap((componentType) =>
        normalized.map((normalize) => ({ componentType, normalized: normalize })),
    );
for (const morph of [false, true]) {
    for (const [semantic, width, allowed] of [
        ["POSITION", 3, formats(morph ? signed : integers, [false, true])],
        ["NORMAL", 3, formats(signed, [true])],
        ["TANGENT", morph ? 3 : 4, formats(signed, [true])],
        ...["TEXCOORD_0", "TEXCOORD_1"].map((semantic) => [
            semantic,
            2,
            morph ? formats(signed, [false]) : formats(integers, [false, true]),
        ]),
    ] as [string, number, { componentType: number; normalized: boolean }[]][]) {
        for (const format of allowed) {
            test(`${morph ? "morph" : "base"} ${semantic} ${format.componentType} normalized=${format.normalized}`, () => {
                const signed = format.componentType === 5120 || format.componentType === 5122;
                const size = format.componentType <= 5121 ? 1 : 2;
                const max = size === 1 ? (signed ? 127 : 255) : signed ? 32767 : 65535;
                const values = [signed ? -max - 1 : 0, max, 1, signed ? -1 : 0];
                const stride = Math.ceil((width * size) / 4) * 4;
                const bytes = new ArrayBuffer(stride * 3);
                const view = new DataView(bytes);
                const expected: number[] = [];
                for (let v = 0; v < 3; v++)
                    for (let c = 0; c < width; c++) {
                        const value = values[(v + c) % values.length];
                        const offset = v * stride + c * size;
                        if (size === 1) {
                            if (signed) view.setInt8(offset, value);
                            else view.setUint8(offset, value);
                        } else if (signed) view.setInt16(offset, value, true);
                        else view.setUint16(offset, value, true);
                        expected.push(
                            Math.fround(
                                format.normalized ? Math.max(value / max, signed ? -1 : 0) : value,
                            ),
                        );
                    }
                const base = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
                const normals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
                const gltf: GltfJson = {
                    asset: { version: "2.0" },
                    extensionsRequired: ["KHR_mesh_quantization"],
                    bufferViews: [
                        { buffer: 0, byteLength: bytes.byteLength, byteStride: stride },
                        { buffer: 1, byteLength: base.byteLength },
                        { buffer: 2, byteLength: normals.byteLength },
                    ],
                    accessors: [
                        { bufferView: 0, count: 3, type: `VEC${width}` as "VEC3", ...format },
                        { bufferView: 1, count: 3, type: "VEC3", componentType: 5126 },
                        { bufferView: 2, count: 3, type: "VEC3", componentType: 5126 },
                    ],
                    meshes: [
                        {
                            primitives: [
                                {
                                    attributes: {
                                        POSITION: 1,
                                        NORMAL: 2,
                                        ...(!morph ? { [semantic]: 0 } : {}),
                                    },
                                    ...(morph ? { targets: [{ [semantic]: 0 }] } : {}),
                                },
                            ],
                        },
                    ],
                    nodes: [{ mesh: 0 }],
                    scenes: [{ nodes: [0] }],
                };
                const result = parse(gltf, [bytes, base.buffer, normals.buffer]);
                expect(result.unsupported).not.toContain("KHR_mesh_quantization");
                const mesh = result.meshes[0];
                const uvMorph = morph && semantic.startsWith("TEXCOORD_");
                if (uvMorph) expect(result.unsupported).toContain(`morph.${semantic}`);
                const actual = uvMorph
                    ? readAccessor(gltf, [bytes, base.buffer, normals.buffer], 0)
                    : !morph && semantic === "TEXCOORD_1"
                      ? mesh.uv1!
                      : morph
                        ? mesh.morphTargets![0][
                              semantic.toLowerCase() as "position" | "normal" | "tangent"
                          ]!
                        : semantic === "TANGENT"
                          ? mesh.tangent!
                          : Float32Array.from({ length: 3 * width }, (_, i) => {
                                const v = Math.floor(i / width),
                                    c = i % width;
                                return mesh.vertices[
                                    v * 8 +
                                        (semantic === "POSITION"
                                            ? c
                                            : semantic === "NORMAL"
                                              ? 4 + c
                                              : c === 0
                                                ? 3
                                                : 7)
                                ];
                            });
                expect(Array.from(actual)).toEqual(expected);
            });
        }
    }
}
