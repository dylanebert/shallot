import { expect, test } from "bun:test";
import { type GltfJson, parse } from "./gltf";

for (const field of ["version", "minVersion"] as const)
    for (const version of ["2.1", "2.10", "3.0", "999999999.0"])
        test(`refuses asset.${field} ${version} by name`, () => {
            expect(() =>
                parse(
                    {
                        asset: {
                            version,
                            ...(field === "minVersion" ? { minVersion: version } : {}),
                        },
                    },
                    [],
                ),
            ).toThrow(`asset.${field}: ${version}`);
        });

test("accepts asset version and minVersion 2.0", () => {
    expect(parse({ asset: { version: "2.0", minVersion: "2.0" } }, []).unsupported).toEqual([]);
});

function primitive(mode = 4): GltfJson {
    return {
        accessors: [{ componentType: 5126, count: 5, type: "VEC3" }],
        meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 0 }, mode }] }],
        nodes: [{ mesh: 0 }],
    };
}

test("accessors without bufferView or sparse initialize to zero", () => {
    const model = parse(primitive(), []);
    expect(Array.from(model.meshes[0].vertices)).toEqual(Array(40).fill(0));
    expect(model.unsupported).toEqual([]);
});

for (const componentType of [5121, 5123, 5125])
    for (const dense of [false, true])
        test(`sparse indices ${componentType} replace ${dense ? "strided dense" : "zero"} values with byte offsets and normalization`, () => {
            const json = primitive();
            const buffer = new ArrayBuffer(128);
            const view = new DataView(buffer);
            for (let i = 0; i < 5; i++)
                for (let c = 0; c < 3; c++) view.setInt16(4 + i * 8 + c * 2, 16384, true);
            if (componentType === 5121) view.setUint8(54, 2);
            else if (componentType === 5123) view.setUint16(54, 2, true);
            else view.setUint32(54, 2, true);
            [-32768, 0, 32767].forEach((v, c) => {
                view.setInt16(68 + c * 2, v, true);
            });
            json.bufferViews = [
                { buffer: 0, byteOffset: 2, byteLength: 40, byteStride: 8 },
                { buffer: 0, byteOffset: 50, byteLength: 8 },
                { buffer: 0, byteOffset: 64, byteLength: 12 },
            ];
            json.accessors![0] = {
                ...(dense ? { bufferView: 0, byteOffset: 2 } : {}),
                componentType: 5122,
                normalized: true,
                count: 5,
                type: "VEC3",
                sparse: {
                    count: 1,
                    indices: { bufferView: 1, byteOffset: 4, componentType },
                    values: { bufferView: 2, byteOffset: 4 },
                },
            };
            const model = parse(json, [buffer]);
            const vertices = model.meshes[0].vertices;
            expect(Array.from(vertices.slice(16, 19))).toEqual([-1, 0, 1]);
            expect(Array.from(vertices.slice(0, 3))).toEqual(
                Array(3).fill(dense ? Math.fround(16384 / 32767) : 0),
            );
            expect(model.unsupported).toEqual([]);
        });

test("sparse indices replace a zero-initialized unsigned index accessor", () => {
    const json = primitive();
    const buffer = new ArrayBuffer(8);
    new DataView(buffer).setUint32(4, 2, true);
    json.bufferViews = [
        { buffer: 0, byteLength: 1 },
        { buffer: 0, byteOffset: 4, byteLength: 4 },
    ];
    json.accessors!.push({
        componentType: 5125,
        count: 3,
        type: "SCALAR",
        sparse: {
            count: 1,
            indices: { bufferView: 0, componentType: 5121 },
            values: { bufferView: 1 },
        },
    });
    json.meshes![0].primitives[0].indices = 1;
    expect(Array.from(parse(json, [buffer]).meshes[0].indices)).toEqual([2, 0, 0]);
});

for (const mode of [5, 6])
    for (const indexed of [false, true])
        test(`${mode === 5 ? "strip" : "fan"}, ${indexed ? "indexed" : "non-indexed"}, uses the Topology equation's vertex order`, () => {
            const json = primitive(mode);
            if (indexed) {
                json.bufferViews = [{ buffer: 0, byteLength: 5 }];
                json.accessors!.push({
                    bufferView: 0,
                    componentType: 5121,
                    count: 5,
                    type: "SCALAR",
                });
                json.meshes![0].primitives[0].indices = 1;
            }
            const source = indexed ? [4, 1, 3, 0, 2] : [0, 1, 2, 3, 4];
            const expected = mode === 5 ? [0, 1, 2, 1, 3, 2, 2, 3, 4] : [1, 2, 0, 2, 3, 0, 3, 4, 0];
            expect(
                Array.from(
                    parse(json, indexed ? [new Uint8Array(source).buffer] : []).meshes[0].indices,
                ),
            ).toEqual(expected.map((i) => source[i]));
        });

for (const [mode, name] of ["POINTS", "LINES", "LINE_LOOP", "LINE_STRIP"].entries())
    test(`reports ${name} by name without reading its unsupported geometry`, () => {
        const model = parse(primitive(mode), []);
        expect(model.unsupported).toEqual([name]);
        expect(model.meshes).toEqual([]);
    });
