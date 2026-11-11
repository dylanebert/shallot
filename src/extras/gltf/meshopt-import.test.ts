import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { decompressedCopies } from "../../../diagnostics/meshopt-records/copies";
import { loadRecords } from "../../../diagnostics/meshopt-records/records";
import { evaluateGltfAnimation, type GltfJson, loadGltf, parseGltf } from "./index";
import { decodeMeshoptView } from "./kernel";

const localFetch: typeof fetch = (async (input: URL | string) => {
    const url = new URL(String(input));
    return url.protocol === "file:" ? new Response(readFileSync(url)) : fetch(url);
}) as typeof fetch;
const copies = await decompressedCopies();
for (const { asset, compressed, decompressed } of copies) {
    test(`${asset}: imported geometry, materials, transforms and sampled animation equal the reference-decompressed copy exactly`, async () => {
        const actual = await loadGltf(compressed, localFetch);
        const expected = await loadGltf(decompressed, localFetch);
        expect(actual).toEqual(expected);
        for (const [i, clip] of actual.animations.entries())
            for (const time of [0, 0.125, 0.5, 1, 4, 10])
                expect(evaluateGltfAnimation(actual.nodes, clip, time)).toEqual(
                    evaluateGltfAnimation(expected.nodes, expected.animations[i], time),
                );
    });
}

test("MeshoptCubeTest/glTF: decoded views equal fallback bytes, triangles up to cyclic rotation", async () => {
    const { compressed } = copies.find((copy) => copy.asset === "MeshoptCubeTest/glTF")!;
    const json = JSON.parse(readFileSync(compressed, "utf8"));
    const records = (await loadRecords()).filter((r) => r.asset === "MeshoptCubeTest/glTF");
    for (const record of records) {
        const view = json.bufferViews[record.index];
        const uri = json.buffers[view.buffer].uri;
        const response = await localFetch(new URL(uri, compressed));
        const fallback = new Uint8Array(
            await response.arrayBuffer(),
            view.byteOffset ?? 0,
            view.byteLength,
        );
        const def = view.extensions.KHR_meshopt_compression;
        const actual = new Uint8Array(record.length);
        decodeMeshoptView(record.index, def, record.source, actual, true);
        if (def.mode !== "TRIANGLES") expect(actual).toEqual(fallback);
        else {
            const a = new DataView(actual.buffer, actual.byteOffset, actual.byteLength);
            const b = new DataView(fallback.buffer, fallback.byteOffset, fallback.byteLength);
            const get = (v: DataView, offset: number) =>
                def.byteStride === 2 ? v.getUint16(offset, true) : v.getUint32(offset, true);
            for (let offset = 0; offset < actual.length; offset += 3 * def.byteStride) {
                const triangle = [0, 1, 2].map((i) => get(a, offset + i * def.byteStride));
                const reference = [0, 1, 2].map((i) => get(b, offset + i * def.byteStride));
                expect(
                    [0, 1, 2].some((rotation) =>
                        triangle.every((v, i) => v === reference[(i + rotation) % 3]),
                    ),
                ).toBe(true);
            }
        }
    }
});

for (const [field, value] of [
    ["mode", "UNKNOWN"],
    ["filter", "COLOR"],
    ["count", 0],
    ["byteStride", 3],
    ["byteOffset", -1],
] as const) {
    test(`meshopt importer refuses invalid ${field} by view name`, async () => {
        const json = fixture();
        const def = json.bufferViews![0].extensions!.EXT_meshopt_compression as Record<
            string,
            unknown
        >;
        def[field] = value;
        await expect(load(json)).rejects.toThrow("bufferView 0:");
    });
}

function fixture(): GltfJson {
    return JSON.parse(
        readFileSync(new URL("./fixtures/required-meshopt.gltf", import.meta.url), "utf8"),
    );
}
async function load(json: GltfJson, requests: string[] = []) {
    return loadGltf("https://example.test/model.gltf", (async (input: URL | string) => {
        requests.push(String(input));
        return String(input) === "https://example.test/model.gltf"
            ? Response.json(json)
            : fetch(input);
    }) as typeof fetch);
}
for (const name of ["EXT_meshopt_compression", "KHR_meshopt_compression"]) {
    test(`${name}: fallback with a URI is never fetched, tagged or untagged`, async () => {
        for (const tagged of [true, false]) {
            const json = fixture();
            const view = json.bufferViews![0];
            view.extensions = { [name]: view.extensions!.EXT_meshopt_compression };
            json.extensionsRequired = [name];
            json.extensionsUsed = [name];
            json.buffers![1].uri = "never-fetch.bin";
            json.buffers![1].extensions = tagged ? { [name]: { fallback: true } } : undefined;
            const requests: string[] = [];
            await load(json, requests);
            expect(requests.some((url) => url.includes("never-fetch"))).toBe(false);
        }
    });
}
for (const owner of ["view", "buffer"]) {
    test(`${owner} carrying both meshopt extensions refuses both by name before fetching`, async () => {
        const json = fixture();
        const target = owner === "view" ? json.bufferViews![0] : json.buffers![1];
        target.extensions = { ["KHR_meshopt_compression"]: {}, ["EXT_meshopt_compression"]: {} };
        const requests: string[] = [];
        await expect(load(json, requests)).rejects.toThrow(
            "KHR_meshopt_compression excludes EXT_meshopt_compression",
        );
        expect(requests).toHaveLength(1);
    });
}
for (const [label, mutate] of [
    [
        "fallback range",
        (j: GltfJson) => {
            j.buffers![1].byteLength = 35;
        },
    ],
    [
        "compressed range",
        (j: GltfJson) => {
            j.buffers![0].byteLength = 1;
        },
    ],
    [
        "parent stride",
        (j: GltfJson) => {
            j.bufferViews![0].byteStride = 16;
        },
    ],
    [
        "decoded length",
        (j: GltfJson) => {
            j.bufferViews![0].byteLength = 32;
        },
    ],
    [
        "untagged view",
        (j: GltfJson) => {
            j.bufferViews!.push({ buffer: 1, byteLength: 4 });
        },
    ],
    [
        "fallback source",
        (j: GltfJson) => {
            (j.bufferViews![0].extensions!.EXT_meshopt_compression as { buffer: number }).buffer =
                1;
        },
    ],
    [
        "optional placeholder",
        (j: GltfJson) => {
            j.extensionsRequired = [];
        },
    ],
] as const) {
    test(`meshopt refuses ${label} by owner name`, async () => {
        const json = fixture();
        mutate(json);
        await expect(load(json)).rejects.toThrow(/buffer(View)? \d/);
    });
}

test("every primitive index path refuses an out-of-range index by accessor and value, compressed or not", async () => {
    for (const mode of [4, 5, 6]) {
        const json: GltfJson = {
            buffers: [{ byteLength: 36 }, { byteLength: 12 }],
            bufferViews: [
                { buffer: 0, byteLength: 36 },
                { buffer: 1, byteLength: 12 },
            ],
            accessors: [
                { bufferView: 0, count: 3, type: "VEC3", componentType: 5126 },
                { bufferView: 1, count: 3, type: "SCALAR", componentType: 5125 },
            ],
            meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1, mode }] }],
            nodes: [{ mesh: 0 }],
        };
        expect(() =>
            parseGltf(json, [new ArrayBuffer(36), new Uint32Array([0, 1, 3]).buffer]),
        ).toThrow("indices accessor 1: index 3 exceeds vertex count 3");
        const fifo = readFileSync(
            new URL("../../../crates/gltf/tests/meshopt/fifo.bin", import.meta.url),
        ).subarray(72, 90);
        json.buffers = [
            {
                byteLength: 36,
                uri: `data:application/octet-stream;base64,${Buffer.alloc(36).toString("base64")}`,
            },
            { byteLength: 12, extensions: { ["KHR_meshopt_compression"]: { fallback: true } } },
            {
                byteLength: fifo.length,
                uri: `data:application/octet-stream;base64,${fifo.toString("base64")}`,
            },
        ];
        json.extensionsRequired = ["KHR_meshopt_compression"];
        json.bufferViews![1].extensions = {
            ["KHR_meshopt_compression"]: {
                buffer: 2,
                byteLength: 18,
                count: 3,
                byteStride: 4,
                mode: "TRIANGLES",
            },
        };
        await expect(load(json)).rejects.toThrow(
            "indices accessor 1: index 4294967295 exceeds vertex count 3",
        );
    }
});
