import { expect, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import manifest from "../../../examples/gltf/assets.json";
import { syncFiles } from "../../../examples/gltf/sync";
import { decodeDraco } from "./draco";
import { type GltfJson, parseGltf } from "./gltf";
import { loadGltf } from "./load";

const assets = fileURLToPath(new URL("../../../examples/gltf/public/assets/", import.meta.url));
await syncFiles(
    manifest.models.find((model) => model.name === "Box")!.files,
    new URL(`${manifest.commit}/`, manifest.source),
    assets,
);
const file = (model: string, variant: string) => {
    const dir = `${assets}Models/${model}/${variant}/`;
    return new URL(`file://${dir}${readdirSync(dir).find((name) => name.endsWith(".gltf"))}`);
};

const cube = (extension: Record<string, unknown>, mode?: number, count = 3): GltfJson => ({
    accessors: [
        { componentType: 5126, count, type: "VEC3" },
        { componentType: 5125, count: 3, type: "SCALAR" },
    ],
    bufferViews: [{ buffer: 0, byteLength: 0 }],
    buffers: [{ uri: "data:,", byteLength: 0 }],
    meshes: [
        {
            primitives: [
                {
                    attributes: { ["POSITION"]: 0 },
                    indices: 1,
                    mode,
                    extensions: { ["KHR_draco_mesh_compression"]: extension },
                },
            ],
        },
    ],
});
const gltfUrl = (json: GltfJson) =>
    `data:model/gltf+json,${encodeURIComponent(JSON.stringify(json))}`;

test("parseGltf refuses an undecoded Draco primitive by name", () => {
    expect(() =>
        parseGltf(cube({ bufferView: 0, attributes: { ["POSITION"]: 0 } }), [new ArrayBuffer(0)]),
    ).toThrow("mesh 0 primitive 0: KHR_draco_mesh_compression is undecoded");
});

test("a strip-mode Draco primitive is refused by name", async () => {
    await expect(
        loadGltf(gltfUrl(cube({ bufferView: 0, attributes: { ["POSITION"]: 0 } }, 5))),
    ).rejects.toThrow("mesh 0 primitive 0: KHR_draco_mesh_compression refuses mode TRIANGLE_STRIP");
});

test("an accessor whose count differs from its decoded data is refused by name", async () => {
    // Box's stream with a NORMAL accessor one short of its decoded points.
    const json = boxDraco();
    const normal = json.accessors![json.meshes![0].primitives[0].attributes.NORMAL!];
    normal.count -= 1;
    await expect(loadGltf(gltfUrl(json))).rejects.toThrow(
        `KHR_draco_mesh_compression NORMAL accessor ${json.meshes![0].primitives[0].attributes.NORMAL} count ${normal.count} VEC3 differs from decoded ${normal.count + 1}`,
    );
});

const boxDraco = () => {
    const url = file("Box", "glTF-Draco");
    const json = JSON.parse(readFileSync(url, "utf8")) as GltfJson;
    const bin = readFileSync(new URL(json.buffers![0].uri!, url)).toString("base64");
    json.buffers![0].uri = `data:application/octet-stream;base64,${bin}`;
    return json;
};

test("an attribute outside the extension is read as usual", async () => {
    const json = boxDraco();
    const prim = json.meshes![0].primitives[0];
    const count = json.accessors![prim.attributes.POSITION].count;
    const colors = Float32Array.from({ length: count * 4 }, (_, i) => (i % 4) / 4);
    json.buffers!.push({
        uri: `data:application/octet-stream;base64,${Buffer.from(colors.buffer).toString("base64")}`,
        byteLength: colors.byteLength,
    });
    json.bufferViews!.push({ buffer: 1, byteLength: colors.byteLength });
    json.accessors!.push({
        bufferView: json.bufferViews!.length - 1,
        componentType: 5126,
        count,
        type: "VEC4",
    });
    prim.attributes.COLOR_0 = json.accessors!.length - 1;
    const scene = await loadGltf(gltfUrl(json));
    expect(scene.meshes[0].color).toEqual(colors);
});

test("an accessor count beyond the admitted decoded size is refused before decoding", async () => {
    const json = boxDraco();
    json.accessors![json.meshes![0].primitives[0].indices!].count = 0x20000000 * 3;
    await expect(loadGltf(gltfUrl(json))).rejects.toThrow(
        "mesh 0 primitive 0: KHR_draco_mesh_compression indices accessor count or type invalid",
    );
});

test("a Draco primitive without indices takes its triangles from the decoded faces", async () => {
    // The pinned encoder's two-face seam fixture: six points, so POSITION's count gives two triangles.
    const stream = readFileSync(
        new URL("../../../crates/draco/tests/fixtures/standard-seams.drc", import.meta.url),
    );
    const decoded = decodeDraco(stream, 2, 6, [{ id: 0, type: "float32" }]);
    expect([...decoded.faces]).not.toEqual([0, 1, 2, 3, 4, 5]);
    const json: GltfJson = {
        accessors: [{ componentType: 5126, count: 6, type: "VEC3" }],
        bufferViews: [{ buffer: 0, byteLength: stream.length }],
        buffers: [
            {
                uri: `data:application/octet-stream;base64,${stream.toString("base64")}`,
                byteLength: stream.length,
            },
        ],
        meshes: [
            {
                primitives: [
                    {
                        attributes: { ["POSITION"]: 0 },
                        extensions: {
                            ["KHR_draco_mesh_compression"]: {
                                bufferView: 0,
                                attributes: { ["POSITION"]: 0 },
                            },
                        },
                    },
                ],
            },
        ],
    };
    const [mesh] = (await loadGltf(gltfUrl(json))).meshes;
    // Without NORMAL the importer unwelds corners; each corner's position is its face's point.
    const corners = Array.from({ length: 6 }, (_, c) =>
        [0, 1, 2].map((k) => mesh.vertices[mesh.indices[c] * 8 + k]),
    );
    const positions = decoded.attributes[0].data;
    expect(corners).toEqual(
        [...decoded.faces].map((point) => [0, 1, 2].map((k) => positions[point * 3 + k])),
    );
});
