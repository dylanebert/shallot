import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { composeMat4 } from "../../engine/utils";
import { type GltfJson, generateSceneTangents, parse } from "./gltf";
import { loadGltf } from "./load";

function fixture(name = "skin-accessors") {
    const json = JSON.parse(
        readFileSync(new URL(`./fixtures/${name}.gltf`, import.meta.url), "utf8"),
    ) as GltfJson;
    const buffers = json.buffers!.map(
        (buffer) => Uint8Array.from(Buffer.from(buffer.uri!.split(",")[1], "base64")).buffer,
    );
    return { json, buffers };
}

test("unused skins outside the default scene are omitted without refusal or unsupported reports", async () => {
    const { json } = fixture("unused-skin");
    const scene = await loadGltf("https://fixture.test/unused.gltf", ((input) =>
        String(input) === "https://fixture.test/unused.gltf"
            ? Promise.resolve(Response.json(json))
            : fetch(input)) as typeof fetch);
    expect(scene.skins).toHaveLength(1);
    expect(scene.skins[0].index).toBe(1);
    expect(scene.skins[0].joints).toEqual([1]);
    expect(scene.instances[0].skin).toBe(0);
    expect(scene.nodes.map((node) => node.index)).toEqual([0, 1]);
    expect(scene.unsupported).toEqual([]);
});

test("skins merge shared and touching joints, include intervening nodes, and keep unrelated siblings separate", () => {
    const { json, buffers } = fixture();
    const scene = parse(json, buffers);
    expect(scene.skins.map((skin) => skin.joints)).toEqual([[1, 3], [3, 4], [4], [5]]);
    expect(scene.skins.map((skin) => skin.skeleton)).toEqual([0, 0, 0, 1]);
    expect(scene.skeletons).toEqual([{ bones: [1, 2, 3, 4] }, { bones: [5] }]);
    expect(scene.instances.map((instance) => instance.skin)).toEqual([0, 1, 2, 3]);
});

test("inverse bind matrices default to identity for every joint and retain authored column-major values", () => {
    const { json, buffers } = fixture();
    const identity = composeMat4(0, 0, 0, 0, 0, 0, 1, 1, 1, 1);
    const scene = parse(json, buffers);
    for (const skin of scene.skins)
        expect(skin.inverseBindMatrices).toEqual(
            new Float32Array(skin.joints.flatMap(() => [...identity])),
        );
    const authored = new Float32Array([...composeMat4(-2, 3, 4, 0, 0, 0, 1, 1, 1, 1), ...identity]);
    json.bufferViews!.push({ buffer: buffers.length, byteLength: authored.byteLength });
    json.accessors!.push({
        bufferView: json.bufferViews!.length - 1,
        componentType: 5126,
        type: "MAT4",
        count: 2,
    });
    json.skins![0].inverseBindMatrices = json.accessors!.length - 1;
    expect(parse(json, [...buffers, authored.buffer]).skins[0].inverseBindMatrices).toEqual(
        authored,
    );
});

test("unorm8 and unorm16 weights dequantize without renormalizing, and uint16 joints retain their values", () => {
    const { json, buffers } = fixture();
    const scene = parse(json, buffers);
    expect(scene.meshes[0].weights).toEqual(
        Float32Array.from([64, 191, 0, 0, 128, 127, 0, 0, 255, 0, 0, 0], (value) => value / 255),
    );
    expect(scene.meshes[1].weights).toEqual(
        Float32Array.from(
            [16384, 49151, 0, 0, 32768, 32767, 0, 0, 65535, 0, 0, 0],
            (value) => value / 65535,
        ),
    );
    const joints = new Uint16Array([0, 257, 0, 0, 256, 1, 0, 0, 0, 65535, 0, 0]);
    json.accessors![1].componentType = 5123;
    json.bufferViews![1].byteLength = joints.byteLength;
    buffers[1] = joints.buffer;
    expect(parse(json, buffers).meshes[0].joints).toEqual(Float32Array.from(joints));
});

test("skin is ignored without both influence streams; further joint and weight sets stay reported by semantic", () => {
    const { json, buffers } = fixture();
    const attributes = json.meshes![0].primitives[0].attributes as Record<string, number>;
    attributes.JOINTS_1 = attributes.JOINTS_0;
    attributes.WEIGHTS_1 = attributes.WEIGHTS_0;
    attributes.JOINTS_2 = attributes.JOINTS_0;
    attributes.WEIGHTS_2 = attributes.WEIGHTS_0;
    expect(parse(json, buffers).unsupported).toEqual([
        "JOINTS_1",
        "JOINTS_2",
        "WEIGHTS_1",
        "WEIGHTS_2",
    ]);
    delete attributes.WEIGHTS_0;
    expect(parse(json, buffers).instances[0].skin).toBe(-1);
    delete attributes.JOINTS_0;
    expect(parse(json, buffers).instances[0].skin).toBe(-1);
});

for (const flat of [false, true])
    test(`influences follow ${flat ? "corner expansion" : "MikkTSpace seam remapping"}`, async () => {
        const { json, buffers } = fixture();
        const add = (
            values: Float32Array | Uint16Array,
            type: "VEC3" | "VEC2" | "VEC4" | "SCALAR",
        ) => {
            json.bufferViews!.push({ buffer: buffers.length, byteLength: values.byteLength });
            buffers.push(values.buffer as ArrayBuffer);
            json.accessors!.push({
                bufferView: json.bufferViews!.length - 1,
                componentType: values instanceof Uint16Array ? 5123 : 5126,
                type,
                count: values.length / { VEC3: 3, VEC2: 2, VEC4: 4, SCALAR: 1 }[type],
            });
            return json.accessors!.length - 1;
        };
        const primitive = json.meshes![0].primitives[0];
        const position = add(
            new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, -1, 0, 0, 0, -1, 0]),
            "VEC3",
        );
        const normal = add(new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]), "VEC3");
        const uv = add(new Float32Array([0, 0, 1, 0, 0, 1, 1, 0, 0, -1]), "VEC2");
        const joints = Float32Array.from({ length: 20 }, (_, i) => i);
        const weights = Float32Array.from({ length: 20 }, (_, i) => i / 20);
        primitive.attributes = {
            POSITION: position,
            TEXCOORD_0: uv,
            JOINTS_0: add(joints, "VEC4"),
            WEIGHTS_0: add(weights, "VEC4"),
            ...(flat ? {} : { NORMAL: normal }),
        };
        primitive.indices = add(new Uint16Array([0, 1, 2, 0, 3, 4]), "SCALAR");
        primitive.material = 0;
        json.materials = [{ normalTexture: { index: 0 } }];
        const scene = parse(json, buffers);
        await generateSceneTangents(scene, json, buffers);
        const mesh = scene.meshes[0];
        expect(mesh.vertices.length / 8).toBe(6);
        for (let corner = 0; corner < 6; corner++) {
            const src = [0, 1, 2, 0, 3, 4][corner] * 4;
            const dest = mesh.indices[corner] * 4;
            expect(mesh.joints!.subarray(dest, dest + 4)).toEqual(joints.subarray(src, src + 4));
            expect(mesh.weights!.subarray(dest, dest + 4)).toEqual(weights.subarray(src, src + 4));
        }
    });
