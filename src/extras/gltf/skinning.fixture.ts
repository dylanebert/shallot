import type { GltfJson } from "./gltf";

/** A triangle whose bind pose is outside the camera; its joint brings it into view at t=1. */
export function skinningFixtureUrl(): string {
    const positions = new Float32Array([49, -1, 0, 51, -1, 0, 50, 1, 0]);
    const joints = new Uint16Array(12);
    const weights = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
    const times = new Float32Array([0, 1]);
    const translations = new Float32Array([0, 0, 0, -50, 0, 0]);
    const streams = [positions, joints, weights, times, translations];
    let offset = 0;
    const bufferViews = streams.map((stream) => {
        const view = { buffer: 0, byteOffset: offset, byteLength: stream.byteLength };
        offset += stream.byteLength;
        return view;
    });
    const bytes = new Uint8Array(offset);
    streams.forEach((stream, i) => {
        bytes.set(new Uint8Array(stream.buffer), bufferViews[i].byteOffset);
    });
    const json: GltfJson = {
        asset: { version: "2.0" },
        extensionsUsed: ["KHR_materials_unlit"],
        buffers: [
            {
                byteLength: bytes.length,
                uri: `data:application/octet-stream;base64,${Buffer.from(bytes).toString("base64")}`,
            },
        ],
        bufferViews,
        accessors: [
            {
                bufferView: 0,
                componentType: 5126,
                type: "VEC3",
                count: 3,
                ...{ min: [49, -1, 0], max: [51, 1, 0] },
            },
            { bufferView: 1, componentType: 5123, type: "VEC4", count: 3 },
            { bufferView: 2, componentType: 5126, type: "VEC4", count: 3 },
            {
                bufferView: 3,
                componentType: 5126,
                type: "SCALAR",
                count: 2,
                ...{ min: [0], max: [1] },
            },
            { bufferView: 4, componentType: 5126, type: "VEC3", count: 2 },
        ],
        // biome-ignore lint/style/useNamingConvention: glTF names its extension and attributes.
        materials: [{ extensions: { KHR_materials_unlit: {} }, doubleSided: true }],
        meshes: [
            {
                primitives: [
                    { attributes: { POSITION: 0, JOINTS_0: 1, WEIGHTS_0: 2 }, material: 0 },
                ],
            },
        ],
        nodes: [{ mesh: 0, skin: 0 }, {}],
        skins: [{ joints: [1] }],
        animations: [
            {
                samplers: [{ input: 3, output: 4, interpolation: "LINEAR" }],
                channels: [{ sampler: 0, target: { node: 1, path: "translation" } }],
            },
        ],
        scenes: [{ nodes: [0, 1] }],
        scene: 0,
    };
    return `data:model/gltf+json;base64,${Buffer.from(JSON.stringify(json)).toString("base64")}`;
}
