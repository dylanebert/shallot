import { describe, expect, test } from "bun:test";
import { decodeDraco, loadDraco } from "./draco";
import { type GltfJson, parse } from "./gltf";

// the vendored Box glTF-Draco fixture (Khronos sample) — a unit cube, 24 vertices / 12 triangles, its
// geometry living entirely in the KHR_draco_mesh_compression bufferView. Small enough to ship in-package.
const dir = `${import.meta.dir}/fixtures`;

describe("draco decode", () => {
    test("decodes the Box glTF-Draco primitive through parse's injected codec", async () => {
        await loadDraco();

        const json = JSON.parse(await Bun.file(`${dir}/box-draco.gltf`).text()) as GltfJson;
        const bin = await Bun.file(`${dir}/Box.bin`).arrayBuffer();
        const scene = parse(json, [bin], decodeDraco);

        // KHR_draco_mesh_compression is handled now, so nothing is reported unsupported
        expect(scene.unsupported).toEqual([]);
        expect(scene.meshes).toHaveLength(1);

        const m = scene.meshes[0];
        expect(m.vertices.length / 8).toBe(24); // VERTEX_FLOATS = 8 per vertex
        expect(m.indices.length).toBe(36); // 12 triangles

        // the decoded geometry is the unit box: bounds ≈ [-0.5, 0.5]³ (within Draco quantization)
        let lo = Number.POSITIVE_INFINITY;
        let hi = Number.NEGATIVE_INFINITY;
        for (let i = 0; i < 24; i++) {
            for (let k = 0; k < 3; k++) {
                const v = m.vertices[i * 8 + k];
                lo = Math.min(lo, v);
                hi = Math.max(hi, v);
            }
        }
        expect(lo).toBeCloseTo(-0.5, 2);
        expect(hi).toBeCloseTo(0.5, 2);

        // indices stay in range and the normal lanes are populated (unit box face normals)
        expect(Math.max(...m.indices)).toBeLessThan(24);
        const normalLen = Math.hypot(m.vertices[4], m.vertices[5], m.vertices[6]);
        expect(normalLen).toBeCloseTo(1, 2);
    });

    test("a non-Draco buffer fails loud through the codec", async () => {
        await loadDraco();
        // the decode-failure boundary (draco.ts): DecodeArrayToMesh rejects non-Draco bytes, so the codec
        // throws with the Draco error rather than returning a degenerate mesh the rest of the pipeline trusts
        expect(() => decodeDraco(new Uint8Array([1, 2, 3, 4]), { POSITION: 0 })).toThrow(
            /Draco decode failed/,
        );
    });
});
