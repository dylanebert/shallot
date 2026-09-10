import { describe, expect, test } from "bun:test";
import { isGlb, parseGlb } from "./glb";
import { type GltfJson, parse } from "./gltf";
import { decodeMeshopt, loadMeshopt } from "./meshopt";

// the Box glTF-Meshopt fixture — the Khronos unit cube run through `gltfpack -c` (EXT_meshopt_compression +
// the companion KHR_mesh_quantization), a .glb container. Geometry lives in compressed bufferViews the codec
// inflates, then dequantizes; the bounds prove the whole chain (decompress → dequant) lands the unit box.
const dir = `${import.meta.dir}/fixtures`;

// a mesh-local position through an instance's TRS (scale → quaternion-rotate → translate) — the world placement
// the GPU applies per instance, here in JS so the test reads the same box the renderer draws
function applyTrs(
    inst: { pos: number[]; rot: number[]; scale: number[] },
    x: number,
    y: number,
    z: number,
): [number, number, number] {
    const sx = x * inst.scale[0];
    const sy = y * inst.scale[1];
    const sz = z * inst.scale[2];
    const [qx, qy, qz, qw] = inst.rot;
    const tx = 2 * (qy * sz - qz * sy);
    const ty = 2 * (qz * sx - qx * sz);
    const tz = 2 * (qx * sy - qy * sx);
    return [
        sx + qw * tx + (qy * tz - qz * ty) + inst.pos[0],
        sy + qw * ty + (qz * tx - qx * tz) + inst.pos[1],
        sz + qw * tz + (qx * ty - qy * tx) + inst.pos[2],
    ];
}

describe("meshopt decode", () => {
    test("decodes the Box glTF-Meshopt geometry through parse's injected codec", async () => {
        await loadMeshopt();

        const bytes = await Bun.file(`${dir}/box-meshopt.glb`).arrayBuffer();
        expect(isGlb(bytes)).toBe(true);
        const { json, bin } = parseGlb(bytes);
        expect(bin).toBeDefined(); // the .glb carries its geometry in the BIN chunk
        const scene = parse(json as GltfJson, [bin as ArrayBuffer], undefined, decodeMeshopt);

        // EXT_meshopt_compression + KHR_mesh_quantization are both handled now, so nothing is unsupported
        expect(scene.unsupported).toEqual([]);
        expect(scene.meshes).toHaveLength(1);

        const m = scene.meshes[0];
        expect(m.indices.length).toBe(36); // 12 triangles
        const verts = m.vertices.length / 8; // VERTEX_FLOATS = 8 per vertex
        expect(Math.max(...m.indices)).toBeLessThan(verts);

        // gltfpack quantizes POSITION to unnormalized integers and bakes the dequant scale + offset into the
        // node transform (mesh-local stays in quantized space, the instance carries the dequant). So the unit
        // box emerges in WORLD space: vertices × instance TRS ≈ [-0.5, 0.5]³ — the proof the decompress +
        // dequant chain is right, not garbage integers.
        const inst = scene.instances[0];
        let lo = Number.POSITIVE_INFINITY;
        let hi = Number.NEGATIVE_INFINITY;
        for (let i = 0; i < verts; i++) {
            const w = applyTrs(
                inst,
                m.vertices[i * 8],
                m.vertices[i * 8 + 1],
                m.vertices[i * 8 + 2],
            );
            for (const v of w) {
                lo = Math.min(lo, v);
                hi = Math.max(hi, v);
            }
        }
        expect(lo).toBeCloseTo(-0.5, 2);
        expect(hi).toBeCloseTo(0.5, 2);

        // the normal lanes are populated + unit length (a normalized-BYTE normal dequant that misread the
        // bytes would land them off the unit sphere)
        const normalLen = Math.hypot(m.vertices[4], m.vertices[5], m.vertices[6]);
        expect(normalLen).toBeCloseTo(1, 2);
    });

    test("a malformed compressed buffer fails loud", async () => {
        await loadMeshopt();
        // the decode-failure boundary: a bad meshopt stream (wrong header byte) rejects in the wasm rather
        // than returning degenerate bytes the accessor read would trust
        expect(() =>
            decodeMeshopt(new Uint8Array([1, 2, 3, 4]), 4, 16, "ATTRIBUTES", "NONE"),
        ).toThrow(/Malformed buffer data/);
    });

    test("an undecodable meshopt asset degrades to a flagged skip, not a crash", () => {
        // graceful degradation (the parse boundary, distinct from the codec throw above): a decoder that can't
        // read an asset's bitstream — a newer vertex-codec version (0xa1) or an unsupported filter — must leave
        // the scene empty and flag the extension, never throw, so one bad asset doesn't break a load. The same
        // shape as a Draco primitive with no codec. Pinned hermetically with a throwing stub (no corpus needed).
        const gltf = {
            buffers: [{ byteLength: 16 }],
            extensionsUsed: ["EXT_meshopt_compression"],
            bufferViews: [
                {
                    buffer: 0,
                    byteOffset: 0,
                    byteLength: 16,
                    extensions: {
                        // biome-ignore lint/style/useNamingConvention: glTF extension name (the JSON key)
                        EXT_meshopt_compression: {
                            buffer: 0,
                            byteOffset: 0,
                            byteLength: 8,
                            byteStride: 4,
                            count: 1,
                            mode: "ATTRIBUTES",
                        },
                    },
                },
            ],
            accessors: [{ bufferView: 0, componentType: 5126, count: 1, type: "VEC3" }],
            meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
            nodes: [{ mesh: 0 }],
            scenes: [{ nodes: [0] }],
            scene: 0,
        } as GltfJson;
        const throwingDecoder = (): Uint8Array => {
            throw new Error("unsupported meshopt bitstream version");
        };
        const scene = parse(gltf, [new ArrayBuffer(16)], undefined, throwingDecoder);
        // the compressed geometry was skipped (not read as garbage), and the extension is flagged
        expect(scene.meshes).toHaveLength(0);
        expect(scene.unsupported.map((u) => u.feature)).toContain("EXT_meshopt_compression");
    });
});
