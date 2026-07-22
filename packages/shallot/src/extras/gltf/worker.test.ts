import { describe, expect, test } from "bun:test";
import { type DecodedGltf, decode } from "./assets";
import type { Targets } from "./target";
import { transferables } from "./worker";

// the pure transfer-list walker the decode worker posts with. The
// worker round-trip itself is the real-Chrome gym `render` `gltf-worker` mode (bun-webgpu has no Worker); here we
// pin the walker against a real decoded payload — every typed-array buffer + every ImageBitmap present, once.
const box = `${import.meta.dir}/fixtures/box-draco.gltf`;
const boxKtx = `${import.meta.dir}/fixtures/box-ktx.gltf`;
const bc7 = { basis: 7, gpu: "bc7-rgba-unorm-srgb" as GPUTextureFormat, blockDim: 4 };
const BC: Targets = {
    albedo: bc7,
    mr: bc7,
    normalTex: { basis: 5, gpu: "bc5-rg-unorm", blockDim: 4 },
    occlusion: { basis: 4, gpu: "bc4-r-unorm", blockDim: 4 },
    emissive: bc7,
};

describe("transferables", () => {
    test("collects every typed-array buffer of an untextured payload, no duplicates", async () => {
        const d = await decode(box);
        const list = transferables(d);

        // the geometry streams + index buffer
        const s = d.geometry.static!;
        for (const a of [s.quant.main, s.quant.position, s.quant.quant, s.indices]) {
            expect(list).toContain(a.buffer);
        }
        // the source mesh buffers + the per-material albedo ref
        for (const m of d.scene.meshes) {
            expect(list).toContain(m.vertices.buffer);
            expect(list).toContain(m.indices.buffer);
        }
        expect(list).toContain(d.textures.albedoRef.buffer);

        // untextured → no images, so every entry is a plain ArrayBuffer
        expect(d.textures.albedo).toHaveLength(0);
        expect(list.every((t) => t instanceof ArrayBuffer)).toBe(true);
        // deduped — a buffer transferred twice throws at postMessage
        expect(new Set(list).size).toBe(list.length);
    });

    test("includes the transcoded KTX2 albedo bytes + mip data", async () => {
        const d = await decode(boxKtx, { targets: BC });
        const list = transferables(d);

        expect(d.textures.albedo).toHaveLength(1);
        const img = d.textures.albedo[0];
        expect(img.kind).toBe("compressed");
        if (img.kind === "compressed") {
            expect(list).toContain(img.bytes.buffer);
            for (const mip of img.image.mips) expect(list).toContain(mip.data.buffer);
        }
        expect(new Set(list).size).toBe(list.length);
    });

    test("dedupes views that share one ArrayBuffer", () => {
        // two views over one buffer (the postMessage hazard the Set guards) — must transfer the buffer once
        const shared = new ArrayBuffer(64);
        const vertices = new Float32Array(shared, 0, 8);
        const indices = new Uint32Array(shared, 32, 8);
        const empty = (): { images: ImageBitmap[]; layer: Int32Array } => ({
            images: [],
            layer: new Int32Array(0),
        });
        const payload = {
            url: "",
            clip: 0,
            textured: false,
            scene: {
                meshes: [{ name: "", vertices, indices, color: [0, 0, 0, 0], material: -1 }],
                instances: [],
                materials: [],
                images: [],
                skinInputs: [],
                unsupported: [],
            },
            geometry: { static: null, skinned: [], live: [] },
            textures: {
                albedo: [],
                albedoRef: new Int32Array(0),
                maps: { mr: empty(), normalTex: empty(), occlusion: empty(), emissive: empty() },
                textured: false,
            },
            vats: [],
            liveMeshes: [],
        } as unknown as DecodedGltf;

        const list = transferables(payload);
        expect(list.filter((t) => t === shared)).toHaveLength(1);
        expect(new Set(list).size).toBe(list.length);
    });

    test("includes a skinned mesh's VAT positions + normals, deduped", () => {
        // the walker's VAT branch (worker.ts) — a baked skinned mesh transfers its per-frame position +
        // normal buffers. The real skinned worker round-trip is the gym `render` `gltf-multi` mode (bun-webgpu has no
        // Worker); this pins the one transferables branch the static + KTX cases above don't reach.
        const positions = new Float32Array(3);
        const normals = new Float32Array(3);
        const empty = (): { images: ImageBitmap[]; layer: Int32Array } => ({
            images: [],
            layer: new Int32Array(0),
        });
        const payload = {
            url: "",
            clip: 0,
            textured: false,
            scene: {
                meshes: [],
                instances: [],
                materials: [],
                images: [],
                skinInputs: [],
                unsupported: [],
            },
            geometry: { static: null, skinned: [], live: [] },
            textures: {
                albedo: [],
                albedoRef: new Int32Array(0),
                maps: { mr: empty(), normalTex: empty(), occlusion: empty(), emissive: empty() },
                textured: false,
            },
            vats: [
                {
                    frameCount: 1,
                    fps: 0,
                    duration: 0,
                    vertCount: 1,
                    positions,
                    normals,
                    aabb: { min: [0, 0, 0], max: [0, 0, 0] },
                },
            ],
            liveMeshes: [],
        } as unknown as DecodedGltf;

        const list = transferables(payload);
        expect(list).toContain(positions.buffer);
        expect(list).toContain(normals.buffer);
        expect(new Set(list).size).toBe(list.length);
    });
});
