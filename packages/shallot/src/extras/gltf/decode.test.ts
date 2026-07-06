import { describe, expect, test } from "bun:test";
import { State } from "../../engine";
import {
    clearGltfCache,
    decode,
    ensureDecoded,
    gltfCacheStats,
    invalidate,
    register,
} from "./assets";
import type { Targets } from "./target";

// the deviceless decode half of the importer (the CPU↔GPU boundary `coding.md` flags for tests). The Box
// glTF-Draco fixture (untextured) exercises fetch + Draco parse + geometry quantization with no GPU and no
// createImageBitmap, so the payload shape is unit-testable here; the upload + spawn half (`register`) is
// GPU-only, gated in the gym `render` `gltf-model` mode.
const box = `${import.meta.dir}/fixtures/box-draco.gltf`;

// the same Box geometry with a KTX2 baseColor (KHR_texture_basisu → box-etc1s.ktx2). The transcode target is
// passed in (the deviceless contract), so this textured path decodes with no GPU device.
const boxKtx = `${import.meta.dir}/fixtures/box-ktx.gltf`;
// the same Box run through `gltfpack -c` — EXT_meshopt_compression geometry + the companion
// KHR_mesh_quantization, a .glb. Geometry-only (untextured), so it decodes deviceless with no transcode target.
const boxMeshopt = `${import.meta.dir}/fixtures/box-meshopt.glb`;
// the per-slot targets pickTargets(device) returns on a BC device; threading a literal in is what lets a
// worker (or this test) transcode off the main thread, no device read. box-ktx is baseColor-only, so only
// `albedo` is exercised here; the data-map targets matter on Sponza-KTX (gym `render` `gltf-model`).
const BC: Targets = {
    albedo: { basis: 7, gpu: "bc7-rgba-unorm-srgb", blockDim: 4 },
    mr: { basis: 7, gpu: "bc7-rgba-unorm", blockDim: 4 },
    normalTex: { basis: 5, gpu: "bc5-rg-unorm", blockDim: 4 },
    occlusion: { basis: 4, gpu: "bc4-r-unorm", blockDim: 4 },
    emissive: { basis: 7, gpu: "bc7-rgba-unorm-srgb", blockDim: 4 },
};

describe("decode", () => {
    test("decodes the Box glTF-Draco fixture to a deviceless payload", async () => {
        const d = await decode(box);

        // one static mesh (24 verts / 12 tris), no skinned meshes
        expect(d.geometry.static).not.toBeNull();
        expect(d.geometry.skinned).toHaveLength(0);
        const s = d.geometry.static!;
        expect(s.slices).toHaveLength(1);
        expect(s.slices[0].meshIndex).toBe(0);
        expect(s.slices[0].indexCount).toBe(36); // 12 triangles
        expect(s.indices.length).toBe(36);
        // the quantized streams: 16 B/vertex main (4 u32) + 8 B/vertex depth (2 u32) over 24 vertices
        expect(s.quant.main.length).toBe(24 * 4);
        expect(s.quant.position.length).toBe(24 * 2);

        // untextured: no albedo images, every data map absent (the palette is packed at union assembly now,
        // not in the deviceless decode — the bucketing is a union-level decision across the active set)
        expect(d.textured).toBe(false);
        expect(d.textures.albedo).toHaveLength(0);
        expect([...d.textures.albedoRef]).toEqual([-1]); // one material, no baseColor image
        expect(d.textures.maps.mr.images).toHaveLength(0);
        expect(d.textures.maps.normalTex.images).toHaveLength(0);

        // no skin → no VAT baked
        expect(d.vats.every((v) => v === null)).toBe(true);

        // one node instance of the box, pointing at mesh 0
        expect(d.scene.instances).toHaveLength(1);
        expect(d.scene.instances[0].mesh).toBe(0);

        // the baked clip, part of the cache key (no opts → 0)
        expect(d.clip).toBe(0);
    });

    test("transcodes a KTX2 baseColor with a passed target — deviceless", async () => {
        const d = await decode(boxKtx, { targets: BC });

        // the baseColor decoded to the target's compressed block format (not the RGBA fallback), with no
        // device — the worker can do this off-thread and hand back the transferable payload
        expect(d.textured).toBe(true);
        expect(d.textures.albedo).toHaveLength(1);
        const img = d.textures.albedo[0];
        expect(img.kind).toBe("compressed");
        if (img.kind === "compressed") {
            expect(img.image.format).toBe("bc7-rgba-unorm-srgb");
            expect(img.image.mips.length).toBeGreaterThan(0);
        }
        // one material → albedo image 0
        expect([...d.textures.albedoRef]).toEqual([0]);
    });

    test("a KTX2 baseColor with no transcode target rejects (the deviceless contract)", async () => {
        // the explicit boundary: a KTX2 baseColor needs the main-thread-resolved target threaded in. Decoding
        // without it must fail loud, not silently fall back — the same contract a worker call site satisfies.
        await expect(decode(boxKtx)).rejects.toThrow(/transcode target/);
    });

    test("decodes the Box glTF-Meshopt fixture deviceless — codec auto-loaded, geometry dequantized", async () => {
        // decode() detects the EXT_meshopt_compression bufferViews, dynamic-imports + instantiates the codec,
        // and decompresses + dequantizes geometry — all with no GPU device (the same deviceless contract Draco
        // satisfies). A .glb container, untextured, so no transcode target needed.
        const d = await decode(boxMeshopt);

        expect(d.textured).toBe(false);
        expect(d.geometry.skinned).toHaveLength(0);
        const s = d.geometry.static;
        expect(s).not.toBeNull();
        expect(s!.slices).toHaveLength(1);
        expect(s!.slices[0].indexCount).toBe(36); // 12 triangles, decompressed
        // the quantized streams: 16 B/vertex main (4 u32) + 8 B/vertex position (2 u32), consistent counts
        const verts = s!.quant.main.length / 4;
        expect(verts).toBeGreaterThan(0);
        expect(s!.quant.position.length).toBe(verts * 2);
        expect(d.clip).toBe(0);
    });
});

// the content-keyed decode cache. The deviceless half — keyed
// (src, clip), it reuses the decoded payload across loads so an editor rebuild never re-decodes. The GPU
// assembly half (register) is device-only, gated in the gym `render` `gltf-model` mode.
describe("asset cache", () => {
    test("ensureDecoded reuses the decode across loads; re-decodes on a new clip / invalidate", async () => {
        clearGltfCache();
        const before = gltfCacheStats().decodes;

        const a = await ensureDecoded(box, 0);
        const b = await ensureDecoded(box, 0);
        // same (src, clip) → the cached payload, no second decode (the rebuild win)
        expect(b).toBe(a);
        expect(gltfCacheStats().decodes - before).toBe(1);
        expect(gltfCacheStats().assets).toBe(1);

        // a different clip is a distinct cache entry → its own decode
        const c = await ensureDecoded(box, 1);
        expect(c).not.toBe(a);
        expect(gltfCacheStats().decodes - before).toBe(2);
        expect(gltfCacheStats().assets).toBe(2);

        // invalidate drops every clip of the src → the next load re-decodes
        invalidate(box);
        expect(gltfCacheStats().assets).toBe(0);
        const d = await ensureDecoded(box, 0);
        expect(d).not.toBe(a);
        expect(gltfCacheStats().decodes - before).toBe(3);

        clearGltfCache();
        expect(gltfCacheStats().assets).toBe(0);
    });

    test("invalidate is targeted — dropping one src preserves another's cached decode", async () => {
        // the contract the editor's live asset-swap relies on: re-saving model A
        // invalidates only A, so model B keeps its cached decode + never re-decodes on the paired rebuild
        clearGltfCache();
        const before = gltfCacheStats().decodes;
        const a = await ensureDecoded(box, 0);
        const k = await ensureDecoded(boxKtx, 0, BC);
        expect(gltfCacheStats().decodes - before).toBe(2);
        expect(gltfCacheStats().assets).toBe(2);

        invalidate(box);
        expect(gltfCacheStats().assets).toBe(1); // only box's entry dropped

        // boxKtx is still a cache hit (same payload object, no re-decode counted)
        const k2 = await ensureDecoded(boxKtx, 0, BC);
        expect(k2).toBe(k);
        expect(gltfCacheStats().decodes - before).toBe(2);

        // box re-decodes to a fresh payload on its next load
        const a2 = await ensureDecoded(box, 0);
        expect(a2).not.toBe(a);
        expect(gltfCacheStats().decodes - before).toBe(3);
        clearGltfCache();
    });

    test("concurrent ensureDecoded of one source decodes once", async () => {
        clearGltfCache();
        const before = gltfCacheStats().decodes;
        const [a, b] = await Promise.all([ensureDecoded(box, 0), ensureDecoded(box, 0)]);
        expect(b).toBe(a);
        expect(gltfCacheStats().decodes - before).toBe(1);
        clearGltfCache();
    });

    test("a failed decode caches nothing and leaves no in-flight slot (stays retryable)", async () => {
        clearGltfCache();
        const before = gltfCacheStats().decodes;
        const missing = `${import.meta.dir}/fixtures/does-not-exist.gltf`;
        await expect(ensureDecoded(missing, 0)).rejects.toThrow();
        // no cache entry, no decode counted (the counter advances only on a successful decode), and — the fix
        // this pins — the in-flight slot cleared, so the next load re-attempts rather than awaiting a cached
        // rejection forever (without the settle-cleanup `inflight` would stay 1, a poisoned source)
        expect(gltfCacheStats().assets).toBe(0);
        expect(gltfCacheStats().decodes).toBe(before);
        expect(gltfCacheStats().inflight).toBe(0);
        await expect(ensureDecoded(missing, 0)).rejects.toThrow();
        expect(gltfCacheStats().inflight).toBe(0);
        clearGltfCache();
    });
});

// the dead-State guard. A decode awaited across a scene switch can
// resolve after its State is torn down; `register` must no-op on the dead State, never throw or spawn into a
// disposed world. (The queued-decode abort half is the Scheduler.abort test; the worker-boundary error half is
// the real-Chrome gym `render` `gltf-worker` mode — bun-webgpu has no Worker.)
describe("disposal guard", () => {
    test("register no-ops on a disposed State (a late decode result is ignored, not thrown)", async () => {
        clearGltfCache();
        const decoded = await decode(box); // a completed deviceless decode resolving onto a torn-down State
        const state = new State();
        state.dispose();
        expect(state.disposed).toBe(true);

        // the guard returns before the device check, so this is a clean no-op even with no GPU (the disposed
        // State is dead regardless): an empty import, nothing registered, no throw, cache untouched
        const imp = await register(state, decoded);
        expect(imp.meshes).toEqual([]);
        expect(imp.instances).toEqual([]);
        expect(state.entities()).toHaveLength(0);
        expect(gltfCacheStats().assets).toBe(0);
    });
});
