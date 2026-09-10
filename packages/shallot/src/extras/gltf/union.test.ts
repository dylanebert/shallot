import { describe, expect, test } from "bun:test";
import type { Ktx2Image } from "./basis";
import type { GltfMaterial } from "./gltf";
import type { DecodedImage, DecodedMap, DecodedTextures } from "./textures";
import {
    type AlbedoDesc,
    planAlbedoBuckets,
    planUnion,
    type UnionAsset,
    uniformBlocks,
} from "./union";

// the pure union planner — bucketing the union of active assets'
// albedo into the shared size-bucketed arrays + rebasing every material's layer indices into them. The
// device assembly (beginUnion) is GPU-only, gated in the gym `render` `gltf-multi` mode; the rebase logic — where
// a multi-asset layout bug would hide — is pure and pinned here.

const bitmap = (size: string): AlbedoDesc => ({ kind: "bitmap", size });
const ktx = (size: string): AlbedoDesc => ({ kind: "compressed", size });

describe("planAlbedoBuckets", () => {
    test("all bitmaps collapse into one resized bucket", () => {
        const p = planAlbedoBuckets([bitmap("1024x1024"), bitmap("512x512"), bitmap("256x256")]);
        expect(p.buckets).toEqual(["bitmap"]);
        // every layer in bucket 0, in image order — the upload resizes them to a common size
        expect(p.loc).toEqual([
            { bucket: 0, layer: 0, spilled: false },
            { bucket: 0, layer: 1, spilled: false },
            { bucket: 0, layer: 2, spilled: false },
        ]);
    });

    test("compressed group by size, one bucket per distinct size", () => {
        const p = planAlbedoBuckets([ktx("1024x1024"), ktx("1024x1024"), ktx("512x512")]);
        expect(p.buckets).toEqual(["compressed", "compressed"]);
        // 1024 is the more populous size → bucket 0 (two layers); 512 → bucket 1
        expect(p.loc).toEqual([
            { bucket: 0, layer: 0, spilled: false },
            { bucket: 0, layer: 1, spilled: false },
            { bucket: 1, layer: 0, spilled: false },
        ]);
    });

    test("bitmap + compressed mix shares no bucket", () => {
        const p = planAlbedoBuckets([bitmap("64x64"), ktx("1024x1024"), ktx("512x512")]);
        expect(p.buckets).toEqual(["bitmap", "compressed", "compressed"]);
        expect(p.loc[0]).toEqual({ bucket: 0, layer: 0, spilled: false }); // bitmap → bucket 0
        // the two compressed sizes take their own dedicated buckets (never the bitmap bucket 0)
        expect(p.loc[1].bucket).toBeGreaterThan(0);
        expect(p.loc[2].bucket).toBeGreaterThan(0);
        expect(p.loc[1].bucket).not.toBe(p.loc[2].bucket);
    });

    test("over-cap distinct compressed sizes spill the rarest to an RGBA bitmap bucket", () => {
        // five distinct sizes, cap 4 → bucket 0 reserved for the spilled RGBA, 3 dedicated compressed buckets
        const p = planAlbedoBuckets([ktx("5x5"), ktx("4x4"), ktx("3x3"), ktx("2x2"), ktx("1x1")]);
        expect(p.buckets[0]).toBe("bitmap");
        expect(p.buckets.filter((b) => b === "compressed")).toHaveLength(3);
        const spilled = p.loc.filter((l) => l.spilled);
        expect(spilled).toHaveLength(2); // the two rarest sizes
        // a spilled image lands in the shared bitmap bucket (0), decoded to RGBA on upload
        expect(spilled.every((l) => l.bucket === 0)).toBe(true);
    });
});

// the data-map compressed-vs-RGBA gate: a slot takes the block-compressed array only when every layer is a
// uniform-size block (it can't resize), else the RGBA fallback. The mixed-size → null path is load-bearing —
// returning the blocks there would hand allocCompressed mismatched dimensions and corrupt the upload.
const block = (w: number, h: number, format = "bc5-rg-unorm"): DecodedImage => ({
    kind: "compressed",
    image: { width: w, height: h, format, blockDim: 4, mips: [] } as unknown as Ktx2Image,
    bytes: new Uint8Array(),
});
const bmp = (): DecodedImage => ({ kind: "bitmap", bitmap: {} as ImageBitmap });

describe("uniformBlocks", () => {
    test("all compressed + uniform size → the block layers (the compressed array path)", () => {
        const r = uniformBlocks([block(1024, 1024), block(1024, 1024)]);
        expect(r).toHaveLength(2);
    });

    test("any bitmap layer → null (a mixed-source slot falls back to RGBA)", () => {
        expect(uniformBlocks([block(1024, 1024), bmp()])).toBeNull();
    });

    test("compressed but mismatched size → null (can't share one compressed array)", () => {
        expect(uniformBlocks([block(1024, 1024), block(512, 512)])).toBeNull();
    });
});

// build a fake decoded-textures payload: albedo images sized for bucketing (the ImageBitmap/Ktx2Image only
// need width/height here), per-material albedo refs, and one mr-map slot. The other slots are empty.
function textures(opts: {
    albedo: AlbedoDesc[];
    albedoRef: number[];
    mrImages: number;
    mrLayer: number[];
}): DecodedTextures {
    const albedo: DecodedImage[] = opts.albedo.map((d) => {
        const [w, h] = d.size.split("x").map(Number);
        return d.kind === "bitmap"
            ? { kind: "bitmap", bitmap: { width: w, height: h } as ImageBitmap }
            : {
                  kind: "compressed",
                  image: { width: w, height: h } as Ktx2Image,
                  bytes: new Uint8Array(),
              };
    });
    const empty: DecodedMap = { images: [], layer: Int32Array.from(opts.albedoRef, () => -1) };
    return {
        albedo,
        albedoRef: Int32Array.from(opts.albedoRef),
        maps: {
            mr: {
                images: Array.from(
                    { length: opts.mrImages },
                    () => ({ kind: "bitmap", bitmap: {} as ImageBitmap }) as DecodedImage,
                ),
                layer: Int32Array.from(opts.mrLayer),
            },
            normalTex: empty,
            occlusion: empty,
            emissive: empty,
        },
        textured: true,
    };
}

describe("planUnion", () => {
    test("two assets concatenate into one palette, layers rebased to the shared arrays", () => {
        const mats = (n: number): GltfMaterial[] =>
            Array.from({ length: n }, () => ({}) as GltfMaterial);
        // asset 0: mat0 has albedo image 0 + mr image 0; mat1 is factor-only
        const a0: UnionAsset = {
            textures: textures({
                albedo: [bitmap("1024x1024")],
                albedoRef: [0, -1],
                mrImages: 1,
                mrLayer: [0, -1],
            }),
            materials: mats(2),
            base: 0,
        };
        // asset 1: mat0 has albedo image 0 + mr image 0 (its own, local layer 0)
        const a1: UnionAsset = {
            textures: textures({
                albedo: [bitmap("1024x1024")],
                albedoRef: [0],
                mrImages: 1,
                mrLayer: [0],
            }),
            materials: mats(1),
            base: 2,
        };
        const u = planUnion([a0, a1], 3);

        // both albedo images are bitmaps → one shared bucket, layer 0 (asset 0) then layer 1 (asset 1)
        expect(u.albedo.buckets).toEqual(["bitmap"]);
        expect([...u.layers.albedo]).toEqual([0, -1, 1]);
        expect([...u.layers.albedoBucket]).toEqual([0, 0, 0]);
        // mr concatenates per slot: asset 0's layer 0 stays 0, asset 1's local layer 0 rebases to 1
        expect([...u.layers.mr]).toEqual([0, -1, 1]);
        // every material slot is filled, asset 1's at base 2
        expect(u.materials).toHaveLength(3);
        expect(u.materials[2]).toBe(a1.materials[0]);
    });
});
