import { describe, expect, test } from "bun:test";
import { type Ktx2Image, loadBasis, transcodeKtx2, transcodeKtx2Rgba } from "./basis";

// a small ETC1S KTX2 fixture (three.js sample texture) — exercises the Basis transcode without a GPU. The
// GPU upload (compressed texture_2d_array) is real-GPU, gated by the gym `render` `gltf-model` ktx variant.
const dir = `${import.meta.dir}/fixtures`;

// BC7 target — every block is 16 bytes over a 4×4 texel tile, so a transcoded level holds exactly
// ceil(w/4)·ceil(h/4)·16 bytes. Deriving the expected size from the dimensions (not a magic number) is the
// invariant the transcode must satisfy.
const BC7: { basis: number; gpu: GPUTextureFormat; blockDim: number } = {
    basis: 7,
    gpu: "bc7-rgba-unorm-srgb",
    blockDim: 4,
};
// the data-map role targets pickTargets returns on a BC device: normal → BC5 (two-channel, 16 B/block),
// AO → BC4 (single-channel, 8 B/block). Both transcode from the same ETC1S source — the role picks the format.
const BC5: { basis: number; gpu: GPUTextureFormat; blockDim: number } = {
    basis: 5,
    gpu: "bc5-rg-unorm",
    blockDim: 4,
};
const BC4: { basis: number; gpu: GPUTextureFormat; blockDim: number } = {
    basis: 4,
    gpu: "bc4-r-unorm",
    blockDim: 4,
};

function blockBytes(width: number, height: number, blockDim: number, bytesPerBlock = 16): number {
    return Math.ceil(width / blockDim) * Math.ceil(height / blockDim) * bytesPerBlock;
}

describe("basis transcode", () => {
    test("transcodes a KTX2/ETC1S file to a BC7 mip chain with block-exact sizes", async () => {
        await loadBasis();
        const bytes = new Uint8Array(await Bun.file(`${dir}/box-etc1s.ktx2`).arrayBuffer());
        const image: Ktx2Image = transcodeKtx2(bytes, BC7);

        expect(image.format).toBe("bc7-rgba-unorm-srgb");
        expect(image.width).toBeGreaterThanOrEqual(4);
        expect(image.mips.length).toBeGreaterThan(0);

        // mip 0 matches the image dimensions; each level's byte length is the block math exactly
        expect(image.mips[0].width).toBe(image.width);
        expect(image.mips[0].height).toBe(image.height);
        for (const mip of image.mips) {
            expect(mip.width).toBeGreaterThanOrEqual(4);
            expect(mip.height).toBeGreaterThanOrEqual(4);
            expect(mip.data.length).toBe(blockBytes(mip.width, mip.height, 4));
        }

        // the chain halves each level and stops at the 4×4 block floor (no 2×2 / 1×1 partial-block mips)
        for (let i = 1; i < image.mips.length; i++) {
            expect(image.mips[i].width).toBe(Math.max(4, image.mips[i - 1].width >> 1));
        }
    });

    test("transcodes a data-map normal to a BC5 two-channel mip chain (16 B/block)", async () => {
        await loadBasis();
        const bytes = new Uint8Array(await Bun.file(`${dir}/box-etc1s.ktx2`).arrayBuffer());
        const image: Ktx2Image = transcodeKtx2(bytes, BC5);

        expect(image.format).toBe("bc5-rg-unorm");
        expect(image.mips.length).toBeGreaterThan(0);
        // BC5 is two BC4 blocks → 16 B per 4×4 tile, same block math as BC7
        for (const mip of image.mips) {
            expect(mip.data.length).toBe(blockBytes(mip.width, mip.height, 4, 16));
        }
    });

    test("transcodes a data-map AO to a BC4 single-channel mip chain (8 B/block)", async () => {
        await loadBasis();
        const bytes = new Uint8Array(await Bun.file(`${dir}/box-etc1s.ktx2`).arrayBuffer());
        const image: Ktx2Image = transcodeKtx2(bytes, BC4);

        expect(image.format).toBe("bc4-r-unorm");
        expect(image.mips.length).toBeGreaterThan(0);
        // BC4 is one block → 8 B per 4×4 tile (half BC7/BC5)
        for (const mip of image.mips) {
            expect(mip.data.length).toBe(blockBytes(mip.width, mip.height, 4, 8));
        }
    });

    test("transcodes the base level to RGBA8 — the mixed-size fallback path", async () => {
        await loadBasis();
        const bytes = new Uint8Array(await Bun.file(`${dir}/box-etc1s.ktx2`).arrayBuffer());
        const { width, height, rgba } = transcodeKtx2Rgba(bytes);
        // raw RGBA8 is exactly 4 bytes per texel — the invariant the resize path relies on (ImageData)
        expect(rgba.length).toBe(width * height * 4);
    });

    test("an invalid KTX2 file throws, not a silent unbindable texture", async () => {
        await loadBasis();
        // the openKtx2 guard (basis.ts): non-KTX2 bytes fail one of its checks (isValid / startTranscoding)
        // and throw loud at the decode boundary, rather than transcoding garbage the GPU can't bind
        expect(() => transcodeKtx2(new Uint8Array([0, 1, 2, 3]), BC7)).toThrow(/KTX2/);
    });
});
