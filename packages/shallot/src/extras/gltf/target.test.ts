import { describe, expect, test } from "bun:test";
import { GltfPlugin } from "./assets";
import { pickTargets } from "./target";

// pickTargets is the device→per-slot-compressed-format boundary (a feature read, no codec, resolved
// main-thread-side per the deviceless-decode contract). A pure read over `device.features`, so a feature-set
// stub exercises every family + the throw with no GPU — the kind of CPU↔GPU format decision `testing.md`
// always pins. On a BC device the data maps specialize (normal → BC5 two-channel, AO → BC4 single-channel, mr
// → BC7); ETC2/ASTC use the family's one color block per slot (the EAC/ASTC two-channel forms are unexercised
// on this hardware, so the conservative color block carries the channels — the normal FS reconstructs Z from
// XY regardless). The color slots (albedo/emissive) stay sRGB, the data slots linear.
function device(...features: string[]): GPUDevice {
    const set = new Set(features);
    return { features: { has: (f: string) => set.has(f) } } as unknown as GPUDevice;
}

describe("pickTargets", () => {
    test("a BC device specializes the data maps (BC5 normal, BC4 AO, BC7 color)", () => {
        expect(pickTargets(device("texture-compression-bc"))).toEqual({
            albedo: { basis: 7, gpu: "bc7-rgba-unorm-srgb", blockDim: 4 },
            mr: { basis: 7, gpu: "bc7-rgba-unorm", blockDim: 4 },
            normalTex: { basis: 5, gpu: "bc5-rg-unorm", blockDim: 4 },
            occlusion: { basis: 4, gpu: "bc4-r-unorm", blockDim: 4 },
            emissive: { basis: 7, gpu: "bc7-rgba-unorm-srgb", blockDim: 4 },
        });
    });

    test("an ETC2 device uses one color block per slot (sRGB color, linear data)", () => {
        expect(pickTargets(device("texture-compression-etc2"))).toEqual({
            albedo: { basis: 1, gpu: "etc2-rgba8unorm-srgb", blockDim: 4 },
            mr: { basis: 1, gpu: "etc2-rgba8unorm", blockDim: 4 },
            normalTex: { basis: 1, gpu: "etc2-rgba8unorm", blockDim: 4 },
            occlusion: { basis: 1, gpu: "etc2-rgba8unorm", blockDim: 4 },
            emissive: { basis: 1, gpu: "etc2-rgba8unorm-srgb", blockDim: 4 },
        });
    });

    test("an ASTC device uses one color block per slot (sRGB color, linear data)", () => {
        expect(pickTargets(device("texture-compression-astc"))).toEqual({
            albedo: { basis: 10, gpu: "astc-4x4-unorm-srgb", blockDim: 4 },
            mr: { basis: 10, gpu: "astc-4x4-unorm", blockDim: 4 },
            normalTex: { basis: 10, gpu: "astc-4x4-unorm", blockDim: 4 },
            occlusion: { basis: 10, gpu: "astc-4x4-unorm", blockDim: 4 },
            emissive: { basis: 10, gpu: "astc-4x4-unorm-srgb", blockDim: 4 },
        });
    });

    test("BC wins when several families are present (priority order)", () => {
        const t = pickTargets(
            device(
                "texture-compression-astc",
                "texture-compression-etc2",
                "texture-compression-bc",
            ),
        );
        expect(t?.normalTex.gpu).toBe("bc5-rg-unorm");
    });

    test("every family the transcode branches on is one GltfPlugin requests", () => {
        // the trap `preferredFeatures` exists for: `family()` reads `device.features.has`, false for a
        // feature the device never *requested*. A family this resolves but the plugin doesn't request
        // reads false on hardware that supports it, and the asset fails to import — so the branches are
        // pinned against `GltfPlugin.preferredFeatures` itself, over the names written out here
        // independently of both. Asserting against COMPRESSION_FAMILIES would prove nothing: it *is*
        // what the plugin declares, so a family swapped out of that list would stay green on both sides.
        const branches: GPUFeatureName[] = [
            "texture-compression-bc",
            "texture-compression-etc2",
            "texture-compression-astc",
        ];
        for (const f of branches) {
            expect(pickTargets(device(f))?.albedo.blockDim).toBe(4);
            expect(GltfPlugin.preferredFeatures).toContain(f);
        }
    });

    test("no texture-compression feature resolves to no targets, never a throw", () => {
        // total on purpose: an import resolves targets before it knows whether the asset carries a KTX2
        // image, so a device with no family must still import geometry. The gate is per-image, in
        // `assets.ts` — `decode.test.ts` "a device with no texture-compression family" owns that pair.
        expect(pickTargets(device())).toBeUndefined();
    });
});
