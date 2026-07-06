// the compressed-texture transcode targets — what to transcode a KTX2/Basis image to, picked from the
// device's compression features. Lives apart from basis.ts so the main thread can resolve the targets
// (a feature read, no codec) without pulling the ~large Basis transcoder glue: basis.ts is reached only via
// dynamic import, keeping the codec code-split, and the pickers must stay on that clean side of the cut.

// the Basis TranscoderFormat enum values we target (basis_transcoder.js KTX2File, the three.js-vendored build).
// BC7_M5 / ETC2 / ASTC_4×4 are the 16-byte/4×4 color blocks; BC5_RG (two-channel, 16 B) and BC4_R (single-
// channel, 8 B) are the data-map formats (gltf normal / occlusion). All are 4×4-block, so the block-row math in
// writeCompressedLayer is uniform — only the per-block byte size differs (derived from the transcoded length).
const ETC2 = 1;
const BC4_R = 4;
const BC5_RG = 5;
const BC7_M5 = 7;
const ASTC_4X4 = 10;

/** the chosen transcode destination: a Basis format enum paired with its WebGPU compressed format. `blockDim`
 *  is the block edge in texels (4 for every BC/ETC2/ASTC target), so a row holds `ceil(width/4)` blocks. */
export interface TranscodeTarget {
    basis: number;
    gpu: GPUTextureFormat;
    blockDim: number;
}

/** one transcode target per texture slot: the color baseColor (sRGB) plus the four data maps, each resolved to
 *  the role's block format. `mr`/`normalTex`/`occlusion` are linear, `emissive` + `albedo` sRGB; on a BC device
 *  the data maps specialize (normal → BC5 two-channel, occlusion → BC4 single-channel, mr → BC7), on ETC2/ASTC
 *  every slot rides the family's one color format (the device-validatable path is BC; the EAC/ASTC two-channel
 *  forms are unexercised on this hardware, so the conservative color block carries the channels instead). */
export interface Targets {
    albedo: TranscodeTarget;
    mr: TranscodeTarget;
    normalTex: TranscodeTarget;
    occlusion: TranscodeTarget;
    emissive: TranscodeTarget;
}

// the compression family the device exposes — the base floor guarantees exactly one (`gpu.md`).
function family(device: GPUDevice): "bc" | "etc2" | "astc" {
    if (device.features.has("texture-compression-bc")) return "bc";
    if (device.features.has("texture-compression-etc2")) return "etc2";
    if (device.features.has("texture-compression-astc")) return "astc";
    throw new Error("[gltf] no texture-compression feature for KTX2 transcode");
}

/**
 * the per-slot transcode targets for a device, role-aware per the family. On a BC device the data maps take
 * their tightest block (Bevy `get_transcoded_formats`: normal/two-channel → BC5, AO/single → BC4, color → BC7);
 * on ETC2/ASTC every slot uses the family's color block (linear or sRGB), which still carries the channels a
 * two-channel form would (the normal FS reconstructs Z from XY regardless). Resolve once on the main thread and
 * thread into the deviceless {@link decode} (the worker has no device).
 *
 * @example
 * const decoded = await decode("sponza/Sponza.gltf", { targets: pickTargets(device) });
 */
export function pickTargets(device: GPUDevice): Targets {
    const f = family(device);
    if (f === "bc") {
        return {
            albedo: { basis: BC7_M5, gpu: "bc7-rgba-unorm-srgb", blockDim: 4 },
            mr: { basis: BC7_M5, gpu: "bc7-rgba-unorm", blockDim: 4 },
            normalTex: { basis: BC5_RG, gpu: "bc5-rg-unorm", blockDim: 4 },
            occlusion: { basis: BC4_R, gpu: "bc4-r-unorm", blockDim: 4 },
            emissive: { basis: BC7_M5, gpu: "bc7-rgba-unorm-srgb", blockDim: 4 },
        };
    }
    const [basis, srgb, linear]: [number, GPUTextureFormat, GPUTextureFormat] =
        f === "etc2"
            ? [ETC2, "etc2-rgba8unorm-srgb", "etc2-rgba8unorm"]
            : [ASTC_4X4, "astc-4x4-unorm-srgb", "astc-4x4-unorm"];
    return {
        albedo: { basis, gpu: srgb, blockDim: 4 },
        mr: { basis, gpu: linear, blockDim: 4 },
        normalTex: { basis, gpu: linear, blockDim: 4 },
        occlusion: { basis, gpu: linear, blockDim: 4 },
        emissive: { basis, gpu: srgb, blockDim: 4 },
    };
}
