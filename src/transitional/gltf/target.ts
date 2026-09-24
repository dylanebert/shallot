// the compressed-texture transcode targets — what to transcode a KTX2/Basis image to, picked from the
// device's compression features. Lives apart from basis.ts so the main thread can resolve the targets
// (a feature read, no codec) without pulling the ~large Basis transcoder glue: basis.ts is reached only via
// dynamic import, keeping the codec code-split, and the pickers must stay on that clean side of the cut.

/** the WebGPU texture-compression families the KTX2 transcode can target, in preference order. Also
 *  `GltfPlugin.preferredFeatures` — a feature the device never requested reads false in
 *  {@link pickTargets}' `device.features` branch, so an app importing KTX2 assets needs that plugin in
 *  its list for a capable device to expose any family here. */
export const COMPRESSION_FAMILIES: readonly GPUFeatureName[] = [
    "texture-compression-bc",
    "texture-compression-etc2",
    "texture-compression-astc",
];

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

// the compression family the device exposes, or undefined where it exposes none. Compression is not on
// the base floor: a device has *requested* a family only when `GltfPlugin` (which declares all three as
// `preferredFeatures`) is in the plugin list, so this read is false for every family in an app that
// imports glTF through `loadGltf` alone. Total on purpose — an import resolves targets before it knows
// whether the asset carries a KTX2 image at all, so the gate belongs at the image (`assets.ts`).
function family(device: GPUDevice): "bc" | "etc2" | "astc" | undefined {
    if (device.features.has("texture-compression-bc")) return "bc";
    if (device.features.has("texture-compression-etc2")) return "etc2";
    if (device.features.has("texture-compression-astc")) return "astc";
    return undefined;
}

/**
 * the per-slot transcode targets for a device, role-aware per the family — or `undefined` where the device
 * exposes no texture-compression family at all (nothing to transcode *to*). Total, never throwing: an import
 * resolves targets up front, before it knows whether the asset even carries a KTX2 image, and a geometry-only
 * or PNG-textured asset needs none. A KTX2 image handed no target is what fails, at the image, naming the
 * three families (`assets.ts`). On a BC device the data maps take their tightest block (Bevy
 * `get_transcoded_formats`: normal/two-channel → BC5, AO/single → BC4, color → BC7); on ETC2/ASTC every slot
 * uses the family's color block (linear or sRGB), which still carries the channels a two-channel form would
 * (the normal FS reconstructs Z from XY regardless). Resolve once on the main thread and thread into the
 * deviceless {@link decode} (the worker has no device).
 *
 * @example
 * const decoded = await decode("sponza/Sponza.gltf", { targets: pickTargets(device) });
 */
export function pickTargets(device: GPUDevice): Targets | undefined {
    const f = family(device);
    if (f === undefined) return undefined;
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
