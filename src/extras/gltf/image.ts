import { CompressedImageFormats, type ImageFormat, type ImageSource } from "../../core/rendering";
import type { BasisTarget } from "./basisu";
import type { GltfScene } from "./gltf";

/** Resolves a KTX2 image against its texture role, sampler and enabled device formats. */
export function basisImageSource(
    image: GltfScene["images"][number],
    isSrgb: boolean,
    mipmaps: boolean,
    read: typeof fetch = fetch,
): ImageSource {
    return async (formats) => {
        const { openBasisImage } = await import("./basisu");
        let bytes = image.bytes;
        if (!bytes) {
            const response = await read(image.uri!);
            if (!response.ok)
                throw new Error(`glTF image load failed: ${image.uri} (${response.status})`);
            bytes = new Uint8Array(await response.arrayBuffer());
        }
        const basis = openBasisImage(bytes);
        try {
            if (basis.transfer !== (isSrgb ? "srgb" : "linear"))
                throw new Error(
                    "KHR_texture_basisu: DFD transferFunction contradicts texture role",
                );
            let target: BasisTarget = "rgba8";
            let format: ImageFormat = "rgba8unorm";
            if (!(basis.levels === 1 && mipmaps)) {
                if (basis.model === "etc1s" && formats & CompressedImageFormats.ETC2) {
                    target = basis.hasAlpha ? "etc2-rgba" : "etc1";
                    format = basis.hasAlpha ? "etc2-rgba8unorm" : "etc2-rgb8unorm";
                } else if (basis.model === "uastc" && formats & CompressedImageFormats.ASTC_LDR) {
                    target = "astc";
                    format = "astc-4x4-unorm";
                } else if (formats & CompressedImageFormats.BC) {
                    target = "bc7";
                    format = "bc7-rgba-unorm";
                }
            }
            return {
                width: basis.width,
                height: basis.height,
                format,
                levels: Array.from({ length: basis.levels }, (_, level) =>
                    basis.transcode(level, target),
                ),
            };
        } finally {
            basis.dispose();
        }
    };
}
