import { Billboard } from "../../core/rendering";
import { component, f32, u32, vec2 } from "../../engine";
import { AlphaModeCode } from "../../standard/rendering";

/** Sprite alpha as [AlphaModeCode, cutoff], stored in Sprite.alphaMode's two lanes. */
export const SpriteAlphaMode = {
    /** Alpha-tested cutout; writes depth and casts holed shadows. Threshold is in 0..1. */
    Mask: (threshold = 0.5): readonly [number, number] => {
        if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1)
            throw new RangeError("SpriteAlphaMode.Mask threshold must be between 0 and 1");
        return [AlphaModeCode.Mask, threshold];
    },
    /** Translucent; blends over the opaque scene and casts no shadows. */
    Blend: [AlphaModeCode.Blend, 0.5] as const,
} as const;

/** which portion of a sprite's image shows, for progress rings and gauges */
export const SpriteFill = {
    /** the whole image (the default) */
    None: 0,
    /** clockwise wedge from 12 o'clock: progress rings */
    Radial: 1,
    /** bottom-up: tanks, vertical gauges */
    Vertical: 2,
    /** left-to-right: bars */
    Horizontal: 3,
} as const;

/** A textured world-space quad centred on its Transform unless it has an Anchor. Remove Sprite to hide it; add it again to show it. */
export const Sprite = component(
    "Sprite",
    {
        /** registered image id (see registerImage) */
        image: u32,
        /** quad size in world units, before the transform's scale */
        size: vec2,
        /** hex sRGB tint multiplied into the texture */
        color: f32,
        /** texture-alpha multiplier; it shrinks the mask or fades the blended sprite */
        opacity: f32,
        /** {@link Billboard} orientation; defaults to Screen. */
        billboard: u32,
        /** {@link SpriteAlphaMode} tuple; defaults to Mask(0.5). */
        alphaMode: vec2,
        /** leading fraction of the image shown, 0..1 */
        fillAmount: f32,
        /** {@link SpriteFill} direction; None shows the whole image. */
        fillMethod: u32,
    },
    {
        defaults: () => ({
            image: 0,
            size: [1, 1],
            color: 0xffffff,
            opacity: 1,
            billboard: Billboard.Screen,
            alphaMode: SpriteAlphaMode.Mask(),
            fillAmount: 1,
            fillMethod: SpriteFill.None,
        }),
    },
);
