import { component, f32, u32, vec2 } from "../../engine";

/** how a sprite quad orients toward the camera */
export const SpriteBillboard = {
    /** camera-plane aligned (the default for icons) */
    Screen: 0,
    /** upright, yawing toward the viewer (foliage, standees) */
    YLocked: 1,
    /** plain transform: the quad lives in the entity's local xy plane (decals, ground markers) */
    World: 2,
} as const;

/** Sprite material alpha mode: Clip maps to Mask(0.5), Alpha maps to Blend. */
export const SpriteBlend = {
    /** alpha-tested cutout at 0.5: depth-written and casts holed shadows (the default) */
    Clip: 0,
    /** translucent: blended over the opaque scene and casts no shadows */
    Alpha: 1,
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

/** A textured world-space quad anchored to an entity's Transform. */
export const Sprite = component(
    "Sprite",
    {
        /** registered image id (see registerImage) */
        image: u32,
        /** quad size in world units, before the transform's scale */
        size: vec2,
        /** 0..1 pivot within the quad; 0.5 0.5 centers, 0.5 0 pins the bottom edge to the entity */
        anchor: vec2,
        /** hex sRGB tint multiplied into the texture */
        color: f32,
        /** texture-alpha multiplier; it shrinks the mask or fades the blended sprite */
        opacity: f32,
        /** drawn when nonzero */
        visible: f32,
        /** billboard orientation */
        billboard: u32,
        /** compositing route */
        blend: u32,
        /** leading fraction of the image shown, 0..1 */
        fill: f32,
        /** fill direction */
        fillMode: u32,
    },
    {
        defaults: () => ({
            image: 0,
            size: [1, 1],
            anchor: [0.5, 0.5],
            color: 0xffffff,
            opacity: 1,
            visible: 1,
            billboard: SpriteBillboard.Screen,
            blend: SpriteBlend.Clip,
            fill: 1,
            fillMode: SpriteFill.None,
        }),
    },
);

/** Encode fill amount and direction in the Sprite material row. */
export function packSpriteFill(amount: number, mode: number): number {
    const a = Math.round(Math.min(1, Math.max(0, amount)) * 0xffff);
    return ((mode & 0xffff) << 16) | a;
}
