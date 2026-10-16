import { component, vec2 } from "../../engine";

/**
 * Normalized pivot of a sprite or text block relative to its Transform.
 * Coordinates run from −0.5 (left/bottom) to 0.5 (right/top); zero is the centre.
 * Without Anchor, sprites are centred and text uses its bottom-left corner.
 */
export const Anchor = Object.assign(
    component("Anchor", { offset: vec2 }, { defaults: () => ({ offset: [0, 0] }) }),
    {
        BOTTOM_LEFT: [-0.5, -0.5] as const,
        BOTTOM_CENTER: [0, -0.5] as const,
        BOTTOM_RIGHT: [0.5, -0.5] as const,
        CENTER_LEFT: [-0.5, 0] as const,
        CENTER: [0, 0] as const,
        CENTER_RIGHT: [0.5, 0] as const,
        TOP_LEFT: [-0.5, 0.5] as const,
        TOP_CENTER: [0, 0.5] as const,
        TOP_RIGHT: [0.5, 0.5] as const,
    },
);
