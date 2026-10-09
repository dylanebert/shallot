import { component, f32, u32, vec4 } from "../../engine";

/** Fixed-tick movement history, including the latest jump press count consumed. */
export const PlayerMotion = component("PlayerMotion", {
    carry: vec4,
    coyote: f32,
    buffer: f32,
    lastJumpPresses: u32,
});
