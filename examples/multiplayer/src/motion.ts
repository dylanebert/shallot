import { component, f32, u32, vec4 } from "@dylanebert/shallot";

/** Fixed-tick movement history, including the latest jump press count consumed. */
export const PlayerMotion = component("PlayerMotion", {
    carry: vec4,
    coyote: f32,
    buffer: f32,
    lastJumpPresses: u32,
});
