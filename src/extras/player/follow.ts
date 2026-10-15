import { u8, vec4 } from "../../engine";

// Membership marks the first captured placement, preventing interpolation from the origin.
// Derived camera history and warning state are not authored or serialized.
export const PlayerFollow = {
    previous: vec4,
    current: vec4,
    missingCameraWarned: u8,
};
