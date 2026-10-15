import { u8, vec4 } from "../../engine";

// Membership marks the first captured placement, preventing interpolation from the origin.
// Camera history and warning state are derived, not authored.
export const PlayerFollow = {
    previous: vec4,
    current: vec4,
    missingCameraWarned: u8,
};
