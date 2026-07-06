import { sparse, u8, vec4 } from "../../engine";

// PlayerFollow holds the camera-follow interpolation state — the two most recent fixed-tick swept poses (the
// camera renders lerp(prev, curr, fixedAlpha)) — plus the once-only missing-camera warn latch. Added on a
// player's first snapshot; that membership doubles as the "first pose captured" flag, so a fresh player
// starts at its spawn pose, not the origin. Derived state: never authored or serialized, re-created on every
// rebuild, so a reload can't desync it from the Body. Internal — a sibling for the systems, never on the barrel.
export const PlayerFollow = {
    prev: sparse(vec4),
    curr: sparse(vec4),
    warned: sparse(u8),
};
