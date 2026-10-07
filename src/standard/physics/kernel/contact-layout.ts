// Generated from Rust resident record offsets by build-kernel.ts.
export const DIR_STRIDE = 46;
export const DIR_COUNT = 19;
export const DIR_BLOCK = 18;
export const DIR_FLAGS = 17;
export const DIR_FRICTION = 35;
export const DIR_RESTITUTION = 40;
export const DIR_ROLLING_RESISTANCE = 41;
export const DIR_TANGENT_VELOCITY = 42;
export const MANIFOLD_STRIDE = 67;
export const M_NORMAL = 56;
export const M_TWIST = 59;
export const M_FRICTION = 60;
export const M_ROLLING = 63;
export const M_POINT_COUNT = 66;
export const M_POINTS = 0;
export const POINT_STRIDE = 14;
export const P_ANCHOR_A = 0;
export const P_ANCHOR_B = 3;
export const P_SEPARATION = 6;
export const P_BASE_SEPARATION = 7;
export const P_NORMAL_IMPULSE = 8;
export const P_TOTAL_NORMAL_IMPULSE = 9;
export const P_NORMAL_VELOCITY = 10;
export const P_FEATURE_ID = 11;
export const P_TRIANGLE_INDEX = 12;
export const P_PERSISTED = 13;
export const ContactField = {
    flags: 17,
    manifoldCount: 19,
    bodySimIndexA: 15,
    bodySimIndexB: 16,
    setIndex: 0,
    colorIndex: 1,
    localIndex: 2,
    bodyIdA: 3,
    prevKeyA: 4,
    nextKeyA: 5,
    bodyIdB: 6,
    prevKeyB: 7,
    nextKeyB: 8,
    shapeIdA: 9,
    shapeIdB: 10,
    childIndex: 11,
    islandId: 12,
    islandIndex: 13,
    contactId: 14,
    generation: 45,
} as const;
