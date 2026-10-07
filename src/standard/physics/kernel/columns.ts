// Body columns (body.rs + bodies.rs). All resident in the persistent body region — the awake
// `BodySim`/`BodyState` are offset-backed views over them (bodycolumns.ts), so no per-step marshal.
export const STATE_STRIDE = 14;
/** Velocity/delta fields; flags occupy word 13 in each resident state. */
export const STATE_LIVE = 13;
export const SIM_STRIDE = 54;
/** Legacy binding slots address the same resident b3BodySim array. */
export const FIN_STRIDE = SIM_STRIDE;
export const SIM2_STRIDE = SIM_STRIDE;
/** Retained body-move bridge: body index, generation, fellAsleep. */
export const MOVE_STRIDE = 11;
// sim2 field offsets.
export const S2_CENTER0 = 14;
export const S2_MIN_EXTENT = 45;
export const S2_BODY_ID = 52;
export const S2_FLAGS = 53;
export * from "./joint-layout";
