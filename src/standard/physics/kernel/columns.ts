// Body columns (body.rs + bodies.rs). All resident in the persistent body region — the awake
// `BodySim`/`BodyState` are offset-backed views over them (bodycolumns.ts), so no per-step marshal.
export const STATE_STRIDE = 14;
/** Velocity/delta fields; flags occupy word 13 in each resident state. */
export const STATE_LIVE = 13;
export const SIM_STRIDE = 54;
// b3BodySim field offsets (body.rs).
export const CENTER0 = 14;
export const MIN_EXTENT = 45;
export const BODY_ID = 52;
export const FLAGS = 53;
export * from "./joint-layout";
