// Body columns (body.rs + bodies.rs). All resident in the persistent body region — the awake
// `BodySim`/`BodyState` are offset-backed views over them (bodycolumns.ts), so no per-step marshal.
export const STATE_STRIDE = 16;
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
// Joint record (kernel/src/joint_abi.rs). One flat f32 record per joint slot: a common header (the
// state indices and body ids via u32 bits, cached invMass/invInertia, the local frames,
// the base constraint frequency + softness) then a per-type payload (distance's
// config, its persistent impulses, and prepare's scratch). Mirror of `joint_abi.rs` — keep in sync.
export const J_FORCE_THRESHOLD = 126;
export const J_TORQUE_THRESHOLD = 127;
export const J_EVENT = 128;
export const JOINT_STRIDE = 130;
export const J_JOINT_ID = 129;
export const J_LOCAL_FRAME_A = 43; // Transform p 43..45 q 46..49
export const J_LOCAL_FRAME_B = 50; // Transform p 50..52 q 53..56
export const J_CONSTRAINT_HERTZ = 57;
export const J_CONSTRAINT_DAMPING = 58;
const J_PAYLOAD = 62;
export const DJ_LENGTH = J_PAYLOAD;
export const DJ_HERTZ = J_PAYLOAD + 1;
export const DJ_DAMPING_RATIO = J_PAYLOAD + 2;
export const DJ_LOWER_SPRING_FORCE = J_PAYLOAD + 3;
export const DJ_UPPER_SPRING_FORCE = J_PAYLOAD + 4;
export const DJ_MIN_LENGTH = J_PAYLOAD + 5;
export const DJ_MAX_LENGTH = J_PAYLOAD + 6;
export const DJ_MAX_MOTOR_FORCE = J_PAYLOAD + 7;
export const DJ_MOTOR_SPEED = J_PAYLOAD + 8;
export const DJ_ENABLE = J_PAYLOAD + 9;
export const DJ_IMPULSE = J_PAYLOAD + 10;
export const DJ_MOTOR_IMPULSE = J_PAYLOAD + 13;
export const DJ_ENABLE_SPRING = 0x1;
export const DJ_ENABLE_LIMIT = 0x2;
export const DJ_ENABLE_MOTOR = 0x4;

// Weld-joint payload (joint_abi.rs weld section). AppConfig, the two persistent impulses, then prepare's
// scratch (frames, angular mass, resolved softnesses, fixedRotation). Marshal writes config + impulses.
export const WJ_LINEAR_HERTZ = J_PAYLOAD;
export const WJ_LINEAR_DAMPING_RATIO = J_PAYLOAD + 1;
export const WJ_ANGULAR_HERTZ = J_PAYLOAD + 2;
export const WJ_ANGULAR_DAMPING_RATIO = J_PAYLOAD + 3;
export const WJ_LINEAR_IMPULSE = J_PAYLOAD + 4; // vec3

// Revolute-joint payload (joint_abi.rs revolute section). AppConfig, the persistent impulses (linear vec3 +
// perp vec2 + four scalar), then prepare's scratch. Marshal writes config + impulses.
export const RJ_HERTZ = J_PAYLOAD;
export const RJ_DAMPING_RATIO = J_PAYLOAD + 1;
export const RJ_MAX_MOTOR_TORQUE = J_PAYLOAD + 2;
export const RJ_MOTOR_SPEED = J_PAYLOAD + 3;
export const RJ_TARGET_ANGLE = J_PAYLOAD + 4;
export const RJ_LOWER_ANGLE = J_PAYLOAD + 5;
export const RJ_UPPER_ANGLE = J_PAYLOAD + 6;
export const RJ_ENABLE = J_PAYLOAD + 7;
export const RJ_LINEAR_IMPULSE = J_PAYLOAD + 8; // vec3
export const RJ_MOTOR_IMPULSE = J_PAYLOAD + 14;
export const RJ_ENABLE_SPRING = 0x1;
export const RJ_ENABLE_MOTOR = 0x2;
export const RJ_ENABLE_LIMIT = 0x4;

// Spherical-joint payload (joint_abi.rs spherical section). AppConfig, the persistent impulses (three vec3 +
// three scalar), then prepare's scratch. Marshal writes config + impulses.
export const SJ_HERTZ = J_PAYLOAD;
export const SJ_DAMPING_RATIO = J_PAYLOAD + 1;
export const SJ_MAX_MOTOR_TORQUE = J_PAYLOAD + 2;
export const SJ_MOTOR_VELOCITY = J_PAYLOAD + 3; // vec3
export const SJ_LOWER_TWIST_ANGLE = J_PAYLOAD + 6;
export const SJ_UPPER_TWIST_ANGLE = J_PAYLOAD + 7;
export const SJ_CONE_ANGLE = J_PAYLOAD + 8;
export const SJ_TARGET_ROTATION = J_PAYLOAD + 9; // quat
export const SJ_ENABLE = J_PAYLOAD + 13;
export const SJ_LINEAR_IMPULSE = J_PAYLOAD + 14; // vec3
export const SJ_MOTOR_IMPULSE = J_PAYLOAD + 20; // vec3
export const SJ_ENABLE_SPRING = 0x1;
export const SJ_ENABLE_MOTOR = 0x2;
export const SJ_ENABLE_CONE_LIMIT = 0x4;
export const SJ_ENABLE_TWIST_LIMIT = 0x8;

// Prismatic-joint payload (joint_abi.rs prismatic section). AppConfig, the persistent impulses (perp vec2 +
// angular vec3 + four scalar), then prepare's scratch. Marshal writes config + impulses.
export const PJ_HERTZ = J_PAYLOAD;
export const PJ_DAMPING_RATIO = J_PAYLOAD + 1;
export const PJ_MAX_MOTOR_FORCE = J_PAYLOAD + 2;
export const PJ_MOTOR_SPEED = J_PAYLOAD + 3;
export const PJ_TARGET_TRANSLATION = J_PAYLOAD + 4;
export const PJ_LOWER_TRANSLATION = J_PAYLOAD + 5;
export const PJ_UPPER_TRANSLATION = J_PAYLOAD + 6;
export const PJ_ENABLE = J_PAYLOAD + 7;
export const PJ_PERP_IMPULSE = J_PAYLOAD + 8; // vec2
export const PJ_MOTOR_IMPULSE = J_PAYLOAD + 14;
export const PJ_ENABLE_SPRING = 0x1;
export const PJ_ENABLE_MOTOR = 0x2;
export const PJ_ENABLE_LIMIT = 0x4;

// Wheel-joint payload (joint_abi.rs wheel section). AppConfig, the persistent impulses (two vec2 + seven
// scalar), then prepare's scratch. Marshal writes config + impulses.
export const WHJ_MAX_SPIN_TORQUE = J_PAYLOAD;
export const WHJ_SPIN_SPEED = J_PAYLOAD + 1;
export const WHJ_LOWER_SUSPENSION_LIMIT = J_PAYLOAD + 2;
export const WHJ_UPPER_SUSPENSION_LIMIT = J_PAYLOAD + 3;
export const WHJ_SUSPENSION_HERTZ = J_PAYLOAD + 4;
export const WHJ_SUSPENSION_DAMPING_RATIO = J_PAYLOAD + 5;
export const WHJ_LOWER_STEERING_LIMIT = J_PAYLOAD + 6;
export const WHJ_UPPER_STEERING_LIMIT = J_PAYLOAD + 7;
export const WHJ_TARGET_STEERING_ANGLE = J_PAYLOAD + 8;
export const WHJ_MAX_STEERING_TORQUE = J_PAYLOAD + 9;
export const WHJ_STEERING_HERTZ = J_PAYLOAD + 10;
export const WHJ_STEERING_DAMPING_RATIO = J_PAYLOAD + 11;
export const WHJ_ENABLE = J_PAYLOAD + 12;
export const WHJ_LINEAR_IMPULSE = J_PAYLOAD + 13; // vec2
export const WHJ_SPIN_IMPULSE = J_PAYLOAD + 17;
export const WHJ_STEERING_SPRING_IMPULSE = J_PAYLOAD + 21;
export const WHJ_ENABLE_SPIN_MOTOR = 0x1;
export const WHJ_ENABLE_SUSPENSION_SPRING = 0x2;
export const WHJ_ENABLE_SUSPENSION_LIMIT = 0x4;
export const WHJ_ENABLE_STEERING = 0x8;
export const WHJ_ENABLE_STEERING_LIMIT = 0x10;

// Motor-joint payload (joint_abi.rs motor section). AppConfig, the four persistent vec3 impulses, then
// prepare's scratch. Marshal writes config + impulses (no enable bitfield — each branch keys on max*>0).
export const MJ_LINEAR_VELOCITY = J_PAYLOAD; // vec3
export const MJ_ANGULAR_VELOCITY = J_PAYLOAD + 3; // vec3
export const MJ_MAX_VELOCITY_FORCE = J_PAYLOAD + 6;
export const MJ_MAX_VELOCITY_TORQUE = J_PAYLOAD + 7;
export const MJ_LINEAR_HERTZ = J_PAYLOAD + 8;
export const MJ_LINEAR_DAMPING_RATIO = J_PAYLOAD + 9;
export const MJ_ANGULAR_HERTZ = J_PAYLOAD + 10;
export const MJ_ANGULAR_DAMPING_RATIO = J_PAYLOAD + 11;
export const MJ_MAX_SPRING_FORCE = J_PAYLOAD + 12;
export const MJ_MAX_SPRING_TORQUE = J_PAYLOAD + 13;
export const MJ_LINEAR_VELOCITY_IMPULSE = J_PAYLOAD + 14; // vec3

// Parallel-joint payload (joint_abi.rs parallel section). AppConfig, the one persistent vec2 impulse, then
// prepare's scratch. Marshal writes config + impulse.
export const PLJ_HERTZ = J_PAYLOAD;
export const PLJ_DAMPING_RATIO = J_PAYLOAD + 1;
export const PLJ_MAX_TORQUE = J_PAYLOAD + 2;
export const PLJ_PERP_IMPULSE = J_PAYLOAD + 3; // vec2
