// AVBD extension surface — the GPU step pipeline + the SAT WGSL, for custom tooling, tests, and the gym
// scenario. The happy path (the `Body` component + `AvbdPlugin`) ships on the `avbd` barrel.

export { COLLIDE_WGSL, HULL_WGSL, MAX_CONTACTS, SPECULATIVE_DISTANCE } from "./collide";
export { HULL_FACE_STRIDE, HULL_HEADER, packHulls } from "./hull";
export { Avbd } from "./index";
export {
    BODY_MARGIN,
    BODY_VEC4,
    COLOR_MARGIN,
    CONSTRAINT_CONTACT,
    CONTACT_VEC4,
    CONTACTS_PER_PAIR,
    JOINT_REC_VEC4,
    type JointDef,
    LDS_CAP,
    LDS_N,
    PAIRS_PER_BODY,
    PENALTY_MIN,
    PhysicsStep,
    SMALL_N,
    type SpringDef,
    type StepParams,
    WORLD,
} from "./step";
