// Physics substrate extension surface — the backend-neutral pieces for custom tooling, tests, and a
// backend plugin's own implementation. The happy path (the `Body` component + a backend plugin like
// `AvbdPlugin`) ships on the main barrel.
export { type Hull, type HullFace, Hulls, UNIT_CUBE_ID } from "./hull";
export {
    type BodyState,
    bodyTraits,
    ComposeSystem,
    ConstraintSystem,
    installBackend,
    type JointDef,
    jointTraits,
    type PhysicsBackend,
    type SpringDef,
    StepSystem,
    springTraits,
    uninstallBackend,
} from "./index";
export { bodyCandidates, cursorRay, forwardRay, grabHit, worldToLocal } from "./pick";
export {
    generateRay,
    qRotate,
    type Ray,
    type RayBody,
    type RayHit,
    rayCapsule,
    raycast,
    rayOBB,
    raySphere,
    screenToRay,
} from "./raycast";
