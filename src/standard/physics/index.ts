export type { DebugDraw, ShapeProxy } from "./api";
export { defaultDebugDraw } from "./api";
export { Character, CharacterPlugin, GroundState } from "./character";
export {
    collideCapsuleAndSphere,
    collideCapsules,
    collideHullAndCapsule,
    collideHullAndSphere,
    collideHulls,
    collideSpheres,
    collideTriangleAndCapsule,
    collideTriangleAndHull,
    collideTriangleAndSphere,
    type FeaturePair,
    type LocalManifold,
    type LocalManifoldPoint,
    type SATCache,
    SeparatingFeature,
} from "./collision/manifold";
export {
    type CollisionPlane,
    clipVector,
    type PlaneResult,
    type PlaneSolverResult,
    solvePlanes,
} from "./collision/mover";
export {
    type DistanceInput,
    type DistanceOutput,
    getSweepTransform,
    type ShapeCastOutput,
    type ShapeCastPairInput,
    type Simplex,
    type SimplexCache,
    type SimplexVertex,
    type Sweep,
    shapeCast,
    shapeDistance,
    type TOIInput,
    type TOIOutput,
    TOIState,
    timeOfImpact,
} from "./collision/queries";
export type { QueryFilter } from "./common/types";
export type { PhysicsContactData, PhysicsManifold, PhysicsManifoldPoint } from "./contact";
export type {
    PhysicsContactEvent,
    PhysicsContactPair,
    PhysicsEventRecords,
    PhysicsEventsState,
    PhysicsHitEvent,
    PhysicsJointEvent,
    PhysicsSensorEvent,
} from "./events";
export { PhysicsEvents } from "./events";
export {
    createPhysicsMesh,
    destroyPhysicsMesh,
    type PhysicsMeshData,
    type PhysicsMeshNode,
} from "./mesh";
export { computeCosSin, rotationFromAxisAngle } from "./rotation";
export {
    applyAngularImpulse,
    applyForce,
    applyForceToCenter,
    applyLinearImpulse,
    applyLinearImpulseToCenter,
    applyTorque,
    applyWind,
    castBodyRay,
    castBodyShape,
    castMover,
    castRay,
    castRayClosest,
    castShape,
    castShapeClosest,
    collideBodyMover,
    collideMover,
    computeBodyAABB,
    defaultExplosionDef,
    drawPhysics,
    type ExplosionDef,
    emptyQueryHit,
    explode,
    getAngularVelocity,
    getContactData,
    getInverseMass,
    getJointLinearSeparation,
    getLocalPoint,
    getMassData,
    getSensorData,
    getWorldCounters,
    getWorldInverseRotationalInertia,
    hashPhysics,
    isAwake,
    overlapAABB,
    overlapBodyShape,
    overlapShape,
    type PhysicsBodyPlaneHit,
    type PhysicsBodyPlanes,
    type PhysicsBodyTOI,
    type PhysicsCastCallback,
    type PhysicsOverlapCallback,
    type PhysicsPlaneHit,
    type PhysicsPlanes,
    type PhysicsQueryHit,
    type PhysicsSensorData,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    SyncPhysicsConstraintsSystem,
    setAngularVelocity,
    setAwake,
    setBodyType,
    setLinearVelocity,
    setMassData,
    setTargetTransform,
    setTransform,
    stepPhysics,
    timeOfImpactBodyMover,
} from "./runtime";
export { shapeEdges, shapeMesh } from "./shape-mesh";
export {
    cloneAndTransformHull,
    createCylinder,
    createHull,
    makeBoxHull,
    makeScaledBoxHull,
} from "./shapes/build-hull";
export type { Capsule, MassData } from "./shapes/geometry";
export type { HullData } from "./shapes/hull";
export { computeCapsuleMass, computeHullMass } from "./shapes/mass";
export type { PhysicsWorldCounters } from "./world/counters";
export type { StepProfile } from "./world/profile";
export {
    PhysicsWorldDefinition,
    type PhysicsWorldDefinitionConfig,
    type WorldCustomFilterCallback,
    type WorldMixCallback,
    type WorldPreSolveCallback,
} from "./world-definition";
