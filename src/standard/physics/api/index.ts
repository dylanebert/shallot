
export type { Manifold, ManifoldPoint } from "../collision/contact";
export type { ShapeProxy } from "../collision/distance";
export {
    type CollisionPlane,
    clipVector,
    type PlaneResult,
    type PlaneSolverResult,
    solvePlanes,
} from "../collision/mover";
export type { AABB, Mat3, Pos, Quat, Transform, Vec3, WorldTransform } from "../common/records";
export {
    type BodyDef,
    BodyType,
    type Capacity,
    defaultBodyDef,
    defaultFilter,
    defaultQueryFilter,
    defaultShapeDef,
    defaultSurfaceMaterial,
    defaultWorldDef,
    type Filter,
    type MotionLocks,
    type QueryFilter,
    type ShapeDef,
    ShapeType,
    type SurfaceMaterial,
    type WorldDef,
} from "../common/types";
export { type InitOptions, init, shutdown, threads } from "../kernel/kernel";
export type { TreeStats } from "../kernel/treecolumns";
export {
    createCone,
    createCylinder,
    createHull,
    createRock,
    makeBoxHull,
    makeCubeHull,
    makeOffsetBoxHull,
    makeTransformedBoxHull,
} from "../shapes/build-hull";
export {
    type CompoundCapsuleDef,
    type CompoundData,
    type CompoundDef,
    type CompoundHullDef,
    type CompoundMeshDef,
    type CompoundSphereDef,
    createCompound,
} from "../shapes/compound";
export type { Capsule, MassData, Sphere } from "../shapes/geometry";
export {
    createGrid,
    createHeightField,
    createWave,
    HEIGHT_FIELD_HOLE,
    type HeightFieldData,
    type HeightFieldDef,
} from "../shapes/heightfield";
export type { HullData } from "../shapes/hull";
export {
    createBoxMesh,
    createGridMesh,
    createHollowBoxMesh,
    createMesh,
    createTorusMesh,
    createWaveMesh,
    type Mesh,
    type MeshData,
    type MeshDef,
    MeshEdgeFlags,
} from "../shapes/mesh";
export { JointType } from "../solver/joint";
export { DebugColor, type DebugDraw, defaultDebugDraw } from "../world/draw";
export { hashWorldState } from "../world/hash";
export type { StepProfile } from "../world/profile";
export type { Counters, WorldState } from "../world/world";
export type {
    BaseJointConfig,
    CustomFilterCallback,
    DistanceJointConfig,
    MotorJointConfig,
    ParallelJointConfig,
    PreSolveCallback,
    PrismaticJointConfig,
    RevoluteJointConfig,
    SoftJointConfig,
    SphericalJointConfig,
    WeldJointConfig,
    WheelJointConfig,
} from "./config";
export { type PhysicsSnapshot, restore, snapshot } from "./snapshot";
