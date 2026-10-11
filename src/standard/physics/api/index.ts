// physics — a from-scratch TypeScript port of Erin Catto's Box3D (github.com/erincatto/box3d, MIT).
// A standalone 3D physics engine. The public surface grows across the port stages; see the README.

export type { Manifold, ManifoldPoint } from "../collision/contact";
export type { ShapeProxy } from "../collision/distance";
export {
    type CollisionPlane,
    clipVector,
    type PlaneResult,
    type PlaneSolverResult,
    solvePlanes,
} from "../collision/mover";
export type { AABB, Mat3, Pos, Quat, Transform, Vec3, WorldTransform } from "../common/math";
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
export {
    createCone,
    createCylinder,
    createHull,
    createRock,
    type HullData,
    makeBoxHull,
    makeCubeHull,
    makeOffsetBoxHull,
    makeTransformedBoxHull,
} from "../shapes/hull";
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
