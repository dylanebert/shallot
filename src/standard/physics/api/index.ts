// physics — a from-scratch TypeScript port of Erin Catto's Box3D (github.com/erincatto/box3d, MIT).
// A standalone 3D physics engine. The public surface grows across the port stages; see the README.

export {
    type BaseJointConfig,
    Body,
    type BodyCastHit,
    type BodyEvents,
    type BodyMoveEvent,
    type BodyPlane,
    type CastCallback,
    type CastHit,
    Contact,
    type ContactData,
    type ContactEvents,
    type ContactHitEvent,
    type ContactTouchEvent,
    DistanceJoint,
    type DistanceJointConfig,
    Joint,
    type JointEvent,
    MotorJoint,
    type MotorJointConfig,
    type MoverFilterCallback,
    type OverlapCallback,
    ParallelJoint,
    type ParallelJointConfig,
    type PlaneResultCallback,
    PrismaticJoint,
    type PrismaticJointConfig,
    type RayResult,
    RevoluteJoint,
    type RevoluteJointConfig,
    type SensorEvents,
    type SensorTouchEvent,
    Shape,
    SphericalJoint,
    type SphericalJointConfig,
    WeldJoint,
    type WeldJointConfig,
    WheelJoint,
    type WheelJointConfig,
    World,
} from "./api";
export {
    type CompoundCapsuleDef,
    type CompoundData,
    type CompoundDef,
    type CompoundHullDef,
    type CompoundMeshDef,
    type CompoundSphereDef,
    createCompound,
} from "../shapes/compound";
export type { Manifold, ManifoldPoint } from "../collision/contact";
export type { ShapeProxy } from "../collision/distance";
export { DebugColor, type DebugDraw, defaultDebugDraw } from "../world/draw";
export type { Capsule, MassData, Sphere } from "../shapes/geometry";
export { hashWorldState } from "../world/hash";
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
export { JointType } from "../solver/joint";
export { type InitOptions, init, shutdown, threads } from "../kernel/kernel";
export type { AABB, Mat3, Pos, Quat, Transform, Vec3, WorldTransform } from "../common/math";
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
export {
    type CollisionPlane,
    clipVector,
    type PlaneResult,
    type PlaneSolverResult,
    solvePlanes,
} from "../collision/mover";
export type { Profile } from "../world/profile";
export type { TreeStats } from "../collision/tree";
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
export type { Counters, WorldState } from "../world/world";
