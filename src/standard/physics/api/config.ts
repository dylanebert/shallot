import type { EntityId } from "../common/ids";
import type { Pos, Quat, Transform, Vec3 } from "../common/records";
import { JointField, jointField } from "../kernel/jointrecords";
import { ShapeField, shapeField } from "../kernel/shaperecords";
import type { Shape as ShapeRecord } from "../shapes/shape";
import type { Joint as JointRecord } from "../solver/joint";
import type { WorldState } from "../world/world";

/** Return false to reject a collision pair. Arguments are generation-aware IDs for the two shapes.
 * Called only when either shape enables custom filtering. Runs synchronously during the step; the
 * world must not be modified from this callback. Kernel re-entry is refused because worlds sharing
 * an instance also share its active driver. */
export type CustomFilterCallback = (shapeA: EntityId, shapeB: EntityId) => boolean;

/** Return false to disable a contact for this step. The point is Box3D's inspection position in
 * world meters. The normal follows the contact or TOI orientation: internally flipped compound
 * sphere/capsule children receive shapes in convex-dispatch order, not contact order.
 * Point and normal are independent values, not writable solver storage. Shape arguments are
 * generation-aware IDs. Called only for enabled pre-solve events, not sensors. Runs synchronously
 * during the step; the world must not be modified. Kernel operations are refused from the callback
 * because that instance has one active driver. */
export type PreSolveCallback = (
    shapeA: EntityId,
    shapeB: EntityId,
    point: Pos,
    normal: Vec3,
) => boolean;

export function makeShapeId(world: WorldState, shape: ShapeRecord): EntityId {
    return {
        index1: shape + 1,
        world0: world.worldId,
        generation: shapeField(world, shape, ShapeField.generation),
    };
}

export function makeJointId(world: WorldState, joint: JointRecord): EntityId {
    return {
        index1: joint + 1,
        world0: world.worldId,
        generation: jointField(world, joint, JointField.generation),
    };
}

/** Options shared by every joint. */
export type BaseJointConfig = {
    localFrameA: Transform;
    localFrameB: Transform;
    collideConnected: boolean;
    constraintHertz: number;
    constraintDampingRatio: number;
    forceThreshold: number;
    torqueThreshold: number;
    drawScale: number;
    userData: unknown;
};

/** Revolute-specific options, on top of the shared base. */
export type RevoluteJointConfig = BaseJointConfig & {
    targetAngle: number;
    enableSpring: boolean;
    hertz: number;
    dampingRatio: number;
    enableLimit: boolean;
    lowerAngle: number;
    upperAngle: number;
    enableMotor: boolean;
    maxMotorTorque: number;
    motorSpeed: number;
};

/** Weld-specific options, on top of the shared base. */
export type WeldJointConfig = BaseJointConfig & {
    linearHertz: number;
    linearDampingRatio: number;
    angularHertz: number;
    angularDampingRatio: number;
};

/** Parallel-specific options, on top of the shared base. */
export type ParallelJointConfig = BaseJointConfig & {
    hertz: number;
    dampingRatio: number;
    maxTorque: number;
};

/** Motor-specific options, on top of the shared base. */
export type MotorJointConfig = BaseJointConfig & {
    linearVelocity: Vec3;
    maxVelocityForce: number;
    angularVelocity: Vec3;
    maxVelocityTorque: number;
    linearHertz: number;
    linearDampingRatio: number;
    maxSpringForce: number;
    angularHertz: number;
    angularDampingRatio: number;
    maxSpringTorque: number;
};

/** Distance-specific options, on top of the shared base. */
export type DistanceJointConfig = BaseJointConfig & {
    length: number;
    enableSpring: boolean;
    hertz: number;
    dampingRatio: number;
    lowerSpringForce: number;
    upperSpringForce: number;
    enableLimit: boolean;
    minLength: number;
    maxLength: number;
    enableMotor: boolean;
    maxMotorForce: number;
    motorSpeed: number;
};

/** Prismatic-specific options, on top of the shared base. */
export type PrismaticJointConfig = BaseJointConfig & {
    enableSpring: boolean;
    hertz: number;
    dampingRatio: number;
    targetTranslation: number;
    enableLimit: boolean;
    lowerTranslation: number;
    upperTranslation: number;
    enableMotor: boolean;
    maxMotorForce: number;
    motorSpeed: number;
};

/** Spherical-specific options, on top of the shared base. */
export type SphericalJointConfig = BaseJointConfig & {
    enableSpring: boolean;
    hertz: number;
    dampingRatio: number;
    targetRotation: Quat;
    enableConeLimit: boolean;
    coneAngle: number;
    enableTwistLimit: boolean;
    lowerTwistAngle: number;
    upperTwistAngle: number;
    enableMotor: boolean;
    maxMotorTorque: number;
    motorVelocity: Vec3;
};

/** Wheel-specific options, on top of the shared base. */
export type WheelJointConfig = BaseJointConfig & {
    enableSuspensionSpring: boolean;
    suspensionHertz: number;
    suspensionDampingRatio: number;
    enableSuspensionLimit: boolean;
    lowerSuspensionLimit: number;
    upperSuspensionLimit: number;
    enableSpinMotor: boolean;
    maxSpinTorque: number;
    spinSpeed: number;
    enableSteering: boolean;
    steeringHertz: number;
    steeringDampingRatio: number;
    targetSteeringAngle: number;
    maxSteeringTorque: number;
    enableSteeringLimit: boolean;
    lowerSteeringLimit: number;
    upperSteeringLimit: number;
};
