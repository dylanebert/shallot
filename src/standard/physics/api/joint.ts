import { NULL_INDEX } from "../common/array";
import type { EntityId } from "../common/ids";
import { f32, type Transform, type Vec3 } from "../common/math";
import {
    DJ_DAMPING_RATIO,
    DJ_ENABLE,
    DJ_ENABLE_LIMIT,
    DJ_ENABLE_MOTOR,
    DJ_ENABLE_SPRING,
    DJ_HERTZ,
    DJ_LENGTH,
    DJ_LOWER_SPRING_FORCE,
    DJ_MAX_LENGTH,
    DJ_MAX_MOTOR_FORCE,
    DJ_MIN_LENGTH,
    DJ_MOTOR_IMPULSE,
    DJ_MOTOR_SPEED,
    DJ_UPPER_SPRING_FORCE,
    J_CONSTRAINT_DAMPING,
    J_CONSTRAINT_HERTZ,
    J_FORCE_THRESHOLD,
    J_LOCAL_FRAME_A,
    J_LOCAL_FRAME_B,
    J_TORQUE_THRESHOLD,
    PJ_DAMPING_RATIO,
    PJ_ENABLE,
    PJ_ENABLE_LIMIT,
    PJ_ENABLE_MOTOR,
    PJ_ENABLE_SPRING,
    PJ_HERTZ,
    PJ_LOWER_TRANSLATION,
    PJ_MAX_MOTOR_FORCE,
    PJ_MOTOR_IMPULSE,
    PJ_MOTOR_SPEED,
    PJ_TARGET_TRANSLATION,
    PJ_UPPER_TRANSLATION,
    RJ_DAMPING_RATIO,
    RJ_ENABLE,
    RJ_ENABLE_LIMIT,
    RJ_ENABLE_MOTOR,
    RJ_ENABLE_SPRING,
    RJ_HERTZ,
    RJ_LOWER_ANGLE,
    RJ_MAX_MOTOR_TORQUE,
    RJ_MOTOR_IMPULSE,
    RJ_MOTOR_SPEED,
    RJ_TARGET_ANGLE,
    RJ_UPPER_ANGLE,
} from "../kernel/columns";
import {
    readJointEventUserData,
    readJointFlag,
    readJointFloat,
    readJointTransform,
    writeJointFloat,
    writeJointTransform,
} from "../kernel/jointcolumns";
import { JointField, jointCapacity, jointField } from "../kernel/jointrecords";
import { kernel } from "../kernel/kernel";
import { distanceJointCurrentLength } from "../solver/distanceJoint";
import {
    destroyJointInternal,
    getJointAngularSeparation,
    getJointConstraintForce,
    getJointConstraintTorque,
    getJointLinearSeparation,
    type Joint as JointRecord,
    type JointType,
    setJointCollideConnected,
    wakeJointBodies,
} from "../solver/joint";
import { prismaticJointSpeed, prismaticJointTranslation } from "../solver/prismaticJoint";
import { revoluteJointAngle } from "../solver/revoluteJoint";
import { makeBodyId } from "../world/body";
import type { WorldState } from "../world/world";
import { Body } from "./body";
import { PhysicsWorld } from "./world";

/** A joint handle connecting two bodies. */
export class Joint {
    /** @internal */
    readonly world: WorldState;
    /** @internal */
    readonly id: EntityId;

    /** @internal use World.createRevoluteJoint */
    constructor(world: WorldState, id: EntityId) {
        this.world = world;
        this.id = id;
    }

    /** @internal */
    protected record(): JointRecord {
        return this.id.index1 - 1;
    }

    /** @returns whether this joint has not been destroyed and its world is alive. */
    isValid(): boolean {
        if (this.world.inUse === false) {
            return false;
        }
        const i = this.id.index1 - 1;
        if (i < 0 || i >= jointCapacity(this.world)) {
            return false;
        }
        const joint = i;
        if (jointField(this.world, joint, JointField.setIndex) === NULL_INDEX) {
            return false;
        }
        return jointField(this.world, joint, JointField.generation) === this.id.generation;
    }

    /** Destroy this joint. Pass `false` to leave the attached bodies asleep. */
    destroy(wakeBodies = true): void {
        destroyJointInternal(this.world, this.record(), wakeBodies);
    }

    /** @returns the joint kind. */
    getType(): JointType {
        return jointField(this.world, this.record(), JointField.type) as JointType;
    }

    /** @returns the two bodies this joint connects. */
    getBodies(): [Body, Body] {
        const joint = this.record();
        return [
            new Body(
                this.world,
                makeBodyId(this.world, jointField(this.world, joint, JointField.bodyIdA)),
            ),
            new Body(
                this.world,
                makeBodyId(this.world, jointField(this.world, joint, JointField.bodyIdB)),
            ),
        ];
    }

    /** @returns the constraint force this joint currently applies (world units). */
    getConstraintForce(): Vec3 {
        const sim = this.record();
        return getJointConstraintForce(this.world, sim);
    }

    /** @returns the constraint torque this joint currently applies (world units). */
    getConstraintTorque(): Vec3 {
        const sim = this.record();
        return getJointConstraintTorque(this.world, sim);
    }

    /** @returns the user data attached to this joint. */
    getUserData(): unknown {
        return this.world.jointUserData[this.record()];
    }

    /** Attach arbitrary user data to this joint. */
    setUserData(userData: unknown): void {
        readJointEventUserData(this.world);
        this.world.jointUserData[this.record()] = userData;
    }

    /** @returns a handle to the world this joint belongs to. */
    getWorld(): PhysicsWorld {
        return PhysicsWorld._wrap(this.world);
    }

    /** @returns body A's local joint frame. */
    getLocalFrameA(): Transform {
        return readJointTransform(this.world, this.record(), J_LOCAL_FRAME_A);
    }

    /** Set body A's local joint frame. */
    setLocalFrameA(frame: Transform): void {
        writeJointTransform(this.world, this.record(), J_LOCAL_FRAME_A, frame);
    }

    /** @returns body B's local joint frame. */
    getLocalFrameB(): Transform {
        return readJointTransform(this.world, this.record(), J_LOCAL_FRAME_B);
    }

    /** Set body B's local joint frame. */
    setLocalFrameB(frame: Transform): void {
        writeJointTransform(this.world, this.record(), J_LOCAL_FRAME_B, frame);
    }

    /** @returns whether the two connected bodies collide. */
    getCollideConnected(): boolean {
        return !!jointField(this.world, this.record(), JointField.collideConnected);
    }

    /** Toggle whether the two connected bodies collide (updates the broad-phase). */
    setCollideConnected(shouldCollide: boolean): void {
        setJointCollideConnected(this.world, this.record(), shouldCollide);
    }

    /** @returns the joint's constraint softness tuning (hertz + damping ratio). */
    getConstraintTuning(): {
        hertz: number;
        dampingRatio: number;
    } {
        return {
            hertz: readJointFloat(this.world, this.record(), J_CONSTRAINT_HERTZ),
            dampingRatio: readJointFloat(this.world, this.record(), J_CONSTRAINT_DAMPING),
        };
    }

    /** Set the joint's constraint softness (hertz + damping ratio). */
    setConstraintTuning(hertz: number, dampingRatio: number): void {
        writeJointFloat(this.world, this.record(), J_CONSTRAINT_HERTZ, f32(hertz));
        writeJointFloat(this.world, this.record(), J_CONSTRAINT_DAMPING, f32(dampingRatio));
    }

    /** @returns the force at which this joint reports as over-stressed. */
    getForceThreshold(): number {
        return readJointFloat(this.world, this.record(), J_FORCE_THRESHOLD);
    }

    /** Set the force at which this joint reports as over-stressed. */
    setForceThreshold(threshold: number): void {
        writeJointFloat(this.world, this.record(), J_FORCE_THRESHOLD, f32(threshold));
    }

    /** @returns the torque at which this joint reports as over-stressed. */
    getTorqueThreshold(): number {
        return readJointFloat(this.world, this.record(), J_TORQUE_THRESHOLD);
    }

    /** Set the torque at which this joint reports as over-stressed. */
    setTorqueThreshold(threshold: number): void {
        writeJointFloat(this.world, this.record(), J_TORQUE_THRESHOLD, f32(threshold));
    }

    /** Wake both bodies this joint connects. */
    wakeBodies(): void {
        wakeJointBodies(this.world, this.record());
    }

    /** @returns the current linear separation error at the joint anchors. */
    getLinearSeparation(): number {
        return getJointLinearSeparation(this.world, this.record());
    }

    /** @returns the current angular separation error at the joint. */
    getAngularSeparation(): number {
        return getJointAngularSeparation(this.world, this.record());
    }
}

/** A revolute (hinge) joint handle. */
export class RevoluteJoint extends Joint {
    /** Enable/disable the angular limit. */
    enableLimit(enable: boolean): void {
        kernel(this.world.ecsState).jointEnable(
            this.world.worldId,
            this.record(),
            RJ_ENABLE_LIMIT,
            enable,
        );
    }

    /** @returns whether the angular limit is enabled. */
    isLimitEnabled(): boolean {
        return readJointFlag(this.world, this.record(), RJ_ENABLE, RJ_ENABLE_LIMIT);
    }

    /** @returns the lower angle limit (radians). */
    getLowerLimit(): number {
        return readJointFloat(this.world, this.record(), RJ_LOWER_ANGLE);
    }

    /** @returns the upper angle limit (radians). */
    getUpperLimit(): number {
        return readJointFloat(this.world, this.record(), RJ_UPPER_ANGLE);
    }

    /** Set the angle limits (radians), clamped to ±0.99π. */
    setLimits(lower: number, upper: number): void {
        kernel(this.world.ecsState).jointSetLimits(this.world.worldId, this.record(), lower, upper);
    }

    /** @returns the current hinge angle (radians). */
    getAngle(): number {
        return revoluteJointAngle(this.world, this.record());
    }

    /** Enable/disable the drive spring. */
    enableSpring(enable: boolean): void {
        kernel(this.world.ecsState).jointEnable(
            this.world.worldId,
            this.record(),
            RJ_ENABLE_SPRING,
            enable,
        );
    }

    /** @returns whether the drive spring is enabled. */
    isSpringEnabled(): boolean {
        return readJointFlag(this.world, this.record(), RJ_ENABLE, RJ_ENABLE_SPRING);
    }

    /** Set the spring target angle (radians). */
    setTargetAngle(target: number): void {
        writeJointFloat(this.world, this.record(), RJ_TARGET_ANGLE, f32(target));
    }

    /** @returns the spring target angle (radians). */
    getTargetAngle(): number {
        return readJointFloat(this.world, this.record(), RJ_TARGET_ANGLE);
    }

    /** Set the spring frequency (Hz). */
    setSpringHertz(hertz: number): void {
        writeJointFloat(this.world, this.record(), RJ_HERTZ, f32(hertz));
    }

    /** @returns the spring frequency (Hz). */
    getSpringHertz(): number {
        return readJointFloat(this.world, this.record(), RJ_HERTZ);
    }

    /** Set the spring damping ratio. */
    setSpringDampingRatio(dampingRatio: number): void {
        writeJointFloat(this.world, this.record(), RJ_DAMPING_RATIO, f32(dampingRatio));
    }

    /** @returns the spring damping ratio. */
    getSpringDampingRatio(): number {
        return readJointFloat(this.world, this.record(), RJ_DAMPING_RATIO);
    }

    /**
     * Enable/disable the motor.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    enableMotor(enable: boolean): void {
        kernel(this.world.ecsState).jointEnable(
            this.world.worldId,
            this.record(),
            RJ_ENABLE_MOTOR,
            enable,
        );
    }

    /** @returns whether the motor is enabled. */
    isMotorEnabled(): boolean {
        return readJointFlag(this.world, this.record(), RJ_ENABLE, RJ_ENABLE_MOTOR);
    }

    /** Set the motor target speed (radians/second), waking the connected bodies. */
    setMotorSpeed(speed: number): void {
        writeJointFloat(this.world, this.record(), RJ_MOTOR_SPEED, f32(speed));
        wakeJointBodies(this.world, this.record());
    }

    /** @returns the motor target speed (radians/second). */
    getMotorSpeed(): number {
        return readJointFloat(this.world, this.record(), RJ_MOTOR_SPEED);
    }

    /**
     * Set the maximum motor torque.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMaxMotorTorque(torque: number): void {
        writeJointFloat(this.world, this.record(), RJ_MAX_MOTOR_TORQUE, f32(torque));
    }

    /** @returns the maximum motor torque. */
    getMaxMotorTorque(): number {
        return readJointFloat(this.world, this.record(), RJ_MAX_MOTOR_TORQUE);
    }

    /** @returns the torque the motor applied last step. */
    getMotorTorque(): number {
        return f32(this.world.invH * readJointFloat(this.world, this.record(), RJ_MOTOR_IMPULSE));
    }
}

/** A distance joint handle. */
export class DistanceJoint extends Joint {
    /** Set the rest length, clamped to [linear slop, huge]; resets accumulated impulses. */
    setLength(length: number): void {
        kernel(this.world.ecsState).distanceJointSetLength(
            this.world.worldId,
            this.record(),
            length,
        );
    }

    /** @returns the rest length. */
    getLength(): number {
        return readJointFloat(this.world, this.record(), DJ_LENGTH);
    }

    /** Enable/disable the length limit. */
    enableLimit(enable: boolean): void {
        kernel(this.world.ecsState).jointEnable(
            this.world.worldId,
            this.record(),
            DJ_ENABLE_LIMIT,
            enable,
        );
    }

    /** @returns whether the length limit is enabled. */
    isLimitEnabled(): boolean {
        return readJointFlag(this.world, this.record(), DJ_ENABLE, DJ_ENABLE_LIMIT);
    }

    /** Set the min/max length range, each clamped to [linear slop, huge]; resets impulses. */
    setLengthRange(minLength: number, maxLength: number): void {
        kernel(this.world.ecsState).jointSetLimits(
            this.world.worldId,
            this.record(),
            minLength,
            maxLength,
        );
    }

    /** @returns the minimum length. */
    getMinLength(): number {
        return readJointFloat(this.world, this.record(), DJ_MIN_LENGTH);
    }

    /** @returns the maximum length. */
    getMaxLength(): number {
        return readJointFloat(this.world, this.record(), DJ_MAX_LENGTH);
    }

    /** @returns the current distance between the anchor points. */
    getCurrentLength(): number {
        return distanceJointCurrentLength(this.world, this.record());
    }

    /** Enable/disable the spring. */
    enableSpring(enable: boolean): void {
        kernel(this.world.ecsState).jointEnable(
            this.world.worldId,
            this.record(),
            DJ_ENABLE_SPRING,
            enable,
        );
    }

    /** @returns whether the spring is enabled. */
    isSpringEnabled(): boolean {
        return readJointFlag(this.world, this.record(), DJ_ENABLE, DJ_ENABLE_SPRING);
    }

    /** Set the spring reaction-force range. */
    setSpringForceRange(lowerForce: number, upperForce: number): void {
        writeJointFloat(this.world, this.record(), DJ_LOWER_SPRING_FORCE, f32(lowerForce));
        writeJointFloat(this.world, this.record(), DJ_UPPER_SPRING_FORCE, f32(upperForce));
    }

    /** @returns the spring reaction-force range. */
    getSpringForceRange(): {
        lowerForce: number;
        upperForce: number;
    } {
        return {
            lowerForce: readJointFloat(this.world, this.record(), DJ_LOWER_SPRING_FORCE),
            upperForce: readJointFloat(this.world, this.record(), DJ_UPPER_SPRING_FORCE),
        };
    }

    /** Set the spring frequency (Hz). */
    setSpringHertz(hertz: number): void {
        writeJointFloat(this.world, this.record(), DJ_HERTZ, f32(hertz));
    }

    /** @returns the spring frequency (Hz). */
    getSpringHertz(): number {
        return readJointFloat(this.world, this.record(), DJ_HERTZ);
    }

    /** Set the spring damping ratio. */
    setSpringDampingRatio(dampingRatio: number): void {
        writeJointFloat(this.world, this.record(), DJ_DAMPING_RATIO, f32(dampingRatio));
    }

    /** @returns the spring damping ratio. */
    getSpringDampingRatio(): number {
        return readJointFloat(this.world, this.record(), DJ_DAMPING_RATIO);
    }

    /**
     * Enable/disable the motor; resets the motor impulse on change.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    enableMotor(enable: boolean): void {
        kernel(this.world.ecsState).jointEnable(
            this.world.worldId,
            this.record(),
            DJ_ENABLE_MOTOR,
            enable,
        );
    }

    /** @returns whether the motor is enabled. */
    isMotorEnabled(): boolean {
        return readJointFlag(this.world, this.record(), DJ_ENABLE, DJ_ENABLE_MOTOR);
    }

    /** Set the motor target speed, waking the connected bodies. */
    setMotorSpeed(speed: number): void {
        writeJointFloat(this.world, this.record(), DJ_MOTOR_SPEED, f32(speed));
        wakeJointBodies(this.world, this.record());
    }

    /** @returns the motor target speed. */
    getMotorSpeed(): number {
        return readJointFloat(this.world, this.record(), DJ_MOTOR_SPEED);
    }

    /** @returns the force the motor applied last step. */
    getMotorForce(): number {
        return f32(this.world.invH * readJointFloat(this.world, this.record(), DJ_MOTOR_IMPULSE));
    }

    /**
     * Set the maximum motor force.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMaxMotorForce(force: number): void {
        writeJointFloat(this.world, this.record(), DJ_MAX_MOTOR_FORCE, f32(force));
    }

    /** @returns the maximum motor force. */
    getMaxMotorForce(): number {
        return readJointFloat(this.world, this.record(), DJ_MAX_MOTOR_FORCE);
    }
}

/** A prismatic (slider) joint handle. */
export class PrismaticJoint extends Joint {
    /** Enable/disable the translation limit; resets limit impulses on change. */
    enableLimit(enable: boolean): void {
        kernel(this.world.ecsState).jointEnable(
            this.world.worldId,
            this.record(),
            PJ_ENABLE_LIMIT,
            enable,
        );
    }

    /** @returns whether the translation limit is enabled. */
    isLimitEnabled(): boolean {
        return readJointFlag(this.world, this.record(), PJ_ENABLE, PJ_ENABLE_LIMIT);
    }

    /** @returns the lower translation limit. */
    getLowerLimit(): number {
        return readJointFloat(this.world, this.record(), PJ_LOWER_TRANSLATION);
    }

    /** @returns the upper translation limit. */
    getUpperLimit(): number {
        return readJointFloat(this.world, this.record(), PJ_UPPER_TRANSLATION);
    }

    /** Set the translation limits (ordered low..high). */
    setLimits(lower: number, upper: number): void {
        kernel(this.world.ecsState).jointSetLimits(this.world.worldId, this.record(), lower, upper);
    }

    /** @returns the current translation along the joint axis. */
    getTranslation(): number {
        return prismaticJointTranslation(this.world, this.record());
    }

    /** Enable/disable the spring; resets the spring impulse on change. */
    enableSpring(enable: boolean): void {
        kernel(this.world.ecsState).jointEnable(
            this.world.worldId,
            this.record(),
            PJ_ENABLE_SPRING,
            enable,
        );
    }

    /** @returns whether the spring is enabled. */
    isSpringEnabled(): boolean {
        return readJointFlag(this.world, this.record(), PJ_ENABLE, PJ_ENABLE_SPRING);
    }

    /** Set the spring target translation. */
    setTargetTranslation(target: number): void {
        writeJointFloat(this.world, this.record(), PJ_TARGET_TRANSLATION, f32(target));
    }

    /** @returns the spring target translation. */
    getTargetTranslation(): number {
        return readJointFloat(this.world, this.record(), PJ_TARGET_TRANSLATION);
    }

    /** Set the spring frequency (Hz). */
    setSpringHertz(hertz: number): void {
        writeJointFloat(this.world, this.record(), PJ_HERTZ, f32(hertz));
    }

    /** @returns the spring frequency (Hz). */
    getSpringHertz(): number {
        return readJointFloat(this.world, this.record(), PJ_HERTZ);
    }

    /** Set the spring damping ratio. */
    setSpringDampingRatio(dampingRatio: number): void {
        writeJointFloat(this.world, this.record(), PJ_DAMPING_RATIO, f32(dampingRatio));
    }

    /** @returns the spring damping ratio. */
    getSpringDampingRatio(): number {
        return readJointFloat(this.world, this.record(), PJ_DAMPING_RATIO);
    }

    /**
     * Enable/disable the motor; resets the motor impulse on change.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    enableMotor(enable: boolean): void {
        kernel(this.world.ecsState).jointEnable(
            this.world.worldId,
            this.record(),
            PJ_ENABLE_MOTOR,
            enable,
        );
    }

    /** @returns whether the motor is enabled. */
    isMotorEnabled(): boolean {
        return readJointFlag(this.world, this.record(), PJ_ENABLE, PJ_ENABLE_MOTOR);
    }

    /** Set the motor target speed, waking the connected bodies. */
    setMotorSpeed(speed: number): void {
        writeJointFloat(this.world, this.record(), PJ_MOTOR_SPEED, f32(speed));
        wakeJointBodies(this.world, this.record());
    }

    /** @returns the motor target speed. */
    getMotorSpeed(): number {
        return readJointFloat(this.world, this.record(), PJ_MOTOR_SPEED);
    }

    /**
     * Set the maximum motor force.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMaxMotorForce(force: number): void {
        writeJointFloat(this.world, this.record(), PJ_MAX_MOTOR_FORCE, f32(force));
    }

    /** @returns the maximum motor force. */
    getMaxMotorForce(): number {
        return readJointFloat(this.world, this.record(), PJ_MAX_MOTOR_FORCE);
    }

    /** @returns the force the motor applied last step. */
    getMotorForce(): number {
        return f32(this.world.invH * readJointFloat(this.world, this.record(), PJ_MOTOR_IMPULSE));
    }

    /** @returns the current translation speed along the joint axis. */
    getSpeed(): number {
        return prismaticJointSpeed(this.world, this.record());
    }
}
