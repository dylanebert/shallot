import { NULL_INDEX } from "../common/array";
import { HUGE, LINEAR_SLOP } from "../common/core";
import type { EntityId } from "../common/ids";
import { clampf, f32, froundConfig, PI, type Transform, type Vec3 } from "../common/math";
import {
    type DistanceJoint as DistanceJointData,
    distanceJointCurrentLength,
} from "../solver/distanceJoint";
import {
    destroyJointInternal,
    getJointAngularSeparation,
    getJointConstraintForce,
    getJointConstraintTorque,
    getJointLinearSeparation,
    getJointSim,
    type Joint as JointRecord,
    type JointSim,
    type JointType,
    setJointCollideConnected,
    wakeJointBodies,
} from "../solver/joint";
import {
    type PrismaticJoint as PrismaticJointData,
    prismaticJointSpeed,
    prismaticJointTranslation,
} from "../solver/prismaticJoint";
import {
    type RevoluteJoint as RevoluteJointData,
    revoluteJointAngle,
} from "../solver/revoluteJoint";
import { makeBodyId } from "../world/body";
import type { WorldState } from "../world/world";
import { Body } from "./body";
import { cloneTransform } from "./config";
import { World } from "./world";

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
        return this.world.joints[this.id.index1 - 1];
    }

    /** @internal the live simulation payload (graph color when awake, else the solver set). */
    protected sim(): JointSim {
        return getJointSim(this.world, this.record());
    }

    /** @returns whether this joint has not been destroyed and its world is alive. */
    isValid(): boolean {
        if (this.world.inUse === false) {
            return false;
        }
        const i = this.id.index1 - 1;
        if (i < 0 || i >= this.world.joints.length) {
            return false;
        }
        const joint = this.world.joints[i];
        if (joint.setIndex === NULL_INDEX) {
            return false;
        }
        return joint.generation === this.id.generation;
    }

    /** Destroy this joint. Pass `false` to leave the attached bodies asleep. */
    destroy(wakeBodies = true): void {
        destroyJointInternal(this.world, this.record(), wakeBodies);
    }

    /** @returns the joint kind. */
    getType(): JointType {
        return this.record().type;
    }

    /** @returns the two bodies this joint connects. */
    getBodies(): [Body, Body] {
        const joint = this.record();
        return [
            new Body(this.world, makeBodyId(this.world, joint.edges[0].bodyId)),
            new Body(this.world, makeBodyId(this.world, joint.edges[1].bodyId)),
        ];
    }

    /** @returns the constraint force this joint currently applies (world units). */
    getConstraintForce(): Vec3 {
        const sim = getJointSim(this.world, this.record());
        return getJointConstraintForce(this.world, sim);
    }

    /** @returns the constraint torque this joint currently applies (world units). */
    getConstraintTorque(): Vec3 {
        const sim = getJointSim(this.world, this.record());
        return getJointConstraintTorque(this.world, sim);
    }

    /** @returns the user data attached to this joint. */
    getUserData(): unknown {
        return this.record().userData;
    }

    /** Attach arbitrary user data to this joint. */
    setUserData(userData: unknown): void {
        this.record().userData = userData;
    }

    /** @returns a handle to the world this joint belongs to. */
    getWorld(): World {
        return World._wrap(this.world);
    }

    /** @returns body A's local joint frame. */
    getLocalFrameA(): Transform {
        return cloneTransform(this.sim().localFrameA);
    }

    /** Set body A's local joint frame. */
    setLocalFrameA(frame: Transform): void {
        // froundConfig returns a fresh deep copy, so the caller's object is never aliased.
        this.sim().localFrameA = froundConfig(frame);
    }

    /** @returns body B's local joint frame. */
    getLocalFrameB(): Transform {
        return cloneTransform(this.sim().localFrameB);
    }

    /** Set body B's local joint frame. */
    setLocalFrameB(frame: Transform): void {
        this.sim().localFrameB = froundConfig(frame);
    }

    /** @returns whether the two connected bodies collide. */
    getCollideConnected(): boolean {
        return this.record().collideConnected;
    }

    /** Toggle whether the two connected bodies collide (updates the broad-phase). */
    setCollideConnected(shouldCollide: boolean): void {
        setJointCollideConnected(this.world, this.record(), shouldCollide);
    }

    /** @returns the joint's constraint softness tuning (hertz + damping ratio). */
    getConstraintTuning(): { hertz: number; dampingRatio: number } {
        const sim = this.sim();
        return { hertz: sim.constraintHertz, dampingRatio: sim.constraintDampingRatio };
    }

    /** Set the joint's constraint softness (hertz + damping ratio). */
    setConstraintTuning(hertz: number, dampingRatio: number): void {
        const sim = this.sim();
        sim.constraintHertz = f32(hertz);
        sim.constraintDampingRatio = f32(dampingRatio);
    }

    /** @returns the force at which this joint reports as over-stressed. */
    getForceThreshold(): number {
        return this.sim().forceThreshold;
    }

    /** Set the force at which this joint reports as over-stressed. */
    setForceThreshold(threshold: number): void {
        this.sim().forceThreshold = f32(threshold);
    }

    /** @returns the torque at which this joint reports as over-stressed. */
    getTorqueThreshold(): number {
        return this.sim().torqueThreshold;
    }

    /** Set the torque at which this joint reports as over-stressed. */
    setTorqueThreshold(threshold: number): void {
        this.sim().torqueThreshold = f32(threshold);
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
    private data(): RevoluteJointData {
        return this.sim().data as RevoluteJointData;
    }

    /** Enable/disable the angular limit. */
    enableLimit(enable: boolean): void {
        const j = this.data();
        if (enable !== j.enableLimit) {
            j.lowerImpulse = 0;
            j.upperImpulse = 0;
        }
        j.enableLimit = enable;
    }

    /** @returns whether the angular limit is enabled. */
    isLimitEnabled(): boolean {
        return this.data().enableLimit;
    }

    /** @returns the lower angle limit (radians). */
    getLowerLimit(): number {
        return this.data().lowerAngle;
    }

    /** @returns the upper angle limit (radians). */
    getUpperLimit(): number {
        return this.data().upperAngle;
    }

    /** Set the angle limits (radians), clamped to ±0.99π. */
    setLimits(lower: number, upper: number): void {
        const lo = f32(lower);
        const hi = f32(upper);
        const lowerAngle = lo < hi ? lo : hi;
        const upperAngle = lo > hi ? lo : hi;
        const bound = f32(f32(0.99) * PI);
        const j = this.data();
        j.lowerAngle = clampf(lowerAngle, -bound, bound);
        j.upperAngle = clampf(upperAngle, -bound, bound);
    }

    /** @returns the current hinge angle (radians). */
    getAngle(): number {
        return revoluteJointAngle(this.world, this.sim());
    }

    /** Enable/disable the drive spring. */
    enableSpring(enable: boolean): void {
        const j = this.data();
        if (enable !== j.enableSpring) {
            j.springImpulse = 0;
        }
        j.enableSpring = enable;
    }

    /** @returns whether the drive spring is enabled. */
    isSpringEnabled(): boolean {
        return this.data().enableSpring;
    }

    /** Set the spring target angle (radians). */
    setTargetAngle(target: number): void {
        this.data().targetAngle = f32(target);
    }

    /** @returns the spring target angle (radians). */
    getTargetAngle(): number {
        return this.data().targetAngle;
    }

    /** Set the spring frequency (Hz). */
    setSpringHertz(hertz: number): void {
        this.data().hertz = f32(hertz);
    }

    /** @returns the spring frequency (Hz). */
    getSpringHertz(): number {
        return this.data().hertz;
    }

    /** Set the spring damping ratio. */
    setSpringDampingRatio(dampingRatio: number): void {
        this.data().dampingRatio = f32(dampingRatio);
    }

    /** @returns the spring damping ratio. */
    getSpringDampingRatio(): number {
        return this.data().dampingRatio;
    }

    /**
     * Enable/disable the motor.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    enableMotor(enable: boolean): void {
        const j = this.data();
        if (enable !== j.enableMotor) {
            j.motorImpulse = 0;
        }
        j.enableMotor = enable;
    }

    /** @returns whether the motor is enabled. */
    isMotorEnabled(): boolean {
        return this.data().enableMotor;
    }

    /**
     * Set the motor target speed (radians/second).
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMotorSpeed(speed: number): void {
        this.data().motorSpeed = f32(speed);
    }

    /** @returns the motor target speed (radians/second). */
    getMotorSpeed(): number {
        return this.data().motorSpeed;
    }

    /**
     * Set the maximum motor torque.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMaxMotorTorque(torque: number): void {
        this.data().maxMotorTorque = f32(torque);
    }

    /** @returns the maximum motor torque. */
    getMaxMotorTorque(): number {
        return this.data().maxMotorTorque;
    }

    /** @returns the torque the motor applied last step. */
    getMotorTorque(): number {
        return f32(this.world.invH * this.data().motorImpulse);
    }
}

/** A distance joint handle. */
export class DistanceJoint extends Joint {
    private data(): DistanceJointData {
        return this.sim().data as DistanceJointData;
    }

    /** Set the rest length, clamped to [linear slop, huge]; resets accumulated impulses. */
    setLength(length: number): void {
        const j = this.data();
        j.length = clampf(f32(length), LINEAR_SLOP, HUGE);
        j.impulse = 0;
        j.lowerImpulse = 0;
        j.upperImpulse = 0;
    }

    /** @returns the rest length. */
    getLength(): number {
        return this.data().length;
    }

    /** Enable/disable the length limit. */
    enableLimit(enable: boolean): void {
        this.data().enableLimit = enable;
    }

    /** @returns whether the length limit is enabled. */
    isLimitEnabled(): boolean {
        return this.data().enableLimit;
    }

    /** Set the min/max length range, each clamped to [linear slop, huge]; resets impulses. */
    setLengthRange(minLength: number, maxLength: number): void {
        const lo = clampf(f32(minLength), LINEAR_SLOP, HUGE);
        const hi = clampf(f32(maxLength), LINEAR_SLOP, HUGE);
        const j = this.data();
        j.minLength = lo < hi ? lo : hi;
        j.maxLength = lo > hi ? lo : hi;
        j.impulse = 0;
        j.lowerImpulse = 0;
        j.upperImpulse = 0;
    }

    /** @returns the minimum length. */
    getMinLength(): number {
        return this.data().minLength;
    }

    /** @returns the maximum length. */
    getMaxLength(): number {
        return this.data().maxLength;
    }

    /** @returns the current distance between the anchor points. */
    getCurrentLength(): number {
        return distanceJointCurrentLength(this.world, this.sim());
    }

    /** Enable/disable the spring. */
    enableSpring(enable: boolean): void {
        this.data().enableSpring = enable;
    }

    /** @returns whether the spring is enabled. */
    isSpringEnabled(): boolean {
        return this.data().enableSpring;
    }

    /** Set the spring reaction-force range. */
    setSpringForceRange(lowerForce: number, upperForce: number): void {
        const j = this.data();
        j.lowerSpringForce = f32(lowerForce);
        j.upperSpringForce = f32(upperForce);
    }

    /** @returns the spring reaction-force range. */
    getSpringForceRange(): { lowerForce: number; upperForce: number } {
        const j = this.data();
        return { lowerForce: j.lowerSpringForce, upperForce: j.upperSpringForce };
    }

    /** Set the spring frequency (Hz). */
    setSpringHertz(hertz: number): void {
        this.data().hertz = f32(hertz);
    }

    /** @returns the spring frequency (Hz). */
    getSpringHertz(): number {
        return this.data().hertz;
    }

    /** Set the spring damping ratio. */
    setSpringDampingRatio(dampingRatio: number): void {
        this.data().dampingRatio = f32(dampingRatio);
    }

    /** @returns the spring damping ratio. */
    getSpringDampingRatio(): number {
        return this.data().dampingRatio;
    }

    /**
     * Enable/disable the motor; resets the motor impulse on change.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    enableMotor(enable: boolean): void {
        const j = this.data();
        if (enable !== j.enableMotor) {
            j.enableMotor = enable;
            j.motorImpulse = 0;
        }
    }

    /** @returns whether the motor is enabled. */
    isMotorEnabled(): boolean {
        return this.data().enableMotor;
    }

    /**
     * Set the motor target speed.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMotorSpeed(speed: number): void {
        this.data().motorSpeed = f32(speed);
    }

    /** @returns the motor target speed. */
    getMotorSpeed(): number {
        return this.data().motorSpeed;
    }

    /** @returns the force the motor applied last step. */
    getMotorForce(): number {
        return f32(this.world.invH * this.data().motorImpulse);
    }

    /**
     * Set the maximum motor force.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMaxMotorForce(force: number): void {
        this.data().maxMotorForce = f32(force);
    }

    /** @returns the maximum motor force. */
    getMaxMotorForce(): number {
        return this.data().maxMotorForce;
    }
}

/** A prismatic (slider) joint handle. */
export class PrismaticJoint extends Joint {
    private data(): PrismaticJointData {
        return this.sim().data as PrismaticJointData;
    }

    /** Enable/disable the translation limit; resets limit impulses on change. */
    enableLimit(enable: boolean): void {
        const j = this.data();
        if (enable !== j.enableLimit) {
            j.lowerImpulse = 0;
            j.upperImpulse = 0;
        }
        j.enableLimit = enable;
    }

    /** @returns whether the translation limit is enabled. */
    isLimitEnabled(): boolean {
        return this.data().enableLimit;
    }

    /** @returns the lower translation limit. */
    getLowerLimit(): number {
        return this.data().lowerTranslation;
    }

    /** @returns the upper translation limit. */
    getUpperLimit(): number {
        return this.data().upperTranslation;
    }

    /** Set the translation limits (ordered low..high). */
    setLimits(lower: number, upper: number): void {
        const lo = f32(lower);
        const hi = f32(upper);
        const j = this.data();
        j.lowerTranslation = lo < hi ? lo : hi;
        j.upperTranslation = lo > hi ? lo : hi;
    }

    /** @returns the current translation along the joint axis. */
    getTranslation(): number {
        return prismaticJointTranslation(this.world, this.sim());
    }

    /** Enable/disable the spring; resets the spring impulse on change. */
    enableSpring(enable: boolean): void {
        const j = this.data();
        if (enable !== j.enableSpring) {
            j.springImpulse = 0;
        }
        j.enableSpring = enable;
    }

    /** @returns whether the spring is enabled. */
    isSpringEnabled(): boolean {
        return this.data().enableSpring;
    }

    /** Set the spring target translation. */
    setTargetTranslation(target: number): void {
        this.data().targetTranslation = f32(target);
    }

    /** @returns the spring target translation. */
    getTargetTranslation(): number {
        return this.data().targetTranslation;
    }

    /** Set the spring frequency (Hz). */
    setSpringHertz(hertz: number): void {
        this.data().hertz = f32(hertz);
    }

    /** @returns the spring frequency (Hz). */
    getSpringHertz(): number {
        return this.data().hertz;
    }

    /** Set the spring damping ratio. */
    setSpringDampingRatio(dampingRatio: number): void {
        this.data().dampingRatio = f32(dampingRatio);
    }

    /** @returns the spring damping ratio. */
    getSpringDampingRatio(): number {
        return this.data().dampingRatio;
    }

    /**
     * Enable/disable the motor; resets the motor impulse on change.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    enableMotor(enable: boolean): void {
        const j = this.data();
        if (enable !== j.enableMotor) {
            j.motorImpulse = 0;
        }
        j.enableMotor = enable;
    }

    /** @returns whether the motor is enabled. */
    isMotorEnabled(): boolean {
        return this.data().enableMotor;
    }

    /**
     * Set the motor target speed.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMotorSpeed(speed: number): void {
        this.data().motorSpeed = f32(speed);
    }

    /** @returns the motor target speed. */
    getMotorSpeed(): number {
        return this.data().motorSpeed;
    }

    /**
     * Set the maximum motor force.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMaxMotorForce(force: number): void {
        this.data().maxMotorForce = f32(force);
    }

    /** @returns the maximum motor force. */
    getMaxMotorForce(): number {
        return this.data().maxMotorForce;
    }

    /** @returns the force the motor applied last step. */
    getMotorForce(): number {
        return f32(this.world.invH * this.data().motorImpulse);
    }

    /** @returns the current translation speed along the joint axis. */
    getSpeed(): number {
        return prismaticJointSpeed(this.world, this.sim());
    }
}
