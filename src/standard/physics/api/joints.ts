import { f32, type Pos, type Quat, type Vec3, vec3 } from "../common/math";
import {
    MJ_ANGULAR_DAMPING_RATIO,
    MJ_ANGULAR_HERTZ,
    MJ_ANGULAR_VELOCITY,
    MJ_LINEAR_DAMPING_RATIO,
    MJ_LINEAR_HERTZ,
    MJ_LINEAR_VELOCITY,
    MJ_MAX_SPRING_FORCE,
    MJ_MAX_SPRING_TORQUE,
    MJ_MAX_VELOCITY_FORCE,
    MJ_MAX_VELOCITY_TORQUE,
    PLJ_DAMPING_RATIO,
    PLJ_HERTZ,
    PLJ_MAX_TORQUE,
    SJ_CONE_ANGLE,
    SJ_DAMPING_RATIO,
    SJ_ENABLE,
    SJ_ENABLE_CONE_LIMIT,
    SJ_ENABLE_MOTOR,
    SJ_ENABLE_SPRING,
    SJ_ENABLE_TWIST_LIMIT,
    SJ_HERTZ,
    SJ_LOWER_TWIST_ANGLE,
    SJ_MAX_MOTOR_TORQUE,
    SJ_MOTOR_IMPULSE,
    SJ_MOTOR_VELOCITY,
    SJ_TARGET_ROTATION,
    SJ_UPPER_TWIST_ANGLE,
    WHJ_ENABLE,
    WHJ_ENABLE_SPIN_MOTOR,
    WHJ_ENABLE_STEERING,
    WHJ_ENABLE_STEERING_LIMIT,
    WHJ_ENABLE_SUSPENSION_LIMIT,
    WHJ_ENABLE_SUSPENSION_SPRING,
    WHJ_LOWER_STEERING_LIMIT,
    WHJ_LOWER_SUSPENSION_LIMIT,
    WHJ_MAX_SPIN_TORQUE,
    WHJ_MAX_STEERING_TORQUE,
    WHJ_SPIN_IMPULSE,
    WHJ_SPIN_SPEED,
    WHJ_STEERING_DAMPING_RATIO,
    WHJ_STEERING_HERTZ,
    WHJ_STEERING_SPRING_IMPULSE,
    WHJ_SUSPENSION_DAMPING_RATIO,
    WHJ_SUSPENSION_HERTZ,
    WHJ_TARGET_STEERING_ANGLE,
    WHJ_UPPER_STEERING_LIMIT,
    WHJ_UPPER_SUSPENSION_LIMIT,
    WJ_ANGULAR_DAMPING_RATIO,
    WJ_ANGULAR_HERTZ,
    WJ_LINEAR_DAMPING_RATIO,
    WJ_LINEAR_HERTZ,
} from "../kernel/columns";
import {
    readJointFlag,
    readJointFloat,
    readJointQuat,
    readJointVec3,
    writeJointFloat,
    writeJointQuat,
    writeJointVec3,
} from "../kernel/jointcolumns";
import { kernel } from "../kernel/kernel";
import { wakeJointBodies } from "../solver/joint";
import { sphericalJointConeAngle, sphericalJointTwistAngle } from "../solver/sphericalJoint";
import { wheelJointSpinSpeed, wheelJointSteeringAngle } from "../solver/wheelJoint";
import type { Body } from "./body";
import { DistanceJoint, Joint } from "./joint";

/** A spring joint from a body anchor to a point fixed in world space. */
export class SoftJoint extends DistanceJoint {
    private readonly _anchorBody: Body;

    /** @internal use World.createSoftJoint */
    constructor(
        world: import("../world/world").WorldState,
        id: import("../common/ids").EntityId,
        anchorBody: Body,
    ) {
        super(world, id);
        this._anchorBody = anchorBody;
    }

    /** @returns the current world-space anchor. */
    getAnchor(): Pos {
        return this._anchorBody.getPosition();
    }

    /** Move the fixed anchor and wake the connected dynamic body. */
    setAnchor(anchor: Pos): void {
        this._anchorBody.setTransform(anchor, {
            v: {
                x: 0,
                y: 0,
                z: 0,
            },
            s: 1,
        });
        this.wakeBodies();
    }
    override destroy(wakeBodies = true): void {
        if (!this.isValid()) return;
        super.destroy(wakeBodies);
        this._anchorBody.destroy();
    }
}

/** A spherical (ball-and-socket) joint handle. */
export class SphericalJoint extends Joint {
    /** Enable/disable the cone (swing) limit; resets the swing impulse on change. */
    enableConeLimit(enable: boolean): void {
        kernel(this.world.ecsState).jointEnable(
            this.world.worldId,
            this.record(),
            SJ_ENABLE_CONE_LIMIT,
            enable,
        );
    }

    /** @returns whether the cone limit is enabled. */
    isConeLimitEnabled(): boolean {
        return readJointFlag(this.world, this.record(), SJ_ENABLE, SJ_ENABLE_CONE_LIMIT);
    }

    /** @returns the cone half-angle limit (radians). */
    getConeLimit(): number {
        return readJointFloat(this.world, this.record(), SJ_CONE_ANGLE);
    }

    /** Set the cone half-angle limit (radians). */
    setConeLimit(angle: number): void {
        writeJointFloat(this.world, this.record(), SJ_CONE_ANGLE, f32(angle));
    }

    /** @returns the current swing (cone) angle (radians). */
    getConeAngle(): number {
        return sphericalJointConeAngle(this.world, this.record());
    }

    /** Enable/disable the twist limit; resets twist impulses on change. */
    enableTwistLimit(enable: boolean): void {
        kernel(this.world.ecsState).jointEnable(
            this.world.worldId,
            this.record(),
            SJ_ENABLE_TWIST_LIMIT,
            enable,
        );
    }

    /** @returns whether the twist limit is enabled. */
    isTwistLimitEnabled(): boolean {
        return readJointFlag(this.world, this.record(), SJ_ENABLE, SJ_ENABLE_TWIST_LIMIT);
    }

    /** @returns the lower twist limit (radians). */
    getLowerTwistLimit(): number {
        return readJointFloat(this.world, this.record(), SJ_LOWER_TWIST_ANGLE);
    }

    /** @returns the upper twist limit (radians). */
    getUpperTwistLimit(): number {
        return readJointFloat(this.world, this.record(), SJ_UPPER_TWIST_ANGLE);
    }

    /** Set the twist limits (radians), clamped to ±0.99π. */
    setTwistLimits(lower: number, upper: number): void {
        kernel(this.world.ecsState).jointSetLimits(this.world.worldId, this.record(), lower, upper);
    }

    /** @returns the current twist angle (radians). */
    getTwistAngle(): number {
        return sphericalJointTwistAngle(this.world, this.record());
    }

    /** Enable/disable the orientation spring; resets the spring impulse on change. */
    enableSpring(enable: boolean): void {
        kernel(this.world.ecsState).jointEnable(
            this.world.worldId,
            this.record(),
            SJ_ENABLE_SPRING,
            enable,
        );
    }

    /** @returns whether the orientation spring is enabled. */
    isSpringEnabled(): boolean {
        return readJointFlag(this.world, this.record(), SJ_ENABLE, SJ_ENABLE_SPRING);
    }

    /** Set the spring target relative rotation. */
    setTargetRotation(target: Quat): void {
        writeJointQuat(this.world, this.record(), SJ_TARGET_ROTATION, target);
    }

    /** @returns the spring target relative rotation. */
    getTargetRotation(): Quat {
        return readJointQuat(this.world, this.record(), SJ_TARGET_ROTATION);
    }

    /** Set the spring frequency (Hz). */
    setSpringHertz(hertz: number): void {
        writeJointFloat(this.world, this.record(), SJ_HERTZ, f32(hertz));
    }

    /** @returns the spring frequency (Hz). */
    getSpringHertz(): number {
        return readJointFloat(this.world, this.record(), SJ_HERTZ);
    }

    /** Set the spring damping ratio. */
    setSpringDampingRatio(dampingRatio: number): void {
        writeJointFloat(this.world, this.record(), SJ_DAMPING_RATIO, f32(dampingRatio));
    }

    /** @returns the spring damping ratio. */
    getSpringDampingRatio(): number {
        return readJointFloat(this.world, this.record(), SJ_DAMPING_RATIO);
    }

    /**
     * Enable/disable the motor; resets the motor impulse on change.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    enableMotor(enable: boolean): void {
        kernel(this.world.ecsState).jointEnable(
            this.world.worldId,
            this.record(),
            SJ_ENABLE_MOTOR,
            enable,
        );
    }

    /** @returns whether the motor is enabled. */
    isMotorEnabled(): boolean {
        return readJointFlag(this.world, this.record(), SJ_ENABLE, SJ_ENABLE_MOTOR);
    }

    /** Set the motor target angular velocity, waking the connected bodies. */
    setMotorVelocity(velocity: Vec3): void {
        writeJointVec3(this.world, this.record(), SJ_MOTOR_VELOCITY, velocity);
        wakeJointBodies(this.world, this.record());
    }

    /** @returns the motor target angular velocity. */
    getMotorVelocity(): Vec3 {
        return readJointVec3(this.world, this.record(), SJ_MOTOR_VELOCITY);
    }

    /**
     * Set the maximum motor torque.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMaxMotorTorque(torque: number): void {
        writeJointFloat(this.world, this.record(), SJ_MAX_MOTOR_TORQUE, f32(torque));
    }

    /** @returns the maximum motor torque. */
    getMaxMotorTorque(): number {
        return readJointFloat(this.world, this.record(), SJ_MAX_MOTOR_TORQUE);
    }

    /** @returns the torque the motor applied last step. */
    getMotorTorque(): Vec3 {
        return vec3.scale(
            this.world.invH,
            readJointVec3(this.world, this.record(), SJ_MOTOR_IMPULSE),
        );
    }
}

/** A weld joint handle. */
export class WeldJoint extends Joint {
    /** Set the linear spring frequency (Hz). */
    setLinearHertz(hertz: number): void {
        writeJointFloat(this.world, this.record(), WJ_LINEAR_HERTZ, f32(hertz));
    }

    /** @returns the linear spring frequency (Hz). */
    getLinearHertz(): number {
        return readJointFloat(this.world, this.record(), WJ_LINEAR_HERTZ);
    }

    /** Set the linear spring damping ratio. */
    setLinearDampingRatio(dampingRatio: number): void {
        writeJointFloat(this.world, this.record(), WJ_LINEAR_DAMPING_RATIO, f32(dampingRatio));
    }

    /** @returns the linear spring damping ratio. */
    getLinearDampingRatio(): number {
        return readJointFloat(this.world, this.record(), WJ_LINEAR_DAMPING_RATIO);
    }

    /** Set the angular spring frequency (Hz). */
    setAngularHertz(hertz: number): void {
        writeJointFloat(this.world, this.record(), WJ_ANGULAR_HERTZ, f32(hertz));
    }

    /** @returns the angular spring frequency (Hz). */
    getAngularHertz(): number {
        return readJointFloat(this.world, this.record(), WJ_ANGULAR_HERTZ);
    }

    /** Set the angular spring damping ratio. */
    setAngularDampingRatio(dampingRatio: number): void {
        writeJointFloat(this.world, this.record(), WJ_ANGULAR_DAMPING_RATIO, f32(dampingRatio));
    }

    /** @returns the angular spring damping ratio. */
    getAngularDampingRatio(): number {
        return readJointFloat(this.world, this.record(), WJ_ANGULAR_DAMPING_RATIO);
    }
}

/** A motor joint handle. */
export class MotorJoint extends Joint {
    /** Set the target relative linear velocity, waking the connected bodies. */
    setLinearVelocity(velocity: Vec3): void {
        writeJointVec3(this.world, this.record(), MJ_LINEAR_VELOCITY, velocity);
        wakeJointBodies(this.world, this.record());
    }

    /** @returns the target relative linear velocity. */
    getLinearVelocity(): Vec3 {
        return readJointVec3(this.world, this.record(), MJ_LINEAR_VELOCITY);
    }

    /** Set the target relative angular velocity, waking the connected bodies. */
    setAngularVelocity(velocity: Vec3): void {
        writeJointVec3(this.world, this.record(), MJ_ANGULAR_VELOCITY, velocity);
        wakeJointBodies(this.world, this.record());
    }

    /** @returns the target relative angular velocity. */
    getAngularVelocity(): Vec3 {
        return readJointVec3(this.world, this.record(), MJ_ANGULAR_VELOCITY);
    }

    /**
     * Set the maximum velocity-drive torque.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMaxVelocityTorque(maxTorque: number): void {
        writeJointFloat(this.world, this.record(), MJ_MAX_VELOCITY_TORQUE, f32(maxTorque));
    }

    /** @returns the maximum velocity-drive torque. */
    getMaxVelocityTorque(): number {
        return readJointFloat(this.world, this.record(), MJ_MAX_VELOCITY_TORQUE);
    }

    /**
     * Set the maximum velocity-drive force.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMaxVelocityForce(maxForce: number): void {
        writeJointFloat(this.world, this.record(), MJ_MAX_VELOCITY_FORCE, f32(maxForce));
    }

    /** @returns the maximum velocity-drive force. */
    getMaxVelocityForce(): number {
        return readJointFloat(this.world, this.record(), MJ_MAX_VELOCITY_FORCE);
    }

    /** Set the linear spring frequency (Hz). */
    setLinearHertz(hertz: number): void {
        writeJointFloat(this.world, this.record(), MJ_LINEAR_HERTZ, f32(hertz));
    }

    /** @returns the linear spring frequency (Hz). */
    getLinearHertz(): number {
        return readJointFloat(this.world, this.record(), MJ_LINEAR_HERTZ);
    }

    /** Set the linear spring damping ratio. */
    setLinearDampingRatio(dampingRatio: number): void {
        writeJointFloat(this.world, this.record(), MJ_LINEAR_DAMPING_RATIO, f32(dampingRatio));
    }

    /** @returns the linear spring damping ratio. */
    getLinearDampingRatio(): number {
        return readJointFloat(this.world, this.record(), MJ_LINEAR_DAMPING_RATIO);
    }

    /** Set the angular spring frequency (Hz). */
    setAngularHertz(hertz: number): void {
        writeJointFloat(this.world, this.record(), MJ_ANGULAR_HERTZ, f32(hertz));
    }

    /** @returns the angular spring frequency (Hz). */
    getAngularHertz(): number {
        return readJointFloat(this.world, this.record(), MJ_ANGULAR_HERTZ);
    }

    /** Set the angular spring damping ratio. */
    setAngularDampingRatio(dampingRatio: number): void {
        writeJointFloat(this.world, this.record(), MJ_ANGULAR_DAMPING_RATIO, f32(dampingRatio));
    }

    /** @returns the angular spring damping ratio. */
    getAngularDampingRatio(): number {
        return readJointFloat(this.world, this.record(), MJ_ANGULAR_DAMPING_RATIO);
    }

    /** Set the maximum spring force (clamped ≥ 0). */
    setMaxSpringForce(maxForce: number): void {
        kernel(this.world.ecsState).motorJointSetMaxSpring(
            this.world.worldId,
            this.record(),
            false,
            maxForce,
        );
    }

    /** @returns the maximum spring force. */
    getMaxSpringForce(): number {
        return readJointFloat(this.world, this.record(), MJ_MAX_SPRING_FORCE);
    }

    /** Set the maximum spring torque (clamped ≥ 0). */
    setMaxSpringTorque(maxTorque: number): void {
        kernel(this.world.ecsState).motorJointSetMaxSpring(
            this.world.worldId,
            this.record(),
            true,
            maxTorque,
        );
    }

    /** @returns the maximum spring torque. */
    getMaxSpringTorque(): number {
        return readJointFloat(this.world, this.record(), MJ_MAX_SPRING_TORQUE);
    }
}

/** A parallel joint handle. */
export class ParallelJoint extends Joint {
    /** Set the spring frequency (Hz). */
    setSpringHertz(hertz: number): void {
        writeJointFloat(this.world, this.record(), PLJ_HERTZ, f32(hertz));
    }

    /** @returns the spring frequency (Hz). */
    getSpringHertz(): number {
        return readJointFloat(this.world, this.record(), PLJ_HERTZ);
    }

    /** Set the spring damping ratio. */
    setSpringDampingRatio(dampingRatio: number): void {
        writeJointFloat(this.world, this.record(), PLJ_DAMPING_RATIO, f32(dampingRatio));
    }

    /** @returns the spring damping ratio. */
    getSpringDampingRatio(): number {
        return readJointFloat(this.world, this.record(), PLJ_DAMPING_RATIO);
    }

    /** Set the maximum corrective torque. */
    setMaxTorque(maxTorque: number): void {
        writeJointFloat(this.world, this.record(), PLJ_MAX_TORQUE, f32(maxTorque));
    }

    /** @returns the maximum corrective torque. */
    getMaxTorque(): number {
        return readJointFloat(this.world, this.record(), PLJ_MAX_TORQUE);
    }
}

/** A wheel joint handle. */
export class WheelJoint extends Joint {
    /** Enable/disable the suspension spring; resets the suspension impulse on change. */
    enableSuspension(enable: boolean): void {
        kernel(this.world.ecsState).jointEnable(
            this.world.worldId,
            this.record(),
            WHJ_ENABLE_SUSPENSION_SPRING,
            enable,
        );
    }

    /** @returns whether the suspension spring is enabled. */
    isSuspensionEnabled(): boolean {
        return readJointFlag(this.world, this.record(), WHJ_ENABLE, WHJ_ENABLE_SUSPENSION_SPRING);
    }

    /** Set the suspension spring frequency (Hz). */
    setSuspensionHertz(hertz: number): void {
        writeJointFloat(this.world, this.record(), WHJ_SUSPENSION_HERTZ, f32(hertz));
    }

    /** @returns the suspension spring frequency (Hz). */
    getSuspensionHertz(): number {
        return readJointFloat(this.world, this.record(), WHJ_SUSPENSION_HERTZ);
    }

    /** Set the suspension spring damping ratio. */
    setSuspensionDampingRatio(dampingRatio: number): void {
        writeJointFloat(this.world, this.record(), WHJ_SUSPENSION_DAMPING_RATIO, f32(dampingRatio));
    }

    /** @returns the suspension spring damping ratio. */
    getSuspensionDampingRatio(): number {
        return readJointFloat(this.world, this.record(), WHJ_SUSPENSION_DAMPING_RATIO);
    }

    /** Enable/disable the suspension limit; resets limit impulses on change. */
    enableSuspensionLimit(enable: boolean): void {
        kernel(this.world.ecsState).jointEnable(
            this.world.worldId,
            this.record(),
            WHJ_ENABLE_SUSPENSION_LIMIT,
            enable,
        );
    }

    /** @returns whether the suspension limit is enabled. */
    isSuspensionLimitEnabled(): boolean {
        return readJointFlag(this.world, this.record(), WHJ_ENABLE, WHJ_ENABLE_SUSPENSION_LIMIT);
    }

    /** @returns the lower suspension limit. */
    getLowerSuspensionLimit(): number {
        return readJointFloat(this.world, this.record(), WHJ_LOWER_SUSPENSION_LIMIT);
    }

    /** @returns the upper suspension limit. */
    getUpperSuspensionLimit(): number {
        return readJointFloat(this.world, this.record(), WHJ_UPPER_SUSPENSION_LIMIT);
    }

    /** Set the suspension limits; resets limit impulses when changed. */
    setSuspensionLimits(lower: number, upper: number): void {
        kernel(this.world.ecsState).jointSetLimits(this.world.worldId, this.record(), lower, upper);
    }

    /**
     * Enable/disable the spin motor; resets the spin impulse on change.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    enableSpinMotor(enable: boolean): void {
        kernel(this.world.ecsState).jointEnable(
            this.world.worldId,
            this.record(),
            WHJ_ENABLE_SPIN_MOTOR,
            enable,
        );
    }

    /** @returns whether the spin motor is enabled. */
    isSpinMotorEnabled(): boolean {
        return readJointFlag(this.world, this.record(), WHJ_ENABLE, WHJ_ENABLE_SPIN_MOTOR);
    }

    /** Set the spin motor target speed, waking the connected bodies. */
    setSpinMotorSpeed(speed: number): void {
        writeJointFloat(this.world, this.record(), WHJ_SPIN_SPEED, f32(speed));
        wakeJointBodies(this.world, this.record());
    }

    /** @returns the spin motor target speed. */
    getSpinMotorSpeed(): number {
        return readJointFloat(this.world, this.record(), WHJ_SPIN_SPEED);
    }

    /**
     * Set the maximum spin torque.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMaxSpinTorque(torque: number): void {
        writeJointFloat(this.world, this.record(), WHJ_MAX_SPIN_TORQUE, f32(torque));
    }

    /** @returns the maximum spin torque. */
    getMaxSpinTorque(): number {
        return readJointFloat(this.world, this.record(), WHJ_MAX_SPIN_TORQUE);
    }

    /** @returns the current spin speed about the spin axis. */
    getSpinSpeed(): number {
        return wheelJointSpinSpeed(this.world, this.record());
    }

    /** @returns the spin torque applied last step. */
    getSpinTorque(): number {
        return f32(this.world.invH * readJointFloat(this.world, this.record(), WHJ_SPIN_IMPULSE));
    }

    /**
     * Enable/disable steering; resets the steering angular impulse on change.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    enableSteering(enable: boolean): void {
        kernel(this.world.ecsState).jointEnable(
            this.world.worldId,
            this.record(),
            WHJ_ENABLE_STEERING,
            enable,
        );
    }

    /** @returns whether steering is enabled. */
    isSteeringEnabled(): boolean {
        return readJointFlag(this.world, this.record(), WHJ_ENABLE, WHJ_ENABLE_STEERING);
    }

    /** Set the steering spring frequency (Hz). */
    setSteeringHertz(hertz: number): void {
        writeJointFloat(this.world, this.record(), WHJ_STEERING_HERTZ, f32(hertz));
    }

    /** @returns the steering spring frequency (Hz). */
    getSteeringHertz(): number {
        return readJointFloat(this.world, this.record(), WHJ_STEERING_HERTZ);
    }

    /** Set the steering spring damping ratio. */
    setSteeringDampingRatio(dampingRatio: number): void {
        writeJointFloat(this.world, this.record(), WHJ_STEERING_DAMPING_RATIO, f32(dampingRatio));
    }

    /** @returns the steering spring damping ratio. */
    getSteeringDampingRatio(): number {
        return readJointFloat(this.world, this.record(), WHJ_STEERING_DAMPING_RATIO);
    }

    /**
     * Set the maximum steering torque.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMaxSteeringTorque(maxTorque: number): void {
        writeJointFloat(this.world, this.record(), WHJ_MAX_STEERING_TORQUE, f32(maxTorque));
    }

    /** @returns the maximum steering torque. */
    getMaxSteeringTorque(): number {
        return readJointFloat(this.world, this.record(), WHJ_MAX_STEERING_TORQUE);
    }

    /** Enable/disable the steering limit; resets limit impulses on change. */
    enableSteeringLimit(enable: boolean): void {
        kernel(this.world.ecsState).jointEnable(
            this.world.worldId,
            this.record(),
            WHJ_ENABLE_STEERING_LIMIT,
            enable,
        );
    }

    /** @returns whether the steering limit is enabled. */
    isSteeringLimitEnabled(): boolean {
        return readJointFlag(this.world, this.record(), WHJ_ENABLE, WHJ_ENABLE_STEERING_LIMIT);
    }

    /** @returns the lower steering limit (radians). */
    getLowerSteeringLimit(): number {
        return readJointFloat(this.world, this.record(), WHJ_LOWER_STEERING_LIMIT);
    }

    /** @returns the upper steering limit (radians). */
    getUpperSteeringLimit(): number {
        return readJointFloat(this.world, this.record(), WHJ_UPPER_STEERING_LIMIT);
    }

    /** Set the steering limits (radians). */
    setSteeringLimits(lower: number, upper: number): void {
        writeJointFloat(this.world, this.record(), WHJ_LOWER_STEERING_LIMIT, f32(lower));
        writeJointFloat(this.world, this.record(), WHJ_UPPER_STEERING_LIMIT, f32(upper));
    }

    /** Set the steering spring target angle (radians), waking the connected bodies. */
    setTargetSteeringAngle(radians: number): void {
        writeJointFloat(this.world, this.record(), WHJ_TARGET_STEERING_ANGLE, f32(radians));
        wakeJointBodies(this.world, this.record());
    }

    /** @returns the steering spring target angle (radians). */
    getTargetSteeringAngle(): number {
        return readJointFloat(this.world, this.record(), WHJ_TARGET_STEERING_ANGLE);
    }

    /** @returns the current steering angle (radians). */
    getSteeringAngle(): number {
        return wheelJointSteeringAngle(this.world, this.record());
    }

    /** @returns the steering torque applied last step. */
    getSteeringTorque(): number {
        return f32(
            this.world.invH *
                readJointFloat(this.world, this.record(), WHJ_STEERING_SPRING_IMPULSE),
        );
    }
}
