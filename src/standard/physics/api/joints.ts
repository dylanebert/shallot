import { clampf, f32, froundConfig, PI, type Quat, type Vec3, vec3 } from "../common/math";
import type { MotorJoint as MotorJointData } from "../solver/motorJoint";
import type { ParallelJoint as ParallelJointData } from "../solver/parallelJoint";
import {
    type SphericalJoint as SphericalJointData,
    sphericalJointConeAngle,
    sphericalJointTwistAngle,
} from "../solver/sphericalJoint";
import type { WeldJoint as WeldJointData } from "../solver/weldJoint";
import {
    type WheelJoint as WheelJointData,
    wheelJointSpinSpeed,
    wheelJointSteeringAngle,
} from "../solver/wheelJoint";
import { cloneQuat } from "./config";
import { Joint } from "./joint";

/** A spherical (ball-and-socket) joint handle. */
export class SphericalJoint extends Joint {
    private data(): SphericalJointData {
        return this.sim().data as SphericalJointData;
    }

    /** Enable/disable the cone (swing) limit; resets the swing impulse on change. */
    enableConeLimit(enable: boolean): void {
        const j = this.data();
        if (enable !== j.enableConeLimit) {
            j.swingImpulse = 0;
        }
        j.enableConeLimit = enable;
    }

    /** @returns whether the cone limit is enabled. */
    isConeLimitEnabled(): boolean {
        return this.data().enableConeLimit;
    }

    /** @returns the cone half-angle limit (radians). */
    getConeLimit(): number {
        return this.data().coneAngle;
    }

    /** Set the cone half-angle limit (radians). */
    setConeLimit(angle: number): void {
        this.data().coneAngle = f32(angle);
    }

    /** @returns the current swing (cone) angle (radians). */
    getConeAngle(): number {
        return sphericalJointConeAngle(this.world, this.sim());
    }

    /** Enable/disable the twist limit; resets twist impulses on change. */
    enableTwistLimit(enable: boolean): void {
        const j = this.data();
        if (enable !== j.enableTwistLimit) {
            j.lowerTwistImpulse = 0;
            j.upperTwistImpulse = 0;
        }
        j.enableTwistLimit = enable;
    }

    /** @returns whether the twist limit is enabled. */
    isTwistLimitEnabled(): boolean {
        return this.data().enableTwistLimit;
    }

    /** @returns the lower twist limit (radians). */
    getLowerTwistLimit(): number {
        return this.data().lowerTwistAngle;
    }

    /** @returns the upper twist limit (radians). */
    getUpperTwistLimit(): number {
        return this.data().upperTwistAngle;
    }

    /** Set the twist limits (radians), clamped to ±0.99π. */
    setTwistLimits(lower: number, upper: number): void {
        const lo = f32(lower);
        const hi = f32(upper);
        const lowerAngle = lo < hi ? lo : hi;
        const upperAngle = lo > hi ? lo : hi;
        const bound = f32(f32(0.99) * PI);
        const j = this.data();
        j.lowerTwistAngle = clampf(lowerAngle, -bound, bound);
        j.upperTwistAngle = clampf(upperAngle, -bound, bound);
    }

    /** @returns the current twist angle (radians). */
    getTwistAngle(): number {
        return sphericalJointTwistAngle(this.world, this.sim());
    }

    /** Enable/disable the orientation spring; resets the spring impulse on change. */
    enableSpring(enable: boolean): void {
        const j = this.data();
        if (enable !== j.enableSpring) {
            j.springImpulse = { x: 0, y: 0, z: 0 };
        }
        j.enableSpring = enable;
    }

    /** @returns whether the orientation spring is enabled. */
    isSpringEnabled(): boolean {
        return this.data().enableSpring;
    }

    /** Set the spring target relative rotation. */
    setTargetRotation(target: Quat): void {
        this.data().targetRotation = froundConfig(target);
    }

    /** @returns the spring target relative rotation. */
    getTargetRotation(): Quat {
        return cloneQuat(this.data().targetRotation);
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
            j.motorImpulse = { x: 0, y: 0, z: 0 };
        }
        j.enableMotor = enable;
    }

    /** @returns whether the motor is enabled. */
    isMotorEnabled(): boolean {
        return this.data().enableMotor;
    }

    /**
     * Set the motor target angular velocity.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMotorVelocity(velocity: Vec3): void {
        this.data().motorVelocity = froundConfig(velocity);
    }

    /** @returns the motor target angular velocity. */
    getMotorVelocity(): Vec3 {
        return { ...this.data().motorVelocity };
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
    getMotorTorque(): Vec3 {
        return vec3.scale(this.world.invH, this.data().motorImpulse);
    }
}

/** A weld joint handle. */
export class WeldJoint extends Joint {
    private data(): WeldJointData {
        return this.sim().data as WeldJointData;
    }

    /** Set the linear spring frequency (Hz). */
    setLinearHertz(hertz: number): void {
        this.data().linearHertz = f32(hertz);
    }

    /** @returns the linear spring frequency (Hz). */
    getLinearHertz(): number {
        return this.data().linearHertz;
    }

    /** Set the linear spring damping ratio. */
    setLinearDampingRatio(dampingRatio: number): void {
        this.data().linearDampingRatio = f32(dampingRatio);
    }

    /** @returns the linear spring damping ratio. */
    getLinearDampingRatio(): number {
        return this.data().linearDampingRatio;
    }

    /** Set the angular spring frequency (Hz). */
    setAngularHertz(hertz: number): void {
        this.data().angularHertz = f32(hertz);
    }

    /** @returns the angular spring frequency (Hz). */
    getAngularHertz(): number {
        return this.data().angularHertz;
    }

    /** Set the angular spring damping ratio. */
    setAngularDampingRatio(dampingRatio: number): void {
        this.data().angularDampingRatio = f32(dampingRatio);
    }

    /** @returns the angular spring damping ratio. */
    getAngularDampingRatio(): number {
        return this.data().angularDampingRatio;
    }
}

/** A motor joint handle. */
export class MotorJoint extends Joint {
    private data(): MotorJointData {
        return this.sim().data as MotorJointData;
    }

    /**
     * Set the target relative linear velocity.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setLinearVelocity(velocity: Vec3): void {
        this.data().linearVelocity = froundConfig(velocity);
    }

    /** @returns the target relative linear velocity. */
    getLinearVelocity(): Vec3 {
        return { ...this.data().linearVelocity };
    }

    /**
     * Set the target relative angular velocity.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setAngularVelocity(velocity: Vec3): void {
        this.data().angularVelocity = froundConfig(velocity);
    }

    /** @returns the target relative angular velocity. */
    getAngularVelocity(): Vec3 {
        return { ...this.data().angularVelocity };
    }

    /**
     * Set the maximum velocity-drive torque.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMaxVelocityTorque(maxTorque: number): void {
        this.data().maxVelocityTorque = f32(maxTorque);
    }

    /** @returns the maximum velocity-drive torque. */
    getMaxVelocityTorque(): number {
        return this.data().maxVelocityTorque;
    }

    /**
     * Set the maximum velocity-drive force.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMaxVelocityForce(maxForce: number): void {
        this.data().maxVelocityForce = f32(maxForce);
    }

    /** @returns the maximum velocity-drive force. */
    getMaxVelocityForce(): number {
        return this.data().maxVelocityForce;
    }

    /** Set the linear spring frequency (Hz). */
    setLinearHertz(hertz: number): void {
        this.data().linearHertz = f32(hertz);
    }

    /** @returns the linear spring frequency (Hz). */
    getLinearHertz(): number {
        return this.data().linearHertz;
    }

    /** Set the linear spring damping ratio. */
    setLinearDampingRatio(dampingRatio: number): void {
        this.data().linearDampingRatio = f32(dampingRatio);
    }

    /** @returns the linear spring damping ratio. */
    getLinearDampingRatio(): number {
        return this.data().linearDampingRatio;
    }

    /** Set the angular spring frequency (Hz). */
    setAngularHertz(hertz: number): void {
        this.data().angularHertz = f32(hertz);
    }

    /** @returns the angular spring frequency (Hz). */
    getAngularHertz(): number {
        return this.data().angularHertz;
    }

    /** Set the angular spring damping ratio. */
    setAngularDampingRatio(dampingRatio: number): void {
        this.data().angularDampingRatio = f32(dampingRatio);
    }

    /** @returns the angular spring damping ratio. */
    getAngularDampingRatio(): number {
        return this.data().angularDampingRatio;
    }

    /** Set the maximum spring force (clamped ≥ 0). */
    setMaxSpringForce(maxForce: number): void {
        const v = f32(maxForce);
        this.data().maxSpringForce = 0 > v ? 0 : v;
    }

    /** @returns the maximum spring force. */
    getMaxSpringForce(): number {
        return this.data().maxSpringForce;
    }

    /** Set the maximum spring torque (clamped ≥ 0). */
    setMaxSpringTorque(maxTorque: number): void {
        const v = f32(maxTorque);
        this.data().maxSpringTorque = 0 > v ? 0 : v;
    }

    /** @returns the maximum spring torque. */
    getMaxSpringTorque(): number {
        return this.data().maxSpringTorque;
    }
}

/** A parallel joint handle. */
export class ParallelJoint extends Joint {
    private data(): ParallelJointData {
        return this.sim().data as ParallelJointData;
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

    /** Set the maximum corrective torque. */
    setMaxTorque(maxTorque: number): void {
        this.data().maxTorque = f32(maxTorque);
    }

    /** @returns the maximum corrective torque. */
    getMaxTorque(): number {
        return this.data().maxTorque;
    }
}

/** A wheel joint handle. */
export class WheelJoint extends Joint {
    private data(): WheelJointData {
        return this.sim().data as WheelJointData;
    }

    /** Enable/disable the suspension spring; resets the suspension impulse on change. */
    enableSuspension(enable: boolean): void {
        const j = this.data();
        if (enable !== j.enableSuspensionSpring) {
            j.enableSuspensionSpring = enable;
            j.suspensionSpringImpulse = 0;
        }
    }

    /** @returns whether the suspension spring is enabled. */
    isSuspensionEnabled(): boolean {
        return this.data().enableSuspensionSpring;
    }

    /** Set the suspension spring frequency (Hz). */
    setSuspensionHertz(hertz: number): void {
        this.data().suspensionHertz = f32(hertz);
    }

    /** @returns the suspension spring frequency (Hz). */
    getSuspensionHertz(): number {
        return this.data().suspensionHertz;
    }

    /** Set the suspension spring damping ratio. */
    setSuspensionDampingRatio(dampingRatio: number): void {
        this.data().suspensionDampingRatio = f32(dampingRatio);
    }

    /** @returns the suspension spring damping ratio. */
    getSuspensionDampingRatio(): number {
        return this.data().suspensionDampingRatio;
    }

    /** Enable/disable the suspension limit; resets limit impulses on change. */
    enableSuspensionLimit(enable: boolean): void {
        const j = this.data();
        if (j.enableSuspensionLimit !== enable) {
            j.lowerSuspensionImpulse = 0;
            j.upperSuspensionImpulse = 0;
            j.enableSuspensionLimit = enable;
        }
    }

    /** @returns whether the suspension limit is enabled. */
    isSuspensionLimitEnabled(): boolean {
        return this.data().enableSuspensionLimit;
    }

    /** @returns the lower suspension limit. */
    getLowerSuspensionLimit(): number {
        return this.data().lowerSuspensionLimit;
    }

    /** @returns the upper suspension limit. */
    getUpperSuspensionLimit(): number {
        return this.data().upperSuspensionLimit;
    }

    /** Set the suspension limits; resets limit impulses when changed. */
    setSuspensionLimits(lower: number, upper: number): void {
        const lo = f32(lower);
        const hi = f32(upper);
        const j = this.data();
        if (lo !== j.lowerSuspensionLimit || hi !== j.upperSuspensionLimit) {
            j.lowerSuspensionLimit = lo;
            j.upperSuspensionLimit = hi;
            j.lowerSuspensionImpulse = 0;
            j.upperSuspensionImpulse = 0;
        }
    }

    /**
     * Enable/disable the spin motor; resets the spin impulse on change.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    enableSpinMotor(enable: boolean): void {
        const j = this.data();
        if (j.enableSpinMotor !== enable) {
            j.spinImpulse = 0;
            j.enableSpinMotor = enable;
        }
    }

    /** @returns whether the spin motor is enabled. */
    isSpinMotorEnabled(): boolean {
        return this.data().enableSpinMotor;
    }

    /**
     * Set the spin motor target speed.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setSpinMotorSpeed(speed: number): void {
        this.data().spinSpeed = f32(speed);
    }

    /** @returns the spin motor target speed. */
    getSpinMotorSpeed(): number {
        return this.data().spinSpeed;
    }

    /**
     * Set the maximum spin torque.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMaxSpinTorque(torque: number): void {
        this.data().maxSpinTorque = f32(torque);
    }

    /** @returns the maximum spin torque. */
    getMaxSpinTorque(): number {
        return this.data().maxSpinTorque;
    }

    /** @returns the current spin speed about the spin axis. */
    getSpinSpeed(): number {
        return wheelJointSpinSpeed(this.world, this.sim());
    }

    /** @returns the spin torque applied last step. */
    getSpinTorque(): number {
        return f32(this.world.invH * this.data().spinImpulse);
    }

    /**
     * Enable/disable steering; resets the steering angular impulse on change.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    enableSteering(enable: boolean): void {
        const j = this.data();
        if (j.enableSteering !== enable) {
            j.angularImpulse = { x: 0, y: 0 };
            j.enableSteering = enable;
        }
    }

    /** @returns whether steering is enabled. */
    isSteeringEnabled(): boolean {
        return this.data().enableSteering;
    }

    /** Set the steering spring frequency (Hz). */
    setSteeringHertz(hertz: number): void {
        this.data().steeringHertz = f32(hertz);
    }

    /** @returns the steering spring frequency (Hz). */
    getSteeringHertz(): number {
        return this.data().steeringHertz;
    }

    /** Set the steering spring damping ratio. */
    setSteeringDampingRatio(dampingRatio: number): void {
        this.data().steeringDampingRatio = f32(dampingRatio);
    }

    /** @returns the steering spring damping ratio. */
    getSteeringDampingRatio(): number {
        return this.data().steeringDampingRatio;
    }

    /**
     * Set the maximum steering torque.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMaxSteeringTorque(maxTorque: number): void {
        this.data().maxSteeringTorque = f32(maxTorque);
    }

    /** @returns the maximum steering torque. */
    getMaxSteeringTorque(): number {
        return this.data().maxSteeringTorque;
    }

    /** Enable/disable the steering limit; resets limit impulses on change. */
    enableSteeringLimit(enable: boolean): void {
        const j = this.data();
        if (j.enableSteeringLimit !== enable) {
            j.lowerSteeringImpulse = 0;
            j.upperSteeringImpulse = 0;
            j.enableSteeringLimit = enable;
        }
    }

    /** @returns whether the steering limit is enabled. */
    isSteeringLimitEnabled(): boolean {
        return this.data().enableSteeringLimit;
    }

    /** @returns the lower steering limit (radians). */
    getLowerSteeringLimit(): number {
        return this.data().lowerSteeringLimit;
    }

    /** @returns the upper steering limit (radians). */
    getUpperSteeringLimit(): number {
        return this.data().upperSteeringLimit;
    }

    /** Set the steering limits (radians). */
    setSteeringLimits(lower: number, upper: number): void {
        const j = this.data();
        j.lowerSteeringLimit = f32(lower);
        j.upperSteeringLimit = f32(upper);
    }

    /**
     * Set the steering spring target angle (radians).
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setTargetSteeringAngle(radians: number): void {
        this.data().targetSteeringAngle = f32(radians);
    }

    /** @returns the steering spring target angle (radians). */
    getTargetSteeringAngle(): number {
        return this.data().targetSteeringAngle;
    }

    /** @returns the current steering angle (radians). */
    getSteeringAngle(): number {
        return wheelJointSteeringAngle(this.world, this.sim());
    }

    /** @returns the steering torque applied last step. */
    getSteeringTorque(): number {
        return f32(this.world.invH * this.data().steeringSpringImpulse);
    }
}
