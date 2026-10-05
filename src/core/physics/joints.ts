import { component, entity, f32, u32, vec4 } from "../../engine";

const FLT_MAX = 3.4028234663852886e38;
const base = {
    /** Body entity references, resolved by the simulation. */
    a: entity,
    b: entity,
    /** Frame origins relative to body origins, not centers of mass; xyz in meters. */
    localAnchorA: vec4,
    localAnchorB: vec4,
    /** Local frame orientations as normalized (x, y, z, w) quaternions. */
    localRotationA: vec4,
    localRotationB: vec4,
    /** Force and torque thresholds for joint events, in N and N·m. */
    forceThreshold: f32,
    torqueThreshold: f32,
    /** Constraint frequency in Hz and dimensionless damping ratio. */
    constraintHertz: f32,
    constraintDampingRatio: f32,
    drawScale: f32,
    /** Boolean flag, authored as 0 or 1. */
    collideConnected: u32,
};
const baseDefaults = () => ({
    a: 0,
    b: 0,
    localAnchorA: [0, 0, 0, 0] as const,
    localAnchorB: [0, 0, 0, 0] as const,
    localRotationA: [0, 0, 0, 1] as const,
    localRotationB: [0, 0, 0, 1] as const,
    forceThreshold: FLT_MAX,
    torqueThreshold: FLT_MAX,
    constraintHertz: 60,
    constraintDampingRatio: 2,
    drawScale: 1,
    collideConnected: 0,
});

/** Distance in meters; spring frequency in Hz, damping dimensionless, forces in N and motor speed in m/s. Enable flags are 0 or 1. */
export const DistanceJoint = component(
    "DistanceJoint",
    {
        ...base,
        length: f32,
        enableSpring: u32,
        lowerSpringForce: f32,
        upperSpringForce: f32,
        hertz: f32,
        dampingRatio: f32,
        enableLimit: u32,
        minLength: f32,
        maxLength: f32,
        enableMotor: u32,
        maxMotorForce: f32,
        motorSpeed: f32,
    },
    {
        defaults: () => ({
            ...baseDefaults(),
            length: 1,
            enableSpring: 0,
            lowerSpringForce: -FLT_MAX,
            upperSpringForce: FLT_MAX,
            hertz: 0,
            dampingRatio: 0,
            enableLimit: 0,
            minLength: 0,
            maxLength: 100000,
            enableMotor: 0,
            maxMotorForce: 0,
            motorSpeed: 0,
        }),
    },
);

/** Suppresses collision between the two bodies without constraining motion. */
export const FilterJoint = component("FilterJoint", { ...base }, { defaults: baseDefaults });

/** Relative velocities in m/s and rad/s; frequencies in Hz, damping dimensionless, force in N and torque in N·m. */
export const MotorJoint = component(
    "MotorJoint",
    {
        ...base,
        linearVelocity: vec4,
        maxVelocityForce: f32,
        angularVelocity: vec4,
        maxVelocityTorque: f32,
        linearHertz: f32,
        linearDampingRatio: f32,
        maxSpringForce: f32,
        angularHertz: f32,
        angularDampingRatio: f32,
        maxSpringTorque: f32,
    },
    {
        defaults: () => ({
            ...baseDefaults(),
            linearVelocity: [0, 0, 0, 0],
            maxVelocityForce: 0,
            angularVelocity: [0, 0, 0, 0],
            maxVelocityTorque: 0,
            linearHertz: 0,
            linearDampingRatio: 0,
            maxSpringForce: 0,
            angularHertz: 0,
            angularDampingRatio: 0,
            maxSpringTorque: 0,
        }),
    },
);

/** Aligns the local frame z axes; frequency in Hz, damping dimensionless and torque in N·m. */
export const ParallelJoint = component(
    "ParallelJoint",
    {
        ...base,
        hertz: f32,
        dampingRatio: f32,
        maxTorque: f32,
    },
    { defaults: () => ({ ...baseDefaults(), hertz: 1, dampingRatio: 1, maxTorque: FLT_MAX }) },
);

/** Slides along frame A's x axis without relative rotation. Translation in meters, frequency in Hz, damping dimensionless, force in N and speed in m/s. Enable flags are 0 or 1. */
export const PrismaticJoint = component(
    "PrismaticJoint",
    {
        ...base,
        enableSpring: u32,
        hertz: f32,
        dampingRatio: f32,
        targetTranslation: f32,
        enableLimit: u32,
        lowerTranslation: f32,
        upperTranslation: f32,
        enableMotor: u32,
        maxMotorForce: f32,
        motorSpeed: f32,
    },
    {
        defaults: () => ({
            ...baseDefaults(),
            enableSpring: 0,
            hertz: 0,
            dampingRatio: 0,
            targetTranslation: 0,
            enableLimit: 0,
            lowerTranslation: 0,
            upperTranslation: 0,
            enableMotor: 0,
            maxMotorForce: 0,
            motorSpeed: 0,
        }),
    },
);

/** Rotates about the local frame z axis. Angles in radians, speed in rad/s, frequency in Hz, damping dimensionless and torque in N·m. Enable flags are 0 or 1. */
export const RevoluteJoint = component(
    "RevoluteJoint",
    {
        ...base,
        targetAngle: f32,
        enableSpring: u32,
        hertz: f32,
        dampingRatio: f32,
        enableLimit: u32,
        lowerAngle: f32,
        upperAngle: f32,
        enableMotor: u32,
        maxMotorTorque: f32,
        motorSpeed: f32,
    },
    {
        defaults: () => ({
            ...baseDefaults(),
            targetAngle: 0,
            enableSpring: 0,
            hertz: 0,
            dampingRatio: 0,
            enableLimit: 0,
            lowerAngle: 0,
            upperAngle: 0,
            enableMotor: 0,
            maxMotorTorque: 0,
            motorSpeed: 0,
        }),
    },
);

/** Ball-and-socket with optional frame alignment, cone/twist limits and motor. Angles in radians, frequency in Hz, damping dimensionless, torque in N·m and motor velocity in world-space rad/s. Enable flags are 0 or 1. */
export const SphericalJoint = component(
    "SphericalJoint",
    {
        ...base,
        enableSpring: u32,
        hertz: f32,
        dampingRatio: f32,
        /** Normalized (x, y, z, w) rotation of frame B relative to frame A. */
        targetRotation: vec4,
        enableConeLimit: u32,
        coneAngle: f32,
        enableTwistLimit: u32,
        lowerTwistAngle: f32,
        upperTwistAngle: f32,
        enableMotor: u32,
        maxMotorTorque: f32,
        motorVelocity: vec4,
    },
    {
        defaults: () => ({
            ...baseDefaults(),
            enableSpring: 0,
            hertz: 0,
            dampingRatio: 0,
            targetRotation: [0, 0, 0, 1],
            enableConeLimit: 0,
            coneAngle: 0,
            enableTwistLimit: 0,
            lowerTwistAngle: 0,
            upperTwistAngle: 0,
            enableMotor: 0,
            maxMotorTorque: 0,
            motorVelocity: [0, 0, 0, 0],
        }),
    },
);

/** Locks the two authored frames together. Frequencies in Hz (zero means rigid), damping ratios dimensionless. Frame B's rotation is never derived from body poses. */
export const WeldJoint = component(
    "WeldJoint",
    {
        ...base,
        linearHertz: f32,
        angularHertz: f32,
        linearDampingRatio: f32,
        angularDampingRatio: f32,
    },
    {
        defaults: () => ({
            ...baseDefaults(),
            linearHertz: 0,
            angularHertz: 0,
            linearDampingRatio: 0,
            angularDampingRatio: 0,
        }),
    },
);

/** Suspension along frame A's x axis, spin about frame B's z axis and optional steering. Translation in meters, angles in radians, speed in rad/s, frequency in Hz, damping dimensionless and torque in N·m. Enable flags are 0 or 1. */
export const WheelJoint = component(
    "WheelJoint",
    {
        ...base,
        enableSuspensionSpring: u32,
        suspensionHertz: f32,
        suspensionDampingRatio: f32,
        enableSuspensionLimit: u32,
        lowerSuspensionLimit: f32,
        upperSuspensionLimit: f32,
        enableSpinMotor: u32,
        maxSpinTorque: f32,
        spinSpeed: f32,
        enableSteering: u32,
        steeringHertz: f32,
        steeringDampingRatio: f32,
        targetSteeringAngle: f32,
        maxSteeringTorque: f32,
        enableSteeringLimit: u32,
        lowerSteeringLimit: f32,
        upperSteeringLimit: f32,
    },
    {
        defaults: () => ({
            ...baseDefaults(),
            enableSuspensionSpring: 1,
            suspensionHertz: 1,
            suspensionDampingRatio: 0.7,
            enableSuspensionLimit: 0,
            lowerSuspensionLimit: 0,
            upperSuspensionLimit: 0,
            enableSpinMotor: 0,
            maxSpinTorque: 0,
            spinSpeed: 0,
            enableSteering: 0,
            steeringHertz: 1,
            steeringDampingRatio: 0.7,
            targetSteeringAngle: 0,
            maxSteeringTorque: 0,
            enableSteeringLimit: 0,
            lowerSteeringLimit: 0,
            upperSteeringLimit: 0,
        }),
    },
);
