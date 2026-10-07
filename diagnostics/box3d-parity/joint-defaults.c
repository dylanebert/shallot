// SPDX-License-Identifier: MIT
#include <box3d/box3d.h>
#include "joint.h"
#include "physics_world.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if !defined(B3_SIMD_SSE2) && !defined(B3_SIMD_NEON)
#error This harness requires SSE2 (oracle authority) or NEON (native-only evidence).
#endif

enum Kind { PARALLEL, DISTANCE, FILTER, MOTOR, PRISMATIC, REVOLUTE, SPHERICAL, WELD, WHEEL, KIND_COUNT };

typedef struct Values
{
    float s[16];
    float p[16];
    uint32_t flags;
} Values;

static uint32_t next(uint32_t* state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static Values values(int kind, int sample)
{
    uint32_t state = 0x47d7f7ccu ^ (uint32_t)kind * 0x9e3779b9u ^ (uint32_t)sample * 0x85ebca6bu;
    Values v;
    for (int i = 0; i < 16; ++i) v.s[i] = (float)((int)(next(&state) & 2047u) - 1024) / 64.0f;
    for (int i = 0; i < 16; ++i) v.p[i] = (float)((next(&state) & 1023u) + 1u) / 64.0f;
    v.flags = next(&state);
    return v;
}

static void emit(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    printf(" %08x", bits);
}

static void emit_vec3(b3Vec3 v) { emit(v.x); emit(v.y); emit(v.z); }
static void emit_quat(b3Quat q) { emit_vec3(q.v); emit(q.s); }
static void emit_transform(b3Transform t) { emit_vec3(t.p); emit_quat(t.q); }
static void ordered(float a, float b, float* lower, float* upper)
{
    *lower = a < b ? a : b;
    *upper = a < b ? b : a;
}

static b3JointId create_joint(int kind, b3WorldId world, b3BodyId a, b3BodyId b)
{
    switch (kind)
    {
        case PARALLEL: {
            b3ParallelJointDef d = b3DefaultParallelJointDef(); d.base.bodyIdA = a; d.base.bodyIdB = b;
            return b3CreateParallelJoint(world, &d);
        }
        case DISTANCE: {
            b3DistanceJointDef d = b3DefaultDistanceJointDef(); d.base.bodyIdA = a; d.base.bodyIdB = b;
            return b3CreateDistanceJoint(world, &d);
        }
        case FILTER: {
            b3FilterJointDef d = b3DefaultFilterJointDef(); d.base.bodyIdA = a; d.base.bodyIdB = b;
            return b3CreateFilterJoint(world, &d);
        }
        case MOTOR: {
            b3MotorJointDef d = b3DefaultMotorJointDef(); d.base.bodyIdA = a; d.base.bodyIdB = b;
            return b3CreateMotorJoint(world, &d);
        }
        case PRISMATIC: {
            b3PrismaticJointDef d = b3DefaultPrismaticJointDef(); d.base.bodyIdA = a; d.base.bodyIdB = b;
            return b3CreatePrismaticJoint(world, &d);
        }
        case REVOLUTE: {
            b3RevoluteJointDef d = b3DefaultRevoluteJointDef(); d.base.bodyIdA = a; d.base.bodyIdB = b;
            return b3CreateRevoluteJoint(world, &d);
        }
        case SPHERICAL: {
            b3SphericalJointDef d = b3DefaultSphericalJointDef(); d.base.bodyIdA = a; d.base.bodyIdB = b;
            return b3CreateSphericalJoint(world, &d);
        }
        case WELD: {
            b3WeldJointDef d = b3DefaultWeldJointDef(); d.base.bodyIdA = a; d.base.bodyIdB = b;
            return b3CreateWeldJoint(world, &d);
        }
        case WHEEL: {
            b3WheelJointDef d = b3DefaultWheelJointDef(); d.base.bodyIdA = a; d.base.bodyIdB = b;
            return b3CreateWheelJoint(world, &d);
        }
        default: return b3_nullJointId;
    }
}

static void set_values(int kind, b3JointId joint, Values v)
{
    b3Transform a = { {v.s[0], v.s[1], v.s[2]}, (v.flags & 2u) ? (b3Quat){{0, 0, 1}, 0} : b3Quat_identity };
    b3Transform b = { {v.s[3], v.s[4], v.s[5]}, (v.flags & 4u) ? (b3Quat){{1, 0, 0}, 0} : b3Quat_identity };
    b3Joint_SetLocalFrameA(joint, a);
    b3Joint_SetLocalFrameB(joint, b);
    b3Joint_SetCollideConnected(joint, (v.flags & 1u) != 0);
    b3Joint_SetConstraintTuning(joint, v.p[0], v.p[1]);
    b3Joint_SetForceThreshold(joint, v.p[2]);
    b3Joint_SetTorqueThreshold(joint, v.p[3]);
    b3Joint_SetUserData(joint, (void*)(uintptr_t)1);

    float lo, hi;
    switch (kind)
    {
        case PARALLEL:
            b3ParallelJoint_SetSpringHertz(joint, v.p[4]);
            b3ParallelJoint_SetSpringDampingRatio(joint, v.p[5]);
            b3ParallelJoint_SetMaxTorque(joint, v.p[6]);
            break;
        case DISTANCE:
            b3DistanceJoint_SetLength(joint, v.p[4]);
            ordered(v.s[6], v.s[7], &lo, &hi); b3DistanceJoint_SetSpringForceRange(joint, lo, hi);
            b3DistanceJoint_EnableSpring(joint, (v.flags & 8u) != 0);
            b3DistanceJoint_SetSpringHertz(joint, v.p[5]);
            b3DistanceJoint_SetSpringDampingRatio(joint, v.p[6]);
            b3DistanceJoint_EnableLimit(joint, (v.flags & 16u) != 0);
            ordered(v.p[7], v.p[8], &lo, &hi); b3DistanceJoint_SetLengthRange(joint, lo, hi);
            b3DistanceJoint_EnableMotor(joint, (v.flags & 32u) != 0);
            b3DistanceJoint_SetMotorSpeed(joint, v.s[8]);
            b3DistanceJoint_SetMaxMotorForce(joint, v.p[9]);
            break;
        case MOTOR:
            b3MotorJoint_SetLinearVelocity(joint, (b3Vec3){v.s[6], v.s[7], v.s[8]});
            b3MotorJoint_SetAngularVelocity(joint, (b3Vec3){v.s[9], v.s[10], v.s[11]});
            b3MotorJoint_SetMaxVelocityForce(joint, v.p[4]);
            b3MotorJoint_SetMaxVelocityTorque(joint, v.p[5]);
            b3MotorJoint_SetLinearHertz(joint, v.p[6]);
            b3MotorJoint_SetLinearDampingRatio(joint, v.p[7]);
            b3MotorJoint_SetMaxSpringForce(joint, v.p[8]);
            b3MotorJoint_SetAngularHertz(joint, v.p[9]);
            b3MotorJoint_SetAngularDampingRatio(joint, v.p[10]);
            b3MotorJoint_SetMaxSpringTorque(joint, v.p[11]);
            break;
        case PRISMATIC:
            b3PrismaticJoint_EnableSpring(joint, (v.flags & 8u) != 0);
            b3PrismaticJoint_SetSpringHertz(joint, v.p[4]);
            b3PrismaticJoint_SetSpringDampingRatio(joint, v.p[5]);
            b3PrismaticJoint_SetTargetTranslation(joint, v.s[6]);
            b3PrismaticJoint_EnableLimit(joint, (v.flags & 16u) != 0);
            ordered(v.s[7], v.s[8], &lo, &hi); b3PrismaticJoint_SetLimits(joint, lo, hi);
            b3PrismaticJoint_EnableMotor(joint, (v.flags & 32u) != 0);
            b3PrismaticJoint_SetMotorSpeed(joint, v.s[9]);
            b3PrismaticJoint_SetMaxMotorForce(joint, v.p[6]);
            break;
        case REVOLUTE:
            b3RevoluteJoint_EnableSpring(joint, (v.flags & 8u) != 0);
            b3RevoluteJoint_SetSpringHertz(joint, v.p[4]);
            b3RevoluteJoint_SetSpringDampingRatio(joint, v.p[5]);
            b3RevoluteJoint_SetTargetAngle(joint, v.s[6] / 8.0f);
            b3RevoluteJoint_EnableLimit(joint, (v.flags & 16u) != 0);
            ordered(v.s[7] / 8.0f, v.s[8] / 8.0f, &lo, &hi); b3RevoluteJoint_SetLimits(joint, lo, hi);
            b3RevoluteJoint_EnableMotor(joint, (v.flags & 32u) != 0);
            b3RevoluteJoint_SetMotorSpeed(joint, v.s[9]);
            b3RevoluteJoint_SetMaxMotorTorque(joint, v.p[6]);
            break;
        case SPHERICAL:
            b3SphericalJoint_EnableConeLimit(joint, (v.flags & 8u) != 0);
            b3SphericalJoint_SetConeLimit(joint, v.p[4] / 8.0f);
            b3SphericalJoint_EnableTwistLimit(joint, (v.flags & 16u) != 0);
            ordered(v.s[6] / 8.0f, v.s[7] / 8.0f, &lo, &hi); b3SphericalJoint_SetTwistLimits(joint, lo, hi);
            b3SphericalJoint_EnableSpring(joint, (v.flags & 32u) != 0);
            b3SphericalJoint_SetTargetRotation(joint, (v.flags & 64u) ? (b3Quat){{0, 1, 0}, 0} : b3Quat_identity);
            b3SphericalJoint_SetSpringHertz(joint, v.p[5]);
            b3SphericalJoint_SetSpringDampingRatio(joint, v.p[6]);
            b3SphericalJoint_EnableMotor(joint, (v.flags & 128u) != 0);
            b3SphericalJoint_SetMotorVelocity(joint, (b3Vec3){v.s[8], v.s[9], v.s[10]});
            b3SphericalJoint_SetMaxMotorTorque(joint, v.p[7]);
            break;
        case WELD:
            b3WeldJoint_SetLinearHertz(joint, v.p[4]);
            b3WeldJoint_SetLinearDampingRatio(joint, v.p[5]);
            b3WeldJoint_SetAngularHertz(joint, v.p[6]);
            b3WeldJoint_SetAngularDampingRatio(joint, v.p[7]);
            break;
        case WHEEL:
            b3WheelJoint_EnableSuspension(joint, (v.flags & 8u) != 0);
            b3WheelJoint_SetSuspensionHertz(joint, v.p[4]);
            b3WheelJoint_SetSuspensionDampingRatio(joint, v.p[5]);
            b3WheelJoint_EnableSuspensionLimit(joint, (v.flags & 16u) != 0);
            ordered(v.s[6], v.s[7], &lo, &hi); b3WheelJoint_SetSuspensionLimits(joint, lo, hi);
            b3WheelJoint_EnableSpinMotor(joint, (v.flags & 32u) != 0);
            b3WheelJoint_SetSpinMotorSpeed(joint, v.s[8]);
            b3WheelJoint_SetMaxSpinTorque(joint, v.p[6]);
            b3WheelJoint_EnableSteering(joint, (v.flags & 64u) != 0);
            b3WheelJoint_SetSteeringHertz(joint, v.p[7]);
            b3WheelJoint_SetSteeringDampingRatio(joint, v.p[8]);
            b3WheelJoint_SetMaxSteeringTorque(joint, v.p[9]);
            b3WheelJoint_EnableSteeringLimit(joint, (v.flags & 128u) != 0);
            ordered(v.s[9] / 8.0f, v.s[10] / 8.0f, &lo, &hi); b3WheelJoint_SetSteeringLimits(joint, lo, hi);
            b3WheelJoint_SetTargetSteeringAngle(joint, v.s[11] / 8.0f);
            break;
    }
}

static void dump(int kind, b3JointId joint)
{
    emit_transform(b3Joint_GetLocalFrameA(joint));
    emit_transform(b3Joint_GetLocalFrameB(joint));
    emit(b3Joint_GetCollideConnected(joint) ? 1.0f : 0.0f);
    float hertz, damping, lower, upper;
    b3Joint_GetConstraintTuning(joint, &hertz, &damping);
    emit(hertz); emit(damping); emit(b3Joint_GetForceThreshold(joint)); emit(b3Joint_GetTorqueThreshold(joint));
    emit((float)b3Joint_GetType(joint));
    emit(b3Joint_GetUserData(joint) != NULL ? 1.0f : 0.0f);
    b3World* world = b3GetWorld(joint.world0);
    emit(b3GetJointFullId(world, joint)->drawScale);
    switch (kind)
    {
        case PARALLEL:
            emit(b3ParallelJoint_GetSpringHertz(joint)); emit(b3ParallelJoint_GetSpringDampingRatio(joint)); emit(b3ParallelJoint_GetMaxTorque(joint));
            break;
        case DISTANCE:
            emit(b3DistanceJoint_GetLength(joint)); emit(b3DistanceJoint_IsSpringEnabled(joint) ? 1.0f : 0.0f);
            b3DistanceJoint_GetSpringForceRange(joint, &lower, &upper); emit(lower); emit(upper);
            emit(b3DistanceJoint_GetSpringHertz(joint)); emit(b3DistanceJoint_GetSpringDampingRatio(joint));
            emit(b3DistanceJoint_IsLimitEnabled(joint) ? 1.0f : 0.0f); emit(b3DistanceJoint_GetMinLength(joint)); emit(b3DistanceJoint_GetMaxLength(joint));
            emit(b3DistanceJoint_IsMotorEnabled(joint) ? 1.0f : 0.0f); emit(b3DistanceJoint_GetMaxMotorForce(joint)); emit(b3DistanceJoint_GetMotorSpeed(joint));
            break;
        case MOTOR:
            emit_vec3(b3MotorJoint_GetLinearVelocity(joint)); emit(b3MotorJoint_GetMaxVelocityForce(joint));
            emit_vec3(b3MotorJoint_GetAngularVelocity(joint)); emit(b3MotorJoint_GetMaxVelocityTorque(joint));
            emit(b3MotorJoint_GetLinearHertz(joint)); emit(b3MotorJoint_GetLinearDampingRatio(joint)); emit(b3MotorJoint_GetMaxSpringForce(joint));
            emit(b3MotorJoint_GetAngularHertz(joint)); emit(b3MotorJoint_GetAngularDampingRatio(joint)); emit(b3MotorJoint_GetMaxSpringTorque(joint));
            break;
        case PRISMATIC:
            emit(b3PrismaticJoint_IsSpringEnabled(joint) ? 1.0f : 0.0f); emit(b3PrismaticJoint_GetSpringHertz(joint)); emit(b3PrismaticJoint_GetSpringDampingRatio(joint)); emit(b3PrismaticJoint_GetTargetTranslation(joint));
            emit(b3PrismaticJoint_IsLimitEnabled(joint) ? 1.0f : 0.0f); emit(b3PrismaticJoint_GetLowerLimit(joint)); emit(b3PrismaticJoint_GetUpperLimit(joint));
            emit(b3PrismaticJoint_IsMotorEnabled(joint) ? 1.0f : 0.0f); emit(b3PrismaticJoint_GetMaxMotorForce(joint)); emit(b3PrismaticJoint_GetMotorSpeed(joint));
            break;
        case REVOLUTE:
            emit(b3RevoluteJoint_IsSpringEnabled(joint) ? 1.0f : 0.0f); emit(b3RevoluteJoint_GetSpringHertz(joint)); emit(b3RevoluteJoint_GetSpringDampingRatio(joint)); emit(b3RevoluteJoint_GetTargetAngle(joint));
            emit(b3RevoluteJoint_IsLimitEnabled(joint) ? 1.0f : 0.0f); emit(b3RevoluteJoint_GetLowerLimit(joint)); emit(b3RevoluteJoint_GetUpperLimit(joint));
            emit(b3RevoluteJoint_IsMotorEnabled(joint) ? 1.0f : 0.0f); emit(b3RevoluteJoint_GetMaxMotorTorque(joint)); emit(b3RevoluteJoint_GetMotorSpeed(joint));
            break;
        case SPHERICAL:
            emit(b3SphericalJoint_IsConeLimitEnabled(joint) ? 1.0f : 0.0f); emit(b3SphericalJoint_GetConeLimit(joint));
            emit(b3SphericalJoint_IsTwistLimitEnabled(joint) ? 1.0f : 0.0f); emit(b3SphericalJoint_GetLowerTwistLimit(joint)); emit(b3SphericalJoint_GetUpperTwistLimit(joint));
            emit(b3SphericalJoint_IsSpringEnabled(joint) ? 1.0f : 0.0f); emit_quat(b3SphericalJoint_GetTargetRotation(joint)); emit(b3SphericalJoint_GetSpringHertz(joint)); emit(b3SphericalJoint_GetSpringDampingRatio(joint));
            emit(b3SphericalJoint_IsMotorEnabled(joint) ? 1.0f : 0.0f); emit_vec3(b3SphericalJoint_GetMotorVelocity(joint)); emit(b3SphericalJoint_GetMaxMotorTorque(joint));
            break;
        case WELD:
            emit(b3WeldJoint_GetLinearHertz(joint)); emit(b3WeldJoint_GetLinearDampingRatio(joint)); emit(b3WeldJoint_GetAngularHertz(joint)); emit(b3WeldJoint_GetAngularDampingRatio(joint));
            break;
        case WHEEL:
            emit(b3WheelJoint_IsSuspensionEnabled(joint) ? 1.0f : 0.0f); emit(b3WheelJoint_GetSuspensionHertz(joint)); emit(b3WheelJoint_GetSuspensionDampingRatio(joint));
            emit(b3WheelJoint_IsSuspensionLimitEnabled(joint) ? 1.0f : 0.0f); emit(b3WheelJoint_GetLowerSuspensionLimit(joint)); emit(b3WheelJoint_GetUpperSuspensionLimit(joint));
            emit(b3WheelJoint_IsSpinMotorEnabled(joint) ? 1.0f : 0.0f); emit(b3WheelJoint_GetSpinMotorSpeed(joint)); emit(b3WheelJoint_GetMaxSpinTorque(joint));
            emit(b3WheelJoint_IsSteeringEnabled(joint) ? 1.0f : 0.0f); emit(b3WheelJoint_GetSteeringHertz(joint)); emit(b3WheelJoint_GetSteeringDampingRatio(joint)); emit(b3WheelJoint_GetMaxSteeringTorque(joint));
            emit(b3WheelJoint_IsSteeringLimitEnabled(joint) ? 1.0f : 0.0f); emit(b3WheelJoint_GetLowerSteeringLimit(joint)); emit(b3WheelJoint_GetUpperSteeringLimit(joint)); emit(b3WheelJoint_GetTargetSteeringAngle(joint));
            break;
    }
    b3JointSim* sim = b3GetJointSimCheckType(joint, b3Joint_GetType(joint));
    emit(sim->constraintSoftness.biasRate); emit(sim->constraintSoftness.massScale); emit(sim->constraintSoftness.impulseScale);
}

int main(void)
{
    b3WorldDef wd = b3DefaultWorldDef(); wd.gravity = b3Vec3_zero;
    b3WorldId world = b3CreateWorld(&wd);
    b3BodyDef bd = b3DefaultBodyDef(); bd.type = b3_dynamicBody;
    b3BodyId a = b3CreateBody(world, &bd); bd.position = (b3Pos){2, 0, 0}; b3BodyId b = b3CreateBody(world, &bd);
    for (int kind = 0; kind < KIND_COUNT; ++kind)
    {
        for (int sample = 0; sample <= 8; ++sample)
        {
            b3JointId joint = create_joint(kind, world, a, b);
            if (sample != 0) set_values(kind, joint, values(kind, sample));
            printf("%d %d", kind, sample); dump(kind, joint); putchar('\n');
            b3DestroyJoint(joint, false);
        }
    }
    b3DestroyWorld(world);
    return 0;
}
