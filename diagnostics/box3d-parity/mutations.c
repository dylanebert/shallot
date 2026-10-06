// SPDX-License-Identifier: MIT
// Native allocation denominator for mutation-allocation.fixture.ts.
#include <box3d/box3d.h>
#include <stdio.h>
#include <stdlib.h>

static int allocations;
static b3ShapeId shape;
static b3JointId distance, spherical, wheel, motor;
static void* allocate(int32_t size, int32_t alignment)
{
    void* p = NULL;
    if (alignment < (int32_t)sizeof(void*)) alignment = (int32_t)sizeof(void*);
    if (posix_memalign(&p, (size_t)alignment, (size_t)size) != 0) abort();
    ++allocations;
    return p;
}
static void release(void* p) { free(p); }
static void mutate(b3WorldId world, b3BodyId a, int i)
{
    bool on = (i & 1) != 0;
    b3Body_SetType(a, on ? b3_staticBody : b3_dynamicBody);
    b3Filter filter = b3DefaultFilter();
    filter.categoryBits = on ? 2 : 1;
    filter.maskBits = on ? 2 : 1;
    b3Shape_SetFilter(shape, filter, true);
    b3Shape_EnableSensorEvents(shape, on);
    b3Shape_EnableContactEvents(shape, on);
    b3Shape_EnableHitEvents(shape, on);
    b3Joint_SetCollideConnected(distance, on);
    b3DistanceJoint_EnableMotor(distance, on);
    b3DistanceJoint_SetLength(distance, 2);
    b3DistanceJoint_SetLengthRange(distance, 1, 4);
    b3SphericalJoint_EnableSpring(spherical, on);
    b3SphericalJoint_EnableMotor(spherical, on);
    b3SphericalJoint_SetTargetRotation(spherical, b3Quat_identity);
    b3SphericalJoint_SetMotorVelocity(spherical, (b3Vec3){0.1f, 0.2f, 0.3f});
    b3Joint_WakeBodies(spherical);
    b3WheelJoint_EnableSteering(wheel, on);
    b3WheelJoint_EnableSuspension(wheel, on);
    b3WheelJoint_EnableSuspensionLimit(wheel, on);
    b3WheelJoint_EnableSteeringLimit(wheel, on);
    b3WheelJoint_EnableSpinMotor(wheel, on);
    b3WheelJoint_SetSuspensionLimits(wheel, -1, 1);
    b3WheelJoint_SetSpinMotorSpeed(wheel, 2);
    b3Joint_WakeBodies(wheel);
    b3WheelJoint_SetTargetSteeringAngle(wheel, 0.1f);
    b3Joint_WakeBodies(wheel);
    b3MotorJoint_SetLinearVelocity(motor, (b3Vec3){0.1f, 0.2f, 0.3f});
    b3Joint_WakeBodies(motor);
    b3MotorJoint_SetAngularVelocity(motor, (b3Vec3){0.1f, 0.2f, 0.3f});
    b3Joint_WakeBodies(motor);
    b3MotorJoint_SetMaxSpringForce(motor, -1);
    b3MotorJoint_SetMaxSpringTorque(motor, 1);
    b3BodyDef bd = b3DefaultBodyDef();
    bd.type = b3_dynamicBody;
    b3BodyId temporary = b3CreateBody(world, &bd);
    b3ShapeDef sd = b3DefaultShapeDef();
    b3Sphere sphere = { .center = {0, 0, 0}, .radius = 1 };
    b3CreateSphereShape(temporary, &sd, &sphere);
    b3DestroyBody(temporary);
}
int main(void)
{
    b3SetAllocator(allocate, release);
    b3WorldDef wd = b3DefaultWorldDef();
    wd.gravity = (b3Vec3){0, 0, 0};
    b3WorldId world = b3CreateWorld(&wd);
    b3BodyDef bd = b3DefaultBodyDef();
    bd.type = b3_dynamicBody;
    b3BodyId a = b3CreateBody(world, &bd);
    bd.position = (b3Pos){4, 0, 0};
    b3BodyId b = b3CreateBody(world, &bd);
    b3DistanceJointDef dj = b3DefaultDistanceJointDef(); dj.base.bodyIdA = a; dj.base.bodyIdB = b;
    distance = b3CreateDistanceJoint(world, &dj);
    b3SphericalJointDef sj = b3DefaultSphericalJointDef(); sj.base.bodyIdA = a; sj.base.bodyIdB = b;
    spherical = b3CreateSphericalJoint(world, &sj);
    b3WheelJointDef whj = b3DefaultWheelJointDef(); whj.base.bodyIdA = a; whj.base.bodyIdB = b;
    wheel = b3CreateWheelJoint(world, &whj);
    b3MotorJointDef mj = b3DefaultMotorJointDef(); mj.base.bodyIdA = a; mj.base.bodyIdB = b;
    motor = b3CreateMotorJoint(world, &mj);
    b3ShapeDef sd = b3DefaultShapeDef();
    b3Sphere sphere = { .center = {0, 0, 0}, .radius = 1 };
    shape = b3CreateSphereShape(a, &sd, &sphere);
    for (int i = 0; i < 1200; ++i) mutate(world, a, i);
    int before = allocations;
    for (int i = 1200; i < 1800; ++i) mutate(world, a, i);
    printf("native warmed mutation allocations: %d for 600 iterations\n", allocations - before);
    b3DestroyWorld(world);
}
