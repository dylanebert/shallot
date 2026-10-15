#include "box3d/box3d.h"
#include "physics_world.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static void vector(b3Vec3 v) { printf("%u,%u,%u", bits(v.x), bits(v.y), bits(v.z)); }
static b3WorldId world(float speed) {
    b3WorldDef d = b3DefaultWorldDef();
    d.gravity = (b3Vec3){0, 0, 0};
    d.maximumLinearSpeed = speed;
    return b3CreateWorld(&d);
}
static void sphere(b3BodyId b, b3Vec3 center) {
    b3ShapeDef d = b3DefaultShapeDef();
    b3Sphere s = {center, 0.5f};
    b3CreateSphereShape(b, &d, &s);
}
static b3BodyId body(b3WorldId w, b3Vec3 p, b3Vec3 axis, float angle) {
    b3BodyDef d = b3DefaultBodyDef();
    d.type = b3_dynamicBody;
    d.position = p;
    d.rotation = b3MakeQuatFromAxisAngle(axis, angle);
    return b3CreateBody(w, &d);
}
int main(void) {
    b3WorldId w = world(400.0f);
    b3BodyId b = body(w, (b3Vec3){2.1f, 3.2f, 4.3f}, (b3Vec3){0, 1, 0}, 0.4f);
    sphere(b, (b3Vec3){0.2f, 0.3f, 0.4f});
    b3Quat q = b3MakeQuatFromAxisAngle((b3Vec3){1, 0, 0}, 0.7f);
    b3Body_SetTargetTransform(b, (b3WorldTransform){{5.5f, 6.5f, 7.5f}, q}, 1.0f / 60.0f, false);
    printf("{\"targetLinear\":["); vector(b3Body_GetLinearVelocity(b));
    printf("],\"targetAngular\":["); vector(b3Body_GetAngularVelocity(b));
    // Preserve the host's f64 authoring-to-f32 storage boundary as well as the native f32 call.
    b3Vec3 offset = b3RotateVector(q, b3Body_GetLocalCenter(b));
    b3Pos hostCenter = {(float)((double)offset.x + 5.6), (float)((double)offset.y + 6.7), (float)((double)offset.z + 7.8)};
    const float hostInvDt = (float)(1.0 / (1.0 / 60.0));
    printf("],\"targetHostLinear\":[");
    vector(b3MulSV(hostInvDt, b3SubPos(hostCenter, b3Body_GetWorldCenter(b))));
    b3Quat q1 = b3Body_GetRotation(b);
    b3Quat dq = {b3Sub(q.v, q1.v), q.s - q1.s};
    printf("],\"targetHostAngular\":[");
    vector(b3MulSV(2.0f * hostInvDt, b3MulQuat(dq, b3Conjugate(q1)).v));
    b3DestroyWorld(w);

    w = world(1.0f);
    b = body(w, (b3Vec3){0, 0, 0}, (b3Vec3){0, 1, 0}, 0.4f);
    sphere(b, (b3Vec3){0, 0, 0});
    b3Vec3 impulse = {1234.5f, 2345.6f, 3456.7f};
    b3Body_ApplyAngularImpulse(b, impulse, false);
    printf("],\"impulseAngular\":["); vector(b3Body_GetAngularVelocity(b));
    b3Body_ApplyLinearImpulseToCenter(b, impulse, false);
    printf("],\"impulseLinear\":["); vector(b3Body_GetLinearVelocity(b));
    b3DestroyWorld(w);

    w = world(400.0f);
    b3BodyId a = body(w, (b3Vec3){1.2f, 2.3f, 3.4f}, (b3Vec3){0, 1, 0}, 0.4f);
    b = body(w, (b3Vec3){4.5f, 5.6f, 6.7f}, (b3Vec3){1, 0, 0}, 0.7f);
    b3Body_SetLinearVelocity(a, (b3Vec3){0.1f, 0.2f, 0.3f});
    b3Body_SetAngularVelocity(a, (b3Vec3){0.4f, 0.5f, 0.6f});
    b3Body_SetLinearVelocity(b, (b3Vec3){0.7f, 0.8f, 0.9f});
    b3Body_SetAngularVelocity(b, (b3Vec3){1.1f, 1.2f, 1.3f});
    sphere(a, (b3Vec3){0.2f, 0.3f, 0.4f});
    sphere(b, (b3Vec3){0.3f, 0.4f, 0.5f});
    b3PrismaticJointDef d = b3DefaultPrismaticJointDef();
    d.base.bodyIdA = a;
    d.base.bodyIdB = b;
    d.base.localFrameA = (b3Transform){{0.6f, 0.7f, 0.8f}, b3MakeQuatFromAxisAngle((b3Vec3){0, 0, 1}, 0.3f)};
    d.base.localFrameB = (b3Transform){{0.9f, 1.1f, 1.2f}, b3Quat_identity};
    b3JointId joint = b3CreatePrismaticJoint(w, &d);
    printf("],\"prismaticSpeed\":%u", bits(b3PrismaticJoint_GetSpeed(joint)));
    b3DestroyWorld(w);

    w = world(400.0f);
    b = body(w, (b3Vec3){0, 0, 0}, (b3Vec3){0, 1, 0}, 0.0f);
    b3ShapeDef shape = b3DefaultShapeDef();
    b3Sphere s = {{0.25f, 0.5f, -0.25f}, 0.5f};
    shape.density = 2.5f;
    b3CreateSphereShape(b, &shape, &s);
    shape.density = 3.0f;
    b3CreateSphereShape(b, &shape, &s);
    shape.density = 1.0f;
    b3Capsule capsule = {{0, -1, 0}, {0, 1, 0}, 0.2f};
    b3CreateCapsuleShape(b, &shape, &capsule);
    printf(",\"shapeMass\":%u,\"materialDefaults\":[", bits(b3Body_GetMass(b)));
    const float values[] = {0.0f, -0.0f, 0.2f, 0.3f, 0.5f, 1.0f, 1e-20f, 1e20f};
    b3World* state = b3GetWorldFromId(w);
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            float a = values[i], c = values[j];
            printf("%s[%u,%u,%u,%u]", i || j ? "," : "", bits(a), bits(c),
                bits(state->frictionCallback(a, 0, c, 0)),
                bits(state->restitutionCallback(a, 0, c, 0)));
        }
    }
    puts("]}");
    b3DestroyWorld(w);
    return 0;
}
