#include "box3d/box3d.h"
#include <stdio.h>
#include <string.h>

static void PrintTransform(b3BodyId body)
{
    b3WorldTransform t = b3Body_GetTransform(body);
    float values[] = {t.p.x,t.p.y,t.p.z,t.q.v.x,t.q.v.y,t.q.v.z,t.q.s};
    for (int j = 0; j < 7; ++j)
    {
        unsigned bits; memcpy(&bits, values + j, 4);
        printf("%s%08x", j ? " " : "", bits);
    }
    printf("\n");
}

int main(void)
{
    for (int c = 0; c < 6; ++c)
    {
        b3WorldDef wd = b3DefaultWorldDef();
        b3WorldId world = b3CreateWorld(&wd);
        b3BodyDef bd = b3DefaultBodyDef();
        bd.type = c == 0 ? b3_staticBody : c == 1 ? b3_kinematicBody : b3_dynamicBody;
        bd.position = (b3Pos){1.25f, 3.5f, -2.75f};
        bd.angularVelocity = (b3Vec3){1, -1, 2};
        bd.linearVelocity = (b3Vec3){0.3f, -0.2f, 0.1f};
        bd.motionLocks.angularX = bd.motionLocks.angularY = bd.motionLocks.angularZ = c == 3;
        bd.isAwake = c != 4;
        bd.isEnabled = c != 5;
        b3BodyId body = b3CreateBody(world, &bd);
        b3ShapeDef sd = b3DefaultShapeDef();
        b3Sphere sphere = { {0, 0, 0}, 1 };
        b3CreateSphereShape(body, &sd, &sphere);
        for (int step = 0; step < 3; ++step)
        {
            b3World_Step(world, 1.0f / 60.0f, 4);
            PrintTransform(body);
        }
        b3Body_SetTransform(body, (b3Pos){4.25f, -2.5f, 8.75f}, (b3Quat){{0.5f, 0.5f, 0.5f}, 0.5f});
        PrintTransform(body);
        b3Body_SetType(body, c == 0 ? b3_dynamicBody : c == 1 ? b3_staticBody : b3_kinematicBody);
        PrintTransform(body);
        b3World_Step(world, 1.0f / 60.0f, 4);
        PrintTransform(body);
        b3DestroyWorld(world);
    }
}
