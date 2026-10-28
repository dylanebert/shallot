#include "box3d/box3d.h"
#include <stdio.h>

int main(void)
{
    printf("[");
    for (int c = 0; c < 6; ++c)
    {
        b3WorldDef wd = b3DefaultWorldDef();
        b3WorldId world = b3CreateWorld(&wd);
        b3BodyDef bd = b3DefaultBodyDef();
        bd.type = c == 0 ? b3_staticBody : c == 1 ? b3_kinematicBody : b3_dynamicBody;
        bd.angularVelocity = (b3Vec3){1, -1, 2};
        bd.isAwake = c != 4;
        bd.isEnabled = c != 5;
        if (c == 3) bd.motionLocks = (b3MotionLocks){.angularX = true, .angularY = true, .angularZ = true};
        b3BodyId body = b3CreateBody(world, &bd);
        b3ShapeDef sd = b3DefaultShapeDef();
        b3Sphere sphere = {{0, 0, 0}, 1};
        b3CreateSphereShape(body, &sd, &sphere);
        for (int step = 0; step < 3; ++step)
        {
            if (step == 1) b3Body_SetAngularVelocity(body, (b3Vec3){3, -2, 4});
            if (step != 1) b3World_Step(world, 1.0f / 60.0f, 4);
            b3Vec3 v = b3Body_GetAngularVelocity(body);
            printf("%s[%.9g,%.9g,%.9g]", c || step ? "," : "", v.x, v.y, v.z);
        }
        b3DestroyWorld(world);
    }
    printf("]\n");
}
