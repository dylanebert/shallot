#include "box3d/box3d.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void scalar(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    printf(" %08x", bits);
}

int main(void)
{
    b3WorldDef worldDef = b3DefaultWorldDef();
    b3WorldId world = b3CreateWorld(&worldDef);
    b3BodyDef bodyDef = b3DefaultBodyDef();
    bodyDef.type = b3_kinematicBody;
    b3BodyId body = b3CreateBody(world, &bodyDef);
    for (int i = 0; i < 3; ++i)
    {
        b3ShapeDef shapeDef = b3DefaultShapeDef();
        shapeDef.userData = (void*)(uintptr_t)(i + 1);
        b3Sphere sphere = { { -0.5f + 0.5f * i, 0, 0 }, 1.0f };
        b3CreateSphereShape(body, &shapeDef, &sphere);
    }
    b3Pos origin = { 10, 2, -3 };
    b3WorldTransform placement = { origin, b3Quat_identity };
    b3Capsule mover = { { 0.8f, -0.2f, 0 }, { 0.8f, 0.2f, 0 }, 0.5f };
    const int capacities[] = { 0, 1, 2, 4 };
    for (int c = 0; c < 4; ++c)
    {
        b3BodyPlaneResult planes[4];
        int count = b3Body_CollideMover(body, planes, capacities[c], origin, &mover, b3DefaultQueryFilter(), placement);
        printf("%d %d", capacities[c], count);
        for (int i = 0; i < count; ++i)
        {
            b3PlaneResult p = planes[i].result;
            printf(" %d", (int)(uintptr_t)b3Shape_GetUserData(planes[i].shapeId));
            scalar(p.plane.normal.x); scalar(p.plane.normal.y); scalar(p.plane.normal.z); scalar(p.plane.offset);
            scalar(p.point.x); scalar(p.point.y); scalar(p.point.z);
            printf(" %d %d %d", p.triangleIndex, p.childIndex, p.materialIndex);
        }
        printf("\n");
    }
    b3DestroyWorld(world);
}
