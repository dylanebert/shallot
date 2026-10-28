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
    b3WorldDef wd = b3DefaultWorldDef();
    b3WorldId world = b3CreateWorld(&wd);
    b3MeshData* mesh = b3CreateTorusMesh(8, 8, 2.0f, 1.0f);
    for (int kind = 0; kind < 4; ++kind)
    {
        b3BodyDef bd = b3DefaultBodyDef();
        bd.type = kind == 3 ? b3_staticBody : b3_dynamicBody;
        b3BodyId body = b3CreateBody(world, &bd);
        for (int i = 0; i < 3; ++i)
        {
            b3ShapeDef sd = b3DefaultShapeDef();
            sd.userData = (void*)(uintptr_t)(i + 1);
            sd.filter.categoryBits = (1ull << i) | (1ull << (40 + i));
            b3Vec3 center = { 0.5f * i, 0, 0 };
            if (kind == 0)
            {
                b3Sphere sphere = { center, 0.5f };
                b3CreateSphereShape(body, &sd, &sphere);
            }
            else if (kind == 1)
            {
                b3Capsule capsule = { { center.x, -0.5f, 0 }, { center.x, 0.5f, 0 }, 0.5f };
                b3CreateCapsuleShape(body, &sd, &capsule);
            }
            else if (kind == 2)
            {
                b3Transform xf = { center, b3Quat_identity };
                b3BoxHull hull = b3MakeTransformedBoxHull(0.5f, 0.5f, 0.5f, xf);
                b3CreateHullShape(body, &sd, &hull.base);
            }
            else
            {
                b3CreateMeshShape(body, &sd, mesh, b3Vec3_one);
            }
        }
        for (int c = 0; c < 10; ++c)
        {
            b3Pos origin = { 10.0, 2, -3 };
            b3Capsule mover = { { 0, -0.2f, 0 }, { 0, 0.2f, 0 }, 0.25f };
            b3WorldTransform start = { { origin.x + 4, 2, -3 }, b3Quat_identity };
            b3WorldTransform end = start;
            b3Vec3 translation = { 8, 0, 0 };
            b3QueryFilter filter = b3DefaultQueryFilter();
            if (c == 1) translation = (b3Vec3){ -8, 0, 0 };
            if (c == 2) start.p = end.p = origin;
            if (c == 3) filter.maskBits = 0;
            if (c == 4) filter.maskBits = 4;
            if (c == 5) { translation = b3Vec3_zero; end.p = origin; }
            if (c == 6) end.q = (b3Quat){ { 0, 0.6f, 0 }, 0.8f };
            if (c == 7) { start.p.x = origin.x - 4; end.p.x = origin.x + 4; translation = b3Vec3_zero; }
            if (c == 8) filter.maskBits = 4ull << 40;
            if (c == 9) filter.categoryBits = 0;
            b3BodyTOIResult out = b3Body_TimeOfImpactMover(body, origin, &mover, translation, filter, start, end);
            printf("%d %d %d", kind, c, B3_IS_NULL(out.shapeId) ? 0 : (int)(uintptr_t)b3Shape_GetUserData(out.shapeId));
            scalar(out.fraction);
            scalar(out.point.x); scalar(out.point.y); scalar(out.point.z);
            scalar(out.normal.x); scalar(out.normal.y); scalar(out.normal.z);
            printf("\n");
        }
        b3DestroyBody(body);
    }
    b3DestroyWorld(world);
    b3DestroyMesh(mesh);
}
