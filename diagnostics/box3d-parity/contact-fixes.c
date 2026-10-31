#include "box3d/box3d.h"
#include "box3d/collision.h"
#include <stdio.h>

static int calls;
static bool accept(b3ShapeId a, b3ShapeId b, b3Pos p, b3Vec3 n, void* context)
{
    (void)a; (void)b; (void)p; (void)n;
    ++calls;
    return *(bool*)context;
}

int main(void)
{
    int mode = 0;
    if (scanf("%d", &mode) != 1) return 1;
    b3WorldDef wd = b3DefaultWorldDef();
    wd.enableSleep = false;
    b3WorldId world = b3CreateWorld(&wd);
    b3BodyDef bd = b3DefaultBodyDef();
    bd.position = (b3Pos){0, -0.5, 0};
    b3BodyId ground = b3CreateBody(world, &bd);
    b3ShapeDef sd = b3DefaultShapeDef();
    b3BoxHull floor = b3MakeBoxHull(10, 0.5f, 10);
    b3CreateHullShape(ground, &sd, &floor.base);
    bd = b3DefaultBodyDef();
    bd.type = b3_dynamicBody;
    bd.position = (b3Pos){0, mode == 2 ? 0.25 : 0.5, 0};
    b3BodyId body = b3CreateBody(world, &bd);
    bool enabled = false;
    if (mode == 2)
    {
        sd.enablePreSolveEvents = true;
        b3Sphere sphere = {b3Vec3_zero, 0.25f};
        b3CreateSphereShape(body, &sd, &sphere);
        b3World_SetPreSolveCallback(world, accept, &enabled);
    }
    else
    {
        sd.enableSpeculativeContact = mode == 1;
        b3BoxHull box = b3MakeBoxHull(0.5f, 0.5f, 0.5f);
        b3CreateHullShape(body, &sd, &box.base);
    }
    for (int i = 0; i < 60; ++i)
    {
        enabled = i != 0;
        b3World_Step(world, 1.0f / 60.0f, 4);
        b3Counters c = b3World_GetCounters(world);
        printf("%d %d\n", calls, c.recycledContactCount);
    }
    b3DestroyWorld(world);
    return 0;
}
