// SPDX-License-Identifier: MIT
// Uses the phase oracle's native bridge and pinned Box3D build.
#define main phase_oracle_main
#include "native.c"
#undef main

static bool filter(b3ShapeId a, b3ShapeId b, void* context)
{
    (void)context;
    return a.index1 != 4 && b.index1 != 4;
}

static bool preSolve(b3ShapeId a, b3ShapeId b, b3Pos point, b3Vec3 normal, void* context)
{
    (void)a; (void)b; (void)context;
    return !(point.x > 1 && point.x < 3 && point.y < 1 && normal.y > 0.9f);
}

int main(int argc, char** argv)
{
    if (argc != 3) return 2;
    bool pre = strcmp(argv[2], "pre") == 0;
    b3WorldDef wd = b3DefaultWorldDef();
    wd.workerCount = atoi(argv[1]);
    b3WorldId world = b3CreateWorld(&wd);
    if (pre) b3World_SetPreSolveCallback(world, preSolve, NULL);
    else b3World_SetCustomFilterCallback(world, filter, NULL);
    b3BodyDef bd = b3DefaultBodyDef();
    b3ShapeDef sd = b3DefaultShapeDef();
    sd.enableCustomFiltering = !pre;
    sd.enablePreSolveEvents = pre;
    sd.enableContactEvents = true;
    sd.enableHitEvents = true;
    b3BoxHull floor = b3MakeBoxHull(6, 0.5f, 3);
    bd.position.y = -0.5f;
    b3CreateHullShape(b3CreateBody(world, &bd), &sd, &floor.base);
    bd.position.x = 20;
    b3CreateHullShape(b3CreateBody(world, &bd), &sd, &floor.base);
    b3Sphere sphere = { .center = {0,0,0}, .radius = 0.5f };
    for (int i = 0; i < 4; ++i) {
        bd = b3DefaultBodyDef();
        bd.type = b3_dynamicBody;
        bd.position = (b3Pos){ i == 3 ? 20 : 2*i, 3, 0 };
        if (i == 3) bd.linearVelocity.y = -100;
        b3CreateSphereShape(b3CreateBody(world, &bd), &sd, &sphere);
    }
    for (int step = 0; step < 90; ++step) {
        b3World_Step(world, 1.0f/60.0f, 4);
        b3Counters c = b3World_GetCounters(world);
        b3ContactEvents e = b3World_GetContactEvents(world);
        b3SensorEvents s = b3World_GetSensorEvents(world);
        printf("%d 0x%016llx %d %d %d %d %d %d\n", step,
            (unsigned long long)b3HashWorldState(b3GetWorldFromId(world)), c.contactCount,
            e.beginCount, e.endCount, e.hitCount, s.beginCount, s.endCount);
    }
    b3DestroyWorld(world);
}
