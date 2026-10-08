// SPDX-License-Identifier: MIT
// Uses the phase oracle's native bridge and pinned Box3D build.
#define main phase_oracle_main
#include "native.c"
#undef main

#include <stdatomic.h>
static atomic_int calls;
static b3WorldId callbackWorld;
static int callbackStep;

static bool eventsVisible(void)
{
    b3ContactEvents c = b3World_GetContactEvents(callbackWorld);
    b3SensorEvents s = b3World_GetSensorEvents(callbackWorld);
    b3BodyEvents b = b3World_GetBodyEvents(callbackWorld);
    b3JointEvents j = b3World_GetJointEvents(callbackWorld);
    return c.beginCount || c.endCount || c.hitCount || s.beginCount || s.endCount || b.moveCount || j.count;
}

static bool filter(b3ShapeId a, b3ShapeId b, void* context)
{
    (void)context;
    atomic_fetch_add(&calls, 1);
    printf("%d F %d %d %d %d\n", callbackStep, a.index1, a.generation, b.index1, b.generation);
    if (eventsVisible() || b3Shape_GetSensorCapacity(a) || b3Shape_GetSensorCapacity(b)) return false;
    if (context) return a.index1 > 98 && b.index1 > 98;
    return a.index1 != 4 && b.index1 != 4 && a.index1 != 6 && b.index1 != 6;
}

static bool preSolve(b3ShapeId a, b3ShapeId b, b3Pos point, b3Vec3 normal, void* context)
{
    (void)a; (void)b; (void)context;
    atomic_fetch_add(&calls, 1);
    printf("%d P %d %d %d %d %u %u %u %u %u %u\n", callbackStep, a.index1, a.generation, b.index1, b.generation,
        bits(point.x), bits(point.y), bits(point.z), bits(normal.x), bits(normal.y), bits(normal.z));
    if (eventsVisible()) return false;
    return !((point.x > 1 && point.x < 3 || point.x > 19 && point.x < 21) && point.y < 1 && normal.y > 0.9f);
}

int main(int argc, char** argv)
{
    if (argc != 3) return 2;
    bool pre = strcmp(argv[2], "pre") == 0;
    bool pressure = strcmp(argv[2], "pressure") == 0;
    b3WorldDef wd = b3DefaultWorldDef();
    wd.workerCount = atoi(argv[1]);
    wd.enableSleep = false;
    if (pressure) wd.gravity = b3Vec3_zero;
    b3WorldId world = b3CreateWorld(&wd);
    callbackWorld = world;
    if (pre) b3World_SetPreSolveCallback(world, preSolve, NULL);
    else b3World_SetCustomFilterCallback(world, filter, pressure ? (void*)1 : NULL);
    b3BodyDef bd = b3DefaultBodyDef();
    b3ShapeDef sd = b3DefaultShapeDef();
    sd.enableCustomFiltering = !pre;
    sd.enablePreSolveEvents = pre;
    sd.enableContactEvents = true;
    sd.enableHitEvents = true;
    sd.enableSensorEvents = true;
    b3CompoundData* compounds[3] = {NULL, NULL, NULL};
    if (pressure) {
        b3Sphere sphere = { .center = {0,0,0}, .radius = 0.5f };
        bd.type = b3_dynamicBody;
        for (int i = 0; i < 100; ++i) b3CreateSphereShape(b3CreateBody(world, &bd), &sd, &sphere);
    } else {
    b3BoxHull floor = b3MakeBoxHull(6, 0.5f, 3);
    bd.position.y = -0.5f;
    b3CreateHullShape(b3CreateBody(world, &bd), &sd, &floor.base);
    bd.position.x = 20;
    b3CreateHullShape(b3CreateBody(world, &bd), &sd, &floor.base);
    b3Sphere sphere = { .center = {0,0,0}, .radius = 0.5f };
    // Target-only flags cover the CCD scheduling gate as well as fast-shape flags.
    sd.enableCustomFiltering = false;
    sd.enablePreSolveEvents = false;
    for (int i = 0; i < 5; ++i) {
        bd = b3DefaultBodyDef();
        bd.type = b3_dynamicBody;
        bd.position = (b3Pos){ i >= 3 ? 20 + 4*(i-3) : 2*i, 3, 0 };
        if (i >= 3) bd.linearVelocity.y = -100;
        bd.isBullet = i == 4;
        b3CreateSphereShape(b3CreateBody(world, &bd), &sd, &sphere);
    }
    bd = b3DefaultBodyDef();
    bd.position = (b3Pos){500, -0.5f, 0};
    b3BoxHull longFloor = b3MakeBoxHull(400, 0.5f, 3);
    b3CreateHullShape(b3CreateBody(world, &bd), &sd, &longFloor.base);
    for (int i = 0; i < 300; ++i) {
        bd = b3DefaultBodyDef();
        bd.type = b3_dynamicBody;
        bd.position = (b3Pos){100 + 2*i, 0.5f, 0};
        b3CreateSphereShape(b3CreateBody(world, &bd), &sd, &sphere);
    }
    sd.isSensor = true;
    sd.enableCustomFiltering = !pre;
    sd.enablePreSolveEvents = pre;
    bd = b3DefaultBodyDef();
    bd.position = (b3Pos){2, 3, 0};
    b3BoxHull volume = b3MakeBoxHull(5, 4, 2);
    b3CreateHullShape(b3CreateBody(world, &bd), &sd, &volume.base);
    bd.position = (b3Pos){22, 1, 0};
    volume = b3MakeBoxHull(4, 0.5f, 2);
    b3CreateHullShape(b3CreateBody(world, &bd), &sd, &volume.base);
    sd.enableCustomFiltering = false;
    sd.enablePreSolveEvents = false;
    sphere.radius = 0.75f;
    for (int i = 0; i < 300; ++i) {
        bd = b3DefaultBodyDef();
        bd.position = (b3Pos){100 + 2*i, 1, 0};
        b3CreateSphereShape(b3CreateBody(world, &bd), &sd, &sphere);
    }
    sd.isSensor = false;
    for (int i = 0; i < 3; ++i) {
        b3CompoundSphereDef childSphere = { .sphere = {{0, 0, 0}, 0.5f}, .material = b3DefaultSurfaceMaterial() };
        b3CompoundCapsuleDef childCapsule = { .capsule = {{-0.5f, 0, 0}, {0.5f, 0, 0}, 0.5f}, .material = b3DefaultSurfaceMaterial() };
        b3BoxHull childBox = b3MakeBoxHull(0.5f, 0.5f, 0.5f);
        b3CompoundHullDef childHull = { .hull = &childBox.base, .transform = {{0.25f, 0.1f, 0}, b3Quat_identity}, .material = b3DefaultSurfaceMaterial() };
        b3CompoundDef compoundDef = {0};
        if (i == 0) { compoundDef.spheres = &childSphere; compoundDef.sphereCount = 1; }
        else if (i == 1) { compoundDef.capsules = &childCapsule; compoundDef.capsuleCount = 1; }
        else { compoundDef.hulls = &childHull; compoundDef.hullCount = 1; }
        compounds[i] = b3CreateCompound(&compoundDef);
        bd = b3DefaultBodyDef();
        bd.position = (b3Pos){8 + 4*i, -0.5f, 0};
        sd.enableCustomFiltering = !pre;
        sd.enablePreSolveEvents = pre;
        b3CreateBakedCompoundShape(b3CreateBody(world, &bd), &sd, compounds[i]);
        bd.type = b3_dynamicBody;
        bd.position.y = 0.49f;
        sd.enableCustomFiltering = false;
        sd.enablePreSolveEvents = false;
        if (i == 2) {
            bd.position.x += 0.25f;
            bd.position.y += 0.1f;
            b3Sphere ball = {{0, 0, 0}, 0.5f};
            b3CreateSphereShape(b3CreateBody(world, &bd), &sd, &ball);
        } else {
            b3BoxHull cube = b3MakeBoxHull(0.5f, 0.5f, 0.5f);
            b3CreateHullShape(b3CreateBody(world, &bd), &sd, &cube.base);
        }
    }
    }
    for (int step = 0; step < 90; ++step) {
        callbackStep = step;
        atomic_store(&calls, 0);
        b3World_Step(world, 1.0f/60.0f, 4);
        b3Counters c = b3World_GetCounters(world);
        b3ContactEvents e = b3World_GetContactEvents(world);
        b3SensorEvents s = b3World_GetSensorEvents(world);
        printf("%d 0x%016llx %d %d %d %d %d %d %d\n", step,
            (unsigned long long)b3HashWorldState(b3GetWorldFromId(world)), c.contactCount,
            e.beginCount, e.endCount, e.hitCount, s.beginCount, s.endCount, atomic_load(&calls));
        for (int i = 0; i < e.beginCount; ++i) {
            b3ContactBeginTouchEvent v = e.beginEvents[i];
            printf("%d B %d %d %d\n", step, v.shapeIdA.index1, v.shapeIdB.index1, v.contactId.index1);
        }
        for (int i = 0; i < e.endCount; ++i) {
            b3ContactEndTouchEvent v = e.endEvents[i];
            printf("%d E %d %d %d\n", step, v.shapeIdA.index1, v.shapeIdB.index1, v.contactId.index1);
        }
        for (int i = 0; i < e.hitCount; ++i) {
            b3ContactHitEvent v = e.hitEvents[i];
            printf("%d H %d %d %d %u %u %u %u %u %u %u %llu %llu\n", step,
                v.shapeIdA.index1, v.shapeIdB.index1, v.contactId.index1,
                bits(v.point.x), bits(v.point.y), bits(v.point.z), bits(v.normal.x), bits(v.normal.y), bits(v.normal.z),
                bits(v.approachSpeed), (unsigned long long)v.userMaterialIdA, (unsigned long long)v.userMaterialIdB);
        }
        for (int i = 0; i < s.beginCount; ++i) {
            printf("%d SB %d %d\n", step, s.beginEvents[i].sensorShapeId.index1, s.beginEvents[i].visitorShapeId.index1);
        }
        for (int i = 0; i < s.endCount; ++i) {
            printf("%d SE %d %d\n", step, s.endEvents[i].sensorShapeId.index1, s.endEvents[i].visitorShapeId.index1);
        }
    }
    b3DestroyWorld(world);
    for (int i = 0; i < 3; ++i) if (compounds[i]) b3DestroyCompound(compounds[i]);
}
