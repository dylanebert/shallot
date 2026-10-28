// SPDX-License-Identifier: MIT
#include "box3d/box3d.h"
#include <stdio.h>

static void Record(b3WorldId worldId)
{
    b3Counters c = b3World_GetCounters(worldId);
    printf("%d %d %d %d %d %d %d %d %d", c.bodyCount, c.shapeCount, c.contactCount, c.jointCount,
           c.islandCount, c.staticTreeHeight, c.treeHeight, c.satCallCount, c.satCacheHitCount);
    for (int i = 0; i < 24; ++i) printf(" %d", c.colorCounts[i]);
    for (int i = 0; i < 8; ++i) printf(" %d", c.manifoldCounts[i]);
    printf(" %d %d %d %d %d\n", c.awakeContactCount, c.recycledContactCount, c.distanceIterations,
           c.pushBackIterations, c.rootIterations);
}
int main(void)
{
    for (int pile = 0; pile < 2; ++pile)
    {
        b3WorldDef wd = b3DefaultWorldDef();
        wd.workerCount = 1;
        wd.enableSleep = false;
        b3WorldId worldId = b3CreateWorld(&wd);
        b3BodyDef bd = b3DefaultBodyDef();
        bd.position.y = -1.0f;
        b3BodyId ground = b3CreateBody(worldId, &bd);
        b3ShapeDef sd = b3DefaultShapeDef();
        b3BoxHull floor = b3MakeBoxHull(20.0f, 1.0f, 20.0f);
        b3CreateHullShape(ground, &sd, &floor.base);
        b3BoxHull box = b3MakeBoxHull(0.5f, 0.5f, 0.5f);
        int count = pile ? 26 : 3;
        for (int i = 0; i < count; ++i)
        {
            bd = b3DefaultBodyDef();
            bd.type = b3_dynamicBody;
            bd.position.y = pile ? 0.5f + 0.02f * i : 0.5f + 1.0f * i;
            b3BodyId body = b3CreateBody(worldId, &bd);
            b3CreateHullShape(body, &sd, &box.base);
        }
        Record(worldId);
        for (int step = 0; step < 60; ++step)
        {
            b3World_Step(worldId, 1.0f / 60.0f, 4);
            Record(worldId);
        }
        b3DestroyWorld(worldId);
    }
    b3WorldDef wd = b3DefaultWorldDef();
    wd.workerCount = 1;
    b3WorldId worldId = b3CreateWorld(&wd);
    b3BodyDef bd = b3DefaultBodyDef();
    bd.position.y = -1.0f;
    b3BodyId ground = b3CreateBody(worldId, &bd);
    b3ShapeDef sd = b3DefaultShapeDef();
    b3BoxHull floor = b3MakeBoxHull(20.0f, 1.0f, 20.0f);
    b3CreateHullShape(ground, &sd, &floor.base);
    bd = b3DefaultBodyDef();
    bd.type = b3_dynamicBody;
    bd.position.y = 2.0f;
    bd.linearVelocity.y = -200.0f;
    b3BodyId bullet = b3CreateBody(worldId, &bd);
    b3Sphere sphere = { {0.0f, 0.0f, 0.0f}, 0.1f };
    b3CreateSphereShape(bullet, &sd, &sphere);
    for (int step = 0; step < 5; ++step) {
        b3World_Step(worldId, 1.0f / 60.0f, 4);
        Record(worldId);
    }
    b3SphericalJointDef jd = b3DefaultSphericalJointDef();
    jd.base.bodyIdA = ground;
    jd.base.bodyIdB = bullet;
    b3CreateSphericalJoint(worldId, &jd);
    b3World_Step(worldId, 1.0f / 60.0f, 4);
    Record(worldId);
    b3DestroyWorld(worldId);
    return 0;
}
