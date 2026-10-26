// SPDX-License-Identifier: MIT
#include "box3d/box3d.h"
#include <stdio.h>
#include <string.h>
static unsigned bits(float v) { unsigned u; memcpy(&u, &v, 4); return u; }
int main(void) {
    b3WorldDef wd = b3DefaultWorldDef(); wd.workerCount = 1;
    b3WorldId world = b3CreateWorld(&wd);
    b3BodyId bodies[12];
    for (int i = 0; i < 12; ++i) {
        b3BodyDef bd = b3DefaultBodyDef(); bd.type = b3_dynamicBody;
        bd.position = (b3Pos){ (float)(i % 4) * 2.0f, 0.25f, (float)(i / 4) * 2.0f };
        bodies[i] = b3CreateBody(world, &bd);
        b3ShapeDef sd = b3DefaultShapeDef(); sd.explosionScale = (float)((i / 3) % 3); sd.filter.categoryBits = 1ull << (i % 2);
        if (i % 3 == 0) { b3Sphere s = {{0.2f, 0.0f, 0.0f}, 0.5f}; b3CreateSphereShape(bodies[i], &sd, &s); }
        else if (i % 3 == 1) { b3Capsule s = {{0, -0.5f, 0}, {0, 0.5f, 0}, 0.25f}; b3CreateCapsuleShape(bodies[i], &sd, &s); }
        else { b3BoxHull h = b3MakeBoxHull(0.5f, 0.75f, 0.25f); b3CreateHullShape(bodies[i], &sd, &h.base); }
    }
    for (int j = 0; j < 3; ++j) {
        b3ExplosionDef d = b3DefaultExplosionDef(); d.position = (b3Pos){0, 0, 0}; d.radius = j == 0 ? 0.5f : 3.0f; d.falloff = j == 0 ? 0.0f : 4.0f; d.maskBits = j == 2 ? 2 : UINT64_MAX; d.impulsePerArea = j == 2 ? -3.0f : 12.0f;
        b3World_Explode(world, &d);
        for (int i = 0; i < 12; ++i) {
            b3Vec3 v = b3Body_GetLinearVelocity(bodies[i]), w = b3Body_GetAngularVelocity(bodies[i]);
            printf("%08x %08x %08x %08x %08x %08x\n", bits(v.x), bits(v.y), bits(v.z), bits(w.x), bits(w.y), bits(w.z));
        }
    }
    b3DestroyWorld(world);
}
