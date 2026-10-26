// SPDX-License-Identifier: MIT
#include "box3d/box3d.h"
#include <stdio.h>
#include <string.h>
static unsigned bits(float v) { unsigned u; memcpy(&u, &v, 4); return u; }
int main(void) {
    for (int j = 0; j < 9; ++j) {
        b3WorldDef wd = b3DefaultWorldDef(); wd.workerCount = 1; wd.gravity = b3Vec3_zero;
        b3WorldId world = b3CreateWorld(&wd);
        b3BodyId bodies[3]; b3ShapeId shapes[3];
        for (int i = 0; i < 3; ++i) {
            b3BodyDef bd = b3DefaultBodyDef(); bd.type = b3_dynamicBody;
            bd.position = (b3Pos){i * 10, 0, 0};
            bd.linearVelocity = (b3Vec3){1, 2, -3}; bd.angularVelocity = (b3Vec3){0.2f, -0.3f, 0.4f};
            bodies[i] = b3CreateBody(world, &bd);
            b3ShapeDef sd = b3DefaultShapeDef();
            if (i == 0) { b3Sphere s = {{0.2f, 0, 0}, 0.5f}; shapes[i] = b3CreateSphereShape(bodies[i], &sd, &s); }
            else if (i == 1) { b3Capsule s = {{0, -0.5f, 0}, {0, 0.5f, 0}, 0.25f}; shapes[i] = b3CreateCapsuleShape(bodies[i], &sd, &s); }
            else { b3BoxHull h = b3MakeBoxHull(0.5f, 0.75f, 0.25f); shapes[i] = b3CreateHullShape(bodies[i], &sd, &h.base); }
            if (j == 5 || j == 6) b3Body_SetAwake(bodies[i], false);
            if (j == 7) b3Body_Disable(bodies[i]);
            if (j == 8) b3Body_SetType(bodies[i], b3_kinematicBody);
            b3Shape_ApplyWind(shapes[i], (b3Vec3){12, -4, 7}, j == 1 ? 0 : 1, j == 2 ? 0 : 1, j == 3 ? 2 : 20, j == 4 || j == 6);
        }
        b3World_Step(world, 1.0f / 60.0f, 4);
        for (int i = 0; i < 3; ++i) {
            b3Vec3 v = b3Body_GetLinearVelocity(bodies[i]), w = b3Body_GetAngularVelocity(bodies[i]);
            printf("%08x %08x %08x %08x %08x %08x\n", bits(v.x), bits(v.y), bits(v.z), bits(w.x), bits(w.y), bits(w.z));
        }
        b3DestroyWorld(world);
    }
}
