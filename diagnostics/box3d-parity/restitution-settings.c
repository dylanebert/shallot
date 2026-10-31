// SPDX-License-Identifier: MIT
#include "box3d/box3d.h"
#include "physics_world.h"
#include <stdio.h>
#include <stdlib.h>

uint64_t b3HashWorldState(b3World *world);

int main(int argc, char **argv) {
  if (argc != 4 && argc != 5)
    return 2;
  int iterations = atoi(argv[1]);
  bool propagation = atoi(argv[2]) != 0;
  bool change = atoi(argv[3]) != 0;
  bool meshFloor = argc == 5 && atoi(argv[4]) != 0;
  b3WorldDef wd = b3DefaultWorldDef();
// For the f555ee42 negative control only: that revision has neither setting.
#ifndef PINNED_PREVIOUS
  wd.restitutionIterations = iterations;
  wd.enableRestitutionPropagation = propagation;
#endif
  b3WorldId world = b3CreateWorld(&wd);
  b3ShapeDef sd = b3DefaultShapeDef();
  sd.density = 1.0f;
  sd.baseMaterial.friction = 0.0f;
  b3Sphere sphere = {.center = {0, 0, 0}, .radius = 1.0f};
  b3BoxHull box = b3MakeBoxHull(1.0f, 1.0f, 1.0f);
  b3MeshData *mesh = meshFloor ? b3CreateGridMesh(4, 4, 4.0f, 1, true) : NULL;
  for (int i = 0; i < 5; ++i) {
    b3BodyDef bd = b3DefaultBodyDef();
    bd.type = i == 0 ? b3_staticBody : b3_dynamicBody;
    bd.position.y = i == 4 ? 11.0f : i == 0 && meshFloor ? 1.0f : 2.0f * i;
    if (i == 4) {
      bd.linearVelocity.y = -12.0f;
      bd.position.x = 0.25f;
      bd.angularVelocity.z = 0.3f;
    }
    sd.baseMaterial.restitution = i == 4 ? 0.75f : 0.0f;
    b3BodyId body = b3CreateBody(world, &bd);
    if (i == 0 && meshFloor)
      b3CreateMeshShape(body, &sd, mesh, b3Vec3_one);
    else if (i == 4)
      b3CreateHullShape(body, &sd, &box.base);
    else
      b3CreateSphereShape(body, &sd, &sphere);
  }
  for (int step = 0; step < 180; ++step) {
#ifndef PINNED_PREVIOUS
    if (change && step == 5) {
      b3World_SetRestitutionIterations(world, iterations == 0 ? 3 : 0);
      b3World_EnableRestitutionPropagation(world, !propagation);
    }
    printf("%d %d ", b3World_GetRestitutionIterations(world),
           b3World_IsRestitutionPropagationEnabled(world));
#endif
    b3World_Step(world, 1.0f / 60.0f, 4);
    printf("%016llx\n",
           (unsigned long long)b3HashWorldState(b3GetWorldFromId(world)));
  }
  b3DestroyWorld(world);
  if (mesh)
    b3DestroyMesh(mesh);
  return 0;
}
