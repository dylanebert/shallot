// SPDX-License-Identifier: MIT
// Native manifold totals, hit/move records and mixed-island wake states.
#include "box3d/box3d.h"
#include <inttypes.h>
#include <stdio.h>

static b3WorldId make_world(float gravity, bool continuous) {
  b3WorldDef d = b3DefaultWorldDef();
  d.gravity.y = gravity;
  d.enableContinuous = continuous;
  return b3CreateWorld(&d);
}
static b3BodyId sphere(b3WorldId w, b3BodyType type, float x, float y, float r,
                       float density, bool events) {
  b3BodyDef bd = b3DefaultBodyDef();
  bd.type = type;
  bd.position = (b3Pos){x, y, 0};
  b3BodyId b = b3CreateBody(w, &bd);
  b3ShapeDef sd = b3DefaultShapeDef();
  sd.density = density;
  sd.enableContactEvents = events;
  b3Sphere s = {{0, 0, 0}, r};
  b3CreateSphereShape(b, &sd, &s);
  return b;
}
static b3ShapeId shape(b3BodyId b) {
  b3ShapeId s;
  b3Body_GetShapes(b, &s, 1);
  return s;
}
// The ECS begin-impulse snapshot sums the native point totals in manifold order.
static float begin_impulse(b3ContactId id) {
  b3ContactData d = b3Contact_GetData(id);
  float sum = 0;
  for (int m = 0; m < d.manifoldCount; m++)
    for (int p = 0; p < d.manifolds[m].pointCount; p++)
      sum += d.manifolds[m].points[p].totalNormalImpulse;
  return sum;
}
static void number(float f) { printf("%.17g", (double)f); }
static void step(b3WorldId w) { b3World_Step(w, 1.0f / 60.0f, 4); }
static void observed_begin(b3WorldId w) {
  b3ContactEvents e = b3World_GetContactEvents(w);
  number(e.beginCount ? begin_impulse(e.beginEvents[0].contactId) : 0);
}
static void lifecycle(void) {
  b3WorldId w = make_world(0, true);
  b3BodyId a = sphere(w, b3_staticBody, 0, 0, 1, 1000, true),
           b = sphere(w, b3_dynamicBody, 0, 1.5f, 1, 1000, true);
  printf("\"lifecycle\":[");
  step(w);
  observed_begin(w);
  b3Body_SetType(a, b3_dynamicBody);
  step(w);
  printf(",");
  observed_begin(w);
  b3Body_Disable(b);
  step(w);
  b3Body_Enable(b);
  step(w);
  printf(",");
  observed_begin(w);
  printf("]");
  b3DestroyWorld(w);
}
static void filters(void) {
  b3WorldId w = make_world(0, true);
  b3BodyId ground = sphere(w, b3_staticBody, 0, 0, 1, 1000, true),
           visitor = sphere(w, b3_dynamicBody, 0, 1.5f, 1, 1000, true);
  b3Shape_EnableSensorEvents(shape(visitor), true);
  b3ShapeDef sd = b3DefaultShapeDef();
  sd.density = 1000;
  sd.isSensor = true;
  sd.enableSensorEvents = true;
  b3Sphere s = {{0, 0, 0}, 3};
  b3CreateSphereShape(ground, &sd, &s);
  printf(",\"filters\":[");
  step(w);
  observed_begin(w);
  b3Filter f = b3Shape_GetFilter(shape(visitor));
  f.maskBits = 0;
  b3Shape_SetFilter(shape(visitor), f, true);
  step(w);
  f.maskBits = UINT64_MAX;
  b3Shape_SetFilter(shape(visitor), f, true);
  step(w);
  printf(",");
  observed_begin(w);
  printf("]");
  b3DestroyWorld(w);
}
static void collision_joint(void) {
  b3WorldId w = make_world(0, true);
  b3BodyId a = sphere(w, b3_dynamicBody, 0, 0, 1, 1000, true),
           b = sphere(w, b3_dynamicBody, 1.5f, 0, 1, 1000, true);
  b3DistanceJointDef jd = b3DefaultDistanceJointDef();
  jd.base.bodyIdA = a;
  jd.base.bodyIdB = b;
  jd.base.collideConnected = true;
  b3JointId j = b3CreateDistanceJoint(w, &jd);
  printf(",\"collisionJoint\":[");
  step(w);
  observed_begin(w);
  b3Joint_SetCollideConnected(j, false);
  step(w);
  b3Joint_SetCollideConnected(j, false);
  step(w);
  b3Joint_SetCollideConnected(j, true);
  step(w);
  printf(",");
  observed_begin(w);
  printf("]");
  b3DestroyWorld(w);
}
static const char *name(b3ShapeId id) {
  static const char *names[] = {"", "ground", "falling", "anchor", "bob"};
  return names[id.index1];
}
static void trace(void) {
  b3WorldId w = make_world(-10, true);
  b3World_SetHitEventThreshold(w, 0);
  sphere(w, b3_staticBody, 0, -0.5f, 0.5f, 1000, true);
  b3BodyId falling = sphere(w, b3_dynamicBody, 0, 0.8f, 0.5f, 1, true);
  b3Body_SetLinearVelocity(falling, (b3Vec3){0, -10, 0});
  b3Shape_EnableHitEvents(shape(falling), true);
  b3SurfaceMaterial material = b3Shape_GetSurfaceMaterial(shape(falling));
  material.userMaterialId = UINT64_C(0xfedcba9876543210);
  b3Shape_SetSurfaceMaterial(shape(falling), material);
  b3BodyId anchor = sphere(w, b3_staticBody, 5, 2, 0.5f, 1000, false),
           bob = sphere(w, b3_dynamicBody, 5, 2, 0.5f, 1, false);
  b3WeldJointDef jd = b3DefaultWeldJointDef();
  jd.base.bodyIdA = anchor;
  jd.base.bodyIdB = bob;
  jd.base.forceThreshold = 0;
  jd.base.torqueThreshold = 0;
  b3CreateWeldJoint(w, &jd);
  printf(",\"trace\":[");
  for (int tick = 0; tick < 4; tick++) {
    step(w);
    if (tick)
      printf(",");
    printf("[%d,[", tick);
    bool comma = false;
    b3ContactEvents e = b3World_GetContactEvents(w);
    for (int i = 0; i < e.beginCount; i++) {
      if (comma)
        printf(",");
      comma = true;
      b3ContactBeginTouchEvent *b = e.beginEvents + i;
      printf("[\"contactBegin\",\"%s\",\"%s\",", name(b->shapeIdA),
             name(b->shapeIdB));
      number(begin_impulse(b->contactId));
      printf("]");
    }
    for (int i = 0; i < e.hitCount; i++) {
      if (comma)
        printf(",");
      comma = true;
      b3ContactHitEvent *h = e.hitEvents + i;
      printf("[\"hit\",\"%s\",\"%s\",{\"x\":", name(h->shapeIdA),
             name(h->shapeIdB));
      number((float)h->point.x);
      printf(",\"y\":");
      number((float)h->point.y);
      printf(",\"z\":");
      number((float)h->point.z);
      printf("},{\"x\":");
      number(h->normal.x);
      printf(",\"y\":");
      number(h->normal.y);
      printf(",\"z\":");
      number(h->normal.z);
      printf("},");
      number(h->approachSpeed);
      printf(",\"%" PRIu64 "\",\"%" PRIu64 "\"]", h->userMaterialIdA,
             h->userMaterialIdB);
    }
    b3JointEvents j = b3World_GetJointEvents(w);
    for (int i = 0; i < j.count; i++) {
      if (comma)
        printf(",");
      comma = true;
      printf("[\"joint\",\"weld\"]");
    }
    b3BodyEvents moves = b3World_GetBodyEvents(w);
    const char *labels[] = {"", "ground", "falling", "anchor", "bob"};
    for (int i = 0; i < moves.moveCount; i++) {
      if (comma)
        printf(",");
      comma = true;
      b3BodyMoveEvent *move = moves.moveEvents + i;
      b3Pos p = move->transform.p;
      printf("[\"move\",\"%s\",[", labels[move->bodyId.index1]);
      number((float)p.x);
      printf(",");
      number((float)p.y);
      printf(",");
      number((float)p.z);
      printf("]]");
    }
    printf("]]");
  }
  printf("]");
  b3DestroyWorld(w);
}
static void retained(void) {
  b3WorldId w = make_world(-10, false);
  sphere(w, b3_staticBody, 0, -0.5f, 0.5f, 1000, false);
  sphere(w, b3_dynamicBody, 0, 3, 0.5f, 1000, true);
  for (int tick = 0; tick < 120; tick++) {
    step(w);
    b3ContactEvents e = b3World_GetContactEvents(w);
    if (e.beginCount) {
      printf(",\"retained\":[%d,", tick);
      number(begin_impulse(e.beginEvents[0].contactId));
      printf("]");
      break;
    }
  }
  b3DestroyWorld(w);
}
static b3BodyId box(b3WorldId w, b3BodyType type, float x, float y, float hx,
                    float hy, float hz) {
  b3BodyDef bd = b3DefaultBodyDef();
  bd.type = type;
  bd.position = (b3Pos){x, y, 0};
  if (type == b3_dynamicBody) {
    bd.angularDamping = 0.5f;
    bd.linearDamping = 0.5f;
  }
  b3BodyId b = b3CreateBody(w, &bd);
  b3ShapeDef sd = b3DefaultShapeDef();
  sd.density = 1;
  b3BoxHull h = b3MakeBoxHull(hx, hy, hz);
  b3CreateHullShape(b, &sd, &h.base);
  return b;
}
static void mixed_sleep(void) {
  b3WorldId w = make_world(-10, false);
  box(w, b3_staticBody, 0, -1, 20, 1, 20);
  b3BodyId torso = box(w, b3_dynamicBody, 0, 3, 0.25f, 0.5f, 0.25f);
  b3BodyId limbs[] = {
      box(w, b3_dynamicBody, -0.75f, 3.5f, 0.5f, 0.125f, 0.125f),
      box(w, b3_dynamicBody, 0.75f, 3.5f, 0.5f, 0.125f, 0.125f),
      box(w, b3_dynamicBody, -0.25f, 2, 0.125f, 0.5f, 0.125f),
      box(w, b3_dynamicBody, 0.25f, 2, 0.125f, 0.5f, 0.125f)};
  for (int i = 0; i < 4; i++) {
    if (i < 2) {
      b3SphericalJointDef d = b3DefaultSphericalJointDef();
      d.base.bodyIdA = torso;
      d.base.bodyIdB = limbs[i];
      d.base.localFrameA.p = (b3Vec3){i == 0 ? -0.25f : 0.25f, 0.5f, 0};
      d.base.localFrameB.p = (b3Vec3){i == 0 ? 0.5f : -0.5f, 0, 0};
      b3CreateSphericalJoint(w, &d);
    } else {
      b3RevoluteJointDef d = b3DefaultRevoluteJointDef();
      d.base.bodyIdA = torso;
      d.base.bodyIdB = limbs[i];
      d.base.localFrameA.p = (b3Vec3){i == 2 ? -0.25f : 0.25f, -0.5f, 0};
      d.base.localFrameB.p = (b3Vec3){0, 0.5f, 0};
      b3CreateRevoluteJoint(w, &d);
    }
  }
  printf(",\"mixedSleep\":{");
  bool allTogether = true;
  for (int tick = 1; tick <= 3600; tick++) {
    step(w);
    bool awake = b3Body_IsAwake(torso);
    for (int i = 0; i < 4; i++)
      allTogether &= b3Body_IsAwake(limbs[i]) == awake;
    if (tick == 300)
      printf("\"awake300\":%s,", awake ? "true" : "false");
    if (!awake) {
      printf("\"step\":%d,\"together\":%s", tick,
             allTogether ? "true" : "false");
      break;
    }
  }
  printf("}");
  b3DestroyWorld(w);
}
int main(void) {
  printf("{");
  lifecycle();
  filters();
  collision_joint();
  trace();
  retained();
  mixed_sleep();
  printf("}\n");
  return 0;
}
