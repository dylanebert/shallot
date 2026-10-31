// SPDX-License-Identifier: MIT
// Capture native contact-phase outputs without reimplementing the solver.
#include "body.h"
#include "contact.h"
#include "contact_solver.h"
#define B3_SIMD_WIDTH 4
#include "core.h"
#if !defined(B3_SIMD_SSE2)
#error Contact phase authority requires SSE2.
#endif
_Static_assert(B3_SIMD_WIDTH == 4,
               "Contact phase authority requires four lanes.");
#include "physics_world.h"
#include "solver.h"
#include "solver_set.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float F(uint32_t bits) {
  float value;
  memcpy(&value, &bits, 4);
  return value;
}
static bool comma;
static void key(const char *name) {
  printf("%s\"%s\":", comma ? "," : "", name);
  comma = true;
}
static void number(float value) {
  uint32_t bits;
  memcpy(&bits, &value, 4);
  printf("\"0x%08x\"", bits);
}
static void array(const char *name, const float *values, int count) {
  key(name);
  printf("[");
  for (int i = 0; i < count; i++) {
    if (i)
      printf(",");
    number(values[i]);
  }
  printf("]");
}
static void scalar(const char *name, float value) {
  key(name);
  number(value);
}
static void integer(const char *name, int value) {
  key(name);
  printf("%d", value);
}
static void velocities(const char *name, const b3BodyState *state) {
  float values[] = {state->linearVelocity.x,  state->linearVelocity.y,
                    state->linearVelocity.z,  state->angularVelocity.x,
                    state->angularVelocity.y, state->angularVelocity.z};
  array(name, values, 6);
}
static void begin(void) {
  comma = false;
  printf("{");
}
static void finish(void) { printf("}\n"); }
static void setup(b3World *world, b3StepContext *context, b3Contact *contacts,
                  int count, b3BodySim *sims, b3BodyState *states,
                  b3TaskContext *task, uint64_t *hits) {
  world->contacts.data = contacts;
  world->contacts.count = count;
  world->solverSets.data = calloc(3, sizeof(b3SolverSet));
  world->solverSets.count = 3;
  world->solverSets.data[b3_awakeSet].bodyStates.data = states;
  world->solverSets.data[b3_awakeSet].bodySims.data = sims;
  world->taskContexts.data = task;
  world->taskContexts.count = 1;
  task->hitEventBitSet.bits = hits;
  task->hitEventBitSet.blockCount = 1;
  task->hitEventBitSet.blockCapacity = 1;
  context->world = world;
  context->graph = &world->constraintGraph;
  context->sims = sims;
  context->states = states;
}
static void capture_scalar(b3StepContext *context) {
  b3ContactConstraint cc = {0};
  b3ManifoldConstraint mc = {0};
  b3ContactSpec spec = {.contactId = 0, .manifoldStart = 0, .manifoldCount = 1};
  b3ContactPrepareSpan spans[2] = {{.start = 0, .count = 1, .contacts = &spec},
                                   {.start = INT_MAX}};
  context->contactPrepareSpans = spans;
  context->contactConstraints = &cc;
  context->manifoldConstraints = &mc;
  b3GraphColor *color = context->graph->colors;
  color->contactConstraints = &cc;
  color->manifoldConstraints = &mc;
  b3SolverBlock block = {.startIndex = 0,
                         .count = 1,
                         .colorIndex = 0,
                         .blockType = b3_graphContactBlock};
  b3PrepareContacts_Mesh(block, context);
  b3WarmStartContacts_Mesh(block, context);
  begin();
  float outCc[35] = {cc.invMassA, cc.invMassB};
  memcpy(outCc + 2, &cc.invIA, 36);
  memcpy(outCc + 11, &cc.invIB, 36);
  memcpy(outCc + 20, &cc.rollingMass, 36);
  memcpy(outCc + 29, &cc.softness, 12);
  outCc[32] = cc.friction;
  outCc[33] = cc.restitution;
  outCc[34] = cc.rollingResistance;
  array("outCc", outCc, 35);
  key("outCcMeta");
  printf("[%u,%u,1,0]", (unsigned)cc.indexA, (unsigned)cc.indexB);
  float outMc[28];
  memcpy(outMc, &mc.normal, 12);
  memcpy(outMc + 3, &mc.tangent1, 12);
  memcpy(outMc + 6, &mc.tangent2, 12);
  memcpy(outMc + 9, &mc.tangentMass, 16);
  memcpy(outMc + 13, &mc.frictionImpulse, 8);
  outMc[15] = mc.twistMass;
  outMc[16] = mc.twistImpulse;
  memcpy(outMc + 17, &mc.rollingImpulse, 12);
  outMc[20] = mc.tangentVelocity1;
  outMc[21] = mc.tangentVelocity2;
  memcpy(outMc + 22, &mc.centerA, 12);
  memcpy(outMc + 25, &mc.centerB, 12);
  array("outMc", outMc, 28);
  key("outMcMeta");
  printf("[%d,0]", mc.pointCount);
  float outMcp[48];
  for (int i = 0; i < mc.pointCount; i++) {
    b3ManifoldConstraintPoint *p = mc.points + i;
    float *out = outMcp + 12 * i;
    memcpy(out, &p->rA, 12);
    memcpy(out + 3, &p->rB, 12);
    out[6] = p->baseSeparation;
    out[7] = p->normalImpulse;
    out[8] = p->totalNormalImpulse;
    out[9] = p->normalMass;
    out[10] = p->relativeVelocity;
    out[11] = p->leverArm;
  }
  array("outMcp", outMcp, 12 * mc.pointCount);
  b3BodyState zero = b3_identityBodyState;
  b3BodyState *a = cc.indexA < 0 ? &zero : context->states + cc.indexA;
  b3BodyState *b = cc.indexB < 0 ? &zero : context->states + cc.indexB;
  velocities("outWsA", a);
  velocities("outWsB", b);
  b3PushContacts_Mesh(block, context);
  b3SolveContacts_Mesh(block, context);
  velocities("outSlvA", a);
  velocities("outSlvB", b);
  float normal[4], total[4];
  for (int i = 0; i < mc.pointCount; i++) {
    normal[i] = mc.points[i].normalImpulse;
    total[i] = mc.points[i].totalNormalImpulse;
  }
  array("outNormalImpulse", normal, mc.pointCount);
  array("outTotalNormalImpulse", total, mc.pointCount);
  scalar("outTwistImpulse", mc.twistImpulse);
  array("outRollingImpulse", &mc.rollingImpulse.x, 3);
  array("outFrictionImpulse", &mc.frictionImpulse.x, 2);
  b3ApplyRestitution_Mesh(block, context);
  b3StoreImpulses_Mesh(block, context, 0);
  velocities("outRestA", a);
  velocities("outRestB", b);
  b3Manifold *m = context->world->contacts.data[0].manifolds;
  array("storedFrictionImpulse", &m->frictionImpulse.x, 3);
  scalar("storedTwistImpulse", m->twistImpulse);
  array("storedRollingImpulse", &m->rollingImpulse.x, 3);
  float velocity[4];
  for (int i = 0; i < m->pointCount; i++) {
    normal[i] = m->points[i].normalImpulse;
    total[i] = m->points[i].totalNormalImpulse;
    velocity[i] = m->points[i].normalVelocity;
  }
  array("storedNormalImpulse", normal, m->pointCount);
  array("storedTotalNormalImpulse", total, m->pointCount);
  array("storedNormalVelocity", velocity, m->pointCount);
  integer("hitFlag", context->world->taskContexts.data[0].hasHitEvents);
  finish();
  free(context->world->solverSets.data);
}
static void capture_wide(b3StepContext *context) {
  int ids[4] = {0, 1, 2, 3};
  b3WidePrepareSpan spans[2] = {{.start = 0, .count = 4, .contacts = ids},
                                {.start = INT_MAX}};
  context->widePrepareSpans = spans;
  context->wideConstraints = calloc(1, b3GetWideContactConstraintByteCount(4));
  context->graph->colors[0].wideConstraints = context->wideConstraints;
  b3SolverBlock block = {.startIndex = 0,
                         .count = 1,
                         .colorIndex = 0,
                         .blockType = b3_graphWideContactBlock};
  b3PrepareContacts_Convex(block, context);
  b3WarmStartContacts_Convex(block, context);
  b3PushContacts_Convex(block, context);
  b3SolveContacts_Convex(block, context);
  b3ApplyRestitution_Convex(block, context);
  b3StoreImpulses_Convex(block, context, 0);
  for (int lane = 0; lane < 4; lane++) {
    b3Contact *c = context->world->contacts.data + lane;
    b3Manifold *m = c->manifolds;
    b3BodyState zero = b3_identityBodyState;
    b3BodyState *a = context->states + c->encodedBodySimA;
    b3BodyState *b =
        c->encodedBodySimB < 0 ? &zero : context->states + c->encodedBodySimB;
    begin();
    float body[] = {
        a->linearVelocity.x,  a->linearVelocity.y,  a->linearVelocity.z,
        a->angularVelocity.x, a->angularVelocity.y, a->angularVelocity.z,
        b->linearVelocity.x,  b->linearVelocity.y,  b->linearVelocity.z,
        b->angularVelocity.x, b->angularVelocity.y, b->angularVelocity.z};
    array("body", body, 12);
    float manifold[] = {m->frictionImpulse.x, m->frictionImpulse.y,
                        m->frictionImpulse.z, m->twistImpulse,
                        m->rollingImpulse.x,  m->rollingImpulse.y,
                        m->rollingImpulse.z};
    array("manifold", manifold, 7);
    key("points");
    printf("[");
    for (int i = 0; i < m->pointCount; i++) {
      if (i)
        printf(",");
      printf("[");
      number(m->points[i].normalImpulse);
      printf(",");
      number(m->points[i].totalNormalImpulse);
      printf(",");
      number(m->points[i].normalVelocity);
      printf("]");
    }
    printf("]");
    integer(
        "hit",
        (context->world->taskContexts.data[0].hitEventBitSet.bits[0] >> lane) &
            1);
    finish();
  }
  free(context->wideConstraints);
  free(context->world->solverSets.data);
}
