// SPDX-License-Identifier: MIT
#include "box3d/box3d.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// mover.cpp's movement phase with no obstacles, no input policy and no pogo hit.
// The public Character supplies velocity rather than the sample's friction/gravity policy.
static uint32_t bits(float value) {
    uint32_t word;
    memcpy(&word, &value, sizeof(word));
    return word;
}

int main(void) {
    b3WorldDef def = b3DefaultWorldDef();
    b3WorldId world = b3CreateWorld(&def);
    b3Pos position = {0.0f, 0.0f, 0.0f};
    b3Vec3 velocity = {3.0f, 0.0f, 0.0f};
    b3Capsule capsule = {{0.0f, -0.5f, 0.0f}, {0.0f, 0.5f, 0.0f}, 0.3f};
    float timeStep = 1.0f / 60.0f;
    float pogoRestLength = 3.0f * capsule.radius;
    float rayLength = pogoRestLength + capsule.radius;
    b3Pos rayOrigin = b3OffsetPos(position, capsule.center1);
    b3QueryFilter pogoFilter = {.categoryBits = 1, .maskBits = ~2u};
    b3RayResult ray = b3World_CastRayClosest(world, rayOrigin, (b3Vec3){0.0f, -rayLength, 0.0f}, pogoFilter);
    if (ray.hit) return 1;
    float pogoVelocity = 0.0f;
    b3Pos target = b3OffsetPos(b3OffsetPos(position, b3MulSV(timeStep, velocity)),
                             b3MulSV(timeStep * pogoVelocity, b3Vec3_axisY));
    b3QueryFilter moverFilter = {.categoryBits = 1, .maskBits = ~0u, .id = 1};
    b3QueryFilter castFilter = {.categoryBits = 1, .maskBits = ~2u, .id = 1};
    b3CollisionPlane planes[8] = {0};
    int planeCount = 0;
    for (int iteration = 0; iteration < 5; ++iteration) {
        b3World_CollideMover(world, position, &capsule, moverFilter, NULL, NULL);
        b3PlaneSolverResult result = b3SolvePlanes(b3SubPos(target, position), planes, planeCount);
        b3Vec3 delta = result.delta;
        float fraction = b3World_CastMover(world, position, &capsule, delta, castFilter, NULL, NULL);
        delta = b3MulSV(fraction, delta);
        position = b3OffsetPos(position, delta);
        if (b3LengthSquared(delta) < 0.01f * 0.01f) break;
    }
    velocity = b3ClipVector(velocity, planes, planeCount);
    printf("{\"timeStep\":%u,\"position\":[%u,%u,%u],\"velocity\":[%u,%u,%u]}\n",
           bits(timeStep), bits(position.x), bits(position.y), bits(position.z),
           bits(velocity.x), bits(velocity.y), bits(velocity.z));
    b3DestroyWorld(world);
    return 0;
}
