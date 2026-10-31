#include "box3d/box3d.h"
#include "physics_world.h"

uint64_t b3HashWorldState(b3World* world);
#include <inttypes.h>
#include <stdio.h>

int main(void)
{
    b3WorldDef wd = b3DefaultWorldDef();
    wd.gravity = b3Vec3_zero;
    wd.isotropicInertiaTolerance = 0.9f;
    b3WorldId world = b3CreateWorld(&wd);
    b3BodyDef bd = b3DefaultBodyDef();
    bd.type = b3_dynamicBody;
    bd.angularVelocity = (b3Vec3){1, -2, 3};
    b3BodyId body = b3CreateBody(world, &bd);
    b3MassData mass = {1, {0, 0, 0}, {{0.4f, 0, 0}, {0, 0.5f, 0}, {0, 0, 0.6f}}};
    b3Body_SetMassData(body, mass);
    for (int i = 0; i < 10; ++i) b3World_Step(world, 1.0f / 60.0f, 4);
    printf("%" PRIu64 "\n", b3HashWorldState(b3GetWorldFromId(world)));
    b3DestroyWorld(world);
}
