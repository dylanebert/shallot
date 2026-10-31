// SPDX-License-Identifier: MIT
#include "physics_world.h"
#include "body.h"
#include "box3d/box3d.h"
#include <stdio.h>

uint64_t b3HashWorldState( b3World* world );

int main( void )
{
    b3WorldDef wd = b3DefaultWorldDef();
    wd.gravity = b3Vec3_zero;
    wd.enableSleep = false;
    b3WorldId world = b3CreateWorld( &wd );
    b3BodyDef bd = b3DefaultBodyDef();
    bd.type = b3_dynamicBody;
    bd.linearVelocity.x = 1.0f;
    bd.safetyFactor = 0.01f;
    b3BodyId body = b3CreateBody( world, &bd );
    b3ShapeDef sd = b3DefaultShapeDef();
    b3Sphere sphere = { { 0, 0, 0 }, 0.5f };
    b3CreateSphereShape( body, &sd, &sphere );
    for ( int i = 0; i < 3; ++i )
    {
        if ( i == 1 ) b3Body_SetSafetyFactor( body, 0.5f );
        if ( i == 2 ) b3Body_SetSafetyFactor( body, 0.01f );
        b3World_Step( world, 1.0f / 60.0f, 4 );
        b3World* w = b3GetWorldFromId( world );
        b3Body* b = b3GetBodyFullId( w, body );
        b3BodySim* sim = b3GetBodySim( w, b );
        printf( "%d %016llx\n", (sim->flags & b3_isFast) != 0, (unsigned long long)b3HashWorldState( w ) );
    }
    b3DestroyWorld( world );

    world = b3CreateWorld( &wd );
    bd = b3DefaultBodyDef();
    b3BodyId a = b3CreateBody( world, &bd );
    b3CreateSphereShape( a, &sd, &sphere );
    bd.type = b3_dynamicBody;
    bd.position.y = 0.9f;
    b3BodyId b = b3CreateBody( world, &bd );
    b3CreateSphereShape( b, &sd, &sphere );
    b3World_Step( world, 1.0f / 60.0f, 4 );
    printf( "%d\n", b3Body_GetContactCapacity( b ) );
    b3FilterJointDef jd = b3DefaultFilterJointDef();
    jd.base.bodyIdA = a;
    jd.base.bodyIdB = b;
    b3CreateFilterJoint( world, &jd );
    printf( "%d\n", b3Body_GetContactCapacity( b ) );
    b3World_Step( world, 1.0f / 60.0f, 4 );
    printf( "%016llx\n", (unsigned long long)b3HashWorldState( b3GetWorldFromId( world ) ) );
    b3DestroyWorld( world );
}
