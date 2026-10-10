// SPDX-License-Identifier: MIT
// Native live-gravity setter oracle used by world-definition.oracle.ts.
#include "physics_world.h"
#include "box3d/box3d.h"

#include <stdio.h>

uint64_t b3HashWorldState( b3World* world );

int main( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.workerCount = 1;
	b3WorldId worldId = b3CreateWorld( &worldDef );
	b3BodyDef bodyDef = b3DefaultBodyDef();
	bodyDef.type = b3_dynamicBody;
	bodyDef.position.y = 10.0f;
	b3BodyId bodyId = b3CreateBody( worldId, &bodyDef );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	b3Sphere sphere = { { 0.0f, 0.0f, 0.0f }, 0.5f };
	b3CreateSphereShape( bodyId, &shapeDef, &sphere );

	b3World_Step( worldId, 1.0f / 60.0f, 4 );
	b3World_SetGravity( worldId, (b3Vec3){ 0.0f, -2.0f, 0.0f } );
	b3World_Step( worldId, 1.0f / 60.0f, 4 );
	printf( "0x%016llx\n", (unsigned long long)b3HashWorldState( b3GetWorldFromId( worldId ) ) );
	b3DestroyWorld( worldId );
	return 0;
}
