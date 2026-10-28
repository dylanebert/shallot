// SPDX-License-Identifier: MIT
#include "physics_world.h"
#include "box3d/box3d.h"
#include <stdio.h>

uint64_t b3HashWorldState( b3World* world );

int main( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.workerCount = 1;
	b3WorldId world = b3CreateWorld( &worldDef );
	b3BodyDef bodyDef = b3DefaultBodyDef();
	b3BodyId sensor = b3CreateBody( world, &bodyDef );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.isSensor = true;
	shapeDef.enableSensorEvents = true;
	b3Sphere sphere = { { 0.0f, 0.0f, 0.0f }, 2.0f };
	b3CreateSphereShape( sensor, &shapeDef, &sphere );
	bodyDef.type = b3_dynamicBody;
	bodyDef.position.y = 1.0f;
	b3BodyId body = b3CreateBody( world, &bodyDef );
	shapeDef.isSensor = false;
	sphere.radius = 0.5f;
	b3CreateSphereShape( body, &shapeDef, &sphere );
	for ( int i = 0; i < 2; ++i )
	{
		b3World_Step( world, 1.0f / 120.0f, 4 );
		b3SensorEvents events = b3World_GetSensorEvents( world );
		printf( "0x%016llx %d %d\n", (unsigned long long)b3HashWorldState( b3GetWorldFromId( world ) ), events.beginCount, events.endCount );
	}
	b3DestroyWorld( world );
	return 0;
}
