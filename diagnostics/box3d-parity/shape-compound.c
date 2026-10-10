// SPDX-License-Identifier: MIT
// Two ECS-authored body-local Shapes must match two native Box3D shapes on one body.
#include "physics_world.h"
#include "box3d/box3d.h"
#include "box3d/collision.h"

#include <stdio.h>

uint64_t b3HashWorldState( b3World* world );

static b3HullData* makeScaledUnitCube( float x, float y, float z )
{
	b3Vec3 points[8] = {
		{ -x, -y, -z }, { x, -y, -z }, { x, y, -z }, { -x, y, -z },
		{ -x, -y, z },  { x, -y, z },  { x, y, z },  { -x, y, z },
	};
	return b3CreateHull( points, 8, 8 );
}

int main( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.workerCount = 1;
	b3WorldId world = b3CreateWorld( &worldDef );

	b3BodyDef floorDef = b3DefaultBodyDef();
	floorDef.position = ( b3Pos ){ 0.0f, -0.5f, 0.0f };
	b3BodyId floor = b3CreateBody( world, &floorDef );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	b3HullData* floorHull = makeScaledUnitCube( 5.0f, 0.5f, 5.0f );
	b3CreateHullShape( floor, &shapeDef, floorHull );
	b3DestroyHull( floorHull );

	b3BodyDef dynamicDef = b3DefaultBodyDef();
	dynamicDef.type = b3_dynamicBody;
	dynamicDef.position = ( b3Pos ){ 0.0f, 0.4f, 0.0f };
	b3BodyId body = b3CreateBody( world, &dynamicDef );
	b3HullData* childHull = makeScaledUnitCube( 0.5f, 0.5f, 0.5f );
	b3CreateHullShape( body, &shapeDef, childHull );
	b3CreateHullShape( body, &shapeDef, childHull );
	b3DestroyHull( childHull );

	b3World_Step( world, 1.0f / 60.0f, 4 );
	b3Pos position = b3Body_GetPosition( body );
	printf( "%016llx %d %.9g\n", (unsigned long long)b3HashWorldState( b3GetWorldFromId( world ) ),
		(int)b3World_GetCounters( world ).shapeCount, position.y );
	b3DestroyWorld( world );
	return 0;
}
