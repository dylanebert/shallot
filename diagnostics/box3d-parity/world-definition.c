// SPDX-License-Identifier: MIT
// Native default-world hash used by world-definition.oracle.ts.
#include "physics_world.h"
#include "box3d/box3d.h"

#include <stdio.h>

uint64_t b3HashWorldState( b3World* world );

int main( void )
{
	b3WorldDef definition = b3DefaultWorldDef();
	definition.workerCount = 1;
	b3WorldId id = b3CreateWorld( &definition );
	b3World* world = b3GetWorldFromId( id );
	printf( "0x%016llx\n", (unsigned long long)b3HashWorldState( world ) );
	b3DestroyWorld( id );
	return 0;
}
