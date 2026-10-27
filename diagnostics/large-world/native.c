// SPDX-License-Identifier: MIT
// Box3D's Large World through shared/benchmarks.c, timed as cost.ts times Shallot's: one line per phase.
#include "box3d/box3d.h"
#include "benchmarks.h"
#include <stdio.h>
#include <time.h>

typedef struct b3World b3World;
b3World* b3GetWorldFromId( b3WorldId );
uint64_t b3HashWorldState( b3World* );

static double now( void )
{
	struct timespec t;
	clock_gettime( CLOCK_MONOTONIC_RAW, &t );
	return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static void report( const char* phase, double start )
{
	printf( "{\"phase\":\"%s\",\"milliseconds\":%.6f,\"byteCount\":%d}\n", phase, now() - start, b3GetByteCount() );
}

int main( void )
{
	b3WorldDef def = b3DefaultWorldDef();
	def.workerCount = 1;
	GetLargeWorldCapacity( &def.capacity );
	double start = now();
	b3WorldId world = b3CreateWorld( &def );
	CreateLargeWorld( world );
	report( "create", start );
	const char* phases[] = { "first step", "second step" };
	for ( int step = 0; step < 2; ++step )
	{
		start = now();
		StepLargeWorld( world, step );
		b3World_Step( world, 1.0f / 60.0f, 4 );
		report( phases[step], start );
	}
	uint64_t hash = 0;
	start = now();
	for ( int i = 0; i < 20; ++i ) hash ^= b3HashWorldState( b3GetWorldFromId( world ) );
	report( "20 hashes", start );
	printf( "hash sentinel: %llu\n", (unsigned long long)hash );
	b3DestroyWorld( world );
	return 0;
}
